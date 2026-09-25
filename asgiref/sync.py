import asyncio
import asyncio.coroutines
import contextvars
import functools
import inspect
import os
import sys
import threading
import warnings
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    ParamSpec,
    TypeVar,
    overload,
)

from .current_thread_executor import CurrentThreadExecutor
from .local import Local, _rehome, _Storage

if TYPE_CHECKING:
    # This is not available to import at runtime
    from _typeshed import OptExcInfo

_F = TypeVar("_F", bound=Callable[..., Any])
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _restore_context(context: contextvars.Context) -> None:
    # Check for changes in contextvars, and set them to the current
    # context for downstream consumers
    for cvar in context:
        cvalue = context.get(cvar)
        # asgiref is deliberately moving this context onto the current thread,
        # so re-home any Local storage to it. This keeps Local data visible
        # across async_to_sync / sync_to_async boundaries while leaving data
        # merely inherited by an unrelated thread isolated (see asgiref.local).
        if isinstance(cvalue, _Storage):
            cvalue = _rehome(cvalue)
        try:
            if cvar.get() != cvalue:
                cvar.set(cvalue)
        except LookupError:
            cvar.set(cvalue)


# Python 3.12 deprecates asyncio.iscoroutinefunction() as an alias for
# inspect.iscoroutinefunction(), whilst also removing the _is_coroutine marker.
# The latter is replaced with the inspect.markcoroutinefunction decorator.
# Until 3.12 is the minimum supported Python version, provide a shim.

if hasattr(inspect, "markcoroutinefunction"):
    iscoroutinefunction = inspect.iscoroutinefunction
    markcoroutinefunction: Callable[[_F], _F] = inspect.markcoroutinefunction
else:
    iscoroutinefunction = asyncio.iscoroutinefunction  # type: ignore[assignment]

    def markcoroutinefunction(func: _F) -> _F:
        func._is_coroutine = asyncio.coroutines._is_coroutine  # type: ignore
        return func


class AsyncSingleThreadContext:
    """Context manager to run async code inside the same thread.

    Normally, AsyncToSync functions run either inside a separate ThreadPoolExecutor or
    the main event loop if it exists. This context manager ensures that all AsyncToSync
    functions execute within the same thread.

    This context manager is re-entrant, so only the outer-most call to
    AsyncSingleThreadContext will set the context.

    Usage:

    >>> import asyncio
    >>> with AsyncSingleThreadContext():
    ...     async_to_sync(asyncio.sleep(1))()
    """

    def __init__(self):
        self.token = None

    def __enter__(self):
        try:
            AsyncToSync.async_single_thread_context.get()
        except LookupError:
            self.token = AsyncToSync.async_single_thread_context.set(self)

        return self

    def __exit__(self, exc, value, tb):
        if not self.token:
            return

        executor = AsyncToSync.context_to_thread_executor.pop(self, None)
        if executor:
            executor.shutdown()

        AsyncToSync.async_single_thread_context.reset(self.token)


class ThreadSensitiveContext:
    """Async context manager to manage context for thread sensitive mode

    This context manager controls which thread pool executor is used when in
    thread sensitive mode. By default, a single thread pool executor is shared
    within a process.

    The ThreadSensitiveContext() context manager may be used to specify a
    thread pool per context.

    By default, only the outer-most ThreadSensitiveContext sets the context.
    With force_new_thread=True, this block uses a new thread even inside another
    context or an AsyncToSync call. On exit, the parent context is restored.
    Use a separate instance for each active force_new_thread=True block.

    Usage:

    >>> import time
    >>> async with ThreadSensitiveContext():
    ...     await sync_to_async(time.sleep, 1)()
    """

    def __init__(self, *, force_new_thread: bool = False) -> None:
        self.force_new_thread = force_new_thread
        self.token: contextvars.Token[ThreadSensitiveContext] | None = None
        self._old_executor: CurrentThreadExecutor | None = None

    async def __aenter__(self):
        if self.force_new_thread:

            """
            (temporary, remove before merge)

            =============
            EXPLANATION 1
            =============

            Ok, story time. Why oh why are we juggling executors here inside `ThreadSensitiveContext`?

            Let's establish some observations:

            - We want to allow `async with transaction.atomic():`
            - We'll implement the async support with something like this (on `Atomic`). Let's start
              WITHOUT transaction-specific isolation, so we can see what's missing.
              These sketches show one entry/exit pair, not a complete implementation. Django's task
              tracking and full entry-failure/cancellation cleanup are omitted. Reusing an `Atomic`
              instance also needs per-entry state, rather than the simple instance attributes below.

            ```
            async def __aenter__(self):
                self._tsc = ThreadSensitiveContext()
                await self._tsc.__aenter__()
                await sync_to_async(self.__enter__)()       # existing sync logic, on the selected thread

            async def __aexit__(self, *exc):
                try:
                    await sync_to_async(self.__exit__)(*exc)
                finally:
                    await self._tsc.__aexit__(*exc)
            ```

            - Note we are first entering ThreadSensitiveContext, and thereafter calling `sync_to_async`.
              Entering the context does not itself create a thread.
            - Which thread does `sync_to_async` use? It depends:
                - If `thread_sensitive==False` => the supplied executor, or the event loop's default executor
                - Otherwise, if a parent `async_to_sync` executor is visible => the waiting parent sync thread
                - Otherwise, the executor of the current `ThreadSensitiveContext`, if one is active
                - Otherwise, a waiting sync thread registered for this event loop by `async_to_sync`, if any
                - Otherwise, asgiref's shared single-worker executor, unless its deadlock check rejects the call

            - Special note about the executor of the current `ThreadSensitiveContext`: If there is no
              executor created yet, `sync_to_async` creates one. Submitting the first call starts its worker.
              Later calls in the same context reuse that worker. In Django ASGI, request handlers are already
              wrapped in `ThreadSensitiveContext`! A plain nested context reuses that request context.
              The context marker is already set, even if its worker has not started yet.

            - The problem with our first example is that the thread being used in the `sync_to_async` call
              IS NOT reserved for this transaction. Other tasks in the request can use it too.
              A thread living longer than a transaction is not itself a problem. Sharing its connection
              with other tasks WHILE THE TRANSACTION IS OPEN is the problem.

            - "So what?", you might say. Well, the issue lies with how Django hands out connections.
              On a sync worker, Django keeps one connection wrapper per database alias, per thread.
              This is thread-local storage, not necessarily a connection pool.
              Now, if we have THE SAME THREAD being used inside and outside of the `transaction.atomic`
              scope, those calls can use the same db connection.

            - This means that multiple async tasks can submit db operations to that same thread and
              connection. The sync calls run one at a time, BUT a transaction spans several calls:
              begin, database work, and commit/rollback. An `await` between these calls lets another
              task submit work on the same connection while the first task's transaction is still open.

            - A ROLLBACK would then undo writes made by those other tasks during that transaction,
              even though they are not part of the same `transaction.atomic` scope. This is a very bad thing.

            - Yes, this is complex. It's not just you. Here is an example of the problem from the
              user perspective:

                  ======================================================================
                     Imagine two tasks started by the same async view. Both use the same
                     database alias.                              
                                                                                                                                           
                     Assume a basic implementation of async atomic() that does not create
                     a separate thread-sensitive context.             
                                                                                                                                           
                     The events below make the order predictable:                                                                          
                                                                                                                                           
                     ```python                                                                                                             
                       transaction_started = asyncio.Event()                                                                               
                       other_write_finished = asyncio.Event()                                                                              
                                                                                                                                           
                       async def task_one():                                                                                               
                           try:                                                                                                            
                               async with transaction.atomic():                                                                            
                                   await Message.objects.acreate(text="From task one")                                                     
                                   transaction_started.set()                                                                               
                                                                                                                                           
                                   # Pause here while the other task writes.                                                               
                                   await other_write_finished.wait()                                                                       
                                                                                                                                           
                                   raise ValueError("Undo my work")                                                                        
                           except ValueError:                                                                                              
                               pass                                                                                                        
                                                                                                                                           
                       async def task_two():                                                                                               
                           await transaction_started.wait()                                                                                
                                                                                                                                           
                           # This is OUTSIDE the atomic block.                                                                             
                           await Message.objects.acreate(text="From task two")                                                             
                                                                                                                                           
                           other_write_finished.set()                                                                                      
                                                                                                                                           
                       await asyncio.gather(task_one(), task_two())                                                                        
                     ```                                                                                                                   
                                                                                                                                           
                     ### What the application author expects                                                                               
                                                                                                                                           
                     - Task one’s message is rolled back.                                                                                  
                     - Task two’s message remains—it was written outside the atomic() block.                                               
                                                                                                                                           
                     ### What the shared database connection actually sees                                                                 
                                                                                                                                           
                     ```text                                                                                                               
                       Task one:  BEGIN                                                                                                    
                       Task one:  INSERT "From task one"                                                                                   
                                  ── task one pauses; the worker is available ──                                                           
                       Task two:  INSERT "From task two"                                                                                   
                                  ── task one resumes and raises ValueError ──                                                             
                       Task one:  ROLLBACK                            
                  ======================================================================

            - What we need is a way to give each INDEPENDENT async transaction its own connection.
              Nested `atomic()` blocks in the owning task must keep that connection and use savepoints.
              Subtasks created inside the transaction still inherit its context, so Django must separately
              enforce transaction ownership and require those subtasks to finish before the transaction exits.
            - BECAUSE Django finds connections by worker thread, we can use its existing connection storage
              by giving each independent transaction its own thread. NOT a new thread for each savepoint.

            - This is what `force_new_thread=True` does. Let's update the impl
              of `async with transaction.atomic():` example above:

            ```
            async def __aenter__(self):
                self._outermost = not in_async_atomic_block()   # Django-side task/transaction tracking

                self._tsc = ThreadSensitiveContext(force_new_thread=self._outermost)  # <====== NEW NEW NEW

                await self._tsc.__aenter__()
                await sync_to_async(self.__enter__)()       # new thread for the independent transaction

            async def __aexit__(self, *exc):
                try:
                    try:
                        await sync_to_async(self.__exit__)(*exc)
                    finally:
                        if self._outermost:
                            await sync_to_async(lambda: get_connection(self.using).close())()
                finally:
                    await self._tsc.__aexit__(*exc)
            ```

            - If `force_new_thread==True`, then a NEW context marker is set. The first `sync_to_async`
              call that selects this context MUST create its executor; later calls reuse it.
              The transaction's entry, database work, exit, and connection cleanup all use that worker.
              Notice that the connection lookup for `close()` is INSIDE the sync function, not on the event loop.

            - The Django machinery will check for a connection wrapper on that worker, find nothing,
              and create one. The underlying database connection is obtained when transaction entry needs it.
              That worker and connection are then used for the transaction, including its nested savepoints.

            - Now we get into some tricky details: when we call `async with transaction.atomic():`, we will
              usually already have a parent `ThreadSensitiveContext`. Its executor stays in the context-to-executor
              dictionary; `self.token` lets us restore that parent context on exit. We do not need to save its pool.
              BUT if we reached this async code through `async_to_sync`, there may ALSO be a `CurrentThreadExecutor`
              pointing back to a waiting sync thread. This is a DIFFERENT executor, and `sync_to_async` checks it
              BEFORE the thread-sensitive context. Just setting our new context would still send work to that
              parent thread! We need to hide this executor, but we don't want to LOSE its reference.

            - So this is why we save that `CurrentThreadExecutor` in `self._old_executor`, clear the visible
              reference, and restore it on exit. This change applies to the current context, not all tasks.
              We must not simply reverse the executor selection order: our new worker might itself call
              `async_to_sync` and wait for async code that calls back into sync code. Those callbacks need
              its NEW `CurrentThreadExecutor`; putting them on its ordinary pool would queue them behind
              the very call that is waiting for them. Clearing only the inherited executor at entry lets
              these later bridges work normally, without sending our transaction back to the parent thread.

            =============
            EXPLANATION 2
            =============

            (AI)

            An async transaction spans several synchronous calls: transaction entry, database operations, and transaction exit. 
            These calls must use the same connection. However, other tasks must not accidentally run their operations on that   
            connection while its transaction is open.                                                                           
                                                                                                                                
            Django already stores connections by worker thread, so an independent transaction can get its own connection by     
            using its own thread-sensitive context.                                                                             
                                                                                                                                
            Entering that context does not immediately create a thread. It sets the context that subsequent sync_to_async calls 
            will use. The first call creates its single-worker executor; later calls reuse it.                                  
                                                                                                                                
            But setting a new context is not sufficient.                                                                        
                                                                                                                                
            If we reached this async code through async_to_sync, there may already be a CurrentThreadExecutor pointing back to  
            the waiting sync thread. SyncToAsync checks for that executor before checking the thread-sensitive context. Without 
            another change, our supposedly independent transaction would still run on the parent thread.                        
                                                                                                                                
            We therefore save that inherited executor in _old_executor and temporarily clear AsyncToSync.executors.current.     
            This lets the existing selection logic reach our new context and use its worker.                                    
                                                                                                                                
            Why not change the selection order instead?                                                                         
                                                                                                                                
            Code running on the new worker may itself call async_to_sync. That worker then waits for async code, which may call 
            back into sync code. Those callbacks must use the new worker’s CurrentThreadExecutor. Queuing them on its ordinary  
            thread pool would deadlock: the worker would be waiting for work queued behind itself.                              
                                                                                                                                
            Clearing the inherited executor at context entry handles both cases. It prevents a return to the parent thread,     
            while allowing bridges created inside the new scope to install their own executors normally.                        
                                                                                                                                
            On exit, self.token restores the parent thread-sensitive context, and _old_executor restores the saved bridge       
            executor. These restore two different pieces of state.  

            """

            if self.token is not None:
                raise RuntimeError("ThreadSensitiveContext is already entered")
            self.token = SyncToAsync.thread_sensitive_context.set(self)
            # A parent AsyncToSync executor would otherwise take priority over
            # this context. Hide it in this task, but let AsyncToSync calls
            # inside the new worker install their own executors as usual.
            self._old_executor = getattr(AsyncToSync.executors, "current", None)
            AsyncToSync.executors.current = None
        else:
            try:
                SyncToAsync.thread_sensitive_context.get()
            except LookupError:
                self.token = SyncToAsync.thread_sensitive_context.set(self)

        return self

    async def __aexit__(self, exc, value, tb):
        if not self.token:
            return

        executor = SyncToAsync.context_to_thread_executor.pop(self, None)
        SyncToAsync.thread_sensitive_context.reset(self.token)
        self.token = None
        if self.force_new_thread:
            AsyncToSync.executors.current = self._old_executor
            self._old_executor = None
        if executor:
            # The executor's worker thread may itself be waiting for this
            # event loop, so a blocking shutdown() here would deadlock it.
            # Join in a dedicated thread, not the loop's default executor:
            # work queued there may itself be needed to unpark the worker,
            # and joins occupying its slots would starve it.
            future: "Future[None]" = Future()

            def join() -> None:
                executor.shutdown()
                try:
                    future.set_result(None)
                except InvalidStateError:
                    # The await below was cancelled while we were joining.
                    pass

            threading.Thread(target=join, daemon=True).start()
            await asyncio.wrap_future(future)


class AsyncToSync(Generic[_P, _R]):
    """
    Utility class which turns an awaitable that only works on the thread with
    the event loop into a synchronous callable that works in a subthread.

    If the call stack contains an async loop, the code runs there.
    Otherwise, the code runs in a new loop in a new thread.

    Either way, this thread then pauses and waits to run any thread_sensitive
    code called from further down the call stack using SyncToAsync, before
    finally exiting once the async task returns.
    """

    # Keeps a reference to the CurrentThreadExecutor in local context, so that
    # any sync_to_async inside the wrapped code can find it.
    executors: "Local" = Local()

    # When we can't find a CurrentThreadExecutor from the context, such as
    # inside create_task, we'll look it up here from the running event loop.
    loop_thread_executors: "Dict[asyncio.AbstractEventLoop, CurrentThreadExecutor]" = {}

    async_single_thread_context: "contextvars.ContextVar[AsyncSingleThreadContext]" = (
        contextvars.ContextVar("async_single_thread_context")
    )

    context_to_thread_executor: (
        "weakref.WeakKeyDictionary[AsyncSingleThreadContext, ThreadPoolExecutor]"
    ) = weakref.WeakKeyDictionary()

    def __init__(
        self,
        awaitable: Callable[_P, Coroutine[Any, Any, _R]] | Callable[_P, Awaitable[_R]],
        force_new_loop: bool = False,
    ):
        if not callable(awaitable) or (
            not iscoroutinefunction(awaitable)
            and not iscoroutinefunction(getattr(awaitable, "__call__", awaitable))
        ):
            # Python does not have very reliable detection of async functions
            # (lots of false negatives) so this is just a warning.
            warnings.warn(
                "async_to_sync was passed a non-async-marked callable", stacklevel=2
            )
        self.awaitable = awaitable
        try:
            self.__self__ = self.awaitable.__self__  # type: ignore[union-attr]
        except AttributeError:
            pass
        self.force_new_loop = force_new_loop

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        __traceback_hide__ = True  # noqa: F841

        main_event_loop = None
        if not self.force_new_loop:
            # There's no event loop in this thread. Look for the threadlocal if
            # we're inside SyncToAsync
            main_event_loop_pid = getattr(
                SyncToAsync.threadlocal, "main_event_loop_pid", None
            )
            # We make sure the parent loop is from the same process - if
            # they've forked, this is not going to be valid any more (#194)
            if main_event_loop_pid and main_event_loop_pid == os.getpid():
                main_event_loop = getattr(
                    SyncToAsync.threadlocal, "main_event_loop", None
                )

        # You can't call AsyncToSync from a thread with a running event loop
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "You cannot use AsyncToSync in the same thread as an async event loop - "
                "just await the async function directly."
            )

        # Make a future for the return information
        call_result: "Future[_R]" = Future()

        # Make a CurrentThreadExecutor we'll use to idle in this thread - we
        # need one for every sync frame, even if there's one above us in the
        # same thread.
        old_executor = getattr(self.executors, "current", None)
        current_executor = CurrentThreadExecutor(old_executor)
        self.executors.current = current_executor

        # Wrapping context in list so it can be reassigned from within
        # `main_wrap`.
        context = [contextvars.copy_context()]

        # Get task context so that parent task knows which task to propagate
        # an asyncio.CancelledError to.
        task_context = getattr(SyncToAsync.threadlocal, "task_context", None)

        # Use call_soon_threadsafe to schedule a synchronous callback on the
        # main event loop's thread if it's there, otherwise make a new loop
        # in this thread.
        try:
            awaitable = self.main_wrap(
                call_result,
                sys.exc_info(),
                task_context,
                context,
                # prepare an awaitable which can be passed as is to self.main_wrap,
                # so that `args` and `kwargs` don't need to be
                # destructured when passed to self.main_wrap
                # (which is required by `ParamSpec`)
                # as that may cause overlapping arguments
                self.awaitable(*args, **kwargs),
            )

            async def new_loop_wrap() -> None:
                loop = asyncio.get_running_loop()
                self.loop_thread_executors[loop] = current_executor
                try:
                    await awaitable
                finally:
                    del self.loop_thread_executors[loop]

            if main_event_loop is not None:
                try:
                    main_event_loop.call_soon_threadsafe(
                        main_event_loop.create_task, awaitable
                    )
                except RuntimeError:
                    running_in_main_event_loop = False
                else:
                    running_in_main_event_loop = True
                    # Run the CurrentThreadExecutor until the future is done.
                    current_executor.run_until_future(call_result)
            else:
                running_in_main_event_loop = False

            if not running_in_main_event_loop:
                loop_executor = None

                if self.async_single_thread_context.get(None):
                    single_thread_context = self.async_single_thread_context.get()

                    if single_thread_context in self.context_to_thread_executor:
                        loop_executor = self.context_to_thread_executor[
                            single_thread_context
                        ]
                    else:
                        loop_executor = ThreadPoolExecutor(max_workers=1)
                        self.context_to_thread_executor[single_thread_context] = (
                            loop_executor
                        )
                else:
                    # Make our own event loop - in a new thread - and run inside that.
                    loop_executor = ThreadPoolExecutor(max_workers=1)

                loop_future = loop_executor.submit(asyncio.run, new_loop_wrap())
                # Run the CurrentThreadExecutor until the future is done.
                current_executor.run_until_future(loop_future)
                # Wait for future and/or allow for exception propagation
                loop_future.result()
        finally:
            _restore_context(context[0])
            # Restore old current thread executor state
            self.executors.current = old_executor

        # Wait for results from the future.
        return call_result.result()

    def __get__(self, parent: Any, objtype: Any) -> Callable[_P, _R]:
        """
        Include self for methods
        """
        func = functools.partial(self.__call__, parent)
        return functools.update_wrapper(func, self.awaitable)

    async def main_wrap(
        self,
        call_result: "Future[_R]",
        exc_info: "OptExcInfo",
        task_context: "Optional[List[asyncio.Task[Any]]]",
        context: list[contextvars.Context],
        awaitable: Coroutine[Any, Any, _R] | Awaitable[_R],
    ) -> None:
        """
        Wraps the awaitable with something that puts the result into the
        result/exception future.
        """

        __traceback_hide__ = True  # noqa: F841

        if context is not None:
            _restore_context(context[0])

        current_task = asyncio.current_task()
        if current_task is not None and task_context is not None:
            task_context.append(current_task)

        try:
            # If we have an exception, run the function inside the except block
            # after raising it so exc_info is correctly populated.
            if exc_info[1]:
                try:
                    raise exc_info[1]
                except BaseException:
                    result = await awaitable
            else:
                result = await awaitable
        except BaseException as e:
            call_result.set_exception(e)
        else:
            call_result.set_result(result)
        finally:
            if current_task is not None and task_context is not None:
                task_context.remove(current_task)
            context[0] = contextvars.copy_context()


class SyncToAsync(Generic[_P, _R]):
    """
    Utility class which turns a synchronous callable into an awaitable that
    runs in a threadpool. It also sets a threadlocal inside the thread so
    calls to AsyncToSync can escape it.

    If thread_sensitive is passed, the code will run in the same thread as any
    outer code. This is needed for underlying Python code that is not
    threadsafe (for example, code which handles SQLite database connections).

    If the outermost program is async (i.e. SyncToAsync is outermost), then
    this will be a dedicated single sub-thread that all sync code runs in,
    one after the other. If the outermost program is sync (i.e. AsyncToSync is
    outermost), this will just be the main thread. This is achieved by idling
    with a CurrentThreadExecutor while AsyncToSync is blocking its sync parent,
    rather than just blocking.

    If executor is passed in, that will be used instead of the loop's default executor.
    In order to pass in an executor, thread_sensitive must be set to False, otherwise
    a TypeError will be raised.
    """

    # Storage for main event loop references
    threadlocal = threading.local()

    # Single-thread executor for thread-sensitive code
    single_thread_executor = ThreadPoolExecutor(max_workers=1)

    # Maintain a contextvar for the current execution context. Optionally used
    # for thread sensitive mode.
    thread_sensitive_context: "contextvars.ContextVar[ThreadSensitiveContext]" = (
        contextvars.ContextVar("thread_sensitive_context")
    )

    # Contextvar that is used to detect if the single thread executor
    # would be awaited on while already being used in the same context
    deadlock_context: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
        "deadlock_context"
    )

    # Maintaining a weak reference to the context ensures that thread pools are
    # erased once the context goes out of scope. This terminates the thread pool.
    context_to_thread_executor: (
        "weakref.WeakKeyDictionary[ThreadSensitiveContext, ThreadPoolExecutor]"
    ) = weakref.WeakKeyDictionary()

    def __init__(
        self,
        func: Callable[_P, _R],
        thread_sensitive: bool = True,
        executor: Optional["ThreadPoolExecutor"] = None,
        context: contextvars.Context | None = None,
    ) -> None:
        if (
            not callable(func)
            or iscoroutinefunction(func)
            or iscoroutinefunction(getattr(func, "__call__", func))
        ):
            raise TypeError("sync_to_async can only be applied to sync functions.")

        functools.update_wrapper(self, func)
        self.func = func
        self.context = context

        self._thread_sensitive = thread_sensitive
        markcoroutinefunction(self)
        if thread_sensitive and executor is not None:
            raise TypeError("executor must not be set when thread_sensitive is True")
        self._executor = executor
        try:
            self.__self__ = func.__self__  # type: ignore
        except AttributeError:
            pass

    async def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        __traceback_hide__ = True  # noqa: F841
        loop = asyncio.get_running_loop()

        # Work out what thread to run the code in
        if self._thread_sensitive:
            current_thread_executor = getattr(AsyncToSync.executors, "current", None)
            if current_thread_executor:
                # If we have a parent sync thread above somewhere, use that
                executor = current_thread_executor
            elif self.thread_sensitive_context.get(None):
                # If we have a way of retrieving the current context, attempt
                # to use a per-context thread pool executor
                thread_sensitive_context = self.thread_sensitive_context.get()

                if thread_sensitive_context in self.context_to_thread_executor:
                    # Re-use thread executor in current context
                    executor = self.context_to_thread_executor[thread_sensitive_context]
                else:
                    # Create new thread executor in current context
                    executor = ThreadPoolExecutor(max_workers=1)
                    self.context_to_thread_executor[thread_sensitive_context] = executor
            elif loop in AsyncToSync.loop_thread_executors:
                # Re-use thread executor for running loop
                executor = AsyncToSync.loop_thread_executors[loop]
            elif self.deadlock_context.get(False):
                raise RuntimeError(
                    "Single thread executor already being used, would deadlock"
                )
            else:
                # Otherwise, we run it in a fixed single thread
                executor = self.single_thread_executor
                self.deadlock_context.set(True)
        else:
            # Use the passed in executor, or the loop's default if it is None
            executor = self._executor

        context = contextvars.copy_context() if self.context is None else self.context
        # ``child`` is the deferred sync function to be run, with its args
        # and kwargs bound.
        child = functools.partial(self.func, *args, **kwargs)

        # On the worker thread, thread_handler runs ``func(child)``. ``func``
        # enters ``context`` (via context.run); then, inside it, ``run_child``
        # re-homes any Local storage to the worker thread so it stays visible
        # there (see _restore_context), and finally calls ``child``.
        def func(child: Callable[[], _R]) -> _R:
            def run_child() -> _R:
                _restore_context(context)
                return child()

            return context.run(run_child)

        task_context: list[asyncio.Task[Any]] = []

        # Run the code in the right thread
        exec_coro = loop.run_in_executor(
            executor,
            functools.partial(
                self.thread_handler,
                loop,
                sys.exc_info(),
                task_context,
                func,
                child,
            ),
        )
        ret: _R
        try:
            ret = await asyncio.shield(exec_coro)
        except asyncio.CancelledError:
            cancel_parent = True
            try:
                task = task_context[0]
                task.cancel()
                try:
                    await task
                    cancel_parent = False
                except asyncio.CancelledError:
                    pass
            except IndexError:
                pass
            if exec_coro.done():
                raise
            if cancel_parent:
                exec_coro.cancel()
            ret = await exec_coro
        finally:
            if self.context is None:
                _restore_context(context)
            self.deadlock_context.set(False)

        return ret

    def __get__(
        self, parent: Any, objtype: Any
    ) -> Callable[_P, Coroutine[Any, Any, _R]]:
        """
        Include self for methods
        """
        func = functools.partial(self.__call__, parent)
        return functools.update_wrapper(func, self.func)

    def thread_handler(self, loop, exc_info, task_context, func, *args, **kwargs):
        """
        Wraps the sync application with exception handling.
        """

        __traceback_hide__ = True  # noqa: F841

        # Set the threadlocal for AsyncToSync
        self.threadlocal.main_event_loop = loop
        self.threadlocal.main_event_loop_pid = os.getpid()
        self.threadlocal.task_context = task_context

        # Run the function
        # If we have an exception, run the function inside the except block
        # after raising it so exc_info is correctly populated.
        if exc_info[1]:
            try:
                raise exc_info[1]
            except BaseException:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


@overload
def async_to_sync(
    *,
    force_new_loop: bool = False,
) -> Callable[
    [Callable[_P, Coroutine[Any, Any, _R]] | Callable[_P, Awaitable[_R]]],
    Callable[_P, _R],
]: ...


@overload
def async_to_sync(
    awaitable: Callable[_P, Coroutine[Any, Any, _R]] | Callable[_P, Awaitable[_R]],
    *,
    force_new_loop: bool = False,
) -> Callable[_P, _R]: ...


def async_to_sync(
    awaitable: None | (
        Callable[_P, Coroutine[Any, Any, _R]] | Callable[_P, Awaitable[_R]]
    ) = None,
    *,
    force_new_loop: bool = False,
) -> (
    Callable[
        [Callable[_P, Coroutine[Any, Any, _R]] | Callable[_P, Awaitable[_R]]],
        Callable[_P, _R],
    ]
    | Callable[_P, _R]
):
    if awaitable is None:
        return lambda f: AsyncToSync(
            f,
            force_new_loop=force_new_loop,
        )
    return AsyncToSync(
        awaitable,
        force_new_loop=force_new_loop,
    )


@overload
def sync_to_async(
    *,
    thread_sensitive: bool = True,
    executor: Optional["ThreadPoolExecutor"] = None,
    context: contextvars.Context | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, Coroutine[Any, Any, _R]]]: ...


@overload
def sync_to_async(
    func: Callable[_P, _R],
    *,
    thread_sensitive: bool = True,
    executor: Optional["ThreadPoolExecutor"] = None,
    context: contextvars.Context | None = None,
) -> Callable[_P, Coroutine[Any, Any, _R]]: ...


def sync_to_async(
    func: Callable[_P, _R] | None = None,
    *,
    thread_sensitive: bool = True,
    executor: Optional["ThreadPoolExecutor"] = None,
    context: contextvars.Context | None = None,
) -> (
    Callable[[Callable[_P, _R]], Callable[_P, Coroutine[Any, Any, _R]]]
    | Callable[_P, Coroutine[Any, Any, _R]]
):
    if func is None:
        return lambda f: SyncToAsync(
            f,
            thread_sensitive=thread_sensitive,
            executor=executor,
            context=context,
        )
    return SyncToAsync(
        func,
        thread_sensitive=thread_sensitive,
        executor=executor,
        context=context,
    )
