import asyncio
import contextvars
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from asgiref.local import Local
from asgiref.sync import (
    AsyncToSync,
    SyncToAsync,
    ThreadSensitiveContext,
    async_to_sync,
    sync_to_async,
)


@pytest.mark.asyncio
async def test_force_new_context_restores_parent() -> None:
    thread = sync_to_async(threading.current_thread)

    async with ThreadSensitiveContext() as parent:
        parent_thread = await thread()
        async with ThreadSensitiveContext(force_new_thread=True) as child:
            child_thread = await thread()
            assert child_thread is not parent_thread
            assert await thread() is child_thread
            async with ThreadSensitiveContext():
                assert await thread() is child_thread
            assert SyncToAsync.thread_sensitive_context.get() is child

            async with ThreadSensitiveContext(force_new_thread=True) as grandchild:
                grandchild_thread = await thread()
                assert grandchild_thread not in (parent_thread, child_thread)
            assert grandchild not in SyncToAsync.context_to_thread_executor
            assert not grandchild_thread.is_alive()
            assert await thread() is child_thread

        assert child not in SyncToAsync.context_to_thread_executor
        assert not child_thread.is_alive()
        assert parent_thread.is_alive()
        assert SyncToAsync.thread_sensitive_context.get() is parent
        assert await thread() is parent_thread


@pytest.mark.asyncio
async def test_force_new_context_without_parent() -> None:
    thread = sync_to_async(threading.current_thread)
    parent_thread = await thread()
    async with ThreadSensitiveContext(force_new_thread=True) as child:
        child_thread = await thread()
        assert child_thread is not parent_thread
    assert child not in SyncToAsync.context_to_thread_executor
    assert SyncToAsync.thread_sensitive_context.get(None) is None
    assert not child_thread.is_alive()
    assert await thread() is parent_thread


@pytest.mark.asyncio
async def test_force_new_context_siblings_are_isolated() -> None:
    thread = sync_to_async(threading.current_thread)
    both_inside = asyncio.Barrier(2)

    async def sibling() -> threading.Thread:
        async with ThreadSensitiveContext(force_new_thread=True):
            first = await thread()
            await both_inside.wait()
            # Tasks inherit the context unless they explicitly create another.
            assert await asyncio.create_task(thread()) is first
            return first

    async with ThreadSensitiveContext():
        parent_thread = await thread()
        first, second = await asyncio.wait_for(
            asyncio.gather(sibling(), sibling()), timeout=5
        )
        assert first is not second
        assert parent_thread not in (first, second)
        assert await thread() is parent_thread


def force_new_context_across_bridges(async_outermost: bool) -> None:
    thread = sync_to_async(threading.current_thread)

    async def inner_async(expected: threading.Thread) -> None:
        assert await thread() is expected
        assert await asyncio.create_task(thread()) is expected
        async with ThreadSensitiveContext(force_new_thread=True):
            other = await thread()
            assert other is not expected
        assert await thread() is expected

    def inner_sync() -> threading.Thread:
        current = threading.current_thread()
        async_to_sync(inner_async)(current)
        return current

    async def outer_async(parent_thread: threading.Thread) -> None:
        parent_executor = AsyncToSync.executors.current
        assert await thread() is parent_thread
        async with ThreadSensitiveContext(force_new_thread=True):
            child_thread = await thread()
            assert child_thread is not parent_thread
            assert await sync_to_async(inner_sync)() is child_thread
        assert AsyncToSync.executors.current is parent_executor
        assert await thread() is parent_thread

    def outer_sync() -> None:
        async_to_sync(outer_async)(threading.current_thread())

    if async_outermost:
        asyncio.run(sync_to_async(outer_sync)())
    else:
        outer_sync()


@pytest.mark.parametrize("async_outermost", [False, True])
def test_force_new_context_across_bridges(async_outermost: bool) -> None:
    # A deadlocked worker can also block interpreter shutdown. Run the bridge
    # checks in a separate process so the parent can stop it after a timeout.
    # Spawn avoids inheriting executor state from the parent's worker threads.
    process = multiprocessing.get_context("spawn").Process(
        target=force_new_context_across_bridges, args=(async_outermost,)
    )
    process.start()
    try:
        process.join(30)
        if process.is_alive():
            pytest.fail("thread-sensitive context deadlocked across sync/async bridges")
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
        process.close()


@pytest.mark.asyncio
async def test_force_new_context_isolates_thread_critical_storage() -> None:
    # Django's ConnectionHandler uses this storage for database wrappers.
    connections = Local(thread_critical=True)
    shared = Local()
    value: contextvars.ContextVar[str] = contextvars.ContextVar("value")
    shared.value = "parent"
    value.set("parent")

    @sync_to_async
    def connection() -> Any:
        if not hasattr(connections, "default"):
            connections.default = object()
        return connections.default

    @sync_to_async
    def update_context() -> None:
        assert shared.value == "parent"
        assert value.get() == "parent"
        shared.value = "child"
        value.set("child")

    async with ThreadSensitiveContext():
        parent = await connection()
        async with ThreadSensitiveContext(force_new_thread=True):
            child = await connection()
            assert child is not parent
            assert await connection() is child
            await update_context()
        assert await connection() is parent
        assert shared.value == "child"
        assert value.get() == "child"


@pytest.mark.parametrize("error", [ValueError, asyncio.CancelledError])
def test_force_new_context_restores_parent_on_error(error: type[BaseException]) -> None:
    async def run() -> None:
        parent_executor = AsyncToSync.executors.current
        async with ThreadSensitiveContext() as parent:
            parent_thread = await sync_to_async(threading.current_thread)()
            with pytest.raises(error):
                async with ThreadSensitiveContext(force_new_thread=True) as child:
                    child_thread = await sync_to_async(threading.current_thread)()
                    raise error()
            assert SyncToAsync.thread_sensitive_context.get() is parent
            assert AsyncToSync.executors.current is parent_executor
            assert child not in SyncToAsync.context_to_thread_executor
            assert not child_thread.is_alive()
            assert await sync_to_async(threading.current_thread)() is parent_thread

    async_to_sync(run)()


@pytest.mark.asyncio
async def test_force_new_context_cancellation_waits_without_blocking() -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    released = []
    threads = []

    def blocking() -> None:
        threads.append(threading.current_thread())
        loop.call_soon_threadsafe(started.set)
        released.append(release.wait(5))

    async with ThreadSensitiveContext() as parent:
        child = ThreadSensitiveContext(force_new_thread=True)

        async def run() -> None:
            try:
                async with child:
                    await sync_to_async(blocking)()
            finally:
                assert SyncToAsync.thread_sensitive_context.get() is parent

        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            # This callback must run while context exit waits for the worker.
            loop.call_soon(release.set)
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert released == [True]
        assert not threads[0].is_alive()
        assert child not in SyncToAsync.context_to_thread_executor


@pytest.mark.asyncio
async def test_force_new_context_leaves_non_thread_sensitive_calls_unchanged() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        thread = sync_to_async(
            threading.current_thread, thread_sensitive=False, executor=executor
        )
        expected = await thread()
        async with ThreadSensitiveContext(force_new_thread=True):
            assert await thread() is expected
            assert await sync_to_async(threading.current_thread)() is not expected


@pytest.mark.asyncio
async def test_force_new_context_reuse() -> None:
    context = ThreadSensitiveContext(force_new_thread=True)
    async with context:
        first = await sync_to_async(threading.current_thread)()
        with pytest.raises(RuntimeError, match="already entered"):
            async with context:
                pass
    async with context:
        second = await sync_to_async(threading.current_thread)()
        assert first is not second


@pytest.mark.asyncio
async def test_force_new_context_without_sync_work() -> None:
    async with ThreadSensitiveContext() as parent:
        async with ThreadSensitiveContext(force_new_thread=True) as child:
            assert SyncToAsync.thread_sensitive_context.get() is child
        assert SyncToAsync.thread_sensitive_context.get() is parent
        assert child not in SyncToAsync.context_to_thread_executor
