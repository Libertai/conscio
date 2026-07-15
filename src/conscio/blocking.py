"""Bounded executors for blocking service work.

The service has several operations that must not run on the asyncio event loop
(filesystem archives, CPU training, and DNS resolution), while some supported
hosts cannot reliably use asyncio's shared default executor.  This module owns
small, dedicated pools and applies backpressure *before* work is submitted.

Cancellation never retries a submitted callable.  Once a job reaches an
executor it continues to occupy its capacity slot until the worker finishes,
even if its awaiting coroutine is cancelled or its deadline expires.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import Any, TypeVar, cast

R = TypeVar("R")

_CURRENT_RUNNER: ContextVar[BoundedBlockingRunner | None] = ContextVar(
    "conscio_blocking_runner",
    default=None,
)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _queue_size(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _deadline_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("deadline must be finite and positive")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("deadline must be finite and positive")
    return seconds


class _BoundedPool:
    """One explicit executor plus a bound on running and queued work."""

    def __init__(self, *, name: str, max_workers: int, max_queue: int) -> None:
        workers = _positive_int(max_workers, f"{name}_workers")
        queue = _queue_size(max_queue, f"{name}_queue")
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"conscio-{name}",
        )
        self._slots = threading.BoundedSemaphore(workers + queue)
        self._futures: set[Future[Any]] = set()
        self._monitors: set[asyncio.Task[Any]] = set()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def run(
        self,
        fn: Callable[..., R],
        *args: Any,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> R:
        """Run one callable once, waiting for bounded submission capacity.

        ``deadline`` is a relative timeout covering both queue wait and result
        wait.  Timing out cannot stop a Python thread, so submitted work keeps
        its slot until it actually finishes.  This prevents a caller from
        repeatedly timing out and overfilling the executor with duplicate work.
        """

        seconds = _deadline_seconds(deadline)
        if self._closed:
            raise RuntimeError("blocking executor is closed")
        if seconds is None:
            return await self._run_once(fn, args, kwargs)
        async with asyncio.timeout(seconds):
            return await self._run_once(fn, args, kwargs)

    async def _run_once(
        self,
        fn: Callable[..., R],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> R:
        while not self._slots.acquire(blocking=False):
            if self._closed:
                raise RuntimeError("blocking executor is closed")
            await asyncio.sleep(0.005)
        if self._closed:
            self._slots.release()
            raise RuntimeError("blocking executor is closed")

        try:
            future = cast(Future[R], self._executor.submit(partial(fn, *args, **kwargs)))
        except BaseException:
            self._slots.release()
            raise

        self._futures.add(future)
        monitor = asyncio.create_task(self._wait_for_result(future))
        self._monitors.add(monitor)
        monitor.add_done_callback(self._consume_monitor_result)
        # Shielding is essential: cancellation of the awaiting task must not
        # cancel the monitor while its callable still runs.
        return await asyncio.shield(monitor)

    async def _wait_for_result(self, future: Future[R]) -> R:
        """Poll from the event loop; never require a worker-thread wakeup.

        A few supported hosts cannot reliably deliver ``call_soon_threadsafe``
        notifications from executor workers. Polling a concurrent future keeps
        the handoff event-loop-owned while the blocking callable remains fully
        off-loop.
        """

        try:
            while not future.done():
                await asyncio.sleep(0.005)
            return future.result()
        finally:
            self._futures.discard(future)
            task = asyncio.current_task()
            if task is not None:
                self._monitors.discard(task)
            self._slots.release()

    @staticmethod
    def _consume_monitor_result(task: asyncio.Task[Any]) -> None:
        """Retrieve detached results after caller timeout/cancellation."""

        if not task.cancelled():
            task.exception()

    def begin_close(self) -> None:
        self._closed = True

    async def finish_close(self, *, deadline: float | None = None) -> None:
        pending = tuple(self._monitors)
        drained = True
        if pending:
            waiter = asyncio.gather(*(asyncio.shield(item) for item in pending), return_exceptions=True)
            try:
                if deadline is None:
                    await waiter
                else:
                    async with asyncio.timeout(deadline):
                        await waiter
            except TimeoutError:
                drained = False
        # A bounded shutdown deadline keeps service stop responsive even if an
        # OS resolver or filesystem call is stuck. Running Python threads are
        # not killable; they finish in the background and cannot accept new
        # work because begin_close() already sealed the pool.
        self._executor.shutdown(wait=drained, cancel_futures=not drained)


class BoundedBlockingRunner:
    """Dedicated bounded IO, CPU, and DNS executors for service injection."""

    def __init__(
        self,
        *,
        io_workers: int = 2,
        io_queue: int = 8,
        cpu_workers: int = 1,
        cpu_queue: int = 1,
        dns_workers: int = 4,
        dns_queue: int = 16,
    ) -> None:
        self._io = _BoundedPool(name="io", max_workers=io_workers, max_queue=io_queue)
        self._cpu = _BoundedPool(name="cpu", max_workers=cpu_workers, max_queue=cpu_queue)
        self._dns = _BoundedPool(name="dns", max_workers=dns_workers, max_queue=dns_queue)
        self._close_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._io.closed and self._cpu.closed and self._dns.closed

    async def run_io(
        self,
        fn: Callable[..., R],
        *args: Any,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> R:
        return await self._io.run(fn, *args, deadline=deadline, **kwargs)

    async def run_cpu(
        self,
        fn: Callable[..., R],
        *args: Any,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> R:
        return await self._cpu.run(fn, *args, deadline=deadline, **kwargs)

    async def run_dns(
        self,
        fn: Callable[..., R],
        *args: Any,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> R:
        return await self._dns.run(fn, *args, deadline=deadline, **kwargs)

    async def close(self, *, deadline: float | None = None) -> None:
        """Stop submissions, drain accepted work once, and join all workers."""

        seconds = _deadline_seconds(deadline)
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(seconds))
        await asyncio.shield(self._close_task)

    async def _close(self, deadline: float | None) -> None:
        pools = (self._io, self._cpu, self._dns)
        for pool in pools:
            pool.begin_close()
        await asyncio.gather(*(pool.finish_close(deadline=deadline) for pool in pools))

    async def __aenter__(self) -> BoundedBlockingRunner:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def current_blocking_runner() -> BoundedBlockingRunner | None:
    """Return the task-local runner injected by the active tool registry."""

    return _CURRENT_RUNNER.get()


@contextmanager
def blocking_runner_context(runner: BoundedBlockingRunner | None):
    """Expose a runner to nested async tool helpers without changing schemas."""

    token = _CURRENT_RUNNER.set(runner)
    try:
        yield
    finally:
        _CURRENT_RUNNER.reset(token)


__all__ = ["BoundedBlockingRunner", "blocking_runner_context", "current_blocking_runner"]
