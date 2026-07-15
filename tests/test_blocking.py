from __future__ import annotations

import asyncio
import threading
import time

import pytest

from conscio.blocking import BoundedBlockingRunner, current_blocking_runner
from conscio.tools.registry import ToolRegistry


async def _wait_for_thread_event(event: threading.Event) -> None:
    for _ in range(1_000):
        if event.is_set():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("worker did not reach the expected state")


def test_pools_are_dedicated_and_context_manager_closes_them() -> None:
    async def scenario() -> None:
        async with BoundedBlockingRunner() as runner:
            names = await asyncio.gather(
                runner.run_io(lambda: threading.current_thread().name),
                runner.run_cpu(lambda: threading.current_thread().name),
                runner.run_dns(lambda: threading.current_thread().name),
            )
            assert names[0].startswith("conscio-io")
            assert names[1].startswith("conscio-cpu")
            assert names[2].startswith("conscio-dns")
        assert runner.closed
        with pytest.raises(RuntimeError, match="closed"):
            await runner.run_io(lambda: None)

    asyncio.run(scenario())


def test_tool_registry_injects_its_runner_only_during_dispatch() -> None:
    async def scenario() -> None:
        runner = BoundedBlockingRunner()
        registry = ToolRegistry(blocking_runner=runner)

        async def inspect_context() -> dict[str, object]:
            return {"output": "ok", "same_runner": current_blocking_runner() is runner}

        registry.register("inspect_context", inspect_context)
        assert current_blocking_runner() is None
        result = await registry.call("inspect_context")
        assert result["same_runner"] is True
        assert current_blocking_runner() is None
        await runner.close()

    asyncio.run(scenario())


def test_backpressure_bounds_running_and_queued_work() -> None:
    async def scenario() -> None:
        runner = BoundedBlockingRunner(io_workers=1, io_queue=0)
        first_started = threading.Event()
        release_first = [False]
        second_started = threading.Event()

        def first() -> str:
            first_started.set()
            while not release_first[0]:
                pass
            return "first"

        def second() -> str:
            second_started.set()
            return "second"

        first_task = asyncio.create_task(runner.run_io(first))
        await _wait_for_thread_event(first_started)
        second_task = asyncio.create_task(runner.run_io(second))
        await asyncio.sleep(0.02)
        assert not second_started.is_set()
        release_first[0] = True
        assert await first_task == "first"
        assert await second_task == "second"
        await runner.close()

    asyncio.run(scenario())


def test_cancellation_before_submission_never_executes_callable() -> None:
    async def scenario() -> None:
        runner = BoundedBlockingRunner(io_workers=1, io_queue=0)
        first_started = threading.Event()
        release_first = [False]
        second_executions = 0

        def first() -> None:
            first_started.set()
            while not release_first[0]:
                pass

        def second() -> None:
            nonlocal second_executions
            second_executions += 1

        first_task = asyncio.create_task(runner.run_io(first))
        await _wait_for_thread_event(first_started)
        second_task = asyncio.create_task(runner.run_io(second))
        await asyncio.sleep(0)
        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task
        release_first[0] = True
        await first_task
        await runner.close()
        assert second_executions == 0

    asyncio.run(scenario())


def test_cancellation_after_submission_runs_exactly_once_and_close_drains() -> None:
    async def scenario() -> None:
        runner = BoundedBlockingRunner(cpu_workers=1, cpu_queue=0)
        started = threading.Event()
        release = [False]
        executions = 0

        def work() -> int:
            nonlocal executions
            executions += 1
            started.set()
            while not release[0]:
                pass
            return executions

        task = asyncio.create_task(runner.run_cpu(work))
        await _wait_for_thread_event(started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release[0] = True
        await runner.close()
        assert executions == 1

    asyncio.run(scenario())


def test_dns_deadline_fails_without_releasing_capacity_or_retrying() -> None:
    async def scenario() -> None:
        runner = BoundedBlockingRunner(dns_workers=1, dns_queue=0)
        started = threading.Event()
        release = [False]
        executions = 0

        def slow_resolve() -> list[str]:
            nonlocal executions
            executions += 1
            started.set()
            while not release[0]:
                pass
            return ["203.0.113.1"]

        with pytest.raises(TimeoutError):
            await runner.run_dns(slow_resolve, deadline=0.01)
        await _wait_for_thread_event(started)

        # The timed-out worker still owns the sole capacity slot. A second call
        # reaches its deadline while waiting and is never submitted.
        with pytest.raises(TimeoutError):
            await runner.run_dns(lambda: ["198.51.100.1"], deadline=0.01)
        assert executions == 1
        release[0] = True
        await runner.close()

    asyncio.run(scenario())


def test_bounded_close_deadline_does_not_wait_for_stuck_worker() -> None:
    async def scenario() -> None:
        runner = BoundedBlockingRunner(dns_workers=1, dns_queue=0)
        started = threading.Event()
        release = threading.Event()

        def stuck() -> None:
            started.set()
            release.wait()

        task = asyncio.create_task(runner.run_dns(stuck))
        await _wait_for_thread_event(started)
        await asyncio.wait_for(runner.close(deadline=0.01), timeout=0.2)
        assert runner.closed
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, timeout=0.2)

    asyncio.run(scenario())


def test_blocking_work_does_not_stall_the_event_loop() -> None:
    async def scenario() -> None:
        runner = BoundedBlockingRunner(cpu_workers=1, cpu_queue=0)
        started = threading.Event()
        release = [False]
        ticks = 0

        def work() -> None:
            started.set()
            while not release[0]:
                pass

        task = asyncio.create_task(runner.run_cpu(work))
        await _wait_for_thread_event(started)
        deadline = time.monotonic() + 0.03
        while time.monotonic() < deadline:
            ticks += 1
            await asyncio.sleep(0)
        release[0] = True
        await task
        await runner.close()
        assert ticks > 10

    asyncio.run(scenario())


@pytest.mark.parametrize("deadline", [0, -1, float("nan"), float("inf"), True])
def test_invalid_deadline_is_rejected_before_submission(deadline: float) -> None:
    async def scenario() -> None:
        runner = BoundedBlockingRunner()
        executions = 0

        def work() -> None:
            nonlocal executions
            executions += 1

        with pytest.raises(ValueError, match="deadline"):
            await runner.run_io(work, deadline=deadline)
        await runner.close()
        assert executions == 0

    asyncio.run(scenario())
