"""Live-eval double gate: `--live` AND CONSCIO_EVAL_LIVE=1 are both required
before `conscio eval --suite ladder|ablations` may make paid LLM calls
(docs/v2-design.md "Live gating"). Anything less exits with an explanation."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from conscio.cli import LIVE_EVAL_GATE_EXPLANATION, _run_live_eval, main
from conscio.eval import LIVE_SUITES
from conscio.eval.types import RunMeta
from conscio.eval.runner import RunResult


def _live_args(*, live: bool, suite: str = "ladder", out: str = "docs/results") -> argparse.Namespace:
    """Namespace matching the `conscio eval` argparse defaults."""
    return argparse.Namespace(
        suite=suite,
        conditions="",
        seeds=1,
        tasks="",
        out=out,
        model="agent-model",
        judge_model="judge-model",
        run_id="gate-test",
        live=live,
    )


def _env_without_gate() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "CONSCIO_EVAL_LIVE"}
    return env


class LiveEvalGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_without_flag_and_without_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, _env_without_gate(), clear=True):
            with self.assertRaises(SystemExit) as ctx:
                await _run_live_eval(_live_args(live=False))
        self.assertEqual(str(ctx.exception), LIVE_EVAL_GATE_EXPLANATION)

    async def test_blocked_with_flag_but_without_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, _env_without_gate(), clear=True):
            with self.assertRaises(SystemExit) as ctx:
                await _run_live_eval(_live_args(live=True))
        self.assertEqual(str(ctx.exception), LIVE_EVAL_GATE_EXPLANATION)

    async def test_blocked_with_env_but_without_flag(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"CONSCIO_EVAL_LIVE": "1"}):
            with self.assertRaises(SystemExit) as ctx:
                await _run_live_eval(_live_args(live=False))
        self.assertEqual(str(ctx.exception), LIVE_EVAL_GATE_EXPLANATION)

    async def test_blocked_when_env_value_is_not_exactly_1(self) -> None:
        for value in ("0", "true", "yes", ""):
            with self.subTest(value=value):
                with unittest.mock.patch.dict(os.environ, {"CONSCIO_EVAL_LIVE": value}):
                    with self.assertRaises(SystemExit) as ctx:
                        await _run_live_eval(_live_args(live=True))
                self.assertEqual(str(ctx.exception), LIVE_EVAL_GATE_EXPLANATION)

    async def test_both_gates_open_runs_battery_without_exit(self) -> None:
        meta = RunMeta(
            run_id="gate-test",
            date="2026-06-12",
            agent_model="agent-model",
            judge_model="judge-model",
            battery_version="test",
            git_commit="0" * 7,
        )
        fake_result = RunResult(records=[], meta=meta, paths={})
        run_battery = unittest.mock.AsyncMock(return_value=fake_result)
        with tempfile.TemporaryDirectory() as tmp:
            with (
                unittest.mock.patch.dict(os.environ, {"CONSCIO_EVAL_LIVE": "1"}),
                unittest.mock.patch("conscio.eval.runner.run_battery", run_battery),
                unittest.mock.patch("conscio.eval.judge.Judge") as judge_cls,
                unittest.mock.patch("conscio.llm.client.LLMClient") as llm_cls,
            ):
                await _run_live_eval(_live_args(live=True, out=tmp))
            run_battery.assert_awaited_once()
            self.assertEqual(run_battery.await_args.kwargs["mode"], "ladder")
            self.assertEqual(run_battery.await_args.kwargs["run_id"], "gate-test")
            self.assertTrue(judge_cls.called)
            self.assertTrue(llm_cls.called)
            self.assertTrue((Path(tmp) / "gate-test").is_dir())


class LiveEvalGateCliDispatchTests(unittest.TestCase):
    """End-to-end through argparse: catches `--live` default flips and routing
    changes, not just the guard expression itself."""

    def test_every_live_suite_is_gated_at_the_cli(self) -> None:
        for suite in LIVE_SUITES:
            with self.subTest(suite=suite):
                argv = ["conscio", "eval", "--suite", suite]
                with (
                    unittest.mock.patch.dict(os.environ, _env_without_gate(), clear=True),
                    unittest.mock.patch.object(sys, "argv", argv),
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        main()
                self.assertEqual(str(ctx.exception), LIVE_EVAL_GATE_EXPLANATION)

    def test_live_flag_alone_is_still_gated_at_the_cli(self) -> None:
        argv = ["conscio", "eval", "--suite", "ladder", "--live"]
        with (
            unittest.mock.patch.dict(os.environ, _env_without_gate(), clear=True),
            unittest.mock.patch.object(sys, "argv", argv),
        ):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(str(ctx.exception), LIVE_EVAL_GATE_EXPLANATION)


if __name__ == "__main__":
    unittest.main()
