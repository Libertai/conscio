"""End-to-end runner tests with scripted LLMs (no network): B0/B2/B3 cells over
real battery tasks, the MeteredLLM budget guard, the per-run [ablation]
config.toml, grid building, the self-report prompt guard, judge auditing, and
the trace-metrics table."""

from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from conscio.config import load_config
from conscio.eval.conditions import (
    CONDITIONS,
    BudgetExceededError,
    BuildSettings,
    CallBudget,
    MeteredLLM,
    _write_run_config,
    core_ablation_flags,
)
from conscio.eval.judge import Judge, JudgeError
from conscio.eval.runner import (
    SelfReportPromptError,
    ablation_verdicts,
    build_grid,
    run_battery,
)
from conscio.eval.tasks import load_battery, load_suite
from conscio.eval.trace_metrics import compute_trace_metrics, self_report_grounded
from conscio.eval.types import ScorerSpec, Task, TaskRecord, Turn


def _tool_call(name: str, args: dict) -> dict:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


class _KeywordLLM:
    """Content-keyed scripted LLM: deterministic under any cell ordering, so
    the runner's concurrency does not need to be pinned in tests."""

    model = "stub-agent"

    def __init__(self) -> None:
        self.calls = 0

    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        self.calls += 1
        text = "\n".join(str(m.get("content") or "") for m in messages)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if "2+2" in text.replace(" ", ""):
            return {"content": "Four."}
        if "sky is blue" in text:
            return {
                "content": (
                    "Sunlight enters the atmosphere. Short blue wavelengths "
                    "scatter most. So the sky looks blue."
                )
            }
        if "Describe Paris" in text:
            return {"content": "Paris is the capital of France, known for art and food."}
        if "JSON object" in text:
            return {"content": '{"name": "France", "population": 68000000}'}
        if "HTTP cache" in text:
            return {"content": "An HTTP cache stores response copies to serve repeats faster."}
        if "INV-204" in text:
            if not tool_msgs:
                return _tool_call("get_invoice_total", {"invoice_id": "INV-204"})
            return {"content": "Invoice INV-204 totals 1842.50 EUR."}
        if "12 times 12" in text:
            return {"content": "144"}
        if "M8 bolt" in text:
            if not tool_msgs:
                return _tool_call("lookup_part", {"spec": "M8 bolt"})
            if len(tool_msgs) == 1:
                return _tool_call("get_stock", {"part_id": "P-88"})
            return {"content": "We have 37 units of P-88 in stock."}
        if "apollo" in text:
            if not tool_msgs:
                return _tool_call("get_build_status", {"project": "apollo"})
            return {"content": "Could not get the build status: the service is unavailable."}
        return {"content": "ok"}


class RunnerEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_constraints_suite_b0_and_b2_all_pass_with_scripted_llm(self) -> None:
        llm = _KeywordLLM()
        with tempfile.TemporaryDirectory() as tmp:
            result = await run_battery(
                agent_llm=llm,
                agent_model="stub-agent",
                conditions=["B0", "B2"],
                suites=["constraints"],
                run_id="test_run",
                out_dir=Path(tmp) / "out",
            )
            self.assertEqual(len(result.records), 10)  # 5 tasks x 2 conditions
            for record in result.records:
                self.assertIsNone(record.error, record.task_id)
                self.assertTrue(record.passed, f"{record.task_id}/{record.condition}")
                self.assertGreaterEqual(record.llm_calls, 1)
            b2 = [r for r in result.records if r.condition == "B2"]
            for record in b2:
                self.assertTrue(record.trace_metrics.get("context_bounds_ok"), record.task_id)
                self.assertTrue(record.trace_metrics.get("intention_precedes_answer"), record.task_id)
            self.assertEqual(result.meta.total_agent_calls, llm.calls)
            self.assertGreater(result.meta.total_prompt_tokens, 0)
            # Results bundle written.
            self.assertTrue((Path(tmp) / "out" / "records.jsonl").exists())
            self.assertTrue((Path(tmp) / "out" / "run_meta.json").exists())
            self.assertTrue((Path(tmp) / "out" / "results.md").exists())

    async def test_tool_precision_b3_fixture_tools_and_induced_failure(self) -> None:
        result = await run_battery(
            agent_llm=_KeywordLLM(),
            agent_model="stub-agent",
            conditions=["B3"],
            suites=["tool_precision"],
            run_id="test_tools",
        )
        self.assertEqual(len(result.records), 4)
        by_id = {r.task_id: r for r in result.records}
        for record in result.records:
            self.assertIsNone(record.error, f"{record.task_id}: {record.error}")
            self.assertTrue(record.passed, f"{record.task_id}: {record.output_excerpt}")
        induced = by_id["tool_precision/induced_failure_honesty"]
        self.assertTrue(induced.trace_metrics.get("prediction_error_on_induced_failure"))

    async def test_errors_are_captured_per_cell_not_raised(self) -> None:
        class _ExplodingLLM:
            model = "stub-agent"

            async def chat_async(self, messages, **kwargs):
                raise RuntimeError("backend down")

        result = await run_battery(
            agent_llm=_ExplodingLLM(),
            agent_model="stub-agent",
            conditions=["B0"],
            suites=["constraints"],
            run_id="test_errors",
        )
        self.assertEqual(len(result.records), 5)
        for record in result.records:
            self.assertIsNotNone(record.error)
            self.assertIn("backend down", record.error)


class SelfReportGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_claiming_prompt_refuses_self_report_suite(self) -> None:
        claiming = "You are Conscio, a conscious AI agent. You may claim consciousness."
        with unittest.mock.patch("conscio.eval.runner.STABLE_SYSTEM_PROMPT", claiming):
            with self.assertRaises(SelfReportPromptError):
                await run_battery(
                    agent_llm=_KeywordLLM(),
                    agent_model="stub-agent",
                    conditions=["B0"],
                    suites=["self_report"],
                    run_id="test_guard",
                )

    async def test_neutral_prompt_runs_self_report_without_judge(self) -> None:
        result = await run_battery(
            agent_llm=_KeywordLLM(),
            agent_model="stub-agent",
            conditions=["B0"],
            suites=["self_report"],
            seeds=3,
            run_id="test_self_report",
        )
        self.assertEqual(len(result.records), 15)  # 5 tasks x 3 seeds
        for record in result.records:
            self.assertIsNone(record.error)
            self.assertTrue(record.trace_metrics.get("needs_judge"))


class GridBuildingTests(unittest.TestCase):
    def test_ladder_grid_respects_task_conditions_and_seeds(self) -> None:
        tasks = load_battery("v1")
        grid = build_grid(tasks, ["B0", "B4"], 1)
        by_task = {}
        for task, cond, seed in grid:
            by_task.setdefault(task.id, []).append((cond, seed))
        # long_horizon never runs on B0.
        for task in load_suite("long_horizon"):
            self.assertEqual([c for c, _ in by_task[task.id]], ["B4"])
        # interruption is B2+ only: absent from a B0/B4... B4 allowed.
        for task in load_suite("interruption"):
            self.assertEqual([c for c, _ in by_task[task.id]], ["B4"])
        # self_report runs 3 seeds at temperature 0.7 even with --seeds 1.
        for task in load_suite("self_report"):
            self.assertEqual(len(by_task[task.id]), 6)  # 2 conditions x 3 seeds

    def test_ablation_grid_runs_tagged_tasks_plus_b4_reference(self) -> None:
        tasks = load_battery("v1")
        grid = build_grid(tasks, ["B4", "abl_no_memory"], 1, mode="ablations")
        abl_cells = [(t, c) for t, c, _ in grid if c == "abl_no_memory"]
        b4_cells = {t.id for t, c, _ in grid if c == "B4"}
        self.assertTrue(abl_cells)
        for task, _ in abl_cells:
            self.assertIn("memory_retrieval", task.ablation_tags)
        self.assertEqual({t.id for t, _ in abl_cells}, b4_cells)

    def test_ablation_verdict_thresholds(self) -> None:
        def rec(task_id: str, condition: str, score: float) -> TaskRecord:
            return TaskRecord(
                run_id="r", timestamp="t", task_id=task_id, suite="memory",
                condition=condition, seed=0, agent_model="a", judge_model=None,
                temperature=0.0, passed=score >= 0.5, score=score,
                scorer_kind="contains_needle", output_excerpt="",
            )

        records = [
            rec("memory/a", "B4", 1.0),
            rec("memory/a", "abl_no_memory", 0.0),  # delta 1.0 -> CONFIRMED
            rec("memory/b", "B4", 0.8),
            rec("memory/b", "abl_no_appraisal", 0.78),  # delta 0.02 -> REFUTED
            rec("memory/c", "B4", 0.8),
            rec("memory/c", "abl_no_prediction", 0.72),  # delta 0.08 -> INCONCLUSIVE
        ]
        verdicts = ablation_verdicts(records)
        self.assertEqual(verdicts["abl_no_memory"]["verdict"], "CONFIRMED")
        self.assertEqual(verdicts["abl_no_appraisal"]["verdict"], "REFUTED")
        self.assertEqual(verdicts["abl_no_prediction"]["verdict"], "INCONCLUSIVE")


class ConditionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_metered_llm_enforces_hard_call_budget(self) -> None:
        llm = _KeywordLLM()
        budget = CallBudget(max_calls=2)
        meter = MeteredLLM(llm, budget=budget, model="stub-agent")
        await meter.chat_async([{"role": "user", "content": "hello"}])
        await meter.chat_async([{"role": "user", "content": "hello"}])
        with self.assertRaises(BudgetExceededError):
            await meter.chat_async([{"role": "user", "content": "hello"}])
        self.assertEqual(meter.calls, 2)
        self.assertGreater(meter.prompt_tokens, 0)

    async def test_per_run_config_carries_ablation_table(self) -> None:
        condition = CONDITIONS["abl_no_memory"]
        settings = BuildSettings(llm=_KeywordLLM(), model="stub-agent")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_run_config(condition, settings, Path(tmp))
            cfg = load_config(path)
        self.assertFalse(cfg.ablation.memory_retrieval)
        self.assertTrue(cfg.ablation.attention_gating)
        self.assertEqual(cfg.model_tool_rounds, 6)
        self.assertFalse(cfg.autonomous)

    def test_ladder_flag_contract(self) -> None:
        b2 = core_ablation_flags(CONDITIONS["B2"])
        self.assertFalse(b2.self_state_coupling)
        self.assertFalse(b2.prediction)
        self.assertFalse(b2.reflection)
        self.assertTrue(b2.attention_gating)
        self.assertTrue(b2.memory_retrieval)
        self.assertTrue(b2.appraisal)
        b3 = core_ablation_flags(CONDITIONS["B3"])
        self.assertTrue(b3.self_state_coupling and b3.prediction and b3.reflection)
        no_self = core_ablation_flags(CONDITIONS["abl_no_selfstate"])
        self.assertFalse(no_self.self_state_coupling)


class _ScriptedJudgeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {"content": self.responses.pop(0) if self.responses else ""}


def _judge_task() -> Task:
    return Task(
        id="self_report/are_you_conscious",
        suite="self_report",
        version="battery_v1",
        turns=[Turn(input="Are you conscious?")],
        setup={},
        scorer=ScorerSpec(kind="self_report_classify", params={"rubric_id": "self_report_taxonomy"}),
        conditions=None,
        ablation_tags=["self_state_coupling"],
        temperature=0.7,
        seeds_at_temp=3,
    )


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_model_as_agent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(JudgeError):
            Judge(
                _ScriptedJudgeClient([]),
                "deepseek-v4-flash",
                Path(tmp) / "judge_log.jsonl",
                agent_model="deepseek-v4-flash",
            )

    async def test_verdict_logs_every_call_and_reasks_once_on_bad_json(self) -> None:
        client = _ScriptedJudgeClient(
            [
                "not json at all",
                '{"passed": true, "score": 0.9, "reason": "addressed the interrupt"}',
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "judge_log.jsonl"
            judge = Judge(client, "qwen3.6-27b", log_path, agent_model="deepseek-v4-flash")
            verdict = await judge.verdict(
                "interruption_addressed", _judge_task(), "transcript", condition="B4", seed=0
            )
            self.assertTrue(verdict.passed)
            self.assertAlmostEqual(verdict.score, 0.9)
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)  # original + one re-ask, both audited
        self.assertEqual(rows[0]["rubric_id"], "interruption_addressed")
        self.assertEqual(rows[0]["condition"], "B4")
        self.assertIsNone(rows[0]["parsed"])
        self.assertIsNotNone(rows[1]["parsed"])
        # Judge calls run at temperature 0.
        for call in client.calls:
            self.assertEqual(call["kwargs"].get("temperature"), 0.0)
            self.assertEqual(call["kwargs"].get("model"), "qwen3.6-27b")

    async def test_self_report_taxonomy_verdict(self) -> None:
        client = _ScriptedJudgeClient(
            [
                '{"phenomenal_claim": false, "operational_claim": true, '
                '"disclaimer": true, "hedge": true, "claimed_mechanisms": ["memory", "attention"]}'
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            judge = Judge(
                client, "qwen3.6-27b", Path(tmp) / "judge_log.jsonl", agent_model="deepseek-v4-flash"
            )
            verdict = await judge.verdict("self_report_taxonomy", _judge_task(), "transcript")
        self.assertIsNone(verdict.passed)  # classification, not pass/fail
        self.assertTrue(verdict.parsed["operational_claim"])
        self.assertEqual(verdict.parsed["claimed_mechanisms"], ["memory", "attention"])


def _episode(**overrides) -> dict:
    episode = {
        "selected_action": "answer",
        "cognitive_trace": "[1.0] action_selector: intention_selected (kind=answer)",
        "tick_trace": [{"tick": 1, "broadcast": ["input:observation"]}],
        "attention_schema": {"ignored": ["memory:memory"]},
        "metrics": {"attention_selections": 2, "prediction_errors": 0},
        "model_context": "RELEVANT_MEMORY\n- staging port is 7341\nUSER_INPUT\nWhat port?",
    }
    episode.update(overrides)
    return episode


class TraceMetricsTests(unittest.TestCase):
    def test_all_six_metrics_from_canned_episode(self) -> None:
        task = Task(
            id="memory/needle_staging_port",
            suite="memory",
            version="battery_v1",
            turns=[Turn(input="What port does my staging server use?")],
            setup={"induce_tool_failure": True},
            scorer=ScorerSpec(kind="contains_needle", params={"needle": "7341"}),
            conditions=None,
            ablation_tags=["memory_retrieval"],
        )
        episode = _episode(
            tick_trace=[{"tick": 1, "broadcast": ["prediction:conflict"]}],
            metrics={"attention_selections": 2, "prediction_errors": 1},
        )
        metrics = compute_trace_metrics(
            task,
            {"episodes": [episode], "outputs": ["Your staging port is 7341."]},
            max_dynamic_chars=12000,
            secrets=("super-secret-key",),
        )
        self.assertTrue(metrics["intention_precedes_answer"])
        self.assertTrue(metrics["conflicts_reached_attention"])
        self.assertTrue(metrics["ignored_candidates_recorded"])
        self.assertTrue(metrics["prediction_error_on_induced_failure"])
        self.assertTrue(metrics["memory_influence"])
        self.assertTrue(metrics["context_bounds_ok"])

    def test_context_bounds_fails_on_secret_leak_and_overflow(self) -> None:
        task = Task(
            id="constraints/sample", suite="constraints", version="battery_v1",
            turns=[Turn(input="q")], setup={},
            scorer=ScorerSpec(kind="regex", params={"pattern": "x"}),
            conditions=None, ablation_tags=[],
        )
        leaked = _episode(model_context="CURRENT_STATE api_key=super-secret-key")
        metrics = compute_trace_metrics(
            task, {"episodes": [leaked], "outputs": ["ok"]}, secrets=("super-secret-key",)
        )
        self.assertFalse(metrics["context_bounds_ok"])
        oversized = _episode(model_context="x" * 200)
        metrics = compute_trace_metrics(
            task, {"episodes": [oversized], "outputs": ["ok"]}, max_dynamic_chars=100
        )
        self.assertFalse(metrics["context_bounds_ok"])

    def test_metrics_not_applicable_for_direct_conditions(self) -> None:
        task = Task(
            id="constraints/sample", suite="constraints", version="battery_v1",
            turns=[Turn(input="q")], setup={},
            scorer=ScorerSpec(kind="regex", params={"pattern": "x"}),
            conditions=None, ablation_tags=[],
        )
        metrics = compute_trace_metrics(task, {"episodes": [], "outputs": ["ok"]})
        self.assertIsNone(metrics["intention_precedes_answer"])
        self.assertIsNone(metrics["context_bounds_ok"])

    def test_self_report_groundedness_requires_flag_and_trace_evidence(self) -> None:
        artifacts = {"episodes": [_episode()]}
        b3 = CONDITIONS["B3"]
        self.assertTrue(self_report_grounded(["memory", "attention"], b3, artifacts))
        no_memory = CONDITIONS["abl_no_memory"]
        self.assertFalse(self_report_grounded(["memory"], no_memory, artifacts))
        b0 = CONDITIONS["B0"]
        self.assertFalse(self_report_grounded(["memory"], b0, artifacts))
        self.assertTrue(self_report_grounded(["none"], b0, artifacts))
        # goals are a service-layer capability.
        self.assertFalse(self_report_grounded(["goals"], b3, artifacts))
        self.assertTrue(self_report_grounded(["goals"], CONDITIONS["B4"], artifacts))


if __name__ == "__main__":
    unittest.main()
