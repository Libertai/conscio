"""Ablation-flag tests: each flag off reproduces the corresponding v1-ish
behavior through the same runtime (one engine with flags, not forks)."""
from __future__ import annotations

import os
import tempfile
import unittest

from conscio.config import AblationFlags
from conscio.core.cognition import InputEvent, SelfState
from conscio.core.runtime import CognitiveRuntime
from conscio.core.workspace import EntryType, WorkspaceEntry
from conscio.memory.store import MemoryStore
from conscio.tools import ToolRegistry


class ScriptedLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append([dict(message) for message in messages])
        if self.responses:
            return self.responses.pop(0)
        return {"content": "done"}


def tool_call(name: str, arguments: str) -> dict:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _workspace_section(prompt: str) -> str:
    start = prompt.index("WORKSPACE\n") + len("WORKSPACE\n")
    end = prompt.index("\n\nUSER_INPUT")
    return prompt[start:end]


class AblationTests(unittest.IsolatedAsyncioTestCase):
    async def test_attention_gating_off_workspace_section_equals_read_output(self) -> None:
        # attention_gating=False: broadcast still happens (SSE), but the
        # prompt's WORKSPACE section falls back to the v1 read() rendering.
        with tempfile.TemporaryDirectory() as tmp:
            fake = ScriptedLLM([{"content": "ok"}])
            runtime = CognitiveRuntime(
                llm=fake,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "gating.db")),
                ablation=AblationFlags(attention_gating=False),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="hello", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        section = _workspace_section(fake.calls[0][1]["content"])
        expected = runtime.prompt_assembler._format_workspace(runtime.workspace, None)
        self.assertEqual(section, expected)
        # Broadcast still ran for SSE/observability.
        self.assertGreaterEqual(result.metrics.global_broadcasts, 1)

    async def test_attention_gating_on_workspace_section_is_broadcast_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = ScriptedLLM([{"content": "ok"}])
            runtime = CognitiveRuntime(
                llm=fake,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "gating-on.db")),
            )
            await runtime.initialize()
            try:
                await runtime.run_episode(InputEvent(content="hello", source="user"))
            finally:
                await runtime.close()

        section = _workspace_section(fake.calls[0][1]["content"])
        # Every rendered line is a broadcast (GLOBAL) winner.
        broadcast_lines = {
            f"- {entry.source}/{entry.type.value}" for entry in runtime.workspace.global_entries
        }
        for line in section.splitlines():
            prefix = line.split(":", 1)[0]
            self.assertIn(prefix, broadcast_lines)

    async def test_attention_gating_on_autonomous_prompt_has_broadcast_workspace(self) -> None:
        # Design §7/§9: tick-1 selection populates the WORKSPACE section of the
        # autonomous initial prompt too, not just the chat one.
        with tempfile.TemporaryDirectory() as tmp:
            fake = ScriptedLLM([{"content": "done"}])
            runtime = CognitiveRuntime(
                llm=fake,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "auto-gating-on.db")),
            )
            await runtime.initialize()
            try:
                await runtime.run_episode(
                    InputEvent(
                        content="Autonomous heartbeat: pick a concrete next step.",
                        source="autonomous",
                        event_type="heartbeat",
                    )
                )
            finally:
                await runtime.close()

        prompt = fake.calls[0][1]["content"]
        self.assertTrue(prompt.startswith("WORKSPACE\n"))
        section = prompt[len("WORKSPACE\n") : prompt.index("\n\nACTIVE_GOAL")]
        self.assertNotEqual(section.strip(), "none")
        # Every rendered line is a broadcast (GLOBAL) winner.
        broadcast_lines = {
            f"  - {entry.source}/{entry.type.value}" for entry in runtime.workspace.global_entries
        }
        for line in section.splitlines():
            prefix = line.split(":", 1)[0]
            self.assertIn(prefix, broadcast_lines)

    async def test_attention_gating_off_autonomous_prompt_has_no_workspace(self) -> None:
        # abl_no_attention: the autonomous prompt falls back to the v1-ish
        # rendering with no WORKSPACE section — the gated/ablated prompts differ.
        with tempfile.TemporaryDirectory() as tmp:
            fake = ScriptedLLM([{"content": "done"}])
            runtime = CognitiveRuntime(
                llm=fake,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "auto-gating-off.db")),
                ablation=AblationFlags(attention_gating=False),
            )
            await runtime.initialize()
            try:
                await runtime.run_episode(
                    InputEvent(
                        content="Autonomous heartbeat: pick a concrete next step.",
                        source="autonomous",
                        event_type="heartbeat",
                    )
                )
            finally:
                await runtime.close()

        prompt = fake.calls[0][1]["content"]
        self.assertNotIn("WORKSPACE\n", prompt)
        self.assertTrue(prompt.startswith("ACTIVE_GOAL"))

    async def test_memory_retrieval_off_skips_memory_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "no-memory.db")),
                ablation=AblationFlags(memory_retrieval=False),
            )
            await runtime.initialize()
            try:
                await runtime.run_episode(InputEvent(content="First message", source="user"))
                second = await runtime.run_episode(InputEvent(content="Second message", source="user"))
            finally:
                await runtime.close()

        self.assertFalse(
            any(e.type == EntryType.MEMORY for e in runtime.workspace.view(second.episode_id))
        )

        # Control: with the flag on (same DB pattern), episodic memory surfaces.
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "with-memory.db")),
            )
            await runtime.initialize()
            try:
                await runtime.run_episode(InputEvent(content="First message", source="user"))
                second = await runtime.run_episode(InputEvent(content="Second message", source="user"))
            finally:
                await runtime.close()

        self.assertTrue(
            any(e.type == EntryType.MEMORY for e in runtime.workspace.view(second.episode_id))
        )

    async def test_prediction_off_records_no_conflicts_for_failed_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry()

            async def bash(input: str = "") -> dict:
                return {"output": "boom", "error": True, "exit_code": 1}

            tools.register("bash", bash, "Execute shell commands.")
            llm = ScriptedLLM([
                tool_call("bash", '{"input": "explode"}'),
                {"content": "Done despite the failure."},
            ])
            runtime = CognitiveRuntime(
                llm=llm,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "no-prediction.db")),
                tools=tools,
                ablation=AblationFlags(prediction=False),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="Run it.", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertEqual(result.metrics.prediction_errors, 0)
        self.assertFalse(
            any(e.type == EntryType.CONFLICT for e in runtime.workspace.view(result.episode_id))
        )

    async def test_reflection_off_answers_with_violation_logged(self) -> None:
        # v1-ish: the constraint violation is recorded, not corrected.
        with tempfile.TemporaryDirectory() as tmp:
            llm = ScriptedLLM([{"content": "The answer is four."}])
            runtime = CognitiveRuntime(
                llm=llm,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "no-reflection.db")),
                ablation=AblationFlags(reflection=False),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(
                    InputEvent(content="Answer in one word: what is 2+2?", source="user")
                )
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertEqual(result.output, "The answer is four.")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result.metrics.reflections, 0)
        self.assertGreaterEqual(result.metrics.constraint_violations, 1)
        self.assertTrue(any(check["passed"] is False for check in result.constraint_report))

    async def test_self_state_coupling_off_drops_state_terms_from_attention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ablated = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "no-coupling.db")),
                ablation=AblationFlags(self_state_coupling=False),
            )
            coupled = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "coupling.db")),
            )

        self.assertFalse(ablated.attention.coupling)
        self.assertTrue(coupled.attention.coupling)
        entry = WorkspaceEntry(content="candidate", source="test", salience=0.5, novelty=0.5)
        calm = SelfState(uncertainty=0.0)
        anxious = SelfState(uncertainty=1.0)
        self.assertEqual(
            ablated.attention.score(entry, calm), ablated.attention.score(entry, anxious)
        )
        self.assertNotEqual(
            coupled.attention.score(entry, calm), coupled.attention.score(entry, anxious)
        )

    async def test_appraisal_off_returns_neutral_constants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "no-appraisal.db")),
                ablation=AblationFlags(appraisal=False),
            )

        self.assertFalse(runtime.appraisal.enabled)
        scores = runtime.appraisal.appraise(
            "URGENT: an error happened now!", source="user", type=EntryType.OBSERVATION
        )
        self.assertEqual(scores, dict(runtime.appraisal.NEUTRAL))


if __name__ == "__main__":
    unittest.main()
