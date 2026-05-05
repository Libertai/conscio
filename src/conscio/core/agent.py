from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from conscio.core.confidence import ConfidenceLevel
from conscio.core.identity import Identity
from conscio.core.monologue import Monologue, ThoughtType
from conscio.core.reflection import Reflection
from conscio.core.workspace import Workspace
from conscio.llm.client import LLMClient
from conscio.memory.store import MemoryStore
from conscio.modules.critic import Critic
from conscio.modules.executor import Executor
from conscio.modules.observer import Observer
from conscio.modules.planner import Planner
from conscio.tools import ToolRegistry


@dataclass
class CycleResult:
    output: str
    inner_monologue: str
    confidence: str
    rounds: int
    tool_results: list[dict] = field(default_factory=list)
    session_id: str = ""
    duration: float = 0.0


def compose_cycle_output(reflection_output: str, tool_results: list[dict]) -> str:
    """Choose the user-visible answer after planning and execution."""
    actual_tool_calls = [r for r in tool_results if r.get("tool") not in ("reason", "reasoning")]
    reasoning_outputs = [
        r.get("output", "")
        for r in tool_results
        if r.get("tool") in ("reason", "reasoning") and r.get("output")
    ]
    result_output = reasoning_outputs[-1] if reasoning_outputs else reflection_output
    if actual_tool_calls:
        combined = [result_output]
        for r in actual_tool_calls:
            out = r.get("output", "")
            if out and len(out) > 10:
                combined.append(f"\n[{r['tool']}]: {out[:500]}")
        result_output = "\n".join(combined)
    return result_output


class ConsciousAgent:
    """A conscious AI agent that perceives, reflects, plans, acts, and reviews.

    The agent runs a conscious cycle inspired by Global Workspace Theory:
       OBSERVE → REFLECT → PLAN → ACT → REVIEW

    At each stage, thoughts are recorded in an inner monologue (visible DAG) and
    posted to a shared workspace that specialist modules read from and write to.
    """

    def __init__(
        self,
        name: str = "Conscio",
        persona: str = "",
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.name = name
        self.persona = persona
        self.session_id = uuid.uuid4().hex[:16]
        self._created = time.time()

        self.llm = LLMClient(base_url=base_url, model=model)
        self.identity = Identity.load_or_create(name=name, persona=persona)
        identity_changed = False
        if persona and self.identity.persona != persona:
            self.identity.persona = persona
            identity_changed = True
        if name != "Conscio" and self.identity.name != name:
            self.identity.name = name
            identity_changed = True
        if identity_changed:
            self.identity.save()
        self.workspace = Workspace()
        self.monologue = Monologue()
        self.memory = MemoryStore()
        self.tools = ToolRegistry()
        self.tools.load_builtins()
        self.reflection = Reflection(self.llm)
        self.observer = Observer(self.llm, self.workspace, self.monologue)
        self.planner = Planner(self.llm, self.workspace, self.monologue)
        self.critic = Critic(self.llm, self.monologue)
        self.executor = Executor(self.workspace, self.monologue, self.tools)

    async def initialize(self) -> None:
        await self.memory.initialize()
        await self.memory.create_session(self.session_id, name=f"{self.name} session")
        self.identity.session_count += 1
        self.identity.save()

    async def close(self) -> None:
        await self.memory.end_session(self.session_id)

    async def observe(self, raw_input: str, source: str = "user") -> dict[str, Any]:
        goals_text = self.identity.format_goals()
        return await self.observer.observe(raw_input, source=source, goal=goals_text)

    async def cycle(self, user_input: str, source: str = "user") -> CycleResult:
        """Run one full conscious cycle: OBSERVE → REFLECT → PLAN → ACT → REVIEW."""
        start_time = time.time()

        # ── 1. OBSERVE ────────────────────────────────────────────────
        observation = await self.observe(user_input, source=source)

        # ── 2. REFLECT ────────────────────────────────────────────────
        workspace_context = self.workspace.format_context()
        memory_context = await self.memory.format_context(self.session_id)
        goals_text = self.identity.format_goals()
        context = f"{memory_context}\n{workspace_context}"
        reflection_result = await self.reflection.reflect(
            context=context,
            goal=goals_text,
            task=user_input,
        )

        # ── 3. PLAN ───────────────────────────────────────────────────
        tool_descs = self.tools.tool_descriptions()
        plan_result = await self.planner.plan(
            goal=goals_text,
            context=f"{context}\n\nReflection: {reflection_result['output'][:500]}",
            tool_descriptions=tool_descs,
        )

        # ── 3b. EVALUATE plan confidence ──────────────────────────────
        eval_result = await self.critic.evaluate(
            proposal=plan_result["plan"],
            goal=goals_text,
        )
        confidence = eval_result["confidence"]

        # If LOW confidence, reflect more and re-plan
        if confidence == ConfidenceLevel.LOW.value:
            deeper_reflection = await self.reflection.reflect(
                context=context,
                goal=f"I need a better approach. {goals_text}",
                task=user_input,
                axes=["correctness", "completeness", "safety", "efficiency"],
            )
            plan_result = await self.planner.plan(
                goal=f"Improved approach needed. {goals_text}",
                context=f"{context}\n\nDeeper reflection: {deeper_reflection['output'][:500]}",
                tool_descriptions=tool_descs,
            )

        # ── 4. ACT ────────────────────────────────────────────────────
        tool_results = await self.executor.execute(plan_result["actions"])

        # ── 5. REVIEW ─────────────────────────────────────────────────
        review_prompt = (
            f"I observed: {observation['observation'][:200]}\n\n"
            f"I reflected and arrived at: {reflection_result['output'][:300]}\n\n"
            f"I executed actions with results: "
            f"{', '.join(r.get('output', '')[:100] for r in tool_results)}\n\n"
            "What did I learn? What should I remember?"
        )
        review_messages = [
            {
                "role": "system",
                "content": "You are reviewing your own thought process. Summarize what happened, what you learned, and what to remember.",
            },
            {"role": "user", "content": review_prompt},
        ]
        review = await self.llm.chat_async(review_messages, temperature=0.5, max_tokens=300)
        learning = review["content"]

        self.monologue.think(
            question="What did I learn from this cycle?",
            answer=learning,
            type=ThoughtType.LEARNING,
        )

        # ── Persist to memory ─────────────────────────────────────────
        await self.memory.add_episode(
            session_id=self.session_id,
            summary=f"Input: {user_input[:100]} → {learning[:200]}",
            outcome=learning[:200],
            confidence=confidence,
        )
        await self.memory.save_thoughts(self.session_id, self.monologue.to_dicts())

        # ── Update identity ───────────────────────────────────────────
        self.identity.evolve(learning)
        self.identity.add_to_history(f"Cycle: {user_input[:80]}")
        self.identity.save()

        # Determine final answer: prefer tool outputs for real tool calls,
        # otherwise use the refined reflection output.
        result_output = compose_cycle_output(reflection_result["output"], tool_results)

        duration = time.time() - start_time

        return CycleResult(
            output=result_output,
            inner_monologue=self.monologue.format(),
            confidence=confidence,
            rounds=reflection_result["rounds"],
            tool_results=tool_results,
            session_id=self.session_id,
            duration=duration,
        )
