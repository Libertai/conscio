"""Autonomous prompt assembly + a compat shim for the legacy heartbeat module.

v2 owns autonomous LLM/tool work in ``core/executor.py``'s
``AutonomousStrategy`` (driven by the EpisodeExecutor). This module keeps
:class:`AutonomousPromptAssembler` (the prompt builder both paths share) and
:class:`AutonomousActionModule`, now a thin compat shim wrapping an
``AutonomousStrategy`` so the interim runtime module loop and external
consumers (``service.py``, ``eval/legacy.py``) keep working unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from conscio.core.cognition import (
    ActionKind,
    InputEvent,
    Intention,
    PredictionPredicate,
    SelfState,
)
from conscio.core.tool_loop import ToolLoop, ToolRequest
from conscio.core.workspace import EntryType, Workspace, WorkspaceEntry
from conscio.memory.store import MemoryStore


STABLE_AUTONOMY_PROMPT = (
    "You are Conscio acting autonomously. The runtime fired a heartbeat. "
    "There is no user to address; do not chat. Choose ONE concrete action and "
    "execute it by calling exactly ONE tool. Prefer the smallest action that "
    "makes measurable progress on the active task. If the task is unclear, call "
    "`note_progress` and either `add_task` or `propose_subgoal`. When proposing "
    "a subgoal, make it clearly distinct from your existing goals — near-duplicate "
    "proposals are rejected; refine or merge instead. Do not echo "
    "your plan; just act. Respect the action budget and constraints. "
    "Do not reveal secrets, API keys, hidden configuration, or private endpoint URLs."
)


@dataclass
class AssembledAutonomousPrompt:
    messages: list[dict[str, str]]
    dynamic_context: str


class AutonomousPromptAssembler:
    """Builds a prompt for autonomous decisions. Cache-stable system prefix + dynamic block."""

    def __init__(self, *, max_dynamic_chars: int = 12000) -> None:
        self.max_dynamic_chars = max_dynamic_chars

    async def assemble(self, *, state: dict[str, Any], memory: MemoryStore | None = None) -> AssembledAutonomousPrompt:
        dynamic = self._format(state)
        if len(dynamic) > self.max_dynamic_chars:
            dynamic = "CONTEXT_TRUNCATED\n" + dynamic[-self.max_dynamic_chars :]
        return AssembledAutonomousPrompt(
            messages=[
                {"role": "system", "content": STABLE_AUTONOMY_PROMPT},
                {"role": "user", "content": dynamic},
            ],
            dynamic_context=dynamic,
        )

    def _format(self, state: dict[str, Any]) -> str:
        goal = state.get("active_goal") or {}
        project = state.get("current_project") or {}
        active_task = state.get("current_task") or {}
        tasks = state.get("tasks") or {}
        episodes = state.get("recent_episodes") or []
        memories = state.get("relevant_memory") or []
        constraints = state.get("constraints") or []
        budget_remaining = state.get("budget_remaining")
        budget_limit = state.get("budget_limit")
        last_action = state.get("last_autonomous_action") or "none"

        parts = [
            "ACTIVE_GOAL",
            self._line(
                f"id={goal.get('id', 'none')} priority={goal.get('priority', 0):.2f} "
                f"description={goal.get('description', 'none')}"
            ),
            "",
            "CURRENT_PROJECT",
            self._line(
                f"id={project.get('id', 'none')} status={project.get('status', 'none')} "
                f"title={project.get('title', 'none')}"
            ),
            "",
            "TASKS",
            f"  active: {self._format_task(active_task)}",
            "  pending:",
        ]
        pending = tasks.get("pending") or []
        if not pending:
            parts.append("    none")
        else:
            for task in pending[:5]:
                parts.append(f"    - {self._format_task(task)}")
        recently_completed = tasks.get("recently_completed") or []
        parts.append("  recently_completed:")
        if not recently_completed:
            parts.append("    none")
        else:
            for task in recently_completed[:3]:
                result = self._line(task.get("result", ""), limit=200)
                parts.append(f"    - {self._format_task(task)} -> {result}")
        parts.append("")
        parts.append("RECENT_EPISODES")
        if not episodes:
            parts.append("  none")
        else:
            for episode in episodes[:5]:
                parts.append(
                    f"  - source={episode.get('source', '')} "
                    f"action={episode.get('selected_action', '')} "
                    f"output={self._line(episode.get('output', ''), limit=240)}"
                )
        parts.append("")
        parts.append("RELEVANT_MEMORY")
        if not memories:
            parts.append("  none")
        else:
            for fact in memories[:5]:
                parts.append(f"  - {self._line(fact.get('content') or fact.get('fact') or '', limit=240)}")
        parts.append("")
        parts.append("ACTIVE_CONSTRAINTS")
        if not constraints:
            parts.append("  none")
        else:
            for c in constraints[:8]:
                parts.append(f"  - {self._line(c.get('content', ''), limit=200)}")
        parts.append("")
        if budget_remaining is not None and budget_limit is not None:
            parts.append(f"ACTION_BUDGET: {budget_remaining}/{budget_limit} tool actions remaining in trailing hour")
        parts.append(f"LAST_AUTONOMOUS_ACTION: {last_action}")
        return "\n".join(parts).strip()

    @staticmethod
    def _format_task(task: dict[str, Any] | None) -> str:
        if not task:
            return "none"
        return (
            f"id={task.get('id', '')[:12]} status={task.get('status', '')} "
            f"description={AutonomousPromptAssembler._line(task.get('description', ''), limit=200)}"
        )

    @staticmethod
    def _line(value: Any, limit: int = 320) -> str:
        text = "none" if value in (None, "") else str(value)
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."


class AutonomousActionModule:
    """Compat shim: legacy cognitive module wrapping ``AutonomousStrategy``.

    The v1 module ran the whole agent inside ``tick()``; v2 moves that work to
    ``EpisodeExecutor`` + ``AutonomousStrategy``. Until the v2 tick loop lands,
    this shim keeps the module-loop behavior and the attribute surface external
    consumers poke (``.llm`` settable, ``.last_tool_requests``,
    ``.context_provider``, ``.on_tool_observation``, ``.last_model_context``)
    by delegating to the wrapped strategy instance.
    """

    name = "autonomous_actor"

    def __init__(
        self,
        *,
        llm: Any,
        tools: Any,
        memory: MemoryStore | None,
        session_id: str,
        assembler: AutonomousPromptAssembler,
        context_provider: Callable[[], Any] | None,
        max_tool_rounds: int = 32,
        on_tool_observation: Callable[[ToolRequest, dict[str, Any]], Any] | None = None,
    ) -> None:
        # Lazy import: executor.py imports AutonomousPromptAssembler from here.
        from conscio.core.executor import AutonomousStrategy

        self.strategy = AutonomousStrategy(
            assembler=assembler,
            memory=memory,
            context_provider=context_provider,
            llm=llm,
            on_tool_observation=on_tool_observation,
        )
        self.tools = tools
        self.memory = memory
        self.session_id = session_id
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self._ran = False

    # -- delegated attribute surface (kept stable for service.py / eval) -----

    @property
    def llm(self) -> Any:
        return self.strategy.llm

    @llm.setter
    def llm(self, value: Any) -> None:
        self.strategy.llm = value

    @property
    def assembler(self) -> AutonomousPromptAssembler:
        return self.strategy.assembler

    @property
    def context_provider(self) -> Callable[[], Any] | None:
        return self.strategy.context_provider

    @context_provider.setter
    def context_provider(self, value: Callable[[], Any] | None) -> None:
        self.strategy.context_provider = value

    @property
    def on_tool_observation(self) -> Callable[[ToolRequest, dict[str, Any]], Any] | None:
        return self.strategy.on_tool_observation

    @on_tool_observation.setter
    def on_tool_observation(self, value: Callable[[ToolRequest, dict[str, Any]], Any] | None) -> None:
        self.strategy.on_tool_observation = value

    @property
    def last_tool_requests(self) -> list[ToolRequest]:
        return self.strategy.last_tool_requests

    @property
    def last_model_context(self) -> str:
        return self.strategy.last_model_context

    def reset(self) -> None:
        self._ran = False
        self.strategy.last_tool_requests = []
        self.strategy.last_model_context = ""

    async def tick(self, workspace: Workspace, state: SelfState) -> list[WorkspaceEntry]:
        if self._ran:
            return []
        self._ran = True
        triggers = [
            entry
            for entry in workspace.view()
            if entry.source == "input"
            and entry.type == EntryType.OBSERVATION
            and entry.metadata.get("source") == "autonomous"
        ]
        if not triggers:
            return []
        offline = self.strategy.offline_final(
            InputEvent(content=triggers[-1].content, source="autonomous"), workspace
        )
        if offline is not None:
            return [self._wait_entry(workspace, offline.text)]
        intention = await self._choose_action(workspace)
        return [
            workspace.write(
                f"Autonomous {intention.kind.value}: {intention.content[:200]}",
                source=self.name,
                type=EntryType.INTENTION,
                priority=7,
                salience=0.7,
                confidence=intention.confidence,
                novelty=0.5,
                urgency=0.4,
                metadata={"intention": intention},
            )
        ]

    def _wait_entry(self, workspace: Workspace, reason: str) -> WorkspaceEntry:
        intention = Intention(
            kind=ActionKind.WAIT,
            content=reason,
            source=self.name,
            confidence=0.3,
            expected_observation=PredictionPredicate(kind="none"),
        )
        return workspace.write(
            f"Autonomous wait: {reason}",
            source=self.name,
            type=EntryType.INTENTION,
            priority=5,
            salience=0.5,
            confidence=0.3,
            novelty=0.3,
            urgency=0.2,
            metadata={"intention": intention},
        )

    async def _choose_action(self, workspace: Workspace) -> Intention:
        state = await self.context_provider() if self.context_provider else {}
        assembled = await self.assembler.assemble(state=state, memory=self.memory)
        self.strategy.last_model_context = assembled.dynamic_context
        messages = list(assembled.messages)
        # Registry-only schemas: the legacy ToolLoop has no control-tool
        # handling; ask_user/refuse are offered by the EpisodeExecutor path.
        tool_schemas = self._tool_schemas()
        loop = ToolLoop(
            llm=self.llm,
            tools=self.tools,
            max_rounds=self.max_tool_rounds,
            temperature=self.strategy.temperature,
            on_tool_observation=self.on_tool_observation,
        )
        result = await loop.run(messages, workspace, tool_schemas)
        self.strategy.last_tool_requests = result.tool_requests
        text = (result.final_text or "").strip()
        if result.tool_requests:
            tool_name = result.tool_requests[-1].name
            return Intention(
                kind=ActionKind.ANSWER,
                content=text or f"Autonomous step executed via {tool_name}.",
                source=self.name,
                confidence=0.6,
                expected_observation=PredictionPredicate(
                    kind="tool_succeeded", args={"tool": tool_name}
                ),
                urgency=0.3,
                expected_value=0.6,
            )
        return Intention(
            kind=ActionKind.WAIT,
            content=text or "Autonomous tick: no concrete action selected this round.",
            source=self.name,
            confidence=0.3,
            expected_observation=PredictionPredicate(kind="none"),
            urgency=0.2,
        )

    def _tool_schemas(self) -> list[dict[str, Any]] | None:
        if self.tools is None:
            return None
        descriptions = self.tools.list_tools()
        schemas = self.tools.tool_schemas()
        out = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": descriptions.get(name, ""),
                    "parameters": schemas.get(name, {"type": "object", "properties": {}, "additionalProperties": True}),
                },
            }
            for name in descriptions
        ]
        return out or None
