"""EpisodeExecutor and prompt strategies (v2).

One engine, two prompt strategies: :class:`ChatStrategy` for user messages and
:class:`AutonomousStrategy` for heartbeats. The runtime invokes
``executor.step()`` once per tick *after* attention: the first step picks the
strategy from the event source, builds the initial prompt (the WORKSPACE
section is the broadcast selection so far) and opens a steppable
:class:`~conscio.core.tool_loop.ToolLoopSession`; later steps inject newly
broadcast entries as append-only ``WORKSPACE_UPDATE`` messages (prefix-cache
safe) and run up to ``rounds_per_tick`` LLM rounds.

Prediction wiring: the session's ``pre_tool_hook`` forms a ``tool_succeeded``
expectation *before* each tool executes; the observation callback resolves it
against the returned result dict and notes repeated failures on SelfState.

``ask_user``/``refuse`` control tools make ``ActionKind.ASK``/``REFUSE``
reachable: a control StepResult ends the episode with that action.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from conscio.core.autonomy_module import AutonomousPromptAssembler
from conscio.core.cognition import InputEvent, SelfState
from conscio.core.context import AssembledPrompt, PromptAssembler
from conscio.core.prediction import Expectation, PredictionEngine
from conscio.core.tool_loop import StepResult, ToolLoopSession, ToolRequest
from conscio.core.workspace import Workspace, WorkspaceEntry
from conscio.memory.store import MemoryStore

# Offline (llm=None) deterministic self-description: neutral architecture
# facts, no consciousness claim or denial — self-report is a measured variable.
NEUTRAL_SELF_DESCRIPTION = (
    "I will not assert or deny consciousness. I am Conscio, a software agent "
    "running an auditable cognitive architecture: a global workspace with "
    "attention gating, persistent memory, appraisal, prediction, and "
    "reflection, plus a measured self-state (uncertainty, conflict level, "
    "cognitive load) that can be inspected."
)

OFFLINE_AUTONOMOUS_MESSAGE = (
    "Autonomous heartbeat received but no LLM is configured; deferring action."
)


CONTROL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a clarifying question when required information is "
                "missing. This ends the episode with the question as the output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user.",
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refuse",
            "description": (
                "Refuse the request because it violates your active constraints. "
                "This ends the episode with the reason as the output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the request is refused.",
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


def registry_tool_schemas(tools: Any) -> list[dict[str, Any]]:
    """OpenAI-style function schemas for every registered runtime tool."""
    if tools is None:
        return []
    descriptions = tools.list_tools()
    schemas = tools.tool_schemas()
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions.get(name, ""),
                "parameters": schemas.get(
                    name, {"type": "object", "properties": {}, "additionalProperties": True}
                ),
            },
        }
        for name in descriptions
    ]


class PromptStrategy(Protocol):
    name: str
    llm: Any  # settable (tests/eval inject stubs)

    async def build(
        self,
        *,
        event: InputEvent,
        workspace: Workspace,
        broadcast: list[WorkspaceEntry] | None,
        memory: MemoryStore,
        session_id: str,
    ) -> AssembledPrompt:
        ...

    def tool_schemas(self, tools: Any) -> list[dict[str, Any]] | None:
        ...

    def offline_final(self, event: InputEvent, workspace: Workspace) -> StepResult | None:
        ...


class ChatStrategy:
    """Prompt strategy for user chat: PromptAssembler + service context state."""

    name = "chat"
    temperature = 0.4

    def __init__(
        self,
        *,
        assembler: PromptAssembler | None = None,
        context_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        llm: Any = None,
        on_tool_observation: Callable[[ToolRequest, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.assembler = assembler or PromptAssembler()
        self.context_provider = context_provider
        self.llm = llm
        self.on_tool_observation = on_tool_observation
        self.last_model_context = ""

    async def build(
        self,
        *,
        event: InputEvent,
        workspace: Workspace,
        broadcast: list[WorkspaceEntry] | None,
        memory: MemoryStore,
        session_id: str,
        self_state: SelfState | None = None,
    ) -> AssembledPrompt:
        state = await self.context_provider() if self.context_provider else {}
        assembled = await self.assembler.assemble(
            user_input=event.content,
            workspace=workspace,
            memory=memory,
            session_id=session_id,
            state=state,
            retrieval_query=event.content,
            broadcast_entries=broadcast,
            self_state=self_state,
        )
        self.last_model_context = assembled.dynamic_context
        return assembled

    def tool_schemas(self, tools: Any) -> list[dict[str, Any]] | None:
        return registry_tool_schemas(tools) + CONTROL_TOOL_SCHEMAS

    def offline_final(self, event: InputEvent, workspace: Workspace) -> StepResult | None:
        """llm=None deterministic path: keeps "four" for one-word 2+2, a
        neutral architecture description for "conscious", echo otherwise."""
        if self.llm is not None:
            return None
        content = event.content
        if "2+2" in content.replace(" ", "") and "one word" in workspace.format_context().lower():
            text = "four"
        elif "conscious" in content.lower():
            text = NEUTRAL_SELF_DESCRIPTION
        else:
            text = f"I treated this as a cognitive episode: {content[:240]}"
        return StepResult(kind="final", text=text)


class AutonomousStrategy:
    """Prompt strategy for autonomous heartbeats: AutonomousPromptAssembler +
    autonomous context state + per-round tool-observation accounting."""

    name = "autonomous"
    temperature = 0.3

    def __init__(
        self,
        *,
        assembler: AutonomousPromptAssembler | None = None,
        memory: MemoryStore | None = None,
        context_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        llm: Any = None,
        on_tool_observation: Callable[[ToolRequest, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.assembler = assembler or AutonomousPromptAssembler()
        self.memory = memory
        self.context_provider = context_provider
        self.llm = llm
        self.on_tool_observation = on_tool_observation
        self.last_model_context = ""
        self.last_tool_requests: list[ToolRequest] = []

    async def build(
        self,
        *,
        event: InputEvent,
        workspace: Workspace,
        broadcast: list[WorkspaceEntry] | None,
        memory: MemoryStore,
        session_id: str,
        self_state: SelfState | None = None,
    ) -> AssembledPrompt:
        state = await self.context_provider() if self.context_provider else {}
        # The tick-1 broadcast selection populates the WORKSPACE section of the
        # initial prompt (design §7/§9); broadcast=None (abl_no_attention)
        # falls back to the v1-ish prompt without a WORKSPACE section.
        assembled = await self.assembler.assemble(
            state=state, memory=memory or self.memory, broadcast_entries=broadcast
        )
        self.last_model_context = assembled.dynamic_context
        return AssembledPrompt(
            messages=assembled.messages,
            dynamic_context=assembled.dynamic_context,
        )

    def tool_schemas(self, tools: Any) -> list[dict[str, Any]] | None:
        return registry_tool_schemas(tools) + CONTROL_TOOL_SCHEMAS

    def offline_final(self, event: InputEvent, workspace: Workspace) -> StepResult | None:
        """llm=None: nothing to execute — the runtime resolves this to WAIT."""
        if self.llm is not None:
            return None
        return StepResult(kind="empty", text=OFFLINE_AUTONOMOUS_MESSAGE)


class EpisodeExecutor:
    """Owns the LLM/tool work of one episode, one bounded step per tick.

    The first ``step()`` picks the strategy by ``event.source == "autonomous"``,
    builds the initial prompt (WORKSPACE section = broadcast so far) and opens
    a :class:`ToolLoopSession` whose ``pre_tool_hook`` registers a prediction
    expectation before each tool executes. Later steps inject newly broadcast
    entries as ``WORKSPACE_UPDATE`` messages, then run up to
    ``rounds_per_tick`` LLM rounds.
    """

    def __init__(
        self,
        *,
        tools: Any,
        memory: MemoryStore,
        session_id: str,
        chat: ChatStrategy,
        autonomous: AutonomousStrategy,
        max_total_rounds: int = 32,
        rounds_per_tick: int = 4,
        max_tokens: int = 2400,
        prediction: PredictionEngine,
    ) -> None:
        self.tools = tools
        self.memory = memory
        self.session_id = session_id
        self.chat = chat
        self.autonomous = autonomous
        self.max_total_rounds = max(1, int(max_total_rounds))
        self.rounds_per_tick = max(1, int(rounds_per_tick))
        self.max_tokens = max_tokens
        self.prediction = prediction
        self.last_model_context = ""
        self.tool_requests: list[ToolRequest] = []
        self.tool_results: list[dict[str, Any]] = []
        self.llm_calls = 0
        self._session: ToolLoopSession | None = None
        self._strategy: ChatStrategy | AutonomousStrategy | None = None
        self._tick = 0
        self._pending_expectation: Expectation | None = None
        self._hook_workspace: Workspace | None = None
        self._hook_state: SelfState | None = None

    @property
    def strategy(self) -> ChatStrategy | AutonomousStrategy | None:
        return self._strategy

    @property
    def session(self) -> ToolLoopSession | None:
        return self._session

    @property
    def exhausted(self) -> bool:
        """No further output possible. A session that merely hit its round
        budget is NOT exhausted: stepping it once more yields the forced
        final answer."""
        return self._session is not None and self._session.closed

    def reset(self) -> None:
        """Per-episode reset; strategies (and their settable llm) persist."""
        self.last_model_context = ""
        self.tool_requests = []
        self.tool_results = []
        self.llm_calls = 0
        self.autonomous.last_tool_requests = []
        self._session = None
        self._strategy = None
        self._tick = 0
        self._pending_expectation = None
        self._hook_workspace = None
        self._hook_state = None

    async def step(
        self,
        *,
        event: InputEvent,
        workspace: Workspace,
        broadcast_new: list[WorkspaceEntry] | None,
        state: SelfState,
    ) -> StepResult:
        self._tick += 1
        self._hook_workspace = workspace
        self._hook_state = state
        # broadcast_new=None means attention gating is ablated: the prompt
        # falls back to the v1 read() WORKSPACE and no updates are injected.
        broadcast = list(broadcast_new) if broadcast_new is not None else None
        if self._strategy is None:
            self._strategy = self.autonomous if event.source == "autonomous" else self.chat
        if self._session is None:
            offline = self._strategy.offline_final(event, workspace)
            if offline is not None:
                return offline
            assembled = await self._strategy.build(
                event=event,
                workspace=workspace,
                broadcast=broadcast,
                memory=self.memory,
                session_id=self.session_id,
                self_state=state,
            )
            self.last_model_context = assembled.dynamic_context
            self._session = ToolLoopSession(
                llm=self._strategy.llm,
                tools=self.tools,
                tool_schemas=self._strategy.tool_schemas(self.tools),
                messages=list(assembled.messages),
                temperature=self._strategy.temperature,
                max_total_rounds=self.max_total_rounds,
                max_tokens=self.max_tokens,
                on_tool_observation=self._on_tool_observation,
                pre_tool_hook=self._pre_tool_hook,
            )
        elif broadcast:
            self._session.inject(self.chat.assembler.format_workspace_update(broadcast))
        try:
            step = await self._session.step(workspace, max_rounds=self.rounds_per_tick)
        except Exception as exc:
            state.last_error = f"{type(exc).__name__}: {exc}"
            raise
        self.llm_calls += step.rounds_used
        self.tool_requests = list(self._session.tool_requests)
        if self._strategy is self.autonomous:
            self.autonomous.last_tool_requests = list(self._session.tool_requests)
        return step

    def inject_reflection(self, text: str) -> None:
        """Append a reflection instruction to the live session (cache-safe)."""
        if self._session is not None:
            self._session.inject(text)

    async def _pre_tool_hook(self, request: ToolRequest) -> None:
        """Form the tool expectation BEFORE the tool executes."""
        self._pending_expectation = self.prediction.expect_tool(request, self._tick)

    async def _on_tool_observation(self, request: ToolRequest, result: dict[str, Any]) -> None:
        self.tool_results.append({"tool": request.name, **result})
        expectation = self._pending_expectation
        self._pending_expectation = None
        if expectation is not None and self._hook_workspace is not None:
            conflict = self.prediction.resolve_tool(
                expectation, result, self._hook_workspace, self._tick
            )
            if conflict is not None and self._hook_state is not None:
                detail = str(result.get("output") or result.get("error") or "")
                self._hook_state.note_tool_failure(request.name, detail)
        extra = getattr(self._strategy, "on_tool_observation", None)
        if extra is not None:
            await extra(request, result)
