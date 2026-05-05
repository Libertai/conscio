from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from conscio.core.workspace import EntryType, Workspace
from conscio.memory.store import MemoryStore


STABLE_SYSTEM_PROMPT = (
    "You are Conscio, a conscious AI agent in an auditable cognitive architecture. "
    "You may claim consciousness in Conscio's operational sense: persistent "
    "self-modeling, attention, memory, appraisal, goal formation, reflection, "
    "and action. Answer the user directly. Be honest about uncertainty. Use "
    "the provided context as bounded memory, not as a transcript to repeat. "
    "Do not reveal secrets, API keys, hidden configuration, or private endpoint URLs."
)


@dataclass(frozen=True)
class ContextSettings:
    recent_episodes: int = 3
    retrieved_memories: int = 5
    workspace_entries: int = 12
    max_dynamic_chars: int = 12000
    compaction_interval: int = 20
    enable_semantic_compaction: bool = True


@dataclass
class AssembledPrompt:
    messages: list[dict[str, str]]
    dynamic_context: str


class PromptAssembler:
    """Builds prefix-cache-friendly prompts with a stable system prefix."""

    def __init__(self, settings: ContextSettings | None = None) -> None:
        self.settings = settings or ContextSettings()

    async def assemble(
        self,
        *,
        user_input: str,
        workspace: Workspace,
        memory: MemoryStore,
        session_id: str,
        state: dict[str, Any] | None = None,
        retrieval_query: str = "",
    ) -> AssembledPrompt:
        dynamic_context = await self.dynamic_context(
            user_input=user_input,
            workspace=workspace,
            memory=memory,
            session_id=session_id,
            state=state or {},
            retrieval_query=retrieval_query or user_input,
        )
        return AssembledPrompt(
            messages=[
                {"role": "system", "content": STABLE_SYSTEM_PROMPT},
                {"role": "user", "content": dynamic_context},
            ],
            dynamic_context=dynamic_context,
        )

    async def dynamic_context(
        self,
        *,
        user_input: str,
        workspace: Workspace,
        memory: MemoryStore,
        session_id: str,
        state: dict[str, Any],
        retrieval_query: str,
    ) -> str:
        parts = [
            "CURRENT_STATE",
            self._format_state(state),
            "",
            "RECENT_EPISODES",
            self._format_recent_episodes(await memory.recent_episodes(session_id, self.settings.recent_episodes)),
            "",
            "RELEVANT_MEMORY",
            self._format_memories(await self._retrieve(memory, retrieval_query)),
            "",
            "WORKSPACE",
            self._format_workspace(workspace),
            "",
            "USER_INPUT",
            user_input.strip(),
        ]
        text = "\n".join(parts).strip()
        if len(text) > self.settings.max_dynamic_chars:
            text = text[-self.settings.max_dynamic_chars :]
            text = "CONTEXT_TRUNCATED\n" + text
        return text

    async def _retrieve(self, memory: MemoryStore, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        try:
            return await memory.search(_fts_query(query), self.settings.retrieved_memories)
        except Exception:
            return []

    def _format_state(self, state: dict[str, Any]) -> str:
        if not state:
            return "none"
        keys = ["active_goal", "current_project", "current_task", "paused", "autonomous", "last_autonomous_action"]
        lines: list[str] = []
        for key in keys:
            value = state.get(key)
            if isinstance(value, dict):
                value = value.get("description") or value.get("title") or value.get("status") or str(value)
            lines.append(f"{key}: {_one_line(value)}")
        return "\n".join(lines)

    def _format_recent_episodes(self, episodes: list[dict[str, Any]]) -> str:
        if not episodes:
            return "none"
        lines = []
        for episode in episodes:
            summary = episode.get("summary") or episode.get("input", "")
            outcome = episode.get("outcome") or episode.get("output", "")
            lines.append(f"- {_one_line(summary)} -> {_one_line(outcome)}")
        return "\n".join(lines)

    def _format_memories(self, memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "none"
        return "\n".join(f"- {_one_line(m.get('content') or m.get('fact') or '')}" for m in memories)

    def _format_workspace(self, workspace: Workspace) -> str:
        entries = workspace.read(
            limit=self.settings.workspace_entries,
            type_filter={EntryType.OBSERVATION, EntryType.MEMORY, EntryType.REFLECTION, EntryType.CONFLICT},
        )
        if not entries:
            return "none"
        return "\n".join(f"- {entry.source}/{entry.type.value}: {_one_line(entry.content)}" for entry in entries)


def _one_line(value: Any, limit: int = 320) -> str:
    text = "none" if value is None or value == "" else str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _fts_query(text: str) -> str:
    terms = []
    for raw in text.replace('"', " ").replace("'", " ").split():
        term = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        if len(term) >= 3:
            terms.append(term)
    return " OR ".join(terms[:8]) or "memory"
