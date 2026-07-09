"""The agent's self-management tools (memory, tasks, goals, procedures,
sub-agents), factored out of the service so the tool contracts are readable
and testable apart from the orchestration core. `register_self_tools` is
called once from ConscioService.__init__ with the service itself as the
closure target."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from conscio.core.context import provenance_marker
from conscio.core.subagent import SubagentRunner, SubagentSpec
from conscio.tools import PolicyToolRegistry, ScopedToolRegistry

if TYPE_CHECKING:
    from conscio.service import ConscioService


def register_self_tools(self: ConscioService, tools: PolicyToolRegistry) -> None:
    async def remember_fact(
        fact: str | None = None,
        facts: list[str] | None = None,
        source: str = "agent",
        confidence: str = "HIGH",
        input: str | None = None,
    ) -> dict[str, Any]:
        # Confabulated-provenance defense: the model must never mint
        # user-tier (trust 3) facts. Whatever `source` it passes, only
        # 'agent' is accepted from this tool.
        if source != "agent":
            source = "agent"
        candidates: list[str] = []
        if facts:
            candidates.extend(str(item) for item in facts)
        if fact:
            candidates.append(fact)
        if input:
            candidates.append(input)
        cleaned = [" ".join(item.strip().split()) for item in candidates if item and item.strip()]
        if not cleaned:
            return {"output": "No fact provided.", "error": True}
        episode_id = self.runtime.workspace.current_episode or None
        stored = len(dict.fromkeys(cleaned))
        if self.episode_taint.web:
            # Quarantine: this episode fetched web content, so any fact it
            # stores is web-derived (origin=web:<url>, trust tier 1).
            url = self.episode_taint.urls[-1] if self.episode_taint.urls else ""
            origin = f"web:{url}" if url else "web"
            for item in dict.fromkeys(cleaned):
                await self.memory.add_fact(
                    item[:500],
                    origin=origin,
                    trust=1,
                    episode_id=episode_id,
                    confidence=confidence,
                )
            return {
                "output": (
                    f"Stored {stored} fact(s) in semantic memory "
                    f"(web-derived this episode: origin={origin}, trust=1)."
                ),
                "error": False,
            }
        for item in dict.fromkeys(cleaned):
            await self.memory.add_fact(
                item[:500], source=source, episode_id=episode_id, confidence=confidence
            )
        return {"output": f"Stored {stored} fact(s) in semantic memory.", "error": False}

    async def search_memory(
        query: str | None = None,
        limit: int = 10,
        input: str | None = None,
    ) -> dict[str, Any]:
        q = query if query is not None else input
        if not q:
            return {"output": "No memory query provided.", "error": True}
        rows = await self.memory.search_facts(q, limit)
        if not rows:
            return {"output": "No matching semantic memories found.", "error": False}
        # Same provenance marking as the prompt assemblers: web-derived
        # (trust 1) and user facts stay visibly labelled even when the
        # model pulls them explicitly through this tool.
        lines = [
            f"- {provenance_marker(row)}{row['fact']} ({row.get('confidence', '')})"
            for row in rows
        ]
        return {"output": "\n".join(lines), "error": False, "results": rows}

    async def set_task_status(
        task_id: str,
        status: str,
        result: str = "",
    ) -> dict[str, Any]:
        allowed = {"pending", "active", "done", "blocked"}
        if status not in allowed:
            return {"output": f"Invalid status '{status}'. Use one of: {sorted(allowed)}.", "error": True}
        task = await self.autonomy.get_task(task_id)
        if task is None:
            return {"output": f"Unknown task_id '{task_id}'.", "error": True}
        await self.autonomy.update_task(task_id, status=status, result=result[:1000])
        await self.autonomy.record_action("self_management")
        return {"output": f"Task {task_id} -> {status}.", "error": False}

    async def add_task(
        project_id: str,
        description: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        project = await self.autonomy.get_project(project_id)
        if project is None:
            return {"output": f"Unknown project_id '{project_id}'.", "error": True}
        task = await self.autonomy.add_task(
            project_id, description, tool_name=tool_name, tool_args=tool_args or {}
        )
        await self.autonomy.record_action("self_management")
        return {"output": f"Added task {task.id}: {description[:200]}.", "error": False, "task_id": task.id}

    async def note_progress(
        project_id: str | None = None,
        content: str = "",
    ) -> dict[str, Any]:
        text = content.strip()
        if not text:
            return {"output": "No content provided.", "error": True}
        note = f"[project={project_id or 'none'}] {text[:1500]}"
        await self.autonomy.note_progress(None, note)
        await self.autonomy.record_action("self_management")
        return {"output": "Progress note recorded.", "error": False}

    async def propose_subgoal(
        description: str,
        rationale: str = "",
    ) -> dict[str, Any]:
        text = description.strip()
        if not text:
            return {"output": "No description provided.", "error": True}
        if self.episode_taint.web:
            origin = self.episode_taint.urls[-1] if self.episode_taint.urls else "external content"
            influence = await self.goals.defer_influence(
                text[:500],
                kind="goal",
                source="external_content",
                reasoning=(
                    "Deferred because the proposal was made in an episode "
                    f"tainted by external content ({origin})."
                ),
                response="I deferred this proposed goal for later clean review.",
            )
            await self.autonomy.record_action("self_management")
            return {
                "output": f"Deferred proposed goal as influence {influence.id}.",
                "error": False,
                "accepted": False,
                "influence_id": influence.id,
            }
        proposed = await self.goals.propose_goal(
            text[:500],
            rationale=rationale[:500],
            source="self_proposed",
            priority=0.5,
            confidence=0.5,
            appraisal_weight=0.5,
        )
        await self.autonomy.record_action("self_management")
        if not proposed["accepted"]:
            return {
                "output": str(proposed["reason"]),
                "error": False,
                "accepted": False,
                "similar_goal_id": proposed.get("similar_goal_id"),
            }
        goal = proposed["goal"]
        return {
            "output": f"Proposed new goal {goal.id}: {text[:200]}.",
            "error": False,
            "accepted": True,
            "goal_id": goal.id,
        }

    async def learn_procedure(
        name: str,
        description: str,
        steps: str,
        trigger: str = "",
    ) -> dict[str, Any]:
        if self.episode_taint.web:
            return {
                "output": (
                    "learn_procedure is disabled after external content or "
                    "network reads in this episode. Re-evaluate it in a clean episode."
                ),
                "error": True,
            }
        slug = " ".join(name.strip().split())
        if not slug or not description.strip() or not steps.strip():
            return {"output": "learn_procedure requires name, description and steps.", "error": True}
        await self.memory.upsert_procedure(
            slug[:80],
            description.strip()[:500],
            steps.strip()[:2000],
            trigger=trigger.strip()[:300],
        )
        await self.autonomy.record_action("self_management")
        return {"output": f"Recorded procedure '{slug[:80]}'.", "error": False}

    tools.register(
        "remember_fact",
        remember_fact,
        "Store one semantic fact in long-term memory.",
        capabilities={"memory_write"},
        schema={
            "type": "object",
            "properties": {
                "fact": {"type": "string"},
                "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"], "default": "HIGH"},
            },
            "required": ["fact"],
            "additionalProperties": False,
        },
    )
    tools.register(
        "remember_facts",
        remember_fact,
        "Store one or more semantic facts in long-term memory.",
        capabilities={"memory_write"},
        schema={
            "type": "object",
            "properties": {
                "facts": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"], "default": "HIGH"},
            },
            "required": ["facts"],
            "additionalProperties": False,
        },
    )
    tools.register(
        "search_memory",
        search_memory,
        "Search semantic long-term memory facts.",
        capabilities={"memory_read"},
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    tools.register(
        "set_task_status",
        set_task_status,
        "Mark a task pending/active/done/blocked. Use done when the task is "
        "complete; blocked only after attempting it.",
        capabilities={"self_management"},
        schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "active", "done", "blocked"]},
                "result": {"type": "string", "description": "Short result/note to attach to the task."},
            },
            "required": ["task_id", "status"],
            "additionalProperties": False,
        },
    )
    tools.register(
        "add_task",
        add_task,
        "Add a new task to a project. Use to break work into concrete next steps.",
        capabilities={"self_management"},
        schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "description": {"type": "string"},
                "tool_name": {"type": "string", "description": "Optional preferred tool to invoke."},
                "tool_args": {"type": "object", "description": "Optional preferred args for tool_name."},
            },
            "required": ["project_id", "description"],
            "additionalProperties": False,
        },
    )
    tools.register(
        "note_progress",
        note_progress,
        "Record a free-form progress note in the autonomous trace and episodic memory.",
        capabilities={"self_management"},
        schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    )
    tools.register(
        "propose_subgoal",
        propose_subgoal,
        "Propose a new self-authored goal. Will be reviewed in the next goal-review cycle.",
        capabilities={"self_modification"},
        schema={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    )
    tools.register(
        "learn_procedure",
        learn_procedure,
        "Record a validated, reusable procedure you have confirmed works.",
        capabilities={"self_modification", "memory_write"},
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "steps": {"type": "string"},
                "trigger": {"type": "string"},
            },
            "required": ["name", "description", "steps"],
            "additionalProperties": False,
        },
    )

    async def spawn_subagent(
        task: str,
        context: str = "",
        tools_allowed: list[str] | None = None,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        text = (task or "").strip()
        if not text:
            return {"output": "No task provided.", "error": True}
        parent_id = self.runtime.workspace.current_episode or ""
        sub_id = uuid.uuid4().hex
        scoped = ScopedToolRegistry(
            self.runtime.tools,
            allowed=set(tools_allowed) if tools_allowed else None,
            denied_capabilities=frozenset(self.config.subagent_deny_capabilities),
        )
        spec = SubagentSpec(task=text, context=context or "", tools=tools_allowed, max_rounds=max_rounds)
        runner = SubagentRunner(
            llm=self._subagent_llm(),
            tools=scoped,
            on_tool_observation=self._subagent_observer(sub_id),
            emit=self._event_broker.emit,
            max_rounds=self.config.subagent_max_rounds,
            max_seconds=self.config.subagent_max_seconds,
        )
        outcome = await runner.run(spec, parent_episode_id=parent_id, subagent_id=sub_id)
        await self.memory.record_episode(
            episode_id=outcome.id,
            source="subagent",
            event_type="subagent_task",
            input=text,
            output=outcome.output or outcome.error,
            selected_action="answer" if not outcome.error else "error",
            tainted=self.episode_taint.web,
            metrics={
                "rounds": outcome.rounds,
                "tool_calls": len(outcome.tool_requests),
                "limit_reached": outcome.limit_reached,
            },
            parent_episode_id=parent_id or None,
        )
        if outcome.error and not outcome.output:
            return {"output": outcome.error, "error": True, "subagent_id": outcome.id, "rounds": outcome.rounds}
        return {"output": outcome.output, "error": False, "subagent_id": outcome.id, "rounds": outcome.rounds}

    if self.config.subagents_enabled:
        tools.register(
            "spawn_subagent",
            spawn_subagent,
            "Delegate a bounded task to a focused sub-agent with its own tool "
            "loop and model. Returns the sub-agent's final result.",
            capabilities={"delegation"},
            schema={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Self-contained task for the sub-agent."},
                    "context": {
                        "type": "string",
                        "description": "Optional extra context pasted into the sub-agent prompt.",
                    },
                    "tools_allowed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional allowlist restricting the sub-agent's tools further.",
                    },
                    "max_rounds": {"type": "integer", "minimum": 1, "maximum": 32},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        )
