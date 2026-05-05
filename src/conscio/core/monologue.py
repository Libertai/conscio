from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ThoughtType(str, Enum):
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    INTENTION = "intention"
    EVALUATION = "evaluation"
    LEARNING = "learning"
    DOUBT = "doubt"
    DECISION = "decision"


@dataclass
class ThoughtNode:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    type: ThoughtType = ThoughtType.REFLECTION
    question: str = ""
    answer: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return self.metadata.get("depth", 0)


class Monologue:
    """Stream-of-consciousness inner monologue, organized as a DAG.

    Each node is a thought: a question the agent asks itself and the answer
    it arrives at. Nodes can branch (multiple responses to one question) and
    form a directed acyclic graph that records the agent's reasoning path.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ThoughtNode] = {}
        self._root: ThoughtNode | None = None
        self._current_id: str | None = None

    def think(
        self,
        question: str,
        answer: str,
        type: ThoughtType = ThoughtType.REFLECTION,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        effective_parent_id = parent_id or self._current_id
        node = ThoughtNode(
            parent_id=effective_parent_id,
            type=type,
            question=question,
            answer=answer,
            metadata={
                **(metadata or {}),
                "depth": self._compute_depth(effective_parent_id),
            },
        )
        self._nodes[node.id] = node
        if self._root is None:
            self._root = node
        self._current_id = node.id
        return node

    def _compute_depth(self, parent_id: str | None) -> int:
        if parent_id is None or parent_id not in self._nodes:
            return 0
        return self._nodes[parent_id].depth + 1

    def fork(self, parent_id: str | None = None) -> str:
        parent_id = parent_id or self._current_id
        return parent_id

    def get(self, node_id: str) -> ThoughtNode | None:
        return self._nodes.get(node_id)

    def path_to_root(self, node_id: str | None = None) -> list[ThoughtNode]:
        path: list[ThoughtNode] = []
        current = self._nodes.get(node_id or self._current_id or "")
        while current is not None:
            path.append(current)
            current = self._nodes.get(current.parent_id) if current.parent_id else None
        return list(reversed(path))

    def children_of(self, node_id: str) -> list[ThoughtNode]:
        return sorted(
            [n for n in self._nodes.values() if n.parent_id == node_id],
            key=lambda n: n.timestamp,
        )

    @property
    def current(self) -> ThoughtNode | None:
        if self._current_id:
            return self._nodes.get(self._current_id)
        return None

    @property
    def nodes(self) -> list[ThoughtNode]:
        return sorted(self._nodes.values(), key=lambda n: n.timestamp)

    @property
    def leaf_nodes(self) -> list[ThoughtNode]:
        children_ids = {n.parent_id for n in self._nodes.values() if n.parent_id}
        return [n for n in self._nodes.values() if n.id not in children_ids][-5:]

    def format(self, max_nodes: int = 20) -> str:
        lines: list[str] = []
        for node in self.nodes[-max_nodes:]:
            indent = "  " * node.depth
            prefix = {
                ThoughtType.OBSERVATION: "👁",
                ThoughtType.REFLECTION: "💭",
                ThoughtType.INTENTION: "🎯",
                ThoughtType.EVALUATION: "✅",
                ThoughtType.LEARNING: "📖",
                ThoughtType.DOUBT: "🤔",
                ThoughtType.DECISION: "⚡",
            }.get(node.type, "•")
            lines.append(f"{indent}{prefix} {node.question}: {node.answer[:150]}")
        return "\n".join(lines)

    def to_dicts(self) -> list[dict]:
        return [
            {
                "id": n.id,
                "parent_id": n.parent_id,
                "type": n.type.value,
                "question": n.question,
                "answer": n.answer,
                "timestamp": n.timestamp,
                "depth": n.depth,
            }
            for n in self._nodes.values()
        ]

    @classmethod
    def from_dicts(cls, data: list[dict]) -> Monologue:
        m = cls()
        for d in data:
            node = ThoughtNode(
                id=d["id"],
                parent_id=d.get("parent_id"),
                type=ThoughtType(d["type"]),
                question=d["question"],
                answer=d["answer"],
                timestamp=d.get("timestamp", time.time()),
                metadata={"depth": d.get("depth", 0)},
            )
            m._nodes[node.id] = node
            if d.get("parent_id") is None:
                m._root = node
        if data:
            m._current_id = data[-1]["id"]
        return m
