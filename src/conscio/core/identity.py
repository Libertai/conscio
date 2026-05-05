from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

_HOME_DIR = os.path.expanduser("~/.conscio")
_IDENTITY_PATH = os.path.join(_HOME_DIR, "identity.json")


@dataclass
class Goal:
    description: str
    tier: str = "session"  # core | session | ephemeral
    created: float = field(default_factory=time.time)
    completed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Goal:
        return cls(**d)


@dataclass
class Identity:
    """Persistent self — the agent's persona, goals, and history."""

    name: str
    persona: str = ""
    goals: list[Goal] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    session_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_goal(self, description: str, tier: str = "session") -> Goal:
        goal = Goal(description=description, tier=tier)
        self.goals.append(goal)
        return goal

    def complete_goal(self, description: str) -> None:
        for goal in self.goals:
            if goal.description == description and not goal.completed:
                goal.completed = True
                break

    def active_goals(self) -> list[Goal]:
        return [g for g in self.goals if not g.completed]

    def add_to_history(self, event: str) -> None:
        self.history.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {event}")
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def evolve(self, outcome: str) -> None:
        """Called after each conscious cycle. Allows the identity to update
        based on outcomes (learning, goal adjustment)."""
        self.session_count += 1

    def format_goals(self) -> str:
        goals = self.active_goals()
        if not goals:
            return "No active goals."
        return "\n".join(
            f"  [{g.tier}] {g.description}" for g in goals[-5:]
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "persona": self.persona,
            "goals": [g.to_dict() for g in self.goals],
            "history": self.history[-50:],
            "created_at": self.created_at,
            "session_count": self.session_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Identity:
        return cls(
            name=d["name"],
            persona=d.get("persona", ""),
            goals=[Goal.from_dict(g) for g in d.get("goals", [])],
            history=d.get("history", []),
            created_at=d.get("created_at", time.time()),
            session_count=d.get("session_count", 0),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def load_or_create(cls, name: str = "Conscio", persona: str = "") -> Identity:
        os.makedirs(_HOME_DIR, exist_ok=True)
        if os.path.exists(_IDENTITY_PATH):
            try:
                with open(_IDENTITY_PATH) as f:
                    return cls.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass
        identity = cls(name=name, persona=persona)
        identity.save()
        return identity

    def save(self) -> None:
        os.makedirs(_HOME_DIR, exist_ok=True)
        tmp = _IDENTITY_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, _IDENTITY_PATH)
