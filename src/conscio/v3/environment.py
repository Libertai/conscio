"""Environment boundary for text/tools/time now and other worlds later."""

from __future__ import annotations

from typing import Protocol

from conscio.core.cognition import InputEvent
from conscio.v3.contracts import ActionOutcome, ActionProposal, CognitiveEvent


class EnvironmentAdapter(Protocol):
    async def observe(self, event: InputEvent, *, episode_id: str) -> CognitiveEvent: ...

    async def act(self, proposal: ActionProposal, *, episode_id: str) -> ActionOutcome: ...


class TextEnvironmentAdapter:
    """Turns message, tool, system, and heartbeat inputs into typed observations.

    Tool execution remains owned by the existing policy-gated executor.  ``act``
    is therefore used for environment-native actions only and explicitly marks
    unsupported proposals as unsuccessful rather than executing them implicitly.
    """

    async def observe(self, event: InputEvent, *, episode_id: str) -> CognitiveEvent:
        return CognitiveEvent(
            event_type=event.event_type,
            source=event.source,
            episode_id=episode_id,
            payload={"content": event.content, "metadata": dict(event.metadata)},
        )

    async def act(self, proposal: ActionProposal, *, episode_id: str) -> ActionOutcome:
        return ActionOutcome(
            proposal_id=proposal.proposal_id,
            action=proposal.action,
            succeeded=False,
            observation="Action is delegated to the policy-gated runtime executor.",
        )
