"""Conscio V3 persistent recurrent cognitive core.

V3 is developed beside the V2 engine.  ``V3CognitiveRuntime`` preserves the
public ``CognitiveRuntime`` result contract while adding typed, append-only
causal records and a checkpointed recurrent state.
"""

from conscio.v3.contracts import (
    ActionOutcome,
    ActionProposal,
    AffectiveState,
    Broadcast,
    CandidateContent,
    CognitiveEvent,
    CoreCheckpoint,
    ExecutionIntent,
    ExecutionOutcome,
    ExecutionReconciliation,
    ExecutionRecovery,
    Prediction,
    deterministic_execution_event_id,
)
from conscio.v3.runtime import V3CognitiveRuntime

__all__ = [
    "ActionOutcome",
    "ActionProposal",
    "AffectiveState",
    "Broadcast",
    "CandidateContent",
    "CognitiveEvent",
    "CoreCheckpoint",
    "ExecutionIntent",
    "ExecutionOutcome",
    "ExecutionReconciliation",
    "ExecutionRecovery",
    "Prediction",
    "V3CognitiveRuntime",
    "deterministic_execution_event_id",
]
