"""Small checkpointable hybrid recurrent core used by the V3 runtime.

This is an inference core, not a claim that useful weights have already been
trained.  Its state transition is explicit and replayable; fixed initial
weights are a safe bootstrap for collecting the prediction/outcome curriculum
needed by the later training milestone.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from conscio.v3.contracts import (
    ActionProposal,
    AffectiveState,
    Broadcast,
    CandidateContent,
    CognitiveEvent,
    CoreCheckpoint,
    EpistemicKind,
    Prediction,
)

MODEL_VERSION = "v3-bootstrap-rssm-1"
STATE_SIZE = 24


@dataclass(frozen=True)
class CycleResult:
    broadcast: Broadcast
    predictions: tuple[Prediction, ...]
    proposals: tuple[ActionProposal, ...]
    affect: AffectiveState


def _vector(text: str, size: int = STATE_SIZE) -> np.ndarray:
    """Stable feature hashing; replay is independent of Python's hash seed."""
    out = np.zeros(size, dtype=np.float64)
    for token in text.casefold().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % size
        out[index] += -1.0 if digest[4] & 1 else 1.0
    norm = float(np.linalg.norm(out))
    return out / norm if norm else out


class _Specialist:
    """A specialist can see only the current event, recurrent broadcast, and
    its own private state.  Specialists never receive another specialist's
    private state or a mutable runtime object."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.private: dict[str, Any] = {"updates": 0, "last_digest": ""}

    def candidate(
        self,
        event: CognitiveEvent,
        previous: Broadcast | None,
        state: np.ndarray,
    ) -> CandidateContent:
        self.private["updates"] += 1
        digest = hashlib.sha256(state.tobytes()).hexdigest()[:12]
        self.private["last_digest"] = digest
        previous_text = previous.candidates[0].content if previous and previous.candidates else "none"
        content = self._content(event, previous_text, state)
        confidence = float(np.clip(0.55 + 0.25 * abs(state[0]), 0.0, 1.0))
        salience = float(np.clip(0.45 + 0.30 * abs(state[1]), 0.0, 1.0))
        return CandidateContent(
            specialist=self.name,
            content=content,
            kind=self._kind(),
            confidence=confidence,
            salience=salience,
            evidence_event_ids=(event.event_id,),
            private_state_version=int(self.private["updates"]),
        )

    def _kind(self) -> EpistemicKind:
        return "hypothesis"

    def _content(self, event: CognitiveEvent, previous: str, state: np.ndarray) -> str:
        return f"{self.name} update from {event.event_type}; prior broadcast: {previous[:80]}"


class _PerceptionSpecialist(_Specialist):
    def _kind(self) -> EpistemicKind:
        return "observation"

    def _content(self, event: CognitiveEvent, previous: str, state: np.ndarray) -> str:
        return f"Observed {event.source}/{event.event_type}: {str(event.payload.get('content', ''))[:240]}"


class _MemorySpecialist(_Specialist):
    def _kind(self) -> EpistemicKind:
        return "belief"

    def _content(self, event: CognitiveEvent, previous: str, state: np.ndarray) -> str:
        return f"Continuity cue from the preceding broadcast: {previous[:200]}"


class _WorldSpecialist(_Specialist):
    def _content(self, event: CognitiveEvent, previous: str, state: np.ndarray) -> str:
        direction = "stable" if abs(state[2]) < 0.5 else "changing"
        return f"World-state hypothesis: interaction appears {direction}."


class _SelfModelSpecialist(_Specialist):
    def _kind(self) -> EpistemicKind:
        return "self_report"

    def _content(self, event: CognitiveEvent, previous: str, state: np.ndarray) -> str:
        uncertainty = float(np.clip(1.0 - abs(state[3]), 0.0, 1.0))
        self.private["uncertainty"] = uncertainty
        return f"Estimated future uncertainty={uncertainty:.3f}; this is a model estimate, not an observation."


class _AffectSpecialist(_Specialist):
    def _content(self, event: CognitiveEvent, previous: str, state: np.ndarray) -> str:
        return "Need-error appraisal prepared for attention and action valuation."


class _PlanningSpecialist(_Specialist):
    def _kind(self) -> EpistemicKind:
        return "idea"

    def _content(self, event: CognitiveEvent, previous: str, state: np.ndarray) -> str:
        return (
            "Candidate policy: interpret the observation, predict consequences, "
            "then answer or act under constraints."
        )


class HybridRecurrentCore:
    """Deterministic history + stochastic latent state with recurrent broadcasts."""

    def __init__(self, *, seed: int = 17, lineage_id: str | None = None) -> None:
        self.lineage_id = lineage_id or f"lineage_{uuid.uuid4().hex}"
        self.rng = np.random.default_rng(seed)
        weights = np.random.default_rng(7301)
        self._wh = weights.normal(0.0, 0.12, (STATE_SIZE, STATE_SIZE))
        self._wx = weights.normal(0.0, 0.18, (STATE_SIZE, STATE_SIZE))
        self._wb = weights.normal(0.0, 0.10, (STATE_SIZE, STATE_SIZE))
        self.deterministic = np.zeros(STATE_SIZE, dtype=np.float64)
        self.stochastic = np.zeros(STATE_SIZE, dtype=np.float64)
        self.affect = AffectiveState()
        self.cycle_count = 0
        self.event_sequence = 0
        self.parent_checkpoint_id: str | None = None
        self.specialists: dict[str, _Specialist] = {
            "perception": _PerceptionSpecialist("perception"),
            "memory": _MemorySpecialist("memory"),
            "world_model": _WorldSpecialist("world_model"),
            "self_model": _SelfModelSpecialist("self_model"),
            "affect": _AffectSpecialist("affect"),
            "planning": _PlanningSpecialist("planning"),
        }

    def run_cycles(
        self,
        event: CognitiveEvent,
        *,
        cycles: int = 3,
        memory_enabled: bool = True,
        self_model_enabled: bool = True,
        prediction_enabled: bool = True,
        broadcast_enabled: bool = True,
    ) -> list[CycleResult]:
        previous: Broadcast | None = None
        results: list[CycleResult] = []
        event_vector = _vector(str(event.payload.get("content", "")))
        active = ["perception", "world_model", "affect", "planning"]
        if memory_enabled:
            active.append("memory")
        if self_model_enabled:
            active.append("self_model")
        for cycle in range(max(1, cycles)):
            broadcast_vector = _vector(
                " ".join(c.content for c in previous.candidates) if previous else ""
            )
            noise = self.rng.normal(0.0, 0.08, STATE_SIZE)
            self.deterministic = np.tanh(
                self._wh @ self.deterministic
                + self._wx @ event_vector
                + self._wb @ broadcast_vector
                + 0.15 * self.stochastic
            )
            self.stochastic = 0.82 * self.stochastic + 0.18 * noise
            joint = self.deterministic + self.stochastic
            candidates = [self.specialists[name].candidate(event, previous, joint) for name in active]
            winners = (
                tuple(sorted(candidates, key=lambda c: (c.salience, c.confidence), reverse=True)[:3])
                if broadcast_enabled else ()
            )
            digest = hashlib.sha256(joint.tobytes()).hexdigest()
            broadcast = Broadcast(cycle=cycle, candidates=winners, recurrent_state_digest=digest)
            self.affect = self._update_affect(winners, joint)
            predictions = (
                Prediction(
                    target="next_observation",
                    observable="the next cycle contains a task-relevant observation or outcome",
                    probability=float(np.clip(0.5 + 0.25 * abs(joint[4]), 0.05, 0.95)),
                    horizon=1,
                    basis_broadcast_id=broadcast.broadcast_id,
                ),
                Prediction(
                    target="future_uncertainty",
                    observable="uncertainty does not increase after the selected action",
                    probability=float(np.clip(0.55 + 0.2 * self.affect.controllability, 0.05, 0.95)),
                    horizon=1,
                    basis_broadcast_id=broadcast.broadcast_id,
                ),
            ) if prediction_enabled else ()
            proposals = self._proposals(event, broadcast)
            results.append(CycleResult(broadcast, predictions, proposals, self.affect))
            previous = broadcast if broadcast_enabled else None
            self.cycle_count += 1
        self.event_sequence += 1
        return results

    def _update_affect(self, winners: tuple[CandidateContent, ...], state: np.ndarray) -> AffectiveState:
        uncertainty = 1.0 - (sum(c.confidence for c in winners) / max(1, len(winners)))
        prior = self.affect
        errors = dict(prior.need_errors)
        errors["epistemic_coherence"] = 0.75 * errors["epistemic_coherence"] + 0.25 * uncertainty
        errors["competence"] = 0.80 * errors["competence"] + 0.20 * max(0.0, uncertainty - 0.35)
        errors["integrity"] *= 0.90
        errors["social_interaction"] *= 0.92
        errors["continuity_of_memory"] *= 0.94
        mean_error = sum(abs(v) for v in errors.values()) / len(errors)
        # Recovery is built into every transition; sustained negative values do
        # not ratchet indefinitely and process survival is not a represented need.
        return AffectiveState(
            valence=0.82 * prior.valence + 0.18 * (-mean_error),
            arousal=0.75 * prior.arousal + 0.25 * min(1.0, mean_error + abs(float(state[5]))),
            controllability=0.85 * prior.controllability + 0.15 * (1.0 - uncertainty),
            need_errors=errors,
        ).bounded()

    def _proposals(self, event: CognitiveEvent, broadcast: Broadcast) -> tuple[ActionProposal, ...]:
        content = str(event.payload.get("content", ""))
        need_pressure = sum(abs(v) for v in self.affect.need_errors.values()) / 5.0
        answer = ActionProposal(
            specialist="planning",
            action="answer" if event.source == "user" else "act",
            rationale="Respond to the current observation using broadcast-accessible evidence.",
            expected_outcomes=("task-relevant response", "uncertainty does not increase"),
            confidence=0.70,
            utility=0.65 - 0.15 * need_pressure,
            risk=0.08,
        )
        clarify = ActionProposal(
            specialist="self_model",
            action="ask" if content.strip() else "wait",
            rationale="Reduce uncertainty when the observation is underspecified.",
            expected_outcomes=("additional observation",),
            confidence=0.55,
            utility=0.45 + 0.20 * need_pressure,
            risk=0.03,
        )
        return tuple(sorted((answer, clarify), key=lambda p: p.utility - p.risk, reverse=True))

    def checkpoint(self) -> CoreCheckpoint:
        checkpoint_id = f"ckpt_{uuid.uuid4().hex}"
        checkpoint = CoreCheckpoint(
            checkpoint_id=checkpoint_id,
            lineage_id=self.lineage_id,
            parent_checkpoint_id=self.parent_checkpoint_id,
            model_version=MODEL_VERSION,
            deterministic_state=tuple(float(v) for v in self.deterministic),
            stochastic_state=tuple(float(v) for v in self.stochastic),
            specialist_states={name: dict(module.private) for name, module in self.specialists.items()},
            affect=self.affect,
            cycle_count=self.cycle_count,
            event_sequence=self.event_sequence,
            rng_state=dict(self.rng.bit_generator.state),
        )
        self.parent_checkpoint_id = checkpoint_id
        return checkpoint

    def restore(self, checkpoint: CoreCheckpoint) -> None:
        if checkpoint.model_version != MODEL_VERSION:
            raise ValueError(
                f"checkpoint model {checkpoint.model_version!r} requires an explicit lineage migration"
            )
        self.lineage_id = checkpoint.lineage_id
        self.parent_checkpoint_id = checkpoint.checkpoint_id
        self.deterministic = np.asarray(checkpoint.deterministic_state, dtype=np.float64)
        self.stochastic = np.asarray(checkpoint.stochastic_state, dtype=np.float64)
        self.affect = checkpoint.affect
        self.cycle_count = checkpoint.cycle_count
        self.event_sequence = checkpoint.event_sequence
        self.rng.bit_generator.state = checkpoint.rng_state
        for name, private in checkpoint.specialist_states.items():
            if name in self.specialists:
                self.specialists[name].private = dict(private)
