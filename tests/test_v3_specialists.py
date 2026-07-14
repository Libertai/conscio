from __future__ import annotations

import json
from dataclasses import fields, replace

import numpy as np
import pytest

from conscio.v3.contracts import (
    AffectiveState,
    Broadcast,
    CandidateContent,
    CognitiveEvent,
    CoreCheckpoint,
)
from conscio.v3.recurrent_core import HybridRecurrentCore
from conscio.v3.specialists import (
    SPECIALIST_ARCHITECTURE_ID,
    SPECIALIST_ORDER,
    PerceptionSpecialist,
    SpecialistInput,
    default_specialist_factories,
)


def _event(content: str = "Remember blue tools", *, event_id: str = "evt-fixed") -> CognitiveEvent:
    return CognitiveEvent(
        event_type="message",
        source="user",
        payload={"content": content},
        episode_id="episode-fixed",
        event_id=event_id,
        observed_at=100.0,
    )


def _json_checkpoint(checkpoint: CoreCheckpoint) -> CoreCheckpoint:
    payload = json.loads(json.dumps(checkpoint.to_dict(), sort_keys=True))
    payload["affect"] = AffectiveState(**payload["affect"])
    payload["deterministic_state"] = tuple(payload["deterministic_state"])
    payload["stochastic_state"] = tuple(payload["stochastic_state"])
    return CoreCheckpoint(**payload)


def test_specialist_boundary_is_typed_isolated_and_preserves_epistemic_provenance() -> None:
    seen_state_is_immutable = False

    class PayloadMutatingPerception(PerceptionSpecialist):
        def candidate(self, specialist_input: SpecialistInput):  # type: ignore[no-untyped-def]
            nonlocal seen_state_is_immutable
            try:
                specialist_input.recurrent_state[0] = 999.0  # type: ignore[index]
            except TypeError:
                seen_state_is_immutable = True
            specialist_input.event.payload["content"] = "tampered inside perception"
            return super().candidate(specialist_input)

    factories = default_specialist_factories()
    factories["perception"] = PayloadMutatingPerception
    core = HybridRecurrentCore(seed=3, specialist_factories=factories)

    core.run_cycles(_event(), cycles=1)

    assert tuple(field.name for field in fields(SpecialistInput)) == (
        "event",
        "previous_broadcast",
        "recurrent_state",
    )
    assert seen_state_is_immutable
    semantic = core.specialists["semantic_belief"].private_state
    assert semantic["belief_cues"] == ("remember blue tools",)
    assert "tampered" not in semantic["belief_cues"]

    direct_core = HybridRecurrentCore(seed=3)
    event = _event()
    specialist_input = SpecialistInput(
        event=event,
        previous_broadcast=None,
        recurrent_state=(0.0,) * 24,
    )
    candidates = direct_core.specialist_registry.candidates(specialist_input)
    assert tuple(candidate.specialist for candidate in candidates) == SPECIALIST_ORDER
    assert {candidate.kind for candidate in candidates} <= {
        "observation",
        "belief",
        "hypothesis",
        "idea",
        "self_report",
    }
    assert all(candidate.evidence_event_ids == (event.event_id,) for candidate in candidates)
    assert all(candidate.private_state_version == 1 for candidate in candidates)


def test_factory_lesion_removes_module_without_construction_or_execution() -> None:
    constructions = 0

    def forbidden_factory():  # type: ignore[no-untyped-def]
        nonlocal constructions
        constructions += 1
        raise AssertionError("lesioned specialist factory must not execute")

    factories = default_specialist_factories()
    factories["self_model"] = forbidden_factory
    core = HybridRecurrentCore(
        seed=5,
        specialist_factories=factories,
        specialist_lesions={"self_model"},
    )

    result = core.run_cycles(_event(), cycles=2)
    checkpoint = core.checkpoint()

    assert constructions == 0
    assert "self_model" not in core.specialists
    assert "self_model" not in checkpoint.specialist_states
    assert all(candidate.specialist != "self_model" for cycle in result for candidate in cycle.broadcast.candidates)


def test_architecture_identity_includes_concrete_specialist_implementation() -> None:
    class AlternatePerception(PerceptionSpecialist):
        implementation_version = 2

    default = HybridRecurrentCore(seed=5)
    factories = default_specialist_factories()
    factories["perception"] = AlternatePerception
    alternate = HybridRecurrentCore(seed=5, specialist_factories=factories)

    assert default.specialist_architecture_id == SPECIALIST_ARCHITECTURE_ID
    assert alternate.specialist_architecture_id != SPECIALIST_ARCHITECTURE_ID
    assert alternate.runtime_identity != default.runtime_identity
    with pytest.raises(ValueError, match="explicit lineage migration"):
        alternate.restore(default.checkpoint())


def test_runtime_lesions_skip_memory_self_and_prediction_specialist_computation() -> None:
    core = HybridRecurrentCore(seed=7)

    result = core.run_cycles(
        _event(),
        cycles=3,
        memory_enabled=False,
        self_model_enabled=False,
        prediction_enabled=False,
    )

    assert core.specialists["autobiographical_memory"].private_state["updates"] == 0
    assert core.specialists["semantic_belief"].private_state["updates"] == 0
    assert core.specialists["self_model"].private_state["updates"] == 0
    assert core.specialists["world_prediction"].private_state["updates"] == 0
    assert core.specialists["perception"].private_state["updates"] == 3
    assert all(not cycle.predictions for cycle in result)
    audit = core.specialist_execution_audit
    assert audit["self_model"] == {"compute": 0, "expose": 0}
    assert audit["perception"]["compute"] == 3
    with pytest.raises(TypeError):
        audit["perception"]["compute"] = 100  # type: ignore[index]


def test_run_cycle_exclusion_prevents_computation_and_exposure() -> None:
    core = HybridRecurrentCore(seed=9)
    active = set(SPECIALIST_ORDER) - {"action_evaluation"}

    result = core.run_cycle(_event(), 0, None, active)

    assert all(candidate.specialist != "action_evaluation" for candidate in result.all_candidates)
    assert all(candidate.specialist != "action_evaluation" for candidate in result.broadcast.candidates)
    assert core.specialist_execution_audit["action_evaluation"] == {
        "compute": 0,
        "expose": 0,
    }


def test_checkpoint_roundtrip_restores_exact_private_state_and_rejects_schemas() -> None:
    first = HybridRecurrentCore(seed=11, lineage_id="lineage-fixed")
    first.run_cycles(_event(), cycles=3)
    expected_specialist_states = first.specialist_registry.checkpoint_states()
    checkpoint = _json_checkpoint(first.checkpoint())
    second = HybridRecurrentCore(seed=999)

    second.restore(checkpoint)

    assert second.specialist_registry.checkpoint_states() == expected_specialist_states
    assert np.array_equal(second.deterministic, first.deterministic)
    assert np.array_equal(second.stochastic, first.stochastic)
    assert second.rng.bit_generator.state == first.rng.bit_generator.state

    unknown_states = dict(checkpoint.specialist_states)
    unknown_states["unknown"] = unknown_states["perception"]
    with pytest.raises(ValueError, match="unknown=.*unknown"):
        HybridRecurrentCore().restore(replace(checkpoint, specialist_states=unknown_states))

    incompatible_states = {name: dict(snapshot) for name, snapshot in checkpoint.specialist_states.items()}
    incompatible_states["perception"] = {
        **incompatible_states["perception"],
        "schema_version": 999,
    }
    with pytest.raises(ValueError, match="schema version"):
        HybridRecurrentCore().restore(replace(checkpoint, specialist_states=incompatible_states))


def test_checkpoint_restore_rejects_invalid_affect_and_counters() -> None:
    checkpoint = HybridRecurrentCore(seed=12).checkpoint()

    with pytest.raises(ValueError, match="cycle_count"):
        HybridRecurrentCore().restore(replace(checkpoint, cycle_count=-1))
    with pytest.raises(ValueError, match="affect valence"):
        HybridRecurrentCore().restore(replace(checkpoint, affect=replace(checkpoint.affect, valence=2.0)))
    with pytest.raises(ValueError, match="exact supported needs"):
        HybridRecurrentCore().restore(
            replace(
                checkpoint,
                affect=replace(checkpoint.affect, need_errors={"unexpected": 0.1}),
            )
        )


def test_autobiographical_and_semantic_private_states_are_separate() -> None:
    core = HybridRecurrentCore(seed=13)

    core.run_cycles(_event("A blue tool is available"), cycles=3)

    autobiographical = core.specialists["autobiographical_memory"].private_state
    semantic = core.specialists["semantic_belief"].private_state
    assert autobiographical["episode_events"] == (("episode-fixed", ("evt-fixed",)),)
    assert "belief_cues" not in autobiographical
    assert semantic["belief_cues"] == ("a blue tool is available",)
    assert semantic["source_counts"] == (("user", 1),)
    assert "episode_events" not in semantic

    detached = core.specialists["semantic_belief"].private_state
    detached["belief_cues"] = ("external mutation",)
    assert core.specialists["semantic_belief"].private_state["belief_cues"] == ("a blue tool is available",)


def test_prior_broadcast_is_the_only_recurrent_specialist_channel() -> None:
    core = HybridRecurrentCore(seed=17)

    cycles = core.run_cycles(_event(), cycles=2)

    first_broadcast_id = cycles[0].broadcast.broadcast_id
    assert core.specialists["planning"].private_state["last_prior_broadcast_id"] == (first_broadcast_id)
    assert core.specialists["autobiographical_memory"].private_state["last_broadcast_id"] == first_broadcast_id
    assert any("preceding broadcast" in candidate.content for candidate in cycles[1].all_candidates)


def test_between_cycle_broadcast_transform_changes_only_recurrent_input() -> None:
    core = HybridRecurrentCore(seed=19)
    calls = 0

    def transform(broadcast: Broadcast) -> Broadcast:
        nonlocal calls
        calls += 1
        intervention = CandidateContent(
            specialist="perception",
            content="matched hidden broadcast intervention",
            kind="observation",
            confidence=1.0,
            salience=1.0,
            evidence_event_ids=("evt-intervention",),
            candidate_id="cand-intervention",
        )
        return replace(broadcast, candidates=(intervention,))

    cycles = core.run_cycles(_event(), cycles=2, broadcast_transform=transform)

    assert calls == 1
    planning = next(candidate for candidate in cycles[1].all_candidates if candidate.specialist == "planning")
    assert "matched hidden broadcast intervention" in planning.content


def test_seeded_specialist_and_core_behavior_is_stably_deterministic() -> None:
    first = HybridRecurrentCore(seed=23, lineage_id="first")
    second = HybridRecurrentCore(seed=23, lineage_id="second")
    event = _event("deterministic observation")

    first_results = first.run_cycles(event, cycles=4)
    second_results = second.run_cycles(event, cycles=4)

    assert first_results == second_results
    assert np.array_equal(first.deterministic, second.deterministic)
    assert np.array_equal(first.stochastic, second.stochastic)
    assert first.specialist_registry.checkpoint_states() == second.specialist_registry.checkpoint_states()
