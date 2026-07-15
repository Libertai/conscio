from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

from conscio.v3.learning import (
    AdapterState,
    ReplaySample,
    ShadowLearningConfig,
    brier_loss,
    derive_replay_samples,
    deterministic_split,
    train_shadow_adapter,
)
from conscio.v3.recurrent_core import HybridRecurrentCore


def _prediction(
    episode_id: str,
    prediction_id: str,
    probability: float,
    target: str,
    sequence: int,
    **extra: object,
) -> dict:
    return {
        "episode_id": episode_id,
        "event_id": f"event_{prediction_id}",
        "event_type": "prediction",
        "sequence": sequence,
        "payload": {
            "prediction_id": prediction_id,
            "probability": probability,
            "target": target,
            "horizon": 1,
            **extra,
        },
    }


def _outcome(episode_id: str, sequence: int, event_id: str = "outcome", **payload: object) -> dict:
    return {
        "episode_id": episode_id,
        "event_id": f"{event_id}_{episode_id}",
        "event_type": "action_outcome",
        "sequence": sequence,
        "payload": payload,
    }


def _sample(index: int, *, probability: float | None = None, outcome: bool | None = None) -> ReplaySample:
    label = bool(index % 2) if outcome is None else outcome
    raw_probability = (0.65 if label else 0.35) if probability is None else probability
    return ReplaySample(
        sample_id=f"sample_{index}",
        episode_id=f"episode_{index}",
        prediction_id=f"prediction_{index}",
        target="action_success",
        probability=raw_probability,
        outcome=label,
        prediction_event_id=f"prediction_event_{index}",
        resolution_event_id=f"outcome_event_{index}",
        resolution_kind="action_success",
    )


def test_derives_current_v3_targets_from_action_outcome() -> None:
    events = [
        _prediction("episode", "observation", 0.6, "next_observation", 1),
        _prediction("episode", "uncertainty", 0.7, "future_uncertainty", 2),
        _prediction("episode", "success", 0.4, "action_success", 3),
        _outcome(
            "episode",
            4,
            observation="policy-gated output",
            succeeded=False,
            prediction_errors={"runtime_prediction_errors": 0.0},
        ),
    ]

    dataset = derive_replay_samples(reversed(events))
    by_id = {sample.prediction_id: sample for sample in dataset.samples}

    assert not dataset.rejections
    assert by_id["observation"].outcome is True
    assert by_id["observation"].resolution_kind == "observation_presence"
    assert by_id["uncertainty"].outcome is True
    assert by_id["uncertainty"].resolution_kind == "no_prediction_error"
    assert by_id["success"].outcome is False


def test_explicit_resolution_and_outcomes_override_inference() -> None:
    events = [
        _prediction("episode", "direct", 0.2, "novel_target", 1),
        {
            "episode_id": "episode",
            "event_id": "resolution",
            "event_type": "prediction_resolution",
            "sequence": 2,
            "payload": {"prediction_id": "direct", "actual": True},
        },
        _prediction("episode", "mapped", 0.9, "novel_target", 3),
        _outcome(
            "episode",
            4,
            observation="",
            succeeded=False,
            prediction_outcomes={"mapped": {"label": True}},
        ),
    ]

    dataset = derive_replay_samples(events)

    assert [sample.outcome for sample in dataset.samples] == [True, True]
    assert {sample.resolution_kind for sample in dataset.samples} == {
        "explicit_resolution",
        "action_outcome_explicit",
    }


def test_resolved_prediction_can_reconstruct_an_unambiguous_error_label() -> None:
    dataset = derive_replay_samples(
        [_prediction("episode", "prediction", 0.8, "anything", 1, resolved=True, error=0.2)]
    )

    assert len(dataset.samples) == 1
    assert dataset.samples[0].outcome is True


def test_horizon_uses_the_matching_later_outcome() -> None:
    prediction = _prediction("episode", "later", 0.7, "next_observation", 1)
    prediction["payload"]["horizon"] = 2
    events = [
        prediction,
        _outcome("episode", 2, event_id="first", observation="first"),
        _outcome("episode", 3, event_id="second", observation="second"),
    ]

    dataset = derive_replay_samples(events)

    assert len(dataset.samples) == 1
    assert dataset.samples[0].resolution_event_id == "second_episode"


def test_unobserved_action_outcome_cannot_resolve_replay_predictions() -> None:
    events = [
        _prediction("episode", "observation", 0.6, "next_observation", 1),
        _prediction("episode", "success", 0.4, "action_success", 2),
        _outcome(
            "episode",
            3,
            observation="synthetic output must not become evidence",
            succeeded=True,
            observed=False,
            learning_eligible=False,
            prediction_outcomes={"success": True},
        ),
    ]

    dataset = derive_replay_samples(events)

    assert not dataset.samples
    assert len(dataset.rejections) == 2
    assert all("not resolved" in rejection.reason for rejection in dataset.rejections)


@pytest.mark.parametrize(
    ("marker", "observed", "reason"),
    [
        ("false", False, "learning_eligible marker is not boolean"),
        (True, False, "not explicitly observed"),
    ],
)
def test_malformed_or_inconsistent_learning_markers_are_rejected(
    marker: object,
    observed: object,
    reason: str,
) -> None:
    dataset = derive_replay_samples(
        [
            _prediction("episode", "success", 0.4, "action_success", 1),
            _outcome(
                "episode",
                2,
                succeeded=True,
                observed=observed,
                learning_eligible=marker,
            ),
        ]
    )

    assert not dataset.samples
    assert any(reason in rejection.reason for rejection in dataset.rejections)


def test_malformed_ambiguous_and_unrecognized_predictions_are_rejected() -> None:
    events = [
        _prediction("episode", "invalid", 1.2, "action_success", 1),
        _prediction("episode", "unknown", 0.6, "unrecognized", 2),
        _outcome("episode", 3, succeeded=True),
        _prediction("episode", "ambiguous", 0.5, "unrecognized", 4),
        {
            "episode_id": "episode",
            "event_id": "ambiguous_resolution",
            "event_type": "prediction_resolution",
            "sequence": 5,
            "payload": {"prediction_id": "ambiguous", "error": 0.5},
        },
    ]

    dataset = derive_replay_samples(events)

    assert not dataset.samples
    reasons = {rejection.reason for rejection in dataset.rejections}
    assert "prediction probability is outside [0, 1]" in reasons
    assert "prediction was not resolved by a compatible outcome" in reasons
    assert "resolution has no unambiguous binary outcome" in reasons


def test_deterministic_split_is_order_independent_and_episode_disjoint() -> None:
    samples = [_sample(index) for index in range(20)]
    shuffled = samples.copy()
    random.Random(991).shuffle(shuffled)

    first = deterministic_split(samples, validation_fraction=0.25, seed=42)
    second = deterministic_split(shuffled, validation_fraction=0.25, seed=42)

    assert first == second
    assert len(first.train) == 15
    assert len(first.validation) == 5
    assert {item.episode_id for item in first.train}.isdisjoint(item.episode_id for item in first.validation)
    assert deterministic_split(samples, validation_fraction=0.25, seed=43) != first


def test_single_episode_never_leaks_into_validation() -> None:
    samples = [
        ReplaySample(
            sample_id=f"sample_{index}",
            episode_id="one_episode",
            prediction_id=f"prediction_{index}",
            target="action_success",
            probability=0.5,
            outcome=bool(index % 2),
            prediction_event_id=f"prediction_event_{index}",
            resolution_event_id=f"outcome_event_{index}",
            resolution_kind="explicit_resolution",
        )
        for index in range(4)
    ]

    split = deterministic_split(samples)

    assert split.train == tuple(samples)
    assert split.validation == ()


def test_brier_loss_rewards_better_probabilities() -> None:
    samples = [_sample(0, probability=0.1, outcome=False), _sample(1, probability=0.9, outcome=True)]
    identity = AdapterState(base_model_version="base")
    flattened = AdapterState(base_model_version="base", log_scale=-10.0)

    assert brier_loss(identity, samples) < brier_loss(flattened, samples)
    assert brier_loss(identity, samples) == pytest.approx(0.01)


def test_adapter_serialization_is_versioned_and_stable() -> None:
    state = AdapterState(
        base_model_version="v3-bootstrap-rssm-1",
        revision=3,
        log_scale=0.2,
        bias=-0.1,
        trained_examples=80,
        validation_examples=20,
        validation_loss=0.12,
        parent_digest="abc",
    )

    restored = AdapterState.from_json(state.to_json())

    assert restored == state
    assert restored.digest() == state.digest()
    assert json.loads(state.to_json())["schema_version"] == 1

    future = state.to_dict()
    future["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported adapter schema"):
        AdapterState.from_dict(future)

    corrupt = state.to_dict()
    corrupt["bias"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        AdapterState.from_dict(corrupt)


def test_calibration_is_monotonic_and_handles_boundary_probabilities() -> None:
    state = AdapterState(base_model_version="base", log_scale=0.4, bias=-0.2)
    calibrated = [state.calibrate(value) for value in (0.0, 0.2, 0.5, 0.8, 1.0)]

    assert calibrated == sorted(calibrated)
    assert all(0.0 < value < 1.0 for value in calibrated)


def test_shadow_candidate_promotes_only_after_held_out_improvement() -> None:
    samples = [_sample(index) for index in range(80)]
    incumbent = AdapterState(base_model_version="v3-bootstrap-rssm-1", revision=4)
    config = ShadowLearningConfig(
        validation_fraction=0.25,
        min_training_examples=40,
        min_validation_examples=10,
        min_improvement=0.001,
        max_parameter_delta=0.3,
        max_parameter_norm=0.3,
    )

    result = train_shadow_adapter(samples, incumbent=incumbent, config=config)

    assert result.promoted is True
    assert result.selected is result.candidate
    assert result.reason == "promotion gates passed"
    assert result.improvement is not None and result.improvement >= config.min_improvement
    assert result.candidate_validation_loss is not None
    assert result.incumbent_validation_loss is not None
    assert result.candidate_validation_loss < result.incumbent_validation_loss
    assert result.candidate.revision == incumbent.revision + 1
    assert result.candidate.parent_digest == incumbent.digest()
    assert result.candidate.base_model_version == incumbent.base_model_version


def test_training_is_deterministic_and_enforces_delta_and_norm_bounds() -> None:
    samples = [_sample(index) for index in range(80)]
    incumbent = AdapterState(base_model_version="base", log_scale=0.09)
    config = ShadowLearningConfig(
        min_training_examples=1,
        min_validation_examples=1,
        min_improvement=0.0,
        learning_rate=1.0,
        epochs=500,
        max_parameter_delta=0.025,
        max_parameter_norm=0.1,
    )

    first = train_shadow_adapter(samples, incumbent=incumbent, config=config)
    second = train_shadow_adapter(list(reversed(samples)), incumbent=incumbent, config=config)
    delta = np.asarray(first.candidate.parameters) - np.asarray(incumbent.parameters)

    assert first == second
    assert np.linalg.norm(delta) <= config.max_parameter_delta + 1e-12
    assert first.candidate.parameter_norm <= config.max_parameter_norm + 1e-12


def test_insufficient_data_and_improvement_threshold_keep_exact_incumbent() -> None:
    incumbent = AdapterState(base_model_version="base")
    too_small = train_shadow_adapter(
        [_sample(index) for index in range(8)],
        incumbent=incumbent,
        config=ShadowLearningConfig(min_training_examples=20, min_validation_examples=2),
    )
    high_threshold = train_shadow_adapter(
        [_sample(index) for index in range(80)],
        incumbent=incumbent,
        config=ShadowLearningConfig(
            min_training_examples=1,
            min_validation_examples=1,
            min_improvement=1.0,
        ),
    )

    assert too_small.promoted is False
    assert too_small.reason == "insufficient training examples"
    assert too_small.selected is incumbent
    assert high_threshold.promoted is False
    assert high_threshold.reason == "held-out improvement below threshold"
    assert high_threshold.selected is incumbent


def test_training_does_not_mutate_incumbent_or_recurrent_base_weights() -> None:
    core = HybridRecurrentCore(seed=7)
    base_weights = (core._wh.copy(), core._wx.copy(), core._wb.copy())
    incumbent = AdapterState(base_model_version="base")
    before = incumbent.to_json()

    train_shadow_adapter(
        [_sample(index) for index in range(80)],
        incumbent=incumbent,
        config=ShadowLearningConfig(min_training_examples=1, min_validation_examples=1),
    )

    assert incumbent.to_json() == before
    assert np.array_equal(core._wh, base_weights[0])
    assert np.array_equal(core._wx, base_weights[1])
    assert np.array_equal(core._wb, base_weights[2])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_fraction", 1.0),
        ("epochs", 0),
        ("min_improvement", -0.1),
        ("max_parameter_delta", -0.1),
        ("max_parameter_norm", 0.0),
        ("probability_epsilon", 0.5),
    ],
)
def test_invalid_learning_configuration_is_rejected(field: str, value: float) -> None:
    values = {field: value}
    with pytest.raises(ValueError):
        ShadowLearningConfig(**values)


def test_incumbent_outside_norm_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="incumbent parameter norm"):
        train_shadow_adapter(
            [_sample(index) for index in range(10)],
            incumbent=AdapterState(base_model_version="base", bias=0.5),
            config=ShadowLearningConfig(max_parameter_norm=0.1),
        )


def test_replay_sample_validates_probability_and_binary_outcome() -> None:
    with pytest.raises(ValueError, match="probability"):
        _sample(0, probability=math.nan)
    with pytest.raises(TypeError, match="bool"):
        ReplaySample(
            sample_id="sample",
            episode_id="episode",
            prediction_id="prediction",
            target="target",
            probability=0.5,
            outcome=1,  # type: ignore[arg-type]
            prediction_event_id="prediction_event",
            resolution_event_id="resolution_event",
            resolution_kind="explicit",
        )
