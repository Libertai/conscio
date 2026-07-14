from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from conscio.v3.contracts import CognitiveEvent
from conscio.v3.curriculum import generate_synthetic_curriculum
from conscio.v3.recurrent_core import MODEL_VERSION, STATE_SIZE, HybridRecurrentCore
from conscio.v3.world_training import (
    TARGET_FAMILIES,
    WorldCoreWeights,
    WorldTargets,
    WorldTrainingConfig,
    WorldTrainingExample,
    active_world_weights,
    deterministic_episode_split,
    evaluate_shadow_candidate,
    examples_from_curriculum,
    train_shadow_world_model,
    world_loss,
)


def _config(**overrides: object) -> WorldTrainingConfig:
    values: dict[str, object] = {
        "epochs": 180,
        "batch_size": 16,
        "learning_rate": 0.08,
        "gradient_clip": 0.5,
        "max_step_update": 0.025,
        "max_total_delta": 4.0,
        "min_training_examples": 16,
        "min_validation_examples": 8,
        "min_training_episodes": 8,
        "min_validation_episodes": 4,
        "min_improvement": 0.0001,
        "max_target_regression": 0.05,
        **overrides,
    }
    return WorldTrainingConfig(**values)  # type: ignore[arg-type]


def _synthetic(*, seed: int = 404, episodes: int = 64) -> tuple[WorldTrainingExample, ...]:
    return examples_from_curriculum(generate_synthetic_curriculum(seed=seed, episodes=episodes))


def _example(
    episode: str,
    *,
    targets: WorldTargets | None = None,
    observation: tuple[float, ...] | None = None,
) -> WorldTrainingExample:
    zeros = (0.0,) * STATE_SIZE
    return WorldTrainingExample(
        example_id=f"example_{episode}",
        episode_id=episode,
        deterministic_state=zeros,
        stochastic_state=zeros,
        observation_features=observation or zeros,
        broadcast_features=zeros,
        targets=targets or WorldTargets(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def test_curriculum_adapter_is_joint_reproducible_and_target_complete() -> None:
    rows = generate_synthetic_curriculum(seed=91, episodes=6)

    first = examples_from_curriculum(rows)
    reversed_rows = examples_from_curriculum(reversed(rows))

    assert first == reversed_rows
    assert len(first) == 6
    assert all(tuple(item.targets.__dict__) == TARGET_FAMILIES for item in first)
    assert any(np.linalg.norm(item.observation_features) > 0.0 for item in first)
    assert {item.episode_id for item in first} == {row.episode_id for row in rows}


def test_curriculum_adapter_rejects_incomplete_and_ambiguous_windows() -> None:
    rows = list(generate_synthetic_curriculum(seed=12, episodes=1))
    with pytest.raises(ValueError, match="lacks target families"):
        examples_from_curriculum(rows[:-1])

    with pytest.raises(ValueError, match="ambiguous duplicate"):
        examples_from_curriculum([*rows, rows[0]])


def test_episode_split_is_order_independent_and_disjoint() -> None:
    examples = _synthetic(episodes=20)

    first = deterministic_episode_split(examples, validation_fraction=0.25, seed=7)
    second = deterministic_episode_split(reversed(examples), validation_fraction=0.25, seed=7)

    assert first == second
    assert {item.episode_id for item in first.train}.isdisjoint(
        item.episode_id for item in first.validation
    )
    assert len(first.train) == 15
    assert len(first.validation) == 5


def test_joint_training_improves_learnable_train_and_validation_loss() -> None:
    examples = _synthetic(seed=31, episodes=80)

    result = train_shadow_world_model(examples, config=_config())

    assert result.promoted is True
    assert result.reason == "promotion gates passed"
    assert result.training_loss_before is not None
    assert result.training_loss_after is not None
    assert result.candidate_validation_loss is not None
    assert result.incumbent_validation_loss is not None
    assert result.training_loss_after.total < result.training_loss_before.total
    assert result.candidate_validation_loss.total < result.incumbent_validation_loss.total
    assert result.candidate.parent_digest == result.incumbent.digest()
    assert result.candidate.revision == result.incumbent.revision + 1
    assert result.selected is result.candidate


def test_seeded_training_is_exactly_reproducible() -> None:
    examples = _synthetic(seed=707, episodes=48)
    config = _config(epochs=80)

    first = train_shadow_world_model(examples, config=config)
    second = train_shadow_world_model(reversed(examples), config=config)

    assert first.candidate.to_json() == second.candidate.to_json()
    assert first.diagnostics == second.diagnostics
    assert first.candidate_validation_loss == second.candidate_validation_loss


def test_gradient_and_parameter_updates_are_clipped() -> None:
    examples = _synthetic(seed=808, episodes=48)
    config = _config(
        epochs=12,
        learning_rate=100.0,
        gradient_clip=0.001,
        max_step_update=0.0002,
        max_total_delta=0.001,
        min_improvement=0.0,
        max_target_regression=1.0,
    )

    result = train_shadow_world_model(examples, config=config)

    assert result.diagnostics is not None
    assert result.diagnostics.clipped_steps > 0
    assert result.diagnostics.max_applied_update_norm <= config.max_step_update + 1e-12
    assert result.diagnostics.total_parameter_delta <= config.max_total_delta + 1e-12


def test_non_finite_examples_and_weights_are_rejected() -> None:
    bad = (float("nan"),) + (0.0,) * (STATE_SIZE - 1)
    with pytest.raises(ValueError, match="finite"):
        _example("bad", observation=bad)

    serialized = WorldCoreWeights.bootstrap().to_dict()
    serialized["prediction_bias"][0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        WorldCoreWeights.from_dict(serialized)


def test_content_addressed_serialization_restores_exactly_and_detects_tampering() -> None:
    weights = WorldCoreWeights.bootstrap()

    restored = WorldCoreWeights.from_json(weights.to_json())

    assert restored == weights
    assert restored.digest() == weights.digest()
    assert restored.model_version == MODEL_VERSION

    envelope = json.loads(weights.to_json())
    envelope["weights"]["prediction_bias"][0] = 0.125
    with pytest.raises(ValueError, match="digest mismatch"):
        WorldCoreWeights.from_json(json.dumps(envelope))


def test_worse_shadow_candidate_is_not_promoted() -> None:
    incumbent = WorldCoreWeights.bootstrap()
    transition = np.asarray(incumbent.observation_kernel) @ np.zeros(STATE_SIZE)
    expected = incumbent.predict_targets(np.tanh(transition))
    targets = WorldTargets(**expected)
    validation = tuple(_example(f"validation_{index}", targets=targets) for index in range(4))
    candidate = replace(
        incumbent,
        revision=incumbent.revision + 1,
        parent_digest=incumbent.digest(),
        prediction_bias=(0.25,) * len(TARGET_FAMILIES),
    )

    result = evaluate_shadow_candidate(
        incumbent=incumbent,
        candidate=candidate,
        validation=validation,
        config=_config(
            min_validation_examples=1,
            min_validation_episodes=1,
            min_improvement=0.0001,
            max_total_delta=10.0,
        ),
    )

    assert result.promoted is False
    assert result.reason == "held-out improvement below threshold"
    assert result.selected is incumbent
    assert result.candidate is candidate
    assert world_loss(candidate, validation).total > world_loss(incumbent, validation).total


def test_trained_bundle_installs_in_fresh_core_and_guards_checkpoint_lineage() -> None:
    result = train_shadow_world_model(_synthetic(episodes=48), config=_config(epochs=60))
    weights = result.candidate
    core = HybridRecurrentCore(seed=4, weights=weights)
    event = CognitiveEvent(event_type="message", source="user", payload={"content": "status"}, episode_id="e")

    cycle = core.run_cycles(event, cycles=1)[0]
    checkpoint = core.checkpoint()

    assert core.active_weight_bundle is weights
    assert active_world_weights(core) is weights
    assert core.model_version == weights.model_version
    assert checkpoint.model_version == weights.model_version
    assert tuple(prediction.target for prediction in cycle.predictions) == TARGET_FAMILIES
    with pytest.raises(ValueError, match="explicit lineage migration"):
        HybridRecurrentCore().restore(checkpoint)
    restored = HybridRecurrentCore(weights=weights)
    restored.restore(checkpoint)
    assert np.array_equal(restored.deterministic, core.deterministic)


def test_default_core_resolves_to_backward_compatible_bootstrap_bundle() -> None:
    core = HybridRecurrentCore()

    weights = active_world_weights(core)

    assert core.active_weight_bundle is None
    assert weights.model_version == MODEL_VERSION == core.model_version
    assert np.array_equal(np.asarray(weights.history_kernel), core._wh)
    assert np.array_equal(np.asarray(weights.observation_kernel), core._wx)
    assert np.array_equal(np.asarray(weights.broadcast_kernel), core._wb)
