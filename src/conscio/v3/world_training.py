"""Deterministic offline training for the V3 recurrent world-state core.

The language specialist is absent from this module by construction.  Training
consumes numeric, typed transition examples and returns a new immutable weight
bundle; it cannot mutate a live core, execute tools, or update a base LLM.

Every candidate is evaluated on episode-disjoint held-out data.  Promotion is
an explicit selection result, never an in-place overwrite, and the candidate's
content digest names both its model version and its immutable parent weights.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard

import numpy as np

from conscio.v3.recurrent_core import MODEL_VERSION, STATE_SIZE, HybridRecurrentCore

WEIGHT_SCHEMA_VERSION = 1
WEIGHT_KIND = "v3_recurrent_world_core"
TARGET_FAMILIES = (
    "next_observation",
    "tool_outcome",
    "action_effect",
    "homeostatic_affect_change",
    "future_uncertainty",
)
TARGET_COUNT = len(TARGET_FAMILIES)


class TrainingExampleLike(Protocol):
    """Minimal structural interface accepted from future curriculum loaders."""

    @property
    def example_id(self) -> str: ...

    @property
    def episode_id(self) -> str: ...

    @property
    def deterministic_state(self) -> Sequence[float]: ...

    @property
    def stochastic_state(self) -> Sequence[float]: ...

    @property
    def observation_features(self) -> Sequence[float]: ...

    @property
    def broadcast_features(self) -> Sequence[float]: ...

    @property
    def targets(self) -> WorldTargets: ...


class CurriculumExampleLike(Protocol):
    """Structural view of a per-family curriculum record.

    Five records sharing ``(episode_id, step)`` become one joint world-model
    example.  This avoids an import dependency on the curriculum package.
    """

    @property
    def example_id(self) -> str: ...

    @property
    def episode_id(self) -> str: ...

    @property
    def step(self) -> int: ...

    @property
    def target_family(self) -> str: ...

    @property
    def inputs(self) -> Mapping[str, Any]: ...

    @property
    def target(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class WorldTargets:
    """Normalized observable targets for one transition.

    Each value is in ``[0, 1]``.  Binary observations use 0/1; continuous
    quantities (including homeostatic change and future uncertainty) are
    normalized by the curriculum before reaching this safety boundary.
    """

    next_observation: float
    tool_outcome: float
    action_effect: float
    homeostatic_affect_change: float
    future_uncertainty: float

    def __post_init__(self) -> None:
        for name in TARGET_FAMILIES:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"target {name} must be finite and in [0, 1]")

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in TARGET_FAMILIES)


@dataclass(frozen=True)
class WorldTrainingExample:
    example_id: str
    episode_id: str
    deterministic_state: tuple[float, ...]
    stochastic_state: tuple[float, ...]
    observation_features: tuple[float, ...]
    broadcast_features: tuple[float, ...]
    targets: WorldTargets

    def __post_init__(self) -> None:
        if not self.example_id or not self.episode_id:
            raise ValueError("training example and episode identifiers must be non-empty")
        for name in (
            "deterministic_state",
            "stochastic_state",
            "observation_features",
            "broadcast_features",
        ):
            _validate_vector(getattr(self, name), name)


@dataclass(frozen=True)
class WorldCoreWeights:
    """Immutable, versioned parameters for transition and prediction heads."""

    history_kernel: tuple[tuple[float, ...], ...]
    observation_kernel: tuple[tuple[float, ...], ...]
    broadcast_kernel: tuple[tuple[float, ...], ...]
    prediction_kernel: tuple[tuple[float, ...], ...]
    prediction_bias: tuple[float, ...]
    base_model_version: str = MODEL_VERSION
    revision: int = 0
    parent_digest: str | None = None
    trained_examples: int = 0
    validation_examples: int = 0
    validation_loss: float | None = None
    training_seed: int | None = None
    schema_version: int = WEIGHT_SCHEMA_VERSION
    weight_kind: str = WEIGHT_KIND

    def __post_init__(self) -> None:
        if self.schema_version != WEIGHT_SCHEMA_VERSION:
            raise ValueError(f"unsupported world weight schema version: {self.schema_version}")
        if self.weight_kind != WEIGHT_KIND:
            raise ValueError(f"unsupported world weight kind: {self.weight_kind!r}")
        if not self.base_model_version:
            raise ValueError("base_model_version must be non-empty")
        if self.revision < 0 or self.trained_examples < 0 or self.validation_examples < 0:
            raise ValueError("world weight counters must be non-negative")
        if self.revision == 0 and self.parent_digest is not None:
            raise ValueError("bootstrap world weights cannot have a parent digest")
        if self.revision > 0 and not _valid_digest(self.parent_digest):
            raise ValueError("trained world weights require a SHA-256 parent digest")
        if self.training_seed is not None and (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed < 0
        ):
            raise ValueError("training_seed must be a non-negative integer")
        _validate_matrix(self.history_kernel, (STATE_SIZE, STATE_SIZE), "history_kernel")
        _validate_matrix(self.observation_kernel, (STATE_SIZE, STATE_SIZE), "observation_kernel")
        _validate_matrix(self.broadcast_kernel, (STATE_SIZE, STATE_SIZE), "broadcast_kernel")
        _validate_matrix(self.prediction_kernel, (TARGET_COUNT, STATE_SIZE), "prediction_kernel")
        _validate_vector(self.prediction_bias, "prediction_bias", size=TARGET_COUNT)
        if self.validation_loss is not None and (
            not math.isfinite(self.validation_loss) or self.validation_loss < 0.0
        ):
            raise ValueError("validation_loss must be finite and non-negative")

    @classmethod
    def bootstrap(cls) -> WorldCoreWeights:
        """Return the deterministic transition initialization used by the core."""
        rng = np.random.default_rng(7301)
        history = rng.normal(0.0, 0.12, (STATE_SIZE, STATE_SIZE))
        observation = rng.normal(0.0, 0.18, (STATE_SIZE, STATE_SIZE))
        broadcast = rng.normal(0.0, 0.10, (STATE_SIZE, STATE_SIZE))
        head_rng = np.random.default_rng(9119)
        prediction = head_rng.normal(0.0, 0.08, (TARGET_COUNT, STATE_SIZE))
        return cls(
            history_kernel=_matrix_tuple(history),
            observation_kernel=_matrix_tuple(observation),
            broadcast_kernel=_matrix_tuple(broadcast),
            prediction_kernel=_matrix_tuple(prediction),
            prediction_bias=(0.0,) * TARGET_COUNT,
        )

    @property
    def model_version(self) -> str:
        if self.revision == 0 and self.parent_digest is None:
            return self.base_model_version
        return f"{self.base_model_version}.world-r{self.revision}-{self.digest()[:16]}"

    def predict_targets(self, state: np.ndarray) -> Mapping[str, float]:
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (STATE_SIZE,) or not np.all(np.isfinite(values)):
            raise ValueError(f"state must be a finite vector of length {STATE_SIZE}")
        kernel = np.asarray(self.prediction_kernel, dtype=np.float64)
        bias = np.asarray(self.prediction_bias, dtype=np.float64)
        predictions = _sigmoid(kernel @ values + bias)
        return {name: float(predictions[index]) for index, name in enumerate(TARGET_FAMILIES)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "weight_kind": self.weight_kind,
            "base_model_version": self.base_model_version,
            "revision": self.revision,
            "parent_digest": self.parent_digest,
            "trained_examples": self.trained_examples,
            "validation_examples": self.validation_examples,
            "validation_loss": self.validation_loss,
            "training_seed": self.training_seed,
            "history_kernel": [list(row) for row in self.history_kernel],
            "observation_kernel": [list(row) for row in self.observation_kernel],
            "broadcast_kernel": [list(row) for row in self.broadcast_kernel],
            "prediction_kernel": [list(row) for row in self.prediction_kernel],
            "prediction_bias": list(self.prediction_bias),
            "target_families": list(TARGET_FAMILIES),
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Serialize a self-verifying content-addressed envelope."""
        return _canonical_json({"digest": self.digest(), "weights": self.to_dict()})

    @classmethod
    def from_json(cls, payload: str) -> WorldCoreWeights:
        try:
            envelope = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("world weights are not valid JSON") from exc
        if not isinstance(envelope, dict) or not isinstance(envelope.get("weights"), dict):
            raise ValueError("world weight JSON must contain a weights object")
        weights = cls.from_dict(envelope["weights"])
        claimed = envelope.get("digest")
        if not isinstance(claimed, str) or not claimed:
            raise ValueError("world weight JSON has no content digest")
        if not hmac.compare_digest(claimed, weights.digest()):
            raise ValueError("world weight content digest mismatch")
        return weights

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorldCoreWeights:
        targets = data.get("target_families")
        if (tuple(targets) if isinstance(targets, list) else ()) != TARGET_FAMILIES:
            raise ValueError("unsupported or reordered world target families")
        try:
            return cls(
                history_kernel=_matrix_from_value(data["history_kernel"], "history_kernel"),
                observation_kernel=_matrix_from_value(data["observation_kernel"], "observation_kernel"),
                broadcast_kernel=_matrix_from_value(data["broadcast_kernel"], "broadcast_kernel"),
                prediction_kernel=_matrix_from_value(data["prediction_kernel"], "prediction_kernel"),
                prediction_bias=_vector_from_value(data["prediction_bias"], "prediction_bias"),
                base_model_version=str(data["base_model_version"]),
                revision=_strict_int(data["revision"], "revision"),
                parent_digest=None if data.get("parent_digest") is None else str(data["parent_digest"]),
                trained_examples=_strict_int(data["trained_examples"], "trained_examples"),
                validation_examples=_strict_int(data["validation_examples"], "validation_examples"),
                validation_loss=(
                    None if data.get("validation_loss") is None else float(data["validation_loss"])
                ),
                training_seed=(
                    None if data.get("training_seed") is None else _strict_int(data["training_seed"], "training_seed")
                ),
                schema_version=_strict_int(data["schema_version"], "schema_version"),
                weight_kind=str(data["weight_kind"]),
            )
        except KeyError as exc:
            raise ValueError(f"missing world weight field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid world weight bundle: {exc}") from exc


@dataclass(frozen=True)
class WorldTrainingSplit:
    train: tuple[WorldTrainingExample, ...]
    validation: tuple[WorldTrainingExample, ...]


@dataclass(frozen=True)
class WorldLoss:
    total: float
    by_target: tuple[tuple[str, float], ...]

    def target(self, name: str) -> float:
        return dict(self.by_target)[name]


@dataclass(frozen=True)
class WorldTrainingConfig:
    validation_fraction: float = 0.25
    seed: int = 17
    epochs: int = 240
    batch_size: int = 32
    learning_rate: float = 0.025
    gradient_clip: float = 1.0
    max_step_update: float = 0.04
    max_total_delta: float = 2.0
    max_parameter_abs: float = 4.0
    min_training_examples: int = 24
    min_validation_examples: int = 8
    min_training_episodes: int = 2
    min_validation_episodes: int = 2
    min_improvement: float = 0.001
    max_target_regression: float = 0.002

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "learning_rate",
            "gradient_clip",
            "max_step_update",
            "max_total_delta",
            "max_parameter_abs",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "min_training_examples",
            "min_validation_examples",
            "min_training_episodes",
            "min_validation_episodes",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("min_improvement", "max_target_regression"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class TrainingDiagnostics:
    max_raw_gradient_norm: float
    max_applied_update_norm: float
    total_parameter_delta: float
    clipped_steps: int
    steps: int


@dataclass(frozen=True)
class ShadowWorldTrainingResult:
    incumbent: WorldCoreWeights
    candidate: WorldCoreWeights
    selected: WorldCoreWeights
    promoted: bool
    reason: str
    training_loss_before: WorldLoss | None
    training_loss_after: WorldLoss | None
    incumbent_validation_loss: WorldLoss | None
    candidate_validation_loss: WorldLoss | None
    train_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]
    diagnostics: TrainingDiagnostics | None


def deterministic_episode_split(
    examples: Iterable[TrainingExampleLike],
    *,
    validation_fraction: float = 0.25,
    seed: int = 17,
) -> WorldTrainingSplit:
    """Split whole episodes deterministically, independent of input ordering."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    snapshots = tuple(sorted((_snapshot(item) for item in examples), key=_example_sort_key))
    episodes = sorted(
        {item.episode_id for item in snapshots},
        key=lambda episode: (hashlib.sha256(f"{seed}:{episode}".encode()).digest(), episode),
    )
    if len(episodes) < 2:
        return WorldTrainingSplit(snapshots, ())
    validation_count = min(len(episodes) - 1, max(1, round(len(episodes) * validation_fraction)))
    validation_ids = frozenset(episodes[:validation_count])
    return WorldTrainingSplit(
        tuple(item for item in snapshots if item.episode_id not in validation_ids),
        tuple(item for item in snapshots if item.episode_id in validation_ids),
    )


def examples_from_curriculum(
    records: Iterable[CurriculumExampleLike],
) -> tuple[WorldTrainingExample, ...]:
    """Assemble model-neutral curriculum JSON into joint transition examples.

    Within each episode, the first row of every family forms window zero, the
    second row of every family forms window one, and so on.  Family rows are
    ordered by ``(step, example_id)``.  Inputs are converted with stable
    SHA-256 feature hashing; recurrent history and stochastic snapshots depend
    only on the episode and preceding inputs, never on target values.

    Missing families, duplicate ``(family, step)`` rows, unequal family counts,
    or ambiguous labels raise ``ValueError`` rather than silently losing data.
    """
    grouped: dict[str, dict[str, list[CurriculumExampleLike]]] = {}
    seen_positions: set[tuple[str, str, int]] = set()
    for record in records:
        episode_id = str(record.episode_id)
        if not episode_id or not str(record.example_id):
            raise ValueError("curriculum example and episode identifiers must be non-empty")
        family = str(record.target_family)
        if family not in TARGET_FAMILIES:
            raise ValueError(f"unsupported curriculum target family: {family!r}")
        if isinstance(record.step, bool) or not isinstance(record.step, int) or record.step < 0:
            raise ValueError("curriculum step must be a non-negative integer")
        position = (episode_id, family, record.step)
        if position in seen_positions:
            raise ValueError(f"ambiguous duplicate curriculum row at {position}")
        seen_positions.add(position)
        grouped.setdefault(episode_id, {}).setdefault(family, []).append(record)

    assembled: list[WorldTrainingExample] = []
    bootstrap = WorldCoreWeights.bootstrap()
    history_kernel = np.asarray(bootstrap.history_kernel, dtype=np.float64)
    observation_kernel = np.asarray(bootstrap.observation_kernel, dtype=np.float64)
    broadcast_kernel = np.asarray(bootstrap.broadcast_kernel, dtype=np.float64)
    for episode_id, family_groups in sorted(grouped.items()):
        missing = set(TARGET_FAMILIES) - set(family_groups)
        if missing:
            raise ValueError(f"curriculum episode {episode_id!r} lacks target families: {sorted(missing)}")
        ordered = {
            family: sorted(family_groups[family], key=lambda item: (item.step, str(item.example_id)))
            for family in TARGET_FAMILIES
        }
        counts = {family: len(items) for family, items in ordered.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"curriculum episode {episode_id!r} has incomplete target windows: {counts}")
        deterministic = np.zeros(STATE_SIZE, dtype=np.float64)
        preceding = np.zeros(STATE_SIZE, dtype=np.float64)
        for window in range(next(iter(counts.values()))):
            family_records = {family: ordered[family][window] for family in TARGET_FAMILIES}
            input_material = [
                {
                    "family": family,
                    "step": family_records[family].step,
                    "inputs": family_records[family].inputs,
                }
                for family in TARGET_FAMILIES
            ]
            observation = _feature_hash(input_material)
            stochastic = 0.08 * _feature_hash(
                {"episode_id": episode_id, "window": window, "channel": "stochastic"}
            )
            target_values = {
                family: _curriculum_target(family_records[family].target, family)
                for family in TARGET_FAMILIES
            }
            source_ids = sorted(str(record.example_id) for record in family_records.values())
            identity = hashlib.sha256("\n".join(source_ids).encode()).hexdigest()[:16]
            assembled.append(
                WorldTrainingExample(
                    example_id=f"curriculum_{episode_id}_{window}_{identity}",
                    episode_id=episode_id,
                    deterministic_state=tuple(float(value) for value in deterministic),
                    stochastic_state=tuple(float(value) for value in stochastic),
                    observation_features=tuple(float(value) for value in observation),
                    broadcast_features=tuple(float(value) for value in preceding),
                    targets=WorldTargets(**target_values),
                )
            )
            deterministic = np.tanh(
                history_kernel @ deterministic
                + observation_kernel @ observation
                + broadcast_kernel @ preceding
                + 0.15 * stochastic
            )
            preceding = observation
    return tuple(assembled)


def world_loss(weights: WorldCoreWeights, examples: Sequence[TrainingExampleLike]) -> WorldLoss:
    snapshots = tuple(_snapshot(example) for example in examples)
    if not snapshots:
        raise ValueError("world loss requires at least one example")
    arrays = _arrays(snapshots)
    predictions, _ = _forward(_parameter_arrays(weights), arrays)
    errors = np.square(predictions - arrays.targets)
    target_losses = np.mean(errors, axis=0)
    if not np.all(np.isfinite(target_losses)):
        raise FloatingPointError("world loss is non-finite")
    return WorldLoss(
        total=float(np.mean(target_losses)),
        by_target=tuple((name, float(target_losses[index])) for index, name in enumerate(TARGET_FAMILIES)),
    )


def active_world_weights(core: HybridRecurrentCore) -> WorldCoreWeights:
    """Resolve a core's public immutable bundle, including bootstrap defaults."""
    installed = core.active_weight_bundle
    if installed is None:
        return WorldCoreWeights.bootstrap()
    if not isinstance(installed, WorldCoreWeights):
        raise TypeError("active core bundle is not a WorldCoreWeights instance")
    return installed


def train_shadow_world_model(
    examples: Iterable[TrainingExampleLike],
    *,
    incumbent: WorldCoreWeights | None = None,
    config: WorldTrainingConfig | None = None,
) -> ShadowWorldTrainingResult:
    """Train in shadow and select a candidate only after conservative gates."""
    active = incumbent or WorldCoreWeights.bootstrap()
    settings = config or WorldTrainingConfig()
    _validate_incumbent(active, settings)
    split = deterministic_episode_split(
        examples,
        validation_fraction=settings.validation_fraction,
        seed=settings.seed,
    )
    train_episodes = tuple(sorted({item.episode_id for item in split.train}))
    validation_episodes = tuple(sorted({item.episode_id for item in split.validation}))
    insufficiency = _insufficient_reason(split, train_episodes, validation_episodes, settings)
    if insufficiency is not None:
        return ShadowWorldTrainingResult(
            incumbent=active,
            candidate=active,
            selected=active,
            promoted=False,
            reason=insufficiency,
            training_loss_before=None,
            training_loss_after=None,
            incumbent_validation_loss=None,
            candidate_validation_loss=None,
            train_episode_ids=train_episodes,
            validation_episode_ids=validation_episodes,
            diagnostics=None,
        )

    before = world_loss(active, split.train)
    params = _parameter_arrays(active)
    initial = tuple(array.copy() for array in params)
    arrays = _arrays(split.train)
    rng = np.random.default_rng(settings.seed)
    max_gradient = 0.0
    max_update = 0.0
    clipped_steps = 0
    steps = 0
    for _ in range(settings.epochs):
        order = rng.permutation(len(split.train))
        for start in range(0, len(order), settings.batch_size):
            batch = arrays.take(order[start : start + settings.batch_size])
            gradients = _gradients(params, batch)
            raw_norm = _global_norm(gradients)
            if not math.isfinite(raw_norm):
                raise FloatingPointError("world model gradient is non-finite")
            max_gradient = max(max_gradient, raw_norm)
            scale = min(1.0, settings.gradient_clip / max(raw_norm, 1e-30))
            proposed = tuple(-settings.learning_rate * scale * gradient for gradient in gradients)
            proposed_norm = _global_norm(proposed)
            if proposed_norm > settings.max_step_update:
                update_scale = settings.max_step_update / proposed_norm
                proposed = tuple(update * update_scale for update in proposed)
            if scale < 1.0 or proposed_norm > settings.max_step_update:
                clipped_steps += 1
            previous = tuple(array.copy() for array in params)
            params = tuple(array + update for array, update in zip(params, proposed, strict=True))
            params = tuple(np.clip(array, -settings.max_parameter_abs, settings.max_parameter_abs) for array in params)
            params = _project_delta(params, initial, settings.max_total_delta)
            if not all(np.all(np.isfinite(array)) for array in params):
                raise FloatingPointError("world model update produced non-finite parameters")
            max_update = max(
                max_update,
                _global_norm(tuple(after - old for after, old in zip(params, previous, strict=True))),
            )
            steps += 1

    provisional = _weights_from_parameters(
        active,
        params,
        revision=active.revision + 1,
        parent_digest=active.digest(),
        trained_examples=len(split.train),
        validation_examples=len(split.validation),
        validation_loss=None,
        training_seed=settings.seed,
    )
    candidate_validation = world_loss(provisional, split.validation)
    candidate = _weights_from_parameters(
        active,
        params,
        revision=active.revision + 1,
        parent_digest=active.digest(),
        trained_examples=len(split.train),
        validation_examples=len(split.validation),
        validation_loss=candidate_validation.total,
        training_seed=settings.seed,
    )
    after = world_loss(candidate, split.train)
    diagnostics = TrainingDiagnostics(
        max_raw_gradient_norm=max_gradient,
        max_applied_update_norm=max_update,
        total_parameter_delta=_global_norm(tuple(now - base for now, base in zip(params, initial, strict=True))),
        clipped_steps=clipped_steps,
        steps=steps,
    )
    return evaluate_shadow_candidate(
        incumbent=active,
        candidate=candidate,
        validation=split.validation,
        config=settings,
        training_loss_before=before,
        training_loss_after=after,
        train_episode_ids=train_episodes,
        validation_episode_ids=validation_episodes,
        diagnostics=diagnostics,
    )


def evaluate_shadow_candidate(
    *,
    incumbent: WorldCoreWeights,
    candidate: WorldCoreWeights,
    validation: Sequence[TrainingExampleLike],
    config: WorldTrainingConfig | None = None,
    training_loss_before: WorldLoss | None = None,
    training_loss_after: WorldLoss | None = None,
    train_episode_ids: tuple[str, ...] = (),
    validation_episode_ids: tuple[str, ...] = (),
    diagnostics: TrainingDiagnostics | None = None,
) -> ShadowWorldTrainingResult:
    """Evaluate an immutable candidate without installing or mutating it."""
    settings = config or WorldTrainingConfig()
    snapshots = tuple(_snapshot(item) for item in validation)
    actual_validation_episodes = tuple(sorted({item.episode_id for item in snapshots}))
    if set(train_episode_ids) & set(actual_validation_episodes):
        raise ValueError("training and validation episodes must be disjoint")
    if validation_episode_ids and validation_episode_ids != actual_validation_episodes:
        raise ValueError("declared validation episodes do not match validation examples")
    incumbent_loss = world_loss(incumbent, snapshots) if snapshots else None
    candidate_loss = world_loss(candidate, snapshots) if snapshots else None
    reason = _promotion_rejection(incumbent, candidate, incumbent_loss, candidate_loss, snapshots, settings)
    promoted = reason is None
    return ShadowWorldTrainingResult(
        incumbent=incumbent,
        candidate=candidate,
        selected=candidate if promoted else incumbent,
        promoted=promoted,
        reason="promotion gates passed" if promoted else reason or "promotion rejected",
        training_loss_before=training_loss_before,
        training_loss_after=training_loss_after,
        incumbent_validation_loss=incumbent_loss,
        candidate_validation_loss=candidate_loss,
        train_episode_ids=train_episode_ids,
        validation_episode_ids=validation_episode_ids,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class _TrainingArrays:
    history: np.ndarray
    stochastic: np.ndarray
    observation: np.ndarray
    broadcast: np.ndarray
    targets: np.ndarray

    def take(self, indexes: np.ndarray) -> _TrainingArrays:
        return _TrainingArrays(
            self.history[indexes],
            self.stochastic[indexes],
            self.observation[indexes],
            self.broadcast[indexes],
            self.targets[indexes],
        )


def _snapshot(example: TrainingExampleLike) -> WorldTrainingExample:
    if not isinstance(example.targets, WorldTargets):
        raise TypeError("training example targets must be WorldTargets")
    return WorldTrainingExample(
        example_id=str(example.example_id),
        episode_id=str(example.episode_id),
        deterministic_state=tuple(float(value) for value in example.deterministic_state),
        stochastic_state=tuple(float(value) for value in example.stochastic_state),
        observation_features=tuple(float(value) for value in example.observation_features),
        broadcast_features=tuple(float(value) for value in example.broadcast_features),
        targets=example.targets,
    )


def _feature_hash(value: Any) -> np.ndarray:
    """Hash arbitrary JSON leaves into a stable, normalized state vector."""
    result = np.zeros(STATE_SIZE, dtype=np.float64)

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        token = f"{path}={_canonical_json(item)}"
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % STATE_SIZE
        sign = -1.0 if digest[4] & 1 else 1.0
        result[index] += sign
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            numeric = float(item)
            if not math.isfinite(numeric):
                raise ValueError(f"curriculum feature {path} is non-finite")
            numeric_index = int.from_bytes(digest[5:9], "little") % STATE_SIZE
            result[numeric_index] += max(-1.0, min(1.0, numeric))

    visit(value, "root")
    norm = float(np.linalg.norm(result))
    return result / norm if norm else result


def _curriculum_target(target: Mapping[str, Any], family: str) -> float:
    explicit_keys = ("value", "normalized_value", family, "label", "outcome")
    explicit = [_normalized_scalar(target[key], f"target.{key}") for key in explicit_keys if key in target]
    if explicit:
        if any(abs(value - explicit[0]) > 1e-12 for value in explicit[1:]):
            raise ValueError(f"ambiguous normalized labels for curriculum target {family!r}")
        return explicit[0]
    if family == "next_observation":
        observed = target.get("observed")
        if type(observed) is bool:
            return float(observed)
        event_type = target.get("event_type")
        if isinstance(event_type, str) and event_type.strip():
            return 1.0
    elif family == "tool_outcome":
        succeeded = target.get("succeeded")
        if type(succeeded) is bool:
            return float(succeeded)
    elif family == "action_effect":
        succeeded = target.get("succeeded")
        if type(succeeded) is bool:
            return float(succeeded)
        progress = target.get("task_progress_delta")
        if _finite_number(progress):
            return max(0.0, min(1.0, float(progress)))
    elif family == "homeostatic_affect_change":
        improved = target.get("improved")
        if type(improved) is bool:
            return float(improved)
        delta = target.get("delta")
        if isinstance(delta, Mapping):
            components: list[float] = []
            for key, direction in (("valence", 1.0), ("arousal", -1.0), ("controllability", 1.0)):
                if _finite_number(delta.get(key)):
                    components.append(direction * float(delta[key]))
            needs = delta.get("need_errors")
            if isinstance(needs, Mapping):
                components.extend(
                    -float(value) for value in needs.values() if _finite_number(value)
                )
            if components:
                signed_change = sum(components) / len(components)
                return max(0.0, min(1.0, 0.5 + 0.5 * signed_change))
    elif family == "future_uncertainty":
        nonincrease = target.get("nonincrease")
        if type(nonincrease) is bool:
            return float(nonincrease)
    raise ValueError(f"curriculum target {family!r} has no unambiguous normalized label")


def _normalized_scalar(value: Any, name: str) -> float:
    if type(value) is bool:
        return float(value)
    if not _finite_number(value):
        raise ValueError(f"{name} must be a finite normalized scalar")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _finite_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _arrays(examples: Sequence[WorldTrainingExample]) -> _TrainingArrays:
    return _TrainingArrays(
        np.asarray([item.deterministic_state for item in examples], dtype=np.float64),
        np.asarray([item.stochastic_state for item in examples], dtype=np.float64),
        np.asarray([item.observation_features for item in examples], dtype=np.float64),
        np.asarray([item.broadcast_features for item in examples], dtype=np.float64),
        np.asarray([item.targets.as_tuple() for item in examples], dtype=np.float64),
    )


def _parameter_arrays(weights: WorldCoreWeights) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(weights.history_kernel, dtype=np.float64).copy(),
        np.asarray(weights.observation_kernel, dtype=np.float64).copy(),
        np.asarray(weights.broadcast_kernel, dtype=np.float64).copy(),
        np.asarray(weights.prediction_kernel, dtype=np.float64).copy(),
        np.asarray(weights.prediction_bias, dtype=np.float64).copy(),
    )


def _forward(
    params: tuple[np.ndarray, ...], arrays: _TrainingArrays
) -> tuple[np.ndarray, np.ndarray]:
    history_kernel, observation_kernel, broadcast_kernel, prediction_kernel, bias = params
    preactivation = (
        arrays.history @ history_kernel.T
        + arrays.observation @ observation_kernel.T
        + arrays.broadcast @ broadcast_kernel.T
        + 0.15 * arrays.stochastic
    )
    state = np.tanh(preactivation)
    predictions = _sigmoid(state @ prediction_kernel.T + bias)
    return predictions, state


def _gradients(params: tuple[np.ndarray, ...], arrays: _TrainingArrays) -> tuple[np.ndarray, ...]:
    history_kernel, observation_kernel, broadcast_kernel, prediction_kernel, _ = params
    predictions, state = _forward(params, arrays)
    output_gradient = (2.0 / predictions.size) * (predictions - arrays.targets)
    logit_gradient = output_gradient * predictions * (1.0 - predictions)
    prediction_gradient = logit_gradient.T @ state
    bias_gradient = np.sum(logit_gradient, axis=0)
    state_gradient = logit_gradient @ prediction_kernel
    preactivation_gradient = state_gradient * (1.0 - np.square(state))
    history_gradient = preactivation_gradient.T @ arrays.history
    observation_gradient = preactivation_gradient.T @ arrays.observation
    broadcast_gradient = preactivation_gradient.T @ arrays.broadcast
    gradients = (
        history_gradient,
        observation_gradient,
        broadcast_gradient,
        prediction_gradient,
        bias_gradient,
    )
    if not all(np.all(np.isfinite(gradient)) for gradient in gradients):
        raise FloatingPointError("world model gradient is non-finite")
    return gradients


def _weights_from_parameters(
    incumbent: WorldCoreWeights,
    params: tuple[np.ndarray, ...],
    **metadata: Any,
) -> WorldCoreWeights:
    history, observation, broadcast, prediction, bias = params
    return WorldCoreWeights(
        history_kernel=_matrix_tuple(history),
        observation_kernel=_matrix_tuple(observation),
        broadcast_kernel=_matrix_tuple(broadcast),
        prediction_kernel=_matrix_tuple(prediction),
        prediction_bias=tuple(float(value) for value in bias),
        base_model_version=incumbent.base_model_version,
        **metadata,
    )


def _promotion_rejection(
    incumbent: WorldCoreWeights,
    candidate: WorldCoreWeights,
    incumbent_loss: WorldLoss | None,
    candidate_loss: WorldLoss | None,
    validation: Sequence[WorldTrainingExample],
    config: WorldTrainingConfig,
) -> str | None:
    if len(validation) < config.min_validation_examples:
        return "insufficient validation examples"
    if len({item.episode_id for item in validation}) < config.min_validation_episodes:
        return "insufficient validation episodes"
    if candidate.base_model_version != incumbent.base_model_version:
        return "candidate base model version mismatch"
    if candidate.revision != incumbent.revision + 1:
        return "candidate revision does not extend incumbent"
    if candidate.parent_digest != incumbent.digest():
        return "candidate parent digest does not extend incumbent"
    delta = _global_norm(
        tuple(
            candidate_value - incumbent_value
            for candidate_value, incumbent_value in zip(
                _parameter_arrays(candidate), _parameter_arrays(incumbent), strict=True
            )
        )
    )
    if delta > config.max_total_delta + 1e-12:
        return "candidate parameter delta exceeds configured bound"
    if incumbent_loss is None or candidate_loss is None:
        return "validation loss unavailable"
    improvement = incumbent_loss.total - candidate_loss.total
    if not math.isfinite(improvement) or improvement < config.min_improvement:
        return "held-out improvement below threshold"
    incumbent_targets = dict(incumbent_loss.by_target)
    for name, loss in candidate_loss.by_target:
        if loss - incumbent_targets[name] > config.max_target_regression:
            return f"held-out target regression exceeds threshold: {name}"
    return None


def _insufficient_reason(
    split: WorldTrainingSplit,
    train_episodes: tuple[str, ...],
    validation_episodes: tuple[str, ...],
    config: WorldTrainingConfig,
) -> str | None:
    if len(split.train) < config.min_training_examples:
        return "insufficient training examples"
    if len(split.validation) < config.min_validation_examples:
        return "insufficient validation examples"
    if len(train_episodes) < config.min_training_episodes:
        return "insufficient training episodes"
    if len(validation_episodes) < config.min_validation_episodes:
        return "insufficient validation episodes"
    return None


def _validate_incumbent(weights: WorldCoreWeights, config: WorldTrainingConfig) -> None:
    arrays = _parameter_arrays(weights)
    if any(float(np.max(np.abs(array))) > config.max_parameter_abs for array in arrays):
        raise ValueError("incumbent parameter magnitude exceeds configured bound")


def _project_delta(
    params: tuple[np.ndarray, ...],
    initial: tuple[np.ndarray, ...],
    maximum: float,
) -> tuple[np.ndarray, ...]:
    deltas = tuple(value - base for value, base in zip(params, initial, strict=True))
    norm = _global_norm(deltas)
    if norm <= maximum:
        return params
    scale = maximum / norm
    return tuple(base + delta * scale for base, delta in zip(initial, deltas, strict=True))


def _global_norm(arrays: Sequence[np.ndarray]) -> float:
    return math.sqrt(sum(float(np.sum(np.square(array))) for array in arrays))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _validate_vector(values: Sequence[float], name: str, *, size: int = STATE_SIZE) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def _validate_matrix(values: Sequence[Sequence[float]], shape: tuple[int, int], name: str) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def _matrix_tuple(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def _matrix_from_value(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or any(not isinstance(row, list) for row in value):
        raise ValueError(f"{name} must be a nested array")
    return tuple(tuple(float(item) for item in row) for row in value)


def _vector_from_value(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return tuple(float(item) for item in value)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _valid_digest(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _example_sort_key(example: WorldTrainingExample) -> tuple[str, str]:
    return (example.episode_id, example.example_id)


__all__ = [
    "TARGET_FAMILIES",
    "CurriculumExampleLike",
    "ShadowWorldTrainingResult",
    "TrainingDiagnostics",
    "TrainingExampleLike",
    "WorldCoreWeights",
    "WorldLoss",
    "WorldTargets",
    "WorldTrainingConfig",
    "WorldTrainingExample",
    "WorldTrainingSplit",
    "active_world_weights",
    "deterministic_episode_split",
    "evaluate_shadow_candidate",
    "examples_from_curriculum",
    "train_shadow_world_model",
    "world_loss",
]
