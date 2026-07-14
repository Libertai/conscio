"""Offline replay and shadow promotion for a bounded V3 calibration adapter.

This module deliberately has no reference to :mod:`conscio.v3.recurrent_core`.
It consumes immutable event dictionaries and returns immutable adapter state;
callers must explicitly persist and activate a promoted state.  In particular,
it cannot mutate recurrent base weights or a live runtime.

The adapter is a monotonic affine transform in log-odds space.  Its two
parameters are small enough to audit, are projected onto both per-update and
absolute norm bounds, and are promoted only on episode-separated held-out
Brier loss (a strictly proper scoring rule for binary outcomes).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ADAPTER_SCHEMA_VERSION = 1
ADAPTER_KIND = "affine_logit_calibrator"
_LABEL_KEYS = ("outcome", "actual", "label", "observed", "succeeded", "value")
_RESOLUTION_EVENT_TYPES = frozenset({"prediction_resolution", "prediction_outcome"})
_OBSERVATION_TARGETS = frozenset({"next_observation", "observation", "observation_present"})
_SUCCESS_TARGETS = frozenset({"action_success", "tool_success", "action_effect"})
_UNCERTAINTY_TARGETS = frozenset({"future_uncertainty", "uncertainty_nonincrease"})


@dataclass(frozen=True)
class ReplaySample:
    """One auditable binary prediction/outcome pair from the event log."""

    sample_id: str
    episode_id: str
    prediction_id: str
    target: str
    probability: float
    outcome: bool
    prediction_event_id: str
    resolution_event_id: str
    resolution_kind: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.episode_id or not self.prediction_id:
            raise ValueError("replay sample identifiers must be non-empty")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("replay probability must be finite and in [0, 1]")
        if type(self.outcome) is not bool:
            raise TypeError("replay outcome must be a bool")


@dataclass(frozen=True)
class ReplayRejection:
    """An event that could not safely become a supervised replay sample."""

    event_id: str
    episode_id: str
    prediction_id: str | None
    reason: str


@dataclass(frozen=True)
class ReplayDataset:
    samples: tuple[ReplaySample, ...]
    rejections: tuple[ReplayRejection, ...]


@dataclass(frozen=True)
class ReplaySplit:
    train: tuple[ReplaySample, ...]
    validation: tuple[ReplaySample, ...]


@dataclass(frozen=True)
class AdapterState:
    """Versioned state for a two-parameter binary probability calibrator.

    ``log_scale=0`` and ``bias=0`` is the identity transform.  Storing the
    logarithm keeps the scale positive, so calibration cannot reverse the
    ordering of predictions.
    """

    base_model_version: str = "unspecified"
    revision: int = 0
    log_scale: float = 0.0
    bias: float = 0.0
    trained_examples: int = 0
    validation_examples: int = 0
    validation_loss: float | None = None
    parent_digest: str | None = None
    schema_version: int = ADAPTER_SCHEMA_VERSION
    adapter_kind: str = ADAPTER_KIND

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_SCHEMA_VERSION:
            raise ValueError(f"unsupported adapter schema version: {self.schema_version}")
        if self.adapter_kind != ADAPTER_KIND:
            raise ValueError(f"unsupported adapter kind: {self.adapter_kind!r}")
        if not self.base_model_version:
            raise ValueError("base_model_version must be non-empty")
        if self.revision < 0 or self.trained_examples < 0 or self.validation_examples < 0:
            raise ValueError("adapter counters must be non-negative")
        if not math.isfinite(self.log_scale) or not math.isfinite(self.bias):
            raise ValueError("adapter parameters must be finite")
        if self.validation_loss is not None and (
            not math.isfinite(self.validation_loss) or self.validation_loss < 0.0
        ):
            raise ValueError("validation_loss must be a finite non-negative value")

    @property
    def parameters(self) -> tuple[float, float]:
        return (self.log_scale, self.bias)

    @property
    def parameter_norm(self) -> float:
        return math.hypot(*self.parameters)

    def calibrate(self, probability: float, *, epsilon: float = 1e-6) -> float:
        """Return a calibrated probability without modifying this state."""
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be finite and in [0, 1]")
        if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
            raise ValueError("epsilon must be finite and in (0, 0.5)")
        clipped = min(1.0 - epsilon, max(epsilon, probability))
        log_odds = math.log(clipped / (1.0 - clipped))
        transformed = math.exp(self.log_scale) * log_odds + self.bias
        return _sigmoid(transformed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_kind": self.adapter_kind,
            "base_model_version": self.base_model_version,
            "revision": self.revision,
            "log_scale": self.log_scale,
            "bias": self.bias,
            "trained_examples": self.trained_examples,
            "validation_examples": self.validation_examples,
            "validation_loss": self.validation_loss,
            "parent_digest": self.parent_digest,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdapterState:
        schema_version = _strict_int(data.get("schema_version"), "schema_version")
        if schema_version != ADAPTER_SCHEMA_VERSION:
            raise ValueError(f"unsupported adapter schema version: {schema_version}")
        adapter_kind = data.get("adapter_kind")
        if adapter_kind != ADAPTER_KIND:
            raise ValueError(f"unsupported adapter kind: {adapter_kind!r}")
        try:
            return cls(
                base_model_version=str(data["base_model_version"]),
                revision=_strict_int(data["revision"], "revision"),
                log_scale=float(data["log_scale"]),
                bias=float(data["bias"]),
                trained_examples=_strict_int(data["trained_examples"], "trained_examples"),
                validation_examples=_strict_int(data["validation_examples"], "validation_examples"),
                validation_loss=(
                    None if data.get("validation_loss") is None else float(data["validation_loss"])
                ),
                parent_digest=(None if data.get("parent_digest") is None else str(data["parent_digest"])),
                schema_version=schema_version,
                adapter_kind=str(adapter_kind),
            )
        except KeyError as exc:
            raise ValueError(f"missing adapter field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid adapter state: {exc}") from exc

    @classmethod
    def from_json(cls, payload: str) -> AdapterState:
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("adapter state is not valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("adapter JSON must contain an object")
        return cls.from_dict(data)


@dataclass(frozen=True)
class ShadowLearningConfig:
    validation_fraction: float = 0.25
    split_seed: int = 17
    learning_rate: float = 0.08
    epochs: int = 250
    min_training_examples: int = 24
    min_validation_examples: int = 8
    min_improvement: float = 0.002
    max_parameter_delta: float = 0.35
    max_parameter_norm: float = 1.5
    gradient_clip: float = 2.0
    probability_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.min_training_examples < 1 or self.min_validation_examples < 1:
            raise ValueError("minimum example counts must be positive")
        for name in ("learning_rate", "gradient_clip", "max_parameter_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.max_parameter_delta) or self.max_parameter_delta < 0.0:
            raise ValueError("max_parameter_delta must be finite and non-negative")
        if not math.isfinite(self.min_improvement) or self.min_improvement < 0.0:
            raise ValueError("min_improvement must be finite and non-negative")
        if not math.isfinite(self.probability_epsilon) or not 0.0 < self.probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must be finite and in (0, 0.5)")


@dataclass(frozen=True)
class ShadowTrainingResult:
    """Candidate evaluation plus the explicitly selected state."""

    incumbent: AdapterState
    candidate: AdapterState
    selected: AdapterState
    promoted: bool
    reason: str
    train_examples: int
    validation_examples: int
    incumbent_validation_loss: float | None
    candidate_validation_loss: float | None
    improvement: float | None


@dataclass
class _PendingPrediction:
    episode_id: str
    prediction_id: str
    target: str
    probability: float
    event_id: str
    remaining_horizon: int


def derive_replay_samples(events: Iterable[Mapping[str, Any]]) -> ReplayDataset:
    """Derive binary replay samples from persisted ``cognitive_events`` rows.

    Resolution is intentionally conservative.  Explicit prediction outcomes
    take precedence.  Otherwise, only recognized targets can be resolved from
    an ``action_outcome``: observation presence, action/tool success, or absence
    of recorded prediction errors.  Every skipped prediction is returned as a
    rejection so data loss is visible to the training operator.
    """
    indexed = list(enumerate(events))
    ordered = sorted(indexed, key=lambda item: _event_sort_key(item[1], item[0]))
    pending: dict[tuple[str, str], _PendingPrediction] = {}
    samples: list[ReplaySample] = []
    rejections: list[ReplayRejection] = []

    for input_index, event in ordered:
        episode_id = str(event.get("episode_id") or "")
        event_type = str(event.get("event_type") or "")
        event_id = str(event.get("event_id") or f"input_{input_index}")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            if event_type == "prediction":
                rejections.append(ReplayRejection(event_id, episode_id, None, "prediction payload is not an object"))
            continue

        if event_type == "prediction":
            parsed = _parse_prediction(payload, episode_id, event_id)
            if isinstance(parsed, ReplayRejection):
                rejections.append(parsed)
                continue
            key = (episode_id, parsed.prediction_id)
            if key in pending:
                rejections.append(
                    ReplayRejection(event_id, episode_id, parsed.prediction_id, "duplicate unresolved prediction id")
                )
                continue
            label = _label_from_mapping(payload)
            if label is None and payload.get("resolved") is True:
                label = _label_from_error(parsed.probability, payload.get("error"))
            if label is not None:
                samples.append(_sample(parsed, label, event_id, "prediction_payload"))
            else:
                pending[key] = parsed
            continue

        if event_type in _RESOLUTION_EVENT_TYPES:
            prediction_id = str(payload.get("prediction_id") or "")
            key = (episode_id, prediction_id)
            prediction = pending.get(key)
            if prediction is None:
                rejections.append(
                    ReplayRejection(event_id, episode_id, prediction_id or None, "resolution has no pending prediction")
                )
                continue
            label = _label_from_mapping(payload)
            if label is None:
                label = _label_from_error(prediction.probability, payload.get("error"))
            if label is None:
                rejections.append(
                    ReplayRejection(event_id, episode_id, prediction_id, "resolution has no unambiguous binary outcome")
                )
                continue
            samples.append(_sample(prediction, label, event_id, "explicit_resolution"))
            del pending[key]
            continue

        if event_type != "action_outcome":
            continue

        explicit = _explicit_prediction_outcomes(payload)
        for key, prediction in tuple(pending.items()):
            if prediction.episode_id != episode_id:
                continue
            if prediction.prediction_id in explicit:
                samples.append(
                    _sample(prediction, explicit[prediction.prediction_id], event_id, "action_outcome_explicit")
                )
                del pending[key]
                continue
            if prediction.remaining_horizon > 1:
                prediction.remaining_horizon -= 1
                continue
            label, kind = _label_from_action_outcome(prediction, payload)
            if label is None:
                continue
            samples.append(_sample(prediction, label, event_id, kind))
            del pending[key]

    for prediction in pending.values():
        rejections.append(
            ReplayRejection(
                prediction.event_id,
                prediction.episode_id,
                prediction.prediction_id,
                "prediction was not resolved by a compatible outcome",
            )
        )
    return ReplayDataset(
        samples=tuple(sorted(samples, key=lambda sample: sample.sample_id)),
        rejections=tuple(rejections),
    )


def deterministic_split(
    samples: Sequence[ReplaySample],
    *,
    validation_fraction: float = 0.25,
    seed: int = 17,
) -> ReplaySplit:
    """Split by episode using stable hashes, preventing within-episode leakage."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")
    groups: dict[str, list[ReplaySample]] = defaultdict(list)
    for sample in samples:
        groups[sample.episode_id].append(sample)
    if len(groups) < 2:
        return ReplaySplit(tuple(sorted(samples, key=lambda item: item.sample_id)), ())
    ranked_episodes = sorted(
        groups,
        key=lambda episode_id: hashlib.sha256(f"{seed}\0{episode_id}".encode()).digest(),
    )
    validation_group_count = max(1, min(len(groups) - 1, math.ceil(len(groups) * validation_fraction)))
    validation_episodes = set(ranked_episodes[:validation_group_count])
    train = sorted(
        (sample for sample in samples if sample.episode_id not in validation_episodes),
        key=lambda item: item.sample_id,
    )
    validation = sorted(
        (sample for sample in samples if sample.episode_id in validation_episodes),
        key=lambda item: item.sample_id,
    )
    return ReplaySplit(tuple(train), tuple(validation))


def brier_loss(state: AdapterState, samples: Sequence[ReplaySample], *, epsilon: float = 1e-6) -> float:
    """Mean held-out Brier loss; lower is better."""
    if not samples:
        raise ValueError("Brier loss requires at least one sample")
    errors = [
        (state.calibrate(sample.probability, epsilon=epsilon) - float(sample.outcome)) ** 2
        for sample in samples
    ]
    return math.fsum(errors) / len(errors)


def train_shadow_adapter(
    samples: Sequence[ReplaySample],
    *,
    incumbent: AdapterState | None = None,
    config: ShadowLearningConfig | None = None,
) -> ShadowTrainingResult:
    """Fit in shadow and select the candidate only when all gates pass.

    This function has no side effects.  ``selected`` is the candidate on
    promotion and the exact incumbent object otherwise; persistence and runtime
    activation are intentionally separate operator-controlled integration steps.
    """
    incumbent = incumbent or AdapterState()
    config = config or ShadowLearningConfig()
    if incumbent.parameter_norm > config.max_parameter_norm + 1e-12:
        raise ValueError("incumbent parameter norm exceeds configured maximum")
    split = deterministic_split(
        samples,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )
    fitted = _fit_parameters(split.train, incumbent, config)
    incumbent_loss = (
        brier_loss(incumbent, split.validation, epsilon=config.probability_epsilon)
        if split.validation
        else None
    )
    provisional = AdapterState(
        base_model_version=incumbent.base_model_version,
        revision=incumbent.revision + 1,
        log_scale=float(fitted[0]),
        bias=float(fitted[1]),
        trained_examples=len(split.train),
        validation_examples=len(split.validation),
        validation_loss=None,
        parent_digest=incumbent.digest(),
    )
    candidate_loss = (
        brier_loss(provisional, split.validation, epsilon=config.probability_epsilon)
        if split.validation
        else None
    )
    candidate = AdapterState(
        base_model_version=provisional.base_model_version,
        revision=provisional.revision,
        log_scale=provisional.log_scale,
        bias=provisional.bias,
        trained_examples=provisional.trained_examples,
        validation_examples=provisional.validation_examples,
        validation_loss=candidate_loss,
        parent_digest=provisional.parent_digest,
    )
    improvement = (
        incumbent_loss - candidate_loss
        if incumbent_loss is not None and candidate_loss is not None
        else None
    )

    promoted = False
    if len(split.train) < config.min_training_examples:
        reason = "insufficient training examples"
    elif len(split.validation) < config.min_validation_examples:
        reason = "insufficient validation examples"
    elif improvement is None or improvement < config.min_improvement:
        reason = "held-out improvement below threshold"
    else:
        promoted = True
        reason = "promotion gates passed"
    return ShadowTrainingResult(
        incumbent=incumbent,
        candidate=candidate,
        selected=candidate if promoted else incumbent,
        promoted=promoted,
        reason=reason,
        train_examples=len(split.train),
        validation_examples=len(split.validation),
        incumbent_validation_loss=incumbent_loss,
        candidate_validation_loss=candidate_loss,
        improvement=improvement,
    )


def _fit_parameters(
    samples: Sequence[ReplaySample],
    incumbent: AdapterState,
    config: ShadowLearningConfig,
) -> np.ndarray:
    parameters = np.asarray(incumbent.parameters, dtype=np.float64)
    if not samples or config.max_parameter_delta == 0.0:
        return parameters
    probabilities = np.asarray([sample.probability for sample in samples], dtype=np.float64)
    probabilities = np.clip(probabilities, config.probability_epsilon, 1.0 - config.probability_epsilon)
    log_odds = np.log(probabilities / (1.0 - probabilities))
    outcomes = np.asarray([float(sample.outcome) for sample in samples], dtype=np.float64)
    origin = parameters.copy()

    for _ in range(config.epochs):
        scale = math.exp(float(parameters[0]))
        logits = scale * log_odds + parameters[1]
        calibrated = _sigmoid_array(logits)
        derivative = 2.0 * (calibrated - outcomes) * calibrated * (1.0 - calibrated)
        gradient = np.asarray(
            [float(np.mean(derivative * scale * log_odds)), float(np.mean(derivative))],
            dtype=np.float64,
        )
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > config.gradient_clip:
            gradient *= config.gradient_clip / gradient_norm
        proposed = parameters - config.learning_rate * gradient
        parameters = _project_parameters(
            proposed,
            origin=origin,
            max_delta=config.max_parameter_delta,
            max_norm=config.max_parameter_norm,
        )
    return parameters


def _project_parameters(
    proposed: np.ndarray,
    *,
    origin: np.ndarray,
    max_delta: float,
    max_norm: float,
) -> np.ndarray:
    delta = proposed - origin
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > max_delta:
        delta *= max_delta / delta_norm
    bounded = origin + delta
    if float(np.linalg.norm(bounded)) <= max_norm + 1e-15:
        return bounded
    # The incumbent is inside the norm ball.  Bisection along the already
    # delta-bounded segment therefore finds a point in the intersection while
    # preserving both guarantees.
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if float(np.linalg.norm(origin + middle * delta)) <= max_norm:
            low = middle
        else:
            high = middle
    return origin + low * delta


def _parse_prediction(
    payload: Mapping[str, Any], episode_id: str, event_id: str
) -> _PendingPrediction | ReplayRejection:
    prediction_id = str(payload.get("prediction_id") or "")
    if not episode_id or not prediction_id:
        return ReplayRejection(event_id, episode_id, prediction_id or None, "prediction identifiers are missing")
    try:
        probability = float(payload["probability"])
    except (KeyError, TypeError, ValueError):
        return ReplayRejection(event_id, episode_id, prediction_id, "prediction probability is invalid")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return ReplayRejection(event_id, episode_id, prediction_id, "prediction probability is outside [0, 1]")
    target = str(payload.get("target") or "")
    if not target:
        return ReplayRejection(event_id, episode_id, prediction_id, "prediction target is missing")
    try:
        horizon = max(1, int(payload.get("horizon", 1)))
    except (TypeError, ValueError):
        return ReplayRejection(event_id, episode_id, prediction_id, "prediction horizon is invalid")
    return _PendingPrediction(episode_id, prediction_id, target.casefold(), probability, event_id, horizon)


def _sample(
    prediction: _PendingPrediction,
    outcome: bool,
    resolution_event_id: str,
    resolution_kind: str,
) -> ReplaySample:
    identity = "\0".join(
        (prediction.episode_id, prediction.prediction_id, prediction.event_id, resolution_event_id)
    )
    sample_id = f"replay_{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
    return ReplaySample(
        sample_id=sample_id,
        episode_id=prediction.episode_id,
        prediction_id=prediction.prediction_id,
        target=prediction.target,
        probability=prediction.probability,
        outcome=outcome,
        prediction_event_id=prediction.event_id,
        resolution_event_id=resolution_event_id,
        resolution_kind=resolution_kind,
    )


def _event_sort_key(event: Mapping[str, Any], input_index: int) -> tuple[str, float, int]:
    episode_id = str(event.get("episode_id") or "")
    sequence = event.get("sequence")
    if isinstance(sequence, int) and not isinstance(sequence, bool):
        return (episode_id, float(sequence), input_index)
    observed_at = event.get("observed_at")
    if isinstance(observed_at, (int, float)) and not isinstance(observed_at, bool) and math.isfinite(observed_at):
        return (episode_id, float(observed_at), input_index)
    return (episode_id, float(input_index), input_index)


def _label_from_mapping(payload: Mapping[str, Any]) -> bool | None:
    for key in _LABEL_KEYS:
        if key in payload:
            label = _binary_label(payload[key])
            if label is not None:
                return label
    return None


def _binary_label(value: Any) -> bool | None:
    if type(value) is bool:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)) and float(value) in {0.0, 1.0}:
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "success", "succeeded", "1"}:
            return True
        if normalized in {"false", "no", "failure", "failed", "0"}:
            return False
    return None


def _label_from_error(probability: float, error: Any) -> bool | None:
    if not isinstance(error, (int, float)) or isinstance(error, bool) or not math.isfinite(float(error)):
        return None
    value = float(error)
    matches: set[bool] = set()
    for outcome in (False, True):
        difference = abs(probability - float(outcome))
        if math.isclose(value, difference, rel_tol=1e-7, abs_tol=1e-9) or math.isclose(
            value, difference**2, rel_tol=1e-7, abs_tol=1e-9
        ):
            matches.add(outcome)
    return next(iter(matches)) if len(matches) == 1 else None


def _explicit_prediction_outcomes(payload: Mapping[str, Any]) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for field in ("prediction_outcomes", "resolved_predictions"):
        values = payload.get(field)
        if not isinstance(values, Mapping):
            continue
        for prediction_id, value in values.items():
            label = _label_from_mapping(value) if isinstance(value, Mapping) else _binary_label(value)
            if label is not None:
                outcomes[str(prediction_id)] = label
    return outcomes


def _label_from_action_outcome(
    prediction: _PendingPrediction,
    payload: Mapping[str, Any],
) -> tuple[bool | None, str]:
    errors = payload.get("prediction_errors")
    if isinstance(errors, Mapping) and prediction.prediction_id in errors:
        label = _label_from_error(prediction.probability, errors[prediction.prediction_id])
        if label is not None:
            return label, "prediction_error"
    if prediction.target in _OBSERVATION_TARGETS:
        observation = payload.get("observation")
        return isinstance(observation, str) and bool(observation.strip()), "observation_presence"
    if prediction.target in _SUCCESS_TARGETS:
        return _binary_label(payload.get("succeeded")), "action_success"
    if prediction.target in _UNCERTAINTY_TARGETS and isinstance(errors, Mapping) and errors:
        numeric_errors = [
            float(value)
            for value in errors.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        ]
        if len(numeric_errors) == len(errors):
            return all(value <= 0.0 for value in numeric_errors), "no_prediction_error"
    return None, ""


def _strict_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values)
    nonnegative = values >= 0.0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    output[~nonnegative] = exp_values / (1.0 + exp_values)
    return output
