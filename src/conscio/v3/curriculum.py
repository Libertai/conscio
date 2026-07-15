"""Deterministic, provenance-preserving curricula for the V3 recurrent core.

The curriculum boundary is deliberately model-neutral: it emits structured
supervision but never calls an LLM, changes live state, or promotes text into
semantic memory.  Targets derived from the event log remain scoped recorded
observations/outcomes/measurements, with their exact evidence event IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

CURRICULUM_SCHEMA_VERSION = 1
CURRICULUM_FORMAT = "conscio.v3.curriculum.jsonl.v1"

__all__ = [
    "CURRICULUM_FORMAT",
    "CURRICULUM_SCHEMA_VERSION",
    "TARGET_FAMILIES",
    "CurriculumBundle",
    "CurriculumCorruptionError",
    "CurriculumDerivation",
    "CurriculumExample",
    "CurriculumManifest",
    "CurriculumRejection",
    "CurriculumSplit",
    "ExampleProvenance",
    "TargetFamily",
    "build_curriculum_manifest",
    "derive_curriculum_examples",
    "deterministic_curriculum_split",
    "generate_synthetic_curriculum",
    "read_curriculum_jsonl",
    "write_curriculum_jsonl",
]

TargetFamily = Literal[
    "next_observation",
    "tool_outcome",
    "action_effect",
    "homeostatic_affect_change",
    "future_uncertainty",
]
TARGET_FAMILIES: tuple[TargetFamily, ...] = (
    "next_observation",
    "tool_outcome",
    "action_effect",
    "homeostatic_affect_change",
    "future_uncertainty",
)

ProvenanceOrigin = Literal["synthetic", "event_log"]
EpistemicStatus = Literal[
    "synthetic_ground_truth",
    "recorded_observation",
    "recorded_outcome",
    "recorded_measurement",
]
EvidenceKind = Literal[
    "synthetic_rule",
    "environment_observation",
    "tool_execution_outcome",
    "action_execution_outcome",
    "affect_transition",
    "uncertainty_resolution",
]

_TARGET_FAMILY_SET = frozenset(TARGET_FAMILIES)
_EPISTEMIC_STATUS_SET = frozenset(
    {
        "synthetic_ground_truth",
        "recorded_observation",
        "recorded_outcome",
        "recorded_measurement",
    }
)
_EVIDENCE_KIND_SET = frozenset(
    {
        "synthetic_rule",
        "environment_observation",
        "tool_execution_outcome",
        "action_execution_outcome",
        "affect_transition",
        "uncertainty_resolution",
    }
)
_EXTERNAL_EVENT_TYPES = frozenset({"message", "heartbeat", "interrupt", "dry_run", "observation"})
_INTERNAL_SOURCE_PARTS = frozenset(
    {
        "affect",
        "agent",
        "assistant",
        "conscio",
        "llm",
        "model",
        "perception",
        "planning",
        "recurrent",
        "self_model",
        "v3",
        "workspace",
        "world_model",
    }
)
_TOOL_OUTCOME_TYPES = frozenset({"tool_outcome", "tool_result"})
_TRUSTED_TOOL_SOURCES = frozenset({"environment", "policy_executor", "runtime_executor", "tool", "tool_executor"})
_TRUSTED_ACTION_SOURCES = frozenset({"environment", "runtime_executor"})
_TRUSTED_AFFECT_SOURCES = frozenset({"action_evaluation", "affect", "homeostasis"})
_TRUSTED_RESOLUTION_SOURCES = frozenset({"action_evaluation", "environment"})
_TYPED_TOOL_STATUSES = frozenset({"cancelled", "denied", "error", "failed", "failure", "ok", "success", "timeout"})


class CurriculumCorruptionError(ValueError):
    """Raised when a curriculum artifact does not match its manifest."""


@dataclass(frozen=True)
class ExampleProvenance:
    """Epistemic scope and exact evidence for one target.

    ``model_output_as_fact`` is intentionally fixed to false.  It is serialized
    so that consumers can enforce the rule at every dataset boundary rather
    than relying on an undocumented convention.
    """

    origin: ProvenanceOrigin
    epistemic_status: EpistemicStatus
    evidence_kind: EvidenceKind
    source_event_ids: tuple[str, ...] = ()
    model_output_as_fact: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))
        if not isinstance(self.origin, str) or self.origin not in {"synthetic", "event_log"}:
            raise ValueError(f"unsupported provenance origin: {self.origin!r}")
        if not isinstance(self.epistemic_status, str) or self.epistemic_status not in _EPISTEMIC_STATUS_SET:
            raise ValueError(f"unsupported epistemic status: {self.epistemic_status!r}")
        if not isinstance(self.evidence_kind, str) or self.evidence_kind not in _EVIDENCE_KIND_SET:
            raise ValueError(f"unsupported evidence kind: {self.evidence_kind!r}")
        if self.model_output_as_fact:
            raise ValueError("model output cannot be promoted as curriculum fact")
        if any(not isinstance(event_id, str) or not event_id.strip() for event_id in self.source_event_ids):
            raise ValueError("source_event_ids must contain non-empty strings")
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("source_event_ids must be unique")
        if self.origin == "synthetic":
            if self.epistemic_status != "synthetic_ground_truth" or self.evidence_kind != "synthetic_rule":
                raise ValueError("synthetic provenance must remain synthetic ground truth")
            if self.source_event_ids:
                raise ValueError("synthetic examples cannot claim recorded source events")
        else:
            if self.epistemic_status == "synthetic_ground_truth" or self.evidence_kind == "synthetic_rule":
                raise ValueError("event-log provenance cannot claim synthetic ground truth")
            if not self.source_event_ids:
                raise ValueError("event-log provenance requires source event IDs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "epistemic_status": self.epistemic_status,
            "evidence_kind": self.evidence_kind,
            "source_event_ids": list(self.source_event_ids),
            "model_output_as_fact": self.model_output_as_fact,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExampleProvenance:
        source_ids = data.get("source_event_ids", [])
        if not isinstance(source_ids, list) or not all(isinstance(item, str) for item in source_ids):
            raise ValueError("source_event_ids must be an array of strings")
        model_output_as_fact = data.get("model_output_as_fact", False)
        if type(model_output_as_fact) is not bool:
            raise ValueError("model_output_as_fact must be a boolean")
        try:
            return cls(
                origin=cast(ProvenanceOrigin, data["origin"]),
                epistemic_status=cast(EpistemicStatus, data["epistemic_status"]),
                evidence_kind=cast(EvidenceKind, data["evidence_kind"]),
                source_event_ids=tuple(source_ids),
                model_output_as_fact=model_output_as_fact,
            )
        except KeyError as exc:
            raise ValueError(f"missing provenance field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class CurriculumExample:
    """One serializable supervised transition, independent of a framework."""

    example_id: str
    episode_id: str
    step: int
    target_family: TargetFamily
    inputs: dict[str, Any]
    target: dict[str, Any]
    provenance: ExampleProvenance
    schema_version: int = CURRICULUM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.example_id, str)
            or not self.example_id.strip()
            or not isinstance(self.episode_id, str)
            or not self.episode_id.strip()
        ):
            raise ValueError("example_id and episode_id must be non-empty")
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(self.target_family, str) or self.target_family not in _TARGET_FAMILY_SET:
            raise ValueError(f"unsupported target family: {self.target_family!r}")
        if self.schema_version != CURRICULUM_SCHEMA_VERSION:
            raise ValueError(f"unsupported curriculum schema version: {self.schema_version}")
        object.__setattr__(self, "inputs", _json_object(self.inputs, "inputs"))
        object.__setattr__(self, "target", _json_object(self.target, "target"))
        if not self.target:
            raise ValueError("target cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "target_family": self.target_family,
            "inputs": _normalize_json(self.inputs, "inputs"),
            "target": _normalize_json(self.target, "target"),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CurriculumExample:
        try:
            schema_version = _strict_int(data["schema_version"], "schema_version")
            step = _strict_int(data["step"], "step")
            inputs = _json_object(data["inputs"], "inputs")
            target = _json_object(data["target"], "target")
            raw_provenance = data["provenance"]
            if not isinstance(raw_provenance, Mapping):
                raise ValueError("provenance must be an object")
            return cls(
                example_id=str(data["example_id"]),
                episode_id=str(data["episode_id"]),
                step=step,
                target_family=cast(TargetFamily, data["target_family"]),
                inputs=inputs,
                target=target,
                provenance=ExampleProvenance.from_dict(raw_provenance),
                schema_version=schema_version,
            )
        except KeyError as exc:
            raise ValueError(f"missing curriculum example field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class CurriculumRejection:
    event_id: str
    episode_id: str
    reason: str


@dataclass(frozen=True)
class CurriculumDerivation:
    examples: tuple[CurriculumExample, ...]
    rejections: tuple[CurriculumRejection, ...]


@dataclass(frozen=True)
class CurriculumSplit:
    train: tuple[CurriculumExample, ...]
    validation: tuple[CurriculumExample, ...]


@dataclass(frozen=True)
class CurriculumManifest:
    """Content address and structural inventory for a JSONL curriculum."""

    dataset_digest: str
    example_count: int
    episode_count: int
    target_counts: dict[str, int]
    manifest_id: str
    schema_version: int = CURRICULUM_SCHEMA_VERSION
    format: str = CURRICULUM_FORMAT

    def __post_init__(self) -> None:
        if self.schema_version != CURRICULUM_SCHEMA_VERSION:
            raise ValueError(f"unsupported curriculum schema version: {self.schema_version}")
        if self.format != CURRICULUM_FORMAT:
            raise ValueError(f"unsupported curriculum format: {self.format!r}")
        if not _is_sha256(self.dataset_digest):
            raise ValueError("dataset_digest must be a sha256 content address")
        if self.manifest_id != f"curriculum:{self.dataset_digest}":
            raise ValueError("manifest_id does not match dataset_digest")
        if self.example_count < 0 or self.episode_count < 0:
            raise ValueError("manifest counts must be non-negative")
        counts = {str(key): _strict_int(value, f"target_counts[{key}]") for key, value in self.target_counts.items()}
        if any(value < 0 for value in counts.values()):
            raise ValueError("target counts must be non-negative")
        if any(key not in _TARGET_FAMILY_SET for key in counts):
            raise ValueError("manifest contains an unsupported target family")
        if sum(counts.values()) != self.example_count:
            raise ValueError("target counts do not sum to example_count")
        object.__setattr__(self, "target_counts", dict(sorted(counts.items())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "manifest_id": self.manifest_id,
            "dataset_digest": self.dataset_digest,
            "example_count": self.example_count,
            "episode_count": self.episode_count,
            "target_counts": dict(self.target_counts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CurriculumManifest:
        try:
            counts = data["target_counts"]
            if not isinstance(counts, Mapping):
                raise ValueError("target_counts must be an object")
            return cls(
                dataset_digest=str(data["dataset_digest"]),
                example_count=_strict_int(data["example_count"], "example_count"),
                episode_count=_strict_int(data["episode_count"], "episode_count"),
                target_counts={str(key): _strict_int(value, f"target_counts[{key}]") for key, value in counts.items()},
                manifest_id=str(data["manifest_id"]),
                schema_version=_strict_int(data["schema_version"], "schema_version"),
                format=str(data["format"]),
            )
        except KeyError as exc:
            raise ValueError(f"missing curriculum manifest field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class CurriculumBundle:
    manifest: CurriculumManifest
    examples: tuple[CurriculumExample, ...]


def generate_synthetic_curriculum(*, seed: int, episodes: int = 32) -> tuple[CurriculumExample, ...]:
    """Generate deterministic rule-scored text/tool episodes from ``seed``.

    Each episode emits exactly one example for every target family.  Synthetic
    labels are explicitly marked as such and never masquerade as observations
    from the deployed agent.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    rng = random.Random(seed)
    tasks = ("locate_note", "compare_values", "inspect_status", "summarize_record")
    tools = ("memory_lookup", "record_inspector", "status_probe", "text_search")
    examples: list[CurriculumExample] = []
    for episode_index in range(episodes):
        episode_token = hashlib.sha256(f"{seed}\0{episode_index}".encode()).hexdigest()[:16]
        episode_id = f"synthetic_{episode_token}"
        task = rng.choice(tasks)
        tool = rng.choice(tools)
        query_token = f"item-{rng.randrange(1000):03d}"
        tool_succeeded = rng.random() >= 0.22
        result_count = rng.randrange(1, 5) if tool_succeeded else 0
        latency_bucket = rng.choice(("low", "medium", "high"))
        action = "answer_from_observation" if tool_succeeded else "request_clarification"
        progress_delta = 1.0 if tool_succeeded else 0.15
        uncertainty_before = _rounded(rng.uniform(0.45, 0.9))
        uncertainty_after = _rounded(
            max(0.0, uncertainty_before - rng.uniform(0.18, 0.4))
            if tool_succeeded
            else min(1.0, uncertainty_before + rng.uniform(0.02, 0.12))
        )
        valence_before = _rounded(rng.uniform(-0.2, 0.2))
        arousal_before = _rounded(rng.uniform(0.2, 0.6))
        controllability_before = _rounded(rng.uniform(0.35, 0.7))
        competence_before = _rounded(rng.uniform(0.2, 0.65))
        coherence_before = _rounded(rng.uniform(0.25, 0.7))
        affect_before: dict[str, Any] = {
            "valence": valence_before,
            "arousal": arousal_before,
            "controllability": controllability_before,
            "need_errors": {
                "competence": competence_before,
                "epistemic_coherence": coherence_before,
            },
        }
        affect_after: dict[str, Any] = {
            "valence": _rounded(valence_before + (0.18 if tool_succeeded else -0.12)),
            "arousal": _rounded(max(0.0, arousal_before - (0.1 if tool_succeeded else 0.02))),
            "controllability": _rounded(min(1.0, controllability_before + (0.16 if tool_succeeded else -0.08))),
            "need_errors": {
                "competence": _rounded(max(0.0, competence_before - (0.2 if tool_succeeded else -0.08))),
                "epistemic_coherence": _rounded(
                    max(
                        0.0,
                        coherence_before - (0.22 if tool_succeeded else -0.05),
                    )
                ),
            },
        }
        synthetic = ExampleProvenance(
            origin="synthetic",
            epistemic_status="synthetic_ground_truth",
            evidence_kind="synthetic_rule",
        )
        specs: tuple[tuple[TargetFamily, dict[str, Any], dict[str, Any]], ...] = (
            (
                "next_observation",
                {
                    "event_type": "message",
                    "content": f"{task}:{query_token}",
                    "proposed_tool": tool,
                },
                {
                    "event_type": "tool_result",
                    "outcome_token": f"{tool}:{'success' if tool_succeeded else 'failure'}",
                    "result_count": result_count,
                },
            ),
            (
                "tool_outcome",
                {"tool": tool, "arguments": {"query": query_token}, "task": task},
                {
                    "succeeded": tool_succeeded,
                    "result_count": result_count,
                    "latency_bucket": latency_bucket,
                },
            ),
            (
                "action_effect",
                {
                    "action": action,
                    "tool_succeeded": tool_succeeded,
                    "task_progress": 0.0,
                },
                {
                    "task_progress_delta": progress_delta,
                    "observation_emitted": True,
                    "requires_follow_up": not tool_succeeded,
                },
            ),
            (
                "homeostatic_affect_change",
                {"before": affect_before, "tool_succeeded": tool_succeeded},
                {"after": affect_after, "delta": _affect_delta(affect_before, affect_after)},
            ),
            (
                "future_uncertainty",
                {
                    "uncertainty": uncertainty_before,
                    "new_evidence_observed": tool_succeeded,
                    "action": action,
                },
                {
                    "uncertainty": uncertainty_after,
                    "nonincrease": uncertainty_after <= uncertainty_before,
                    "delta": _rounded(uncertainty_after - uncertainty_before),
                },
            ),
        )
        for step, (family, inputs, target) in enumerate(specs):
            examples.append(_make_example(episode_id, step, family, inputs, target, synthetic))
    return tuple(sorted(examples, key=_example_sort_key))


def derive_curriculum_examples(events: Iterable[Mapping[str, Any]]) -> CurriculumDerivation:
    """Conservatively convert append-only ``CognitiveEvent`` history.

    Only allowlisted environment/executor/measurement events can label data.
    Broadcast text, checkpoint model inputs, assistant output, hypotheses, and
    self-reports are ignored.  In particular, ``action_outcome.observation`` is
    never copied because the current runtime may populate it with LLM text.
    """
    indexed = list(enumerate(events))
    ordered = sorted(indexed, key=lambda item: _event_sort_key(item[1], item[0]))
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    rejections: list[CurriculumRejection] = []
    for input_index, event in ordered:
        episode_id = str(event.get("episode_id") or "")
        event_id = str(event.get("event_id") or "")
        if not episode_id:
            rejections.append(CurriculumRejection(event_id or f"input_{input_index}", "", "event has no episode_id"))
            continue
        if not event_id:
            rejections.append(CurriculumRejection(f"input_{input_index}", episode_id, "event has no event_id"))
            continue
        groups[episode_id].append((input_index, event))

    examples: list[CurriculumExample] = []
    for episode_id, episode_events in groups.items():
        external: list[tuple[int, Mapping[str, Any]]] = []
        predictions: dict[str, Mapping[str, Any]] = {}
        previous_affect: tuple[Mapping[str, Any], dict[str, Any]] | None = None
        latest_external: Mapping[str, Any] | None = None

        for position, (_, event) in enumerate(episode_events):
            event_type = str(event.get("event_type") or "")
            event_id = str(event.get("event_id") or f"{episode_id}_input_{position}")
            source = str(event.get("source") or "")
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                recognized = (
                    _EXTERNAL_EVENT_TYPES
                    | _TOOL_OUTCOME_TYPES
                    | {
                        "action_outcome",
                        "affect",
                        "prediction_resolution",
                    }
                )
                if event_type in recognized:
                    rejections.append(
                        CurriculumRejection(event_id, episode_id, "recognized event payload is not an object")
                    )
                continue

            if _is_external_observation(event_type, source, payload):
                external.append((position, event))
                latest_external = event

            if event_type == "prediction":
                prediction_id = str(payload.get("prediction_id") or "")
                if prediction_id:
                    predictions[prediction_id] = event
                continue

            if event_type in _TOOL_OUTCOME_TYPES:
                if source not in _TRUSTED_TOOL_SOURCES:
                    rejections.append(CurriculumRejection(event_id, episode_id, "tool outcome source is not trusted"))
                    continue
                parsed_tool = _parse_tool_outcome(payload)
                if parsed_tool is None:
                    rejections.append(
                        CurriculumRejection(
                            event_id,
                            episode_id,
                            "tool outcome lacks a typed tool and boolean result",
                        )
                    )
                    continue
                tool, arguments, tool_target = parsed_tool
                inputs: dict[str, Any] = {"tool": tool, "arguments": arguments}
                _add_external_context(inputs, latest_external)
                provenance_ids = _source_ids(latest_external, event)
                provenance = ExampleProvenance(
                    origin="event_log",
                    epistemic_status="recorded_outcome",
                    evidence_kind="tool_execution_outcome",
                    source_event_ids=provenance_ids,
                )
                examples.append(
                    _make_example(
                        episode_id,
                        position,
                        "tool_outcome",
                        inputs,
                        tool_target,
                        provenance,
                    )
                )
                continue

            if event_type == "action_outcome":
                if source not in _TRUSTED_ACTION_SOURCES:
                    rejections.append(CurriculumRejection(event_id, episode_id, "action outcome source is not trusted"))
                    continue
                learning_eligible, eligibility_error = _learning_eligibility(
                    payload,
                    observed_key="observed",
                    subject="action outcome",
                )
                if eligibility_error is not None:
                    rejections.append(CurriculumRejection(event_id, episode_id, eligibility_error))
                    continue
                if not learning_eligible:
                    continue
                succeeded = payload.get("succeeded")
                action = payload.get("action")
                if type(succeeded) is not bool or not isinstance(action, str) or not action.strip():
                    rejections.append(
                        CurriculumRejection(
                            event_id,
                            episode_id,
                            "action outcome lacks a typed action and boolean result",
                        )
                    )
                    continue
                inputs = {"action": action, "proposal_id": str(payload.get("proposal_id") or "")}
                _add_external_context(inputs, latest_external)
                provenance = ExampleProvenance(
                    origin="event_log",
                    epistemic_status="recorded_outcome",
                    evidence_kind="action_execution_outcome",
                    source_event_ids=_source_ids(latest_external, event),
                )
                examples.append(
                    _make_example(
                        episode_id,
                        position,
                        "action_effect",
                        inputs,
                        {"succeeded": succeeded},
                        provenance,
                    )
                )
                continue

            if event_type == "affect":
                if source not in _TRUSTED_AFFECT_SOURCES:
                    rejections.append(CurriculumRejection(event_id, episode_id, "affect source is not trusted"))
                    continue
                learning_eligible, eligibility_error = _learning_eligibility(
                    payload,
                    observed_key="outcome_observed",
                    subject="affect transition",
                )
                if eligibility_error is not None:
                    rejections.append(CurriculumRejection(event_id, episode_id, eligibility_error))
                    continue
                if not learning_eligible:
                    continue
                state = _parse_affect(payload)
                if state is None:
                    rejections.append(
                        CurriculumRejection(event_id, episode_id, "affect payload is not a finite typed state")
                    )
                    continue
                if previous_affect is not None:
                    previous_event, before = previous_affect
                    provenance = ExampleProvenance(
                        origin="event_log",
                        epistemic_status="recorded_measurement",
                        evidence_kind="affect_transition",
                        source_event_ids=_source_ids(previous_event, event),
                    )
                    examples.append(
                        _make_example(
                            episode_id,
                            position,
                            "homeostatic_affect_change",
                            {"before": before},
                            {"after": state, "delta": _affect_delta(before, state)},
                            provenance,
                        )
                    )
                previous_affect = (event, state)
                continue

            if event_type == "prediction_resolution" and payload.get("target") == "future_uncertainty":
                if source not in _TRUSTED_RESOLUTION_SOURCES:
                    rejections.append(
                        CurriculumRejection(event_id, episode_id, "uncertainty resolution source is not trusted")
                    )
                    continue
                prediction_id = str(payload.get("prediction_id") or "")
                prediction_event = predictions.get(prediction_id)
                observed = payload.get("observed")
                if prediction_event is None or type(observed) is not bool:
                    rejections.append(
                        CurriculumRejection(
                            event_id,
                            episode_id,
                            "uncertainty resolution lacks a prediction or boolean observation",
                        )
                    )
                    continue
                prediction_payload = prediction_event.get("payload")
                if not isinstance(prediction_payload, Mapping):
                    rejections.append(
                        CurriculumRejection(
                            event_id,
                            episode_id,
                            "uncertainty prediction payload is not an object",
                        )
                    )
                    continue
                probability = _finite_probability(prediction_payload.get("probability"))
                if probability is None or prediction_payload.get("target") != "future_uncertainty":
                    rejections.append(
                        CurriculumRejection(event_id, episode_id, "uncertainty prediction is incompatible")
                    )
                    continue
                uncertainty_target: dict[str, Any] = {"nonincrease": observed}
                error = payload.get("error")
                if _is_finite_number(error):
                    uncertainty_target["brier_error"] = float(cast(int | float, error))
                provenance = ExampleProvenance(
                    origin="event_log",
                    epistemic_status="recorded_measurement",
                    evidence_kind="uncertainty_resolution",
                    source_event_ids=_source_ids(prediction_event, event),
                )
                examples.append(
                    _make_example(
                        episode_id,
                        position,
                        "future_uncertainty",
                        {
                            "probability": probability,
                            "observable": str(prediction_payload.get("observable") or ""),
                            "horizon": _safe_horizon(prediction_payload.get("horizon")),
                        },
                        uncertainty_target,
                        provenance,
                    )
                )

        for pair_index, ((_, current), (_, following)) in enumerate(zip(external, external[1:], strict=False)):
            current_payload = cast(Mapping[str, Any], current["payload"])
            following_payload = cast(Mapping[str, Any], following["payload"])
            provenance = ExampleProvenance(
                origin="event_log",
                epistemic_status="recorded_observation",
                evidence_kind="environment_observation",
                source_event_ids=_source_ids(current, following),
            )
            examples.append(
                _make_example(
                    episode_id,
                    pair_index,
                    "next_observation",
                    {
                        "event_type": str(current.get("event_type") or ""),
                        "source": str(current.get("source") or ""),
                        "content": current_payload["content"],
                    },
                    {
                        "event_type": str(following.get("event_type") or ""),
                        "source": str(following.get("source") or ""),
                        "content": following_payload["content"],
                    },
                    provenance,
                )
            )

    return CurriculumDerivation(
        examples=tuple(sorted(examples, key=_example_sort_key)),
        rejections=tuple(rejections),
    )


def deterministic_curriculum_split(
    examples: Sequence[CurriculumExample],
    *,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> CurriculumSplit:
    """Split whole episodes by stable hash so transitions cannot leak."""
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    example_ids = [example.example_id for example in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("example_id values must be unique")
    groups: dict[str, list[CurriculumExample]] = defaultdict(list)
    for example in examples:
        groups[example.episode_id].append(example)
    if len(groups) < 2:
        return CurriculumSplit(tuple(sorted(examples, key=_example_sort_key)), ())
    ranked = sorted(
        groups,
        key=lambda episode_id: hashlib.sha256(f"{seed}\0{episode_id}".encode()).digest(),
    )
    validation_count = max(1, min(len(groups) - 1, math.ceil(len(groups) * validation_fraction)))
    validation_episodes = set(ranked[:validation_count])
    train = tuple(
        sorted(
            (item for item in examples if item.episode_id not in validation_episodes),
            key=_example_sort_key,
        )
    )
    validation = tuple(
        sorted(
            (item for item in examples if item.episode_id in validation_episodes),
            key=_example_sort_key,
        )
    )
    return CurriculumSplit(train, validation)


def build_curriculum_manifest(examples: Sequence[CurriculumExample]) -> CurriculumManifest:
    """Build a canonical content address and inventory without writing a file."""
    canonical = _canonical_examples(examples)
    digest = _examples_digest(canonical)
    counts = Counter(example.target_family for example in canonical)
    return CurriculumManifest(
        dataset_digest=digest,
        example_count=len(canonical),
        episode_count=len({example.episode_id for example in canonical}),
        target_counts={str(family): count for family, count in counts.items()},
        manifest_id=f"curriculum:{digest}",
    )


def write_curriculum_jsonl(path: str | os.PathLike[str], examples: Sequence[CurriculumExample]) -> CurriculumManifest:
    """Atomically write canonical examples preceded by their manifest."""
    canonical = _canonical_examples(examples)
    manifest = build_curriculum_manifest(canonical)
    records = [
        _canonical_json({"record_type": "manifest", "data": manifest.to_dict()}),
        *(_canonical_json({"record_type": "example", "data": example.to_dict()}) for example in canonical),
    ]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(records) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return manifest


def read_curriculum_jsonl(path: str | os.PathLike[str]) -> CurriculumBundle:
    """Read and hash-verify a curriculum, rejecting truncation or mutation."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CurriculumCorruptionError(f"cannot read curriculum: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise CurriculumCorruptionError("curriculum is empty or contains blank records")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CurriculumCorruptionError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(record, Mapping):
            raise CurriculumCorruptionError(f"line {line_number} is not an object")
        records.append(record)
    if records[0].get("record_type") != "manifest" or any(
        record.get("record_type") != "example" for record in records[1:]
    ):
        raise CurriculumCorruptionError("manifest must be first and all remaining records examples")
    manifest_data = records[0].get("data")
    if not isinstance(manifest_data, Mapping):
        raise CurriculumCorruptionError("manifest data is not an object")
    try:
        manifest = CurriculumManifest.from_dict(manifest_data)
        examples = tuple(
            CurriculumExample.from_dict(cast(Mapping[str, Any], record["data"]))
            for record in records[1:]
            if isinstance(record.get("data"), Mapping)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CurriculumCorruptionError(f"invalid curriculum record: {exc}") from exc
    if len(examples) != len(records) - 1:
        raise CurriculumCorruptionError("an example record has no object data")
    try:
        rebuilt = build_curriculum_manifest(examples)
    except ValueError as exc:
        raise CurriculumCorruptionError(f"invalid curriculum examples: {exc}") from exc
    if rebuilt != manifest:
        raise CurriculumCorruptionError("curriculum content does not match its manifest")
    canonical = _canonical_examples(examples)
    if examples != canonical:
        raise CurriculumCorruptionError("curriculum examples are not in canonical order")
    return CurriculumBundle(manifest=manifest, examples=examples)


def _make_example(
    episode_id: str,
    step: int,
    family: TargetFamily,
    inputs: Mapping[str, Any],
    target: Mapping[str, Any],
    provenance: ExampleProvenance,
) -> CurriculumExample:
    normalized_inputs = _json_object(inputs, "inputs")
    normalized_target = _json_object(target, "target")
    identity = {
        "episode_id": episode_id,
        "step": step,
        "target_family": family,
        "inputs": normalized_inputs,
        "target": normalized_target,
        "provenance": provenance.to_dict(),
    }
    example_id = f"curr_{hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:24]}"
    return CurriculumExample(
        example_id=example_id,
        episode_id=episode_id,
        step=step,
        target_family=family,
        inputs=normalized_inputs,
        target=normalized_target,
        provenance=provenance,
    )


def _canonical_examples(examples: Sequence[CurriculumExample]) -> tuple[CurriculumExample, ...]:
    result = tuple(sorted(examples, key=_example_sort_key))
    ids = [example.example_id for example in result]
    if len(ids) != len(set(ids)):
        raise ValueError("example_id values must be unique")
    return result


def _examples_digest(examples: Sequence[CurriculumExample]) -> str:
    payload = "".join(_canonical_json(example.to_dict()) + "\n" for example in examples)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _example_sort_key(example: CurriculumExample) -> tuple[str, int, str, str]:
    return (example.episode_id, example.step, example.target_family, example.example_id)


def _event_sort_key(event: Mapping[str, Any], input_index: int) -> tuple[str, float, int]:
    episode_id = str(event.get("episode_id") or "")
    sequence = event.get("sequence")
    if isinstance(sequence, int) and not isinstance(sequence, bool):
        return (episode_id, float(sequence), input_index)
    observed_at = event.get("observed_at")
    if _is_finite_number(observed_at):
        return (episode_id, float(cast(int | float, observed_at)), input_index)
    return (episode_id, float(input_index), input_index)


def _is_external_observation(event_type: str, source: str, payload: Mapping[str, Any]) -> bool:
    if event_type not in _EXTERNAL_EVENT_TYPES or not isinstance(payload.get("content"), str):
        return False
    normalized_parts = set(source.casefold().replace(".", "_").replace(":", "_").split("_"))
    return bool(source.strip()) and not normalized_parts.intersection(_INTERNAL_SOURCE_PARTS)


def _learning_eligibility(
    payload: Mapping[str, Any],
    *,
    observed_key: str,
    subject: str,
) -> tuple[bool, str | None]:
    """Validate explicit learning gates while retaining marker-free legacy logs."""

    marker_present = "learning_eligible" in payload
    observed_present = observed_key in payload
    marker = payload.get("learning_eligible")
    observed = payload.get(observed_key)
    if marker_present and type(marker) is not bool:
        return False, f"{subject} learning_eligible marker is not boolean"
    if observed_present and type(observed) is not bool:
        return False, f"{subject} {observed_key} marker is not boolean"
    if marker_present:
        if marker is False:
            return False, None
        if observed is not True:
            return False, f"learning-eligible {subject} is not explicitly observed"
        return True, None
    if observed_present:
        return observed is True, None
    return True, None


def _parse_tool_outcome(
    payload: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    tool = payload.get("tool", payload.get("name"))
    succeeded = payload.get("succeeded")
    if not isinstance(tool, str) or not tool.strip() or type(succeeded) is not bool:
        return None
    raw_arguments = payload.get("arguments", payload.get("args", {}))
    if not isinstance(raw_arguments, Mapping):
        return None
    target: dict[str, Any] = {"succeeded": succeeded}
    status = payload.get("status")
    if isinstance(status, str) and status.strip().casefold() in _TYPED_TOOL_STATUSES:
        target["status"] = status.strip().casefold()
    return tool, _json_object(raw_arguments, "tool arguments"), target


def _parse_affect(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    scalar_names = ("valence", "arousal", "controllability")
    if any(not _is_finite_number(payload.get(name)) for name in scalar_names):
        return None
    valence = float(cast(int | float, payload["valence"]))
    arousal = float(cast(int | float, payload["arousal"]))
    controllability = float(cast(int | float, payload["controllability"]))
    if not -1.0 <= valence <= 1.0 or not 0.0 <= arousal <= 1.0 or not 0.0 <= controllability <= 1.0:
        return None
    raw_needs = payload.get("need_errors")
    if (
        not isinstance(raw_needs, Mapping)
        or not raw_needs
        or any(not isinstance(key, str) or not _is_finite_number(value) for key, value in raw_needs.items())
    ):
        return None
    need_errors = {str(key): float(cast(int | float, value)) for key, value in sorted(raw_needs.items())}
    if any(not -1.0 <= value <= 1.0 for value in need_errors.values()):
        return None
    return {
        "valence": valence,
        "arousal": arousal,
        "controllability": controllability,
        "need_errors": need_errors,
    }


def _affect_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_needs = cast(Mapping[str, Any], before.get("need_errors", {}))
    after_needs = cast(Mapping[str, Any], after.get("need_errors", {}))
    need_names = sorted(set(before_needs) | set(after_needs))
    return {
        "valence": _rounded(float(after["valence"]) - float(before["valence"])),
        "arousal": _rounded(float(after["arousal"]) - float(before["arousal"])),
        "controllability": _rounded(float(after["controllability"]) - float(before["controllability"])),
        "need_errors": {
            name: _rounded(float(after_needs.get(name, 0.0)) - float(before_needs.get(name, 0.0)))
            for name in need_names
        },
    }


def _add_external_context(inputs: dict[str, Any], event: Mapping[str, Any] | None) -> None:
    if event is None:
        return
    payload = event.get("payload")
    if isinstance(payload, Mapping) and isinstance(payload.get("content"), str):
        inputs["preceding_observation"] = payload["content"]


def _source_ids(*events: Mapping[str, Any] | None) -> tuple[str, ...]:
    result: list[str] = []
    for event in events:
        if event is None:
            continue
        event_id = str(event.get("event_id") or "")
        if event_id and event_id not in result:
            result.append(event_id)
    return tuple(result)


def _safe_horizon(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 1


def _finite_probability(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    probability = float(value)
    return probability if 0.0 <= probability <= 1.0 else None


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _rounded(value: float) -> float:
    return round(max(-1.0, min(1.0, value)), 6)


def _strict_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    normalized = _normalize_json(value, name)
    if not isinstance(normalized, dict):  # pragma: no cover - implied by Mapping check
        raise ValueError(f"{name} must be a JSON object")
    return normalized


def _normalize_json(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            result[key] = _normalize_json(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, f"{path}[]") for item in value]
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
