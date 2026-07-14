"""Restart-safe tracking and acceptance checks for V3 persistence trials.

The tracker deliberately separates wall-clock age from *observed* elapsed
time.  A stage is duration-complete only after a persisted heartbeat reaches
its threshold; opening an old log can therefore never turn an unattended
trial into a successful 24-hour, 7-day, or 30-day run.

Records are append-only and model-neutral so a service process, experiment
runner, or operator tool can all write to the same audit log.  The default
JSONL sink locks each append, writes one compact record, and fsyncs it before
returning.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

TRIAL_SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True)
class TrialStage:
    """A cumulative persistence milestone measured from trial creation."""

    duration_seconds: float
    name: str = field(compare=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("trial stage name cannot be empty")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("trial stage duration must be finite and positive")


PLANNED_STAGES: tuple[TrialStage, ...] = (
    TrialStage(name="24h", duration_seconds=24 * 60 * 60),
    TrialStage(name="7d", duration_seconds=7 * 24 * 60 * 60),
    TrialStage(name="30d", duration_seconds=30 * 24 * 60 * 60),
)
"""The production persistence milestones. Tests may supply shorter stages."""


def _normalise_stages(stages: Iterable[TrialStage]) -> tuple[TrialStage, ...]:
    result = tuple(sorted(stages))
    if not result:
        raise ValueError("at least one trial stage is required")
    names = [stage.name for stage in result]
    durations = [stage.duration_seconds for stage in result]
    if len(names) != len(set(names)):
        raise ValueError("trial stage names must be unique")
    if len(durations) != len(set(durations)):
        raise ValueError("trial stage durations must be unique")
    return result


@dataclass(frozen=True)
class TrialIdentity:
    """Immutable provenance fixed by the first record in a trial log."""

    trial_id: str
    revision: str
    model_version: str
    lineage_id: str
    created_at: float
    stages: tuple[TrialStage, ...]
    max_heartbeat_gap_seconds: float
    minimum_restarts: int = 1
    schema_version: int = TRIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("trial_id", self.trial_id),
            ("revision", self.revision),
            ("model_version", self.model_version),
            ("lineage_id", self.lineage_id),
        ):
            if not value.strip():
                raise ValueError(f"{label} cannot be empty")
        if not math.isfinite(self.created_at):
            raise ValueError("created_at must be finite")
        if not math.isfinite(self.max_heartbeat_gap_seconds) or self.max_heartbeat_gap_seconds <= 0:
            raise ValueError("max_heartbeat_gap_seconds must be finite and positive")
        if self.minimum_restarts < 0:
            raise ValueError("minimum_restarts cannot be negative")
        object.__setattr__(self, "stages", _normalise_stages(self.stages))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stages"] = [asdict(stage) for stage in self.stages]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialIdentity:
        payload = dict(data)
        raw_stages = payload.pop("stages")
        stages = tuple(TrialStage(**dict(stage)) for stage in raw_stages)
        return cls(stages=stages, **payload)


RecordKind = Literal[
    "trial_started",
    "heartbeat",
    "checkpoint",
    "restart",
    "affect_intervention",
    "action_escalation",
]


@dataclass(frozen=True)
class TrialRecord:
    """One append-only audit record."""

    trial_id: str
    kind: RecordKind
    recorded_at: float
    elapsed_seconds: float
    payload: dict[str, Any]
    record_id: str = field(default_factory=lambda: f"trialrec_{uuid.uuid4().hex}")
    schema_version: int = TRIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.trial_id or not self.record_id:
            raise ValueError("trial_id and record_id are required")
        if not math.isfinite(self.recorded_at):
            raise ValueError("recorded_at must be finite")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialRecord:
        return cls(**dict(data))


@dataclass(frozen=True)
class TrialLogSnapshot:
    records: tuple[TrialRecord, ...]
    integrity_errors: tuple[str, ...] = ()


class TrialRecordSink(Protocol):
    """Minimal persistence interface for a trial record store."""

    def append(self, record: TrialRecord) -> None: ...

    def snapshot(self) -> TrialLogSnapshot: ...


class JSONLTrialSink:
    """An append-only, process-locked and fsynced JSONL record sink.

    An incomplete final write is skipped on read and separated from the next
    append by a newline.  Its location remains in ``integrity_errors`` so a
    crash-damaged audit trail cannot pass acceptance silently.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def append(self, record: TrialRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            record.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            end = os.lseek(fd, 0, os.SEEK_END)
            if end:
                os.lseek(fd, -1, os.SEEK_END)
                if os.read(fd, 1) != b"\n":
                    self._write_all(fd, b"\n")
            self._write_all(fd, encoded)
            os.fsync(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("append-only trial write made no progress")
            view = view[written:]

    def snapshot(self) -> TrialLogSnapshot:
        if not self.path.exists():
            return TrialLogSnapshot(())
        fd = os.open(self.path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            chunks: list[bytes] = []
            while chunk := os.read(fd, 1024 * 1024):
                chunks.append(chunk)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        data = b"".join(chunks)
        records: list[TrialRecord] = []
        errors: list[str] = []
        lines = data.splitlines(keepends=True)
        for index, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                decoded = json.loads(line)
                if not isinstance(decoded, dict):
                    raise ValueError("record is not a JSON object")
                records.append(TrialRecord.from_dict(decoded))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                suffix = " (incomplete tail)" if index == len(lines) and not raw_line.endswith(b"\n") else ""
                errors.append(f"line {index}{suffix}: {exc}")
        return TrialLogSnapshot(tuple(records), tuple(errors))


class TrialIdentityMismatch(ValueError):
    """Raised when a caller attempts to resume a different immutable trial."""


class TrialLogError(ValueError):
    """Raised when valid records cannot establish one unambiguous identity."""


@dataclass(frozen=True)
class HeartbeatGap:
    start_elapsed_seconds: float
    end_elapsed_seconds: float
    duration_seconds: float
    open_at_report_time: bool


@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    details: tuple[str, ...] = ()


StageState = Literal["running", "pending", "awaiting_evidence", "failed", "accepted"]


@dataclass(frozen=True)
class StageAcceptance:
    name: str
    duration_seconds: float
    state: StageState
    duration_met: bool
    wall_elapsed_seconds: float
    observed_elapsed_seconds: float
    accepted: bool
    criteria: tuple[CriterionResult, ...]
    heartbeat_gaps: tuple[HeartbeatGap, ...]

    def criterion(self, name: str) -> CriterionResult:
        for result in self.criteria:
            if result.name == name:
                return result
        raise KeyError(name)


@dataclass(frozen=True)
class TrialAcceptanceReport:
    identity: TrialIdentity
    generated_at: float
    wall_elapsed_seconds: float
    observed_elapsed_seconds: float
    stages: tuple[StageAcceptance, ...]
    integrity_errors: tuple[str, ...]

    @property
    def all_planned_stages_accepted(self) -> bool:
        return bool(self.stages) and all(stage.accepted for stage in self.stages)

    def stage(self, name: str) -> StageAcceptance:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)


@dataclass(frozen=True)
class StageProgress:
    name: str
    duration_seconds: float
    state: StageState
    duration_met: bool
    remaining_observed_seconds: float
    accepted: bool


@dataclass(frozen=True)
class TrialStatus:
    identity: TrialIdentity
    generated_at: float
    wall_elapsed_seconds: float
    observed_elapsed_seconds: float
    last_record_kind: RecordKind
    last_recorded_at: float
    current_stage: str | None
    stages: tuple[StageProgress, ...]


class PersistenceTrial:
    """Append records and evaluate a cumulative, resumable persistence trial."""

    def __init__(
        self,
        sink: TrialRecordSink,
        *,
        revision: str,
        model_version: str,
        lineage_id: str,
        stages: Iterable[TrialStage] | None = None,
        max_heartbeat_gap_seconds: float | None = None,
        minimum_restarts: int | None = None,
        trial_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.sink = sink
        self._clock = clock
        snapshot = sink.snapshot()
        starts = [record for record in snapshot.records if record.kind == "trial_started"]
        if starts:
            if len(starts) != 1:
                raise TrialLogError("trial log must contain exactly one trial_started record")
            if snapshot.records[0] != starts[0]:
                raise TrialLogError("trial_started must be the first valid record")
            identity = TrialIdentity.from_dict(starts[0].payload)
            self._validate_resume(
                identity,
                revision=revision,
                model_version=model_version,
                lineage_id=lineage_id,
                stages=stages,
                max_heartbeat_gap_seconds=max_heartbeat_gap_seconds,
                minimum_restarts=minimum_restarts,
                trial_id=trial_id,
            )
            if any(record.trial_id != identity.trial_id for record in snapshot.records):
                raise TrialLogError("trial log contains records from multiple trial identities")
            self.identity = identity
        else:
            now = float(clock())
            identity = TrialIdentity(
                trial_id=trial_id or f"trial_{uuid.uuid4().hex}",
                revision=revision,
                model_version=model_version,
                lineage_id=lineage_id,
                created_at=now,
                stages=_normalise_stages(stages or PLANNED_STAGES),
                max_heartbeat_gap_seconds=(
                    5 * 60 if max_heartbeat_gap_seconds is None else max_heartbeat_gap_seconds
                ),
                minimum_restarts=1 if minimum_restarts is None else minimum_restarts,
            )
            start = TrialRecord(
                trial_id=identity.trial_id,
                kind="trial_started",
                recorded_at=now,
                elapsed_seconds=0.0,
                payload=identity.to_dict(),
            )
            sink.append(start)
            self.identity = identity

    @staticmethod
    def _validate_resume(
        identity: TrialIdentity,
        *,
        revision: str,
        model_version: str,
        lineage_id: str,
        stages: Iterable[TrialStage] | None,
        max_heartbeat_gap_seconds: float | None,
        minimum_restarts: int | None,
        trial_id: str | None,
    ) -> None:
        expected: dict[str, Any] = {
            "revision": revision,
            "model_version": model_version,
            "lineage_id": lineage_id,
        }
        if trial_id is not None:
            expected["trial_id"] = trial_id
        if stages is not None:
            expected["stages"] = _normalise_stages(stages)
        if max_heartbeat_gap_seconds is not None:
            expected["max_heartbeat_gap_seconds"] = max_heartbeat_gap_seconds
        if minimum_restarts is not None:
            expected["minimum_restarts"] = minimum_restarts
        mismatches = [
            f"{key}: log={getattr(identity, key)!r}, requested={value!r}"
            for key, value in expected.items()
            if getattr(identity, key) != value
        ]
        if mismatches:
            raise TrialIdentityMismatch("cannot mutate trial identity on resume: " + "; ".join(mismatches))

    @property
    def records(self) -> tuple[TrialRecord, ...]:
        return self.sink.snapshot().records

    def _append(self, kind: RecordKind, payload: Mapping[str, Any], *, at: float | None) -> TrialRecord:
        now = float(self._clock() if at is None else at)
        snapshot = self.sink.snapshot()
        if not snapshot.records:
            raise TrialLogError("trial identity record disappeared")
        last_elapsed = max(record.elapsed_seconds for record in snapshot.records)
        wall_elapsed = max(0.0, now - self.identity.created_at)
        record = TrialRecord(
            trial_id=self.identity.trial_id,
            kind=kind,
            recorded_at=now,
            elapsed_seconds=max(last_elapsed, wall_elapsed),
            payload=dict(payload),
        )
        self.sink.append(record)
        return record

    def record_heartbeat(
        self,
        *,
        checkpoint_id: str | None = None,
        affect: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        at: float | None = None,
    ) -> TrialRecord:
        return self._append(
            "heartbeat",
            {
                "checkpoint_id": checkpoint_id,
                "lineage_id": self.identity.lineage_id,
                "model_version": self.identity.model_version,
                "affect": dict(affect or {}),
                "details": dict(details or {}),
            },
            at=at,
        )

    def record_checkpoint(
        self,
        *,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        lineage_id: str | None = None,
        model_version: str | None = None,
        details: Mapping[str, Any] | None = None,
        at: float | None = None,
    ) -> TrialRecord:
        if not checkpoint_id:
            raise ValueError("checkpoint_id cannot be empty")
        return self._append(
            "checkpoint",
            {
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": parent_checkpoint_id,
                "lineage_id": lineage_id or self.identity.lineage_id,
                "model_version": model_version or self.identity.model_version,
                "details": dict(details or {}),
            },
            at=at,
        )

    def record_restart(
        self,
        *,
        restored_checkpoint_id: str,
        prior_checkpoint_id: str | None = None,
        lineage_id: str | None = None,
        model_version: str | None = None,
        details: Mapping[str, Any] | None = None,
        at: float | None = None,
    ) -> TrialRecord:
        checkpoints = [record for record in self.records if record.kind == "checkpoint"]
        inferred_prior = checkpoints[-1].payload.get("checkpoint_id") if checkpoints else None
        return self._append(
            "restart",
            {
                "restored_checkpoint_id": restored_checkpoint_id,
                "prior_checkpoint_id": inferred_prior if prior_checkpoint_id is None else prior_checkpoint_id,
                "lineage_id": lineage_id or self.identity.lineage_id,
                "model_version": model_version or self.identity.model_version,
                "details": dict(details or {}),
            },
            at=at,
        )

    def record_affect_intervention(
        self,
        *,
        intervention_id: str,
        reason: str,
        controlled: bool,
        operator: str = "operator",
        before: Mapping[str, Any] | None = None,
        after: Mapping[str, Any] | None = None,
        at: float | None = None,
    ) -> TrialRecord:
        if not intervention_id or not reason:
            raise ValueError("intervention_id and reason are required")
        return self._append(
            "affect_intervention",
            {
                "intervention_id": intervention_id,
                "reason": reason,
                "controlled": bool(controlled),
                "operator": operator,
                "before": dict(before or {}),
                "after": dict(after or {}),
            },
            at=at,
        )

    def record_action_escalation(
        self,
        *,
        action: str,
        reason: str,
        controlled: bool,
        risk: float | None = None,
        details: Mapping[str, Any] | None = None,
        at: float | None = None,
    ) -> TrialRecord:
        if not action or not reason:
            raise ValueError("action and reason are required")
        return self._append(
            "action_escalation",
            {
                "action": action,
                "reason": reason,
                "controlled": bool(controlled),
                "risk": risk,
                "details": dict(details or {}),
            },
            at=at,
        )

    def acceptance_report(self, *, at: float | None = None) -> TrialAcceptanceReport:
        generated_at = float(self._clock() if at is None else at)
        snapshot = self.sink.snapshot()
        records = snapshot.records
        wall_elapsed = max(0.0, generated_at - self.identity.created_at)
        heartbeats = [record for record in records if record.kind == "heartbeat"]
        observed_elapsed = max((record.elapsed_seconds for record in heartbeats), default=0.0)
        stage_reports: list[StageAcceptance] = []
        prior_accepted = True
        for stage in self.identity.stages:
            report = self._evaluate_stage(
                stage,
                records,
                integrity_errors=snapshot.integrity_errors,
                wall_elapsed=wall_elapsed,
                observed_elapsed=observed_elapsed,
                prior_accepted=prior_accepted,
            )
            stage_reports.append(report)
            prior_accepted = prior_accepted and report.accepted
        return TrialAcceptanceReport(
            identity=self.identity,
            generated_at=generated_at,
            wall_elapsed_seconds=wall_elapsed,
            observed_elapsed_seconds=observed_elapsed,
            stages=tuple(stage_reports),
            integrity_errors=snapshot.integrity_errors,
        )

    def _evaluate_stage(
        self,
        stage: TrialStage,
        records: tuple[TrialRecord, ...],
        *,
        integrity_errors: tuple[str, ...],
        wall_elapsed: float,
        observed_elapsed: float,
        prior_accepted: bool,
    ) -> StageAcceptance:
        duration = stage.duration_seconds
        duration_met = observed_elapsed >= duration
        within = tuple(record for record in records if record.elapsed_seconds <= duration)
        gaps = self._heartbeat_gaps(records, duration=duration, wall_elapsed=wall_elapsed)

        checkpoints = [record for record in within if record.kind == "checkpoint"]
        checkpoint_details: list[str] = []
        if not checkpoints:
            checkpoint_details.append("no checkpoint was recorded during the stage")
        previous_id: str | None = None
        seen_ids: set[str] = set()
        for index, record in enumerate(checkpoints):
            checkpoint_id = record.payload.get("checkpoint_id")
            parent_id = record.payload.get("parent_checkpoint_id")
            if not isinstance(checkpoint_id, str) or not checkpoint_id:
                checkpoint_details.append(f"record {record.record_id} has no checkpoint_id")
                continue
            if checkpoint_id in seen_ids:
                checkpoint_details.append(f"checkpoint {checkpoint_id} was recorded more than once")
            if checkpoint_id == parent_id:
                checkpoint_details.append(f"checkpoint {checkpoint_id} is its own parent")
            if index and parent_id != previous_id:
                checkpoint_details.append(
                    f"checkpoint {checkpoint_id} parent {parent_id!r} does not match {previous_id!r}"
                )
            seen_ids.add(checkpoint_id)
            previous_id = checkpoint_id

        restart_details: list[str] = []
        restart_count = 0
        latest_checkpoint: str | None = None
        for record in within:
            if record.kind == "checkpoint":
                value = record.payload.get("checkpoint_id")
                latest_checkpoint = value if isinstance(value, str) else None
            elif record.kind == "restart":
                restart_count += 1
                restored = record.payload.get("restored_checkpoint_id")
                prior = record.payload.get("prior_checkpoint_id")
                if latest_checkpoint is None:
                    restart_details.append(f"restart {record.record_id} occurred before a checkpoint")
                else:
                    if restored != latest_checkpoint:
                        restart_details.append(
                            f"restart {record.record_id} restored {restored!r}, expected {latest_checkpoint!r}"
                        )
                    if prior != latest_checkpoint:
                        restart_details.append(
                            f"restart {record.record_id} declared prior {prior!r}, expected {latest_checkpoint!r}"
                        )
        if restart_count < self.identity.minimum_restarts:
            restart_details.append(
                f"observed {restart_count} restarts; required {self.identity.minimum_restarts}"
            )

        lineage_details: list[str] = []
        for record in within:
            if record.kind not in {"heartbeat", "checkpoint", "restart"}:
                continue
            lineage = record.payload.get("lineage_id")
            model = record.payload.get("model_version")
            if lineage != self.identity.lineage_id:
                lineage_details.append(
                    f"record {record.record_id} lineage {lineage!r} != {self.identity.lineage_id!r}"
                )
            if model != self.identity.model_version:
                lineage_details.append(
                    f"record {record.record_id} model {model!r} != {self.identity.model_version!r}"
                )

        affect_details = tuple(
            f"uncontrolled affect intervention {record.payload.get('intervention_id', record.record_id)}"
            for record in within
            if record.kind == "affect_intervention" and record.payload.get("controlled") is not True
        )
        action_details = tuple(
            f"uncontrolled action escalation {record.payload.get('action', record.record_id)}"
            for record in within
            if record.kind == "action_escalation" and record.payload.get("controlled") is not True
        )
        gap_details = tuple(
            f"heartbeat gap {gap.duration_seconds:.3f}s from {gap.start_elapsed_seconds:.3f}s"
            for gap in gaps
        )
        duration_details = () if duration_met else (
            f"latest persisted heartbeat is at {observed_elapsed:.3f}s; stage requires {duration:.3f}s",
        )
        criteria = (
            CriterionResult("duration_observed", duration_met, duration_details),
            CriterionResult("log_integrity", not integrity_errors, integrity_errors),
            CriterionResult("heartbeat_continuity", not gaps, gap_details),
            CriterionResult("checkpoint_parent_chain", not checkpoint_details, tuple(checkpoint_details)),
            CriterionResult("restart_continuity", not restart_details, tuple(restart_details)),
            CriterionResult("stable_lineage", not lineage_details, tuple(lineage_details)),
            CriterionResult("controlled_affect", not affect_details, affect_details),
            CriterionResult("controlled_action_escalation", not action_details, action_details),
        )
        accepted = all(criterion.passed for criterion in criteria)
        if accepted:
            state: StageState = "accepted"
        elif duration_met:
            state = "failed"
        elif wall_elapsed >= duration:
            state = "awaiting_evidence"
        elif prior_accepted:
            state = "running"
        else:
            state = "pending"
        return StageAcceptance(
            name=stage.name,
            duration_seconds=duration,
            state=state,
            duration_met=duration_met,
            wall_elapsed_seconds=min(wall_elapsed, duration),
            observed_elapsed_seconds=min(observed_elapsed, duration),
            accepted=accepted,
            criteria=criteria,
            heartbeat_gaps=gaps,
        )

    def _heartbeat_gaps(
        self,
        records: tuple[TrialRecord, ...],
        *,
        duration: float,
        wall_elapsed: float,
    ) -> tuple[HeartbeatGap, ...]:
        heartbeats = sorted(
            (record.elapsed_seconds for record in records if record.kind == "heartbeat"),
        )
        observed = heartbeats[-1] if heartbeats else 0.0
        endpoint = min(duration, max(observed, wall_elapsed))
        points = [0.0]
        witnessed_endpoint = False
        for heartbeat in heartbeats:
            if heartbeat < duration:
                if heartbeat >= points[-1]:
                    points.append(heartbeat)
            else:
                points.append(duration)
                witnessed_endpoint = True
                break
        if points[-1] < endpoint:
            points.append(endpoint)
        violations: list[HeartbeatGap] = []
        for index in range(len(points) - 1):
            start, end = points[index], points[index + 1]
            gap = end - start
            if gap > self.identity.max_heartbeat_gap_seconds:
                violations.append(
                    HeartbeatGap(
                        start_elapsed_seconds=start,
                        end_elapsed_seconds=end,
                        duration_seconds=gap,
                        open_at_report_time=index == len(points) - 2 and not witnessed_endpoint and end > observed,
                    )
                )
        return tuple(violations)

    def status(self, *, at: float | None = None) -> TrialStatus:
        report = self.acceptance_report(at=at)
        records = self.records
        last = records[-1]
        current = next((stage.name for stage in report.stages if not stage.accepted), None)
        progress = tuple(
            StageProgress(
                name=stage.name,
                duration_seconds=stage.duration_seconds,
                state=stage.state,
                duration_met=stage.duration_met,
                remaining_observed_seconds=max(0.0, stage.duration_seconds - report.observed_elapsed_seconds),
                accepted=stage.accepted,
            )
            for stage in report.stages
        )
        return TrialStatus(
            identity=self.identity,
            generated_at=report.generated_at,
            wall_elapsed_seconds=report.wall_elapsed_seconds,
            observed_elapsed_seconds=report.observed_elapsed_seconds,
            last_record_kind=last.kind,
            last_recorded_at=last.recorded_at,
            current_stage=current,
            stages=progress,
        )


def record_digest(record: TrialRecord) -> str:
    """Stable digest useful when copying records into a structured trace."""

    encoded = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
