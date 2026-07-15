from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from typing import Any, Literal

from conscio.memory.store import (
    CognitiveEventConflictError,
    ExecutionJournalCorruptionError,
    MemoryStore,
)
from conscio.v3.contracts import (
    EXECUTION_JOURNAL_SCHEMA_VERSION,
    CognitiveEvent,
    ExecutionIntent,
    ExecutionOutcome,
    ExecutionReconciliation,
    ExecutionRecovery,
    deterministic_execution_event_id,
)

_UNSET = object()


def _digest(character: str) -> str:
    return "sha256:" + (character * 64)


def _intent(
    *,
    execution_character: str = "1",
    sequence: int = 0,
    action_kind: Literal["tool", "ask", "refuse"] = "tool",
) -> ExecutionIntent:
    return ExecutionIntent(
        execution_id="exec_" + (execution_character * 64),
        proposal_id="proposal-1",
        action_digest=_digest("2"),
        context_digest=_digest("3"),
        runtime_identity=_digest("4"),
        competition_sequence=sequence,
        action_kind=action_kind,
        tool_manifest_digest=_digest("6"),
        tool="lookup",
        arguments_json='{"z":2,"a":1}',
        capabilities=("network_read", "local_read", "network_read"),
    )


def _outcome(
    intent: ExecutionIntent,
    *,
    executed: bool = True,
    succeeded: bool | None = True,
    disposition: str = "succeeded",
) -> ExecutionOutcome:
    return ExecutionOutcome(
        execution_id=intent.execution_id,
        intent_digest=intent.intent_digest,
        proposal_id=intent.proposal_id,
        action_digest=intent.action_digest,
        tool=intent.tool,
        executed=executed,
        succeeded=succeeded,
        disposition=disposition,  # type: ignore[arg-type]
        result_digest=_digest("5"),
        reason_code="tool_returned",
    )


def _recovery(intent: ExecutionIntent) -> ExecutionRecovery:
    return ExecutionRecovery(
        execution_id=intent.execution_id,
        intent_digest=intent.intent_digest,
    )


def _reconciliation(intent: ExecutionIntent) -> ExecutionReconciliation:
    return ExecutionReconciliation(
        execution_id=intent.execution_id,
        intent_digest=intent.intent_digest,
        operator="test-operator",
        reason="verified out of band",
    )


def _event(
    event_type: str,
    intent: ExecutionIntent,
    payload: dict[str, Any],
    *,
    observed_at: float = 1.0,
    source: str | None = None,
    episode_id: str = "episode-1",
    parent_event_id: str | None | object = _UNSET,
) -> CognitiveEvent:
    sources = {
        "execution_intent": "runtime_executor",
        "execution_outcome": "runtime_executor",
        "execution_reconciliation": "operator_reconciliation",
        "execution_recovery": "runtime_recovery",
    }
    if parent_event_id is _UNSET:
        resolved_parent = None if event_type == "execution_intent" else intent.event_id
    elif parent_event_id is None or isinstance(parent_event_id, str):
        resolved_parent = parent_event_id
    else:  # pragma: no cover - test helper misuse
        raise TypeError("parent_event_id must be a string or null")
    return CognitiveEvent(
        event_type=event_type,
        source=source or sources[event_type],
        payload=payload,
        episode_id=episode_id,
        event_id=deterministic_execution_event_id(event_type, intent.execution_id),
        observed_at=observed_at,
        parent_event_id=resolved_parent,
    )


class ExecutionContractTests(unittest.TestCase):
    def test_intent_is_frozen_canonical_and_content_addressed(self) -> None:
        intent = _intent()
        same = ExecutionIntent.from_dict(intent.to_dict())

        self.assertEqual(intent.arguments_json, '{"a":1,"z":2}')
        self.assertEqual(intent.capabilities, ("local_read", "network_read"))
        self.assertEqual(intent.action_kind, "tool")
        self.assertEqual(intent.tool_manifest_digest, _digest("6"))
        self.assertEqual(same, intent)
        self.assertEqual(same.intent_digest, intent.intent_digest)
        self.assertEqual(intent.event_id, deterministic_execution_event_id("execution_intent", intent.execution_id))
        with self.assertRaises(FrozenInstanceError):
            intent.tool = "changed"  # type: ignore[misc]

    def test_outcome_enforces_nullable_not_executed_semantics(self) -> None:
        intent = _intent()
        denied = _outcome(intent, executed=False, succeeded=None, disposition="not_executed")

        self.assertFalse(denied.executed)
        self.assertIsNone(denied.succeeded)
        self.assertEqual(ExecutionOutcome.from_dict(denied.to_dict()), denied)

        invalid = (
            {"executed": False, "succeeded": False, "disposition": "not_executed"},
            {"executed": False, "succeeded": None, "disposition": "failed"},
            {"executed": True, "succeeded": None, "disposition": "failed"},
            {"executed": True, "succeeded": True, "disposition": "failed"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _outcome(intent, **changes)

    def test_recovery_and_reconciliation_are_strict_unknown_outcome_records(self) -> None:
        intent = _intent()
        recovery = _recovery(intent)
        reconciliation = _reconciliation(intent)

        self.assertEqual(ExecutionRecovery.from_dict(recovery.to_dict()), recovery)
        self.assertEqual(ExecutionReconciliation.from_dict(reconciliation.to_dict()), reconciliation)
        self.assertEqual(recovery.schema_version, EXECUTION_JOURNAL_SCHEMA_VERSION)
        self.assertEqual(reconciliation.schema_version, EXECUTION_JOURNAL_SCHEMA_VERSION)
        self.assertEqual(
            recovery.event_id,
            deterministic_execution_event_id("execution_recovery", intent.execution_id),
        )
        self.assertEqual(
            reconciliation.event_id,
            deterministic_execution_event_id("execution_reconciliation", intent.execution_id),
        )
        with self.assertRaises(FrozenInstanceError):
            recovery.executed = True  # type: ignore[misc,assignment]

        invalid_recovery = recovery.to_dict()
        invalid_recovery["executed"] = True
        with self.assertRaisesRegex(ValueError, "cannot claim"):
            ExecutionRecovery.from_dict(invalid_recovery)
        invalid_recovery = recovery.to_dict()
        invalid_recovery["extra"] = True
        with self.assertRaisesRegex(ValueError, "keys differ"):
            ExecutionRecovery.from_dict(invalid_recovery)
        invalid_recovery = recovery.to_dict()
        invalid_recovery["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema_version"):
            ExecutionRecovery.from_dict(invalid_recovery)

        invalid_reconciliation = reconciliation.to_dict()
        invalid_reconciliation["learning_eligible"] = True
        with self.assertRaisesRegex(ValueError, "learning target"):
            ExecutionReconciliation.from_dict(invalid_reconciliation)
        invalid_reconciliation = reconciliation.to_dict()
        invalid_reconciliation["operator"] = " untrimmed"
        with self.assertRaisesRegex(ValueError, "trimmed"):
            ExecutionReconciliation.from_dict(invalid_reconciliation)

    def test_contract_parsers_reject_extra_fields_and_bad_deterministic_ids(self) -> None:
        payload = _intent().to_dict()
        payload["extra"] = True
        with self.assertRaisesRegex(ValueError, "keys differ"):
            ExecutionIntent.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "execution_id"):
            deterministic_execution_event_id("execution_intent", "provider-call-1")
        outcome_payload = _outcome(_intent()).to_dict()
        outcome_payload["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema_version"):
            ExecutionOutcome.from_dict(outcome_payload)

        invalid_kind = _intent().to_dict()
        invalid_kind["action_kind"] = "respond"
        with self.assertRaisesRegex(ValueError, "action_kind"):
            ExecutionIntent.from_dict(invalid_kind)
        invalid_manifest = _intent().to_dict()
        invalid_manifest["tool_manifest_digest"] = "untracked"
        with self.assertRaisesRegex(ValueError, "tool_manifest_digest"):
            ExecutionIntent.from_dict(invalid_manifest)


class ExecutionJournalStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "journal.db")
        self.store = MemoryStore(db_path=self.path)
        await self.store.initialize()

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.tmp.cleanup()

    async def test_idempotent_append_distinguishes_insert_retry_and_conflict(self) -> None:
        intent = _intent()
        first_event = _event("execution_intent", intent, intent.to_dict(), observed_at=1.0)
        retry_event = _event("execution_intent", intent, intent.to_dict(), observed_at=99.0)

        first = await self.store.append_cognitive_event_idempotent(first_event)
        retry = await self.store.append_cognitive_event_idempotent(retry_event)

        self.assertTrue(first.inserted)
        self.assertFalse(retry.inserted)
        self.assertEqual(retry.sequence, first.sequence)
        self.assertEqual(len(await self.store.cognitive_events("episode-1")), 1)

        conflicting_payload = intent.to_dict()
        conflicting_payload["tool"] = "different"
        with self.assertRaises(CognitiveEventConflictError):
            await self.store.append_cognitive_event_idempotent(_event("execution_intent", intent, conflicting_payload))
        with self.assertRaisesRegex(ValueError, "observed_at"):
            await self.store.append_cognitive_event_idempotent(
                _event(
                    "execution_intent",
                    _intent(execution_character="b"),
                    {},
                    observed_at=float("nan"),
                )
            )

    async def test_recovery_is_nonterminal_and_reconciliation_is_terminal(self) -> None:
        intent = _intent()
        await self.store.append_cognitive_event_idempotent(_event("execution_intent", intent, intent.to_dict()))
        recovery = _recovery(intent)
        await self.store.append_cognitive_event_idempotent(
            _event("execution_recovery", intent, recovery.to_dict(), observed_at=2.0)
        )

        unresolved = await self.store.unresolved_execution_intents()

        self.assertEqual([row["event_id"] for row in unresolved], [intent.event_id])

        reconciliation = _reconciliation(intent)
        await self.store.append_cognitive_event_idempotent(
            _event("execution_reconciliation", intent, reconciliation.to_dict(), observed_at=3.0)
        )
        self.assertEqual(await self.store.unresolved_execution_intents(), [])

    async def test_outcomes_are_terminal_including_not_executed(self) -> None:
        successful = _intent(execution_character="6")
        denied = _intent(execution_character="7", sequence=1)
        for intent in (successful, denied):
            await self.store.append_cognitive_event_idempotent(_event("execution_intent", intent, intent.to_dict()))
        success = _outcome(successful)
        not_executed = _outcome(
            denied,
            executed=False,
            succeeded=None,
            disposition="not_executed",
        )
        for intent, outcome in ((successful, success), (denied, not_executed)):
            await self.store.append_cognitive_event_idempotent(
                _event("execution_outcome", intent, outcome.to_dict(), observed_at=2.0)
            )

        self.assertEqual(await self.store.unresolved_execution_intents(), [])

    async def test_query_rejects_wrong_identity_and_multiple_terminals(self) -> None:
        intent = _intent(execution_character="8")
        await self.store.append_cognitive_event_idempotent(_event("execution_intent", intent, intent.to_dict()))
        outcome_payload = _outcome(intent).to_dict()
        outcome_payload["proposal_id"] = "wrong-proposal"
        await self.store.append_cognitive_event(_event("execution_outcome", intent, outcome_payload, observed_at=2.0))
        with self.assertRaisesRegex(ExecutionJournalCorruptionError, "disagrees"):
            await self.store.unresolved_execution_intents()

    async def test_query_rejects_wrong_source(self) -> None:
        intent = _intent(execution_character="c")
        await self.store.append_cognitive_event_idempotent(
            _event(
                "execution_intent",
                intent,
                intent.to_dict(),
                source="operator_reconciliation",
            )
        )

        with self.assertRaisesRegex(ExecutionJournalCorruptionError, "wrong source"):
            await self.store.unresolved_execution_intents()

    async def test_query_rejects_cross_episode_and_wrong_parent_links(self) -> None:
        cross_episode = _intent(execution_character="d")
        await self.store.append_cognitive_event_idempotent(
            _event("execution_intent", cross_episode, cross_episode.to_dict())
        )
        await self.store.append_cognitive_event_idempotent(
            _event(
                "execution_outcome",
                cross_episode,
                _outcome(cross_episode).to_dict(),
                observed_at=2.0,
                episode_id="different-episode",
            )
        )
        with self.assertRaisesRegex(ExecutionJournalCorruptionError, "different episode"):
            await self.store.unresolved_execution_intents()

        other_store = MemoryStore(db_path=os.path.join(self.tmp.name, "wrong-parent.db"))
        await other_store.initialize()
        try:
            wrong_parent = _intent(execution_character="e")
            await other_store.append_cognitive_event_idempotent(
                _event("execution_intent", wrong_parent, wrong_parent.to_dict())
            )
            await other_store.append_cognitive_event_idempotent(
                _event(
                    "execution_recovery",
                    wrong_parent,
                    _recovery(wrong_parent).to_dict(),
                    observed_at=2.0,
                    parent_event_id="evt_unrelated",
                )
            )
            with self.assertRaisesRegex(ExecutionJournalCorruptionError, "wrong parent event"):
                await other_store.unresolved_execution_intents()
        finally:
            await other_store.close()

    async def test_query_rejects_reconciliation_that_claims_an_outcome(self) -> None:
        intent = _intent(execution_character="f")
        await self.store.append_cognitive_event_idempotent(_event("execution_intent", intent, intent.to_dict()))
        malformed = _reconciliation(intent).to_dict()
        malformed["executed"] = True
        await self.store.append_cognitive_event_idempotent(
            _event("execution_reconciliation", intent, malformed, observed_at=2.0)
        )

        with self.assertRaisesRegex(ExecutionJournalCorruptionError, "cannot claim"):
            await self.store.unresolved_execution_intents()

    async def test_query_rejects_terminal_without_intent_and_duplicate_terminal(self) -> None:
        missing = _intent(execution_character="9")
        outcome = _outcome(missing)
        await self.store.append_cognitive_event_idempotent(_event("execution_outcome", missing, outcome.to_dict()))
        with self.assertRaisesRegex(ExecutionJournalCorruptionError, "missing intents"):
            await self.store.unresolved_execution_intents()

        other_store = MemoryStore(db_path=os.path.join(self.tmp.name, "duplicate.db"))
        await other_store.initialize()
        try:
            intent = _intent(execution_character="a")
            await other_store.append_cognitive_event_idempotent(_event("execution_intent", intent, intent.to_dict()))
            terminal = _reconciliation(intent)
            await other_store.append_cognitive_event_idempotent(
                _event("execution_reconciliation", intent, terminal.to_dict(), observed_at=2.0)
            )
            outcome = _outcome(intent)
            await other_store.append_cognitive_event_idempotent(
                _event("execution_outcome", intent, outcome.to_dict(), observed_at=3.0)
            )
            with self.assertRaisesRegex(ExecutionJournalCorruptionError, "multiple terminal"):
                await other_store.unresolved_execution_intents()
        finally:
            await other_store.close()

    async def test_schema_has_execution_event_sequence_index(self) -> None:
        indexes = {row["name"] for row in self.store.fetchall("PRAGMA index_list(cognitive_events)")}
        self.assertIn("idx_cognitive_events_type_sequence", indexes)


if __name__ == "__main__":
    unittest.main()
