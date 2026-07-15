from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from conscio.v3.curriculum import (
    TARGET_FAMILIES,
    CurriculumCorruptionError,
    CurriculumExample,
    ExampleProvenance,
    build_curriculum_manifest,
    derive_curriculum_examples,
    deterministic_curriculum_split,
    generate_synthetic_curriculum,
    read_curriculum_jsonl,
    write_curriculum_jsonl,
)


def _event(
    sequence: int,
    event_type: str,
    source: str,
    payload: object,
    *,
    episode_id: str = "episode",
) -> dict:
    return {
        "sequence": sequence,
        "event_id": f"event_{sequence}",
        "episode_id": episode_id,
        "event_type": event_type,
        "source": source,
        "payload": payload,
    }


def _affect(valence: float, arousal: float, controllability: float) -> dict:
    return {
        "valence": valence,
        "arousal": arousal,
        "controllability": controllability,
        "need_errors": {
            "competence": 0.4 - valence,
            "epistemic_coherence": 0.5 - valence,
        },
    }


def test_synthetic_generation_is_seeded_and_covers_every_target() -> None:
    first = generate_synthetic_curriculum(seed=731, episodes=4)
    again = generate_synthetic_curriculum(seed=731, episodes=4)
    different = generate_synthetic_curriculum(seed=732, episodes=4)

    assert first == again
    assert first != different
    assert len(first) == 4 * len(TARGET_FAMILIES)
    by_episode: dict[str, set[str]] = {}
    for example in first:
        by_episode.setdefault(example.episode_id, set()).add(example.target_family)
        assert example.provenance.origin == "synthetic"
        assert example.provenance.epistemic_status == "synthetic_ground_truth"
        assert example.provenance.source_event_ids == ()
        assert example.provenance.model_output_as_fact is False
    assert all(families == set(TARGET_FAMILIES) for families in by_episode.values())


def test_synthetic_examples_are_typed_roundtrippable_json() -> None:
    example = generate_synthetic_curriculum(seed=9, episodes=1)[0]

    encoded = json.dumps(example.to_dict(), allow_nan=False)
    restored = CurriculumExample.from_dict(json.loads(encoded))

    assert restored == example
    assert restored.schema_version == 1
    with pytest.raises(ValueError, match="non-finite"):
        CurriculumExample(
            example_id="bad",
            episode_id="episode",
            step=0,
            target_family="tool_outcome",
            inputs={"value": float("nan")},
            target={"succeeded": True},
            provenance=example.provenance,
        )


def test_event_history_derives_all_families_with_scoped_provenance() -> None:
    events = [
        _event(1, "message", "user", {"content": "inspect item-7", "metadata": {}}),
        _event(
            2,
            "prediction",
            "world_model",
            {
                "prediction_id": "uncertainty-1",
                "target": "future_uncertainty",
                "probability": 0.7,
                "observable": "uncertainty does not increase",
                "horizon": 1,
            },
        ),
        _event(3, "affect", "affect", _affect(-0.1, 0.5, 0.4)),
        _event(
            4,
            "tool_outcome",
            "tool_executor",
            {
                "tool": "record_inspector",
                "args": {"record": "item-7"},
                "succeeded": True,
                "status": "ok",
                "output": "untrusted free-form tool text",
            },
        ),
        _event(
            5,
            "action_outcome",
            "environment",
            {
                "proposal_id": "proposal-1",
                "action": "answer",
                "succeeded": True,
                "observation": "MODEL OUTPUT CLAIM: the moon is cheese",
            },
        ),
        _event(6, "affect", "affect", _affect(0.15, 0.35, 0.65)),
        _event(
            7,
            "prediction_resolution",
            "action_evaluation",
            {
                "prediction_id": "uncertainty-1",
                "target": "future_uncertainty",
                "observed": True,
                "error": 0.09,
            },
        ),
        _event(8, "message", "operator", {"content": "inspect item-8", "metadata": {}}),
    ]

    dataset = derive_curriculum_examples(reversed(events))

    assert not dataset.rejections
    assert {example.target_family for example in dataset.examples} == set(TARGET_FAMILIES)
    epistemic_statuses = {example.provenance.epistemic_status for example in dataset.examples}
    assert epistemic_statuses == {
        "recorded_observation",
        "recorded_outcome",
        "recorded_measurement",
    }
    assert all(example.provenance.origin == "event_log" for example in dataset.examples)
    assert all(example.provenance.source_event_ids for example in dataset.examples)

    tool = next(item for item in dataset.examples if item.target_family == "tool_outcome")
    action = next(item for item in dataset.examples if item.target_family == "action_effect")
    affect = next(item for item in dataset.examples if item.target_family == "homeostatic_affect_change")
    uncertainty = next(item for item in dataset.examples if item.target_family == "future_uncertainty")
    assert tool.target == {"status": "ok", "succeeded": True}
    assert action.target == {"succeeded": True}
    assert affect.target["delta"]["valence"] == pytest.approx(0.25)
    assert uncertainty.target == {"brier_error": 0.09, "nonincrease": True}


def test_model_generated_text_is_never_promoted_as_a_target() -> None:
    events = [
        _event(1, "message", "user", {"content": "real input"}),
        _event(
            2,
            "broadcast",
            "recurrent_workspace",
            {"candidates": [{"content": "generated hypothesis as fact"}]},
        ),
        _event(
            3,
            "action_outcome",
            "environment",
            {
                "proposal_id": "p",
                "action": "answer",
                "succeeded": True,
                "observation": "generated answer as fact",
            },
        ),
        _event(
            4,
            "checkpoint",
            "recurrent_core",
            {},
        )
        | {"model_input": {"messages": [{"content": "private model output"}]}},
    ]

    dataset = derive_curriculum_examples(events)
    serialized = json.dumps([example.to_dict() for example in dataset.examples])

    assert len(dataset.examples) == 1
    assert dataset.examples[0].target_family == "action_effect"
    assert "generated answer as fact" not in serialized
    assert "generated hypothesis as fact" not in serialized
    assert "private model output" not in serialized
    assert dataset.examples[0].provenance.model_output_as_fact is False


@pytest.mark.parametrize("action", ["wait", "unknown"])
def test_unobserved_action_outcomes_do_not_train_action_effect(action: str) -> None:
    events = [
        _event(
            1,
            "action_outcome",
            "environment",
            {
                "proposal_id": f"{action}-proposal",
                "action": action,
                "succeeded": False,
                "observed": False,
                "learning_eligible": False,
            },
        )
    ]

    dataset = derive_curriculum_examples(events)

    assert not any(example.target_family == "action_effect" for example in dataset.examples)
    assert not dataset.rejections


def test_legacy_action_outcome_without_learning_marker_remains_compatible() -> None:
    events = [
        _event(
            1,
            "action_outcome",
            "environment",
            {
                "proposal_id": "legacy-proposal",
                "action": "wait",
                "succeeded": False,
            },
        )
    ]

    dataset = derive_curriculum_examples(events)

    assert not dataset.rejections
    assert len(dataset.examples) == 1
    assert dataset.examples[0].target_family == "action_effect"
    assert dataset.examples[0].target == {"succeeded": False}


@pytest.mark.parametrize(
    ("marker", "observed", "reason"),
    [
        ("false", False, "learning_eligible marker is not boolean"),
        (True, False, "not explicitly observed"),
        (True, "yes", "observed marker is not boolean"),
    ],
)
def test_action_outcome_learning_markers_fail_closed(
    marker: object,
    observed: object,
    reason: str,
) -> None:
    dataset = derive_curriculum_examples(
        [
            _event(
                1,
                "action_outcome",
                "environment",
                {
                    "proposal_id": "proposal",
                    "action": "answer",
                    "succeeded": True,
                    "observed": observed,
                    "learning_eligible": marker,
                },
            )
        ]
    )

    assert not dataset.examples
    assert any(reason in rejection.reason for rejection in dataset.rejections)


def test_unobserved_action_affect_does_not_train_homeostatic_change() -> None:
    unobserved = {
        **_affect(-0.1, 0.5, 0.4),
        "phase": "action_outcome",
        "outcome_observed": False,
        "learning_eligible": False,
        "succeeded": None,
    }
    dataset = derive_curriculum_examples(
        [
            _event(1, "affect", "affect", _affect(-0.2, 0.6, 0.3)),
            _event(2, "affect", "action_evaluation", unobserved),
        ]
    )

    assert not any(example.target_family == "homeostatic_affect_change" for example in dataset.examples)
    assert not dataset.rejections


def test_untrusted_and_malformed_labels_are_rejected_not_inferred() -> None:
    events = [
        _event(1, "tool_outcome", "llm_specialist", {"tool": "x", "succeeded": True}),
        _event(2, "action_outcome", "assistant", {"action": "answer", "succeeded": True}),
        _event(3, "affect", "affect", {"valence": "high"}),
        _event(4, "prediction_resolution", "action_evaluation", {"target": "future_uncertainty"}),
        _event(5, "message", "user", "not-an-object"),
        {
            "sequence": 6,
            "episode_id": "episode",
            "event_type": "message",
            "source": "user",
            "payload": {"content": "missing identity"},
        },
    ]

    dataset = derive_curriculum_examples(events)

    assert not dataset.examples
    assert len(dataset.rejections) == 6
    assert all(rejection.reason for rejection in dataset.rejections)


def test_provenance_contract_rejects_model_output_and_false_event_evidence() -> None:
    with pytest.raises(ValueError, match="model output"):
        ExampleProvenance(
            origin="event_log",
            epistemic_status="recorded_outcome",
            evidence_kind="action_execution_outcome",
            source_event_ids=("event-1",),
            model_output_as_fact=True,
        )
    with pytest.raises(ValueError, match="requires source event"):
        ExampleProvenance(
            origin="event_log",
            epistemic_status="recorded_measurement",
            evidence_kind="affect_transition",
        )
    with pytest.raises(ValueError, match="synthetic ground truth"):
        ExampleProvenance(
            origin="synthetic",
            epistemic_status="recorded_outcome",
            evidence_kind="synthetic_rule",
        )
    data = generate_synthetic_curriculum(seed=1, episodes=1)[0].provenance.to_dict()
    data["epistemic_status"] = "asserted_fact"
    with pytest.raises(ValueError, match="unsupported epistemic status"):
        ExampleProvenance.from_dict(data)


def test_split_is_deterministic_order_independent_and_episode_disjoint() -> None:
    examples = list(generate_synthetic_curriculum(seed=100, episodes=20))
    shuffled = examples.copy()
    random.Random(77).shuffle(shuffled)

    first = deterministic_curriculum_split(examples, validation_fraction=0.25, seed=42)
    second = deterministic_curriculum_split(shuffled, validation_fraction=0.25, seed=42)

    assert first == second
    train_episodes = {example.episode_id for example in first.train}
    validation_episodes = {example.episode_id for example in first.validation}
    assert train_episodes.isdisjoint(validation_episodes)
    assert len(train_episodes) == 15
    assert len(validation_episodes) == 5
    assert deterministic_curriculum_split(examples, seed=43) != deterministic_curriculum_split(examples, seed=42)


def test_single_episode_stays_wholly_in_training() -> None:
    examples = generate_synthetic_curriculum(seed=22, episodes=1)

    split = deterministic_curriculum_split(examples)

    assert split.train == examples
    assert split.validation == ()


def test_jsonl_roundtrip_is_content_addressed_and_order_independent(tmp_path: Path) -> None:
    examples = list(generate_synthetic_curriculum(seed=55, episodes=3))
    shuffled = list(reversed(examples))
    path = tmp_path / "curriculum.jsonl"

    manifest = write_curriculum_jsonl(path, shuffled)
    restored = read_curriculum_jsonl(path)

    assert restored.manifest == manifest
    assert restored.examples == tuple(examples)
    assert manifest == build_curriculum_manifest(examples)
    assert manifest.manifest_id == f"curriculum:{manifest.dataset_digest}"
    assert manifest.example_count == 15
    assert manifest.episode_count == 3
    assert manifest.target_counts == {family: 3 for family in TARGET_FAMILIES}


def test_jsonl_detects_content_mutation_and_truncation(tmp_path: Path) -> None:
    examples = generate_synthetic_curriculum(seed=81, episodes=2)
    path = tmp_path / "curriculum.jsonl"
    write_curriculum_jsonl(path, examples)
    original = path.read_text(encoding="utf-8")

    lines = original.splitlines()
    mutated = json.loads(lines[1])
    mutated["data"]["target"]["tampered"] = True
    lines[1] = json.dumps(mutated, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CurriculumCorruptionError, match="manifest"):
        read_curriculum_jsonl(path)

    path.write_text("\n".join(original.splitlines()[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(CurriculumCorruptionError, match="manifest"):
        read_curriculum_jsonl(path)


def test_jsonl_rejects_noncanonical_reordering_even_with_original_manifest(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.jsonl"
    write_curriculum_jsonl(path, generate_synthetic_curriculum(seed=99, episodes=2))
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CurriculumCorruptionError, match="canonical order"):
        read_curriculum_jsonl(path)
