from __future__ import annotations

from dataclasses import replace

import pytest

from conscio.v3.action_competition import (
    ActionCandidate,
    AffectSnapshot,
    CompetitionContext,
    ConstraintDisposition,
    LesionMask,
    NeedSnapshot,
    PredictionSignal,
    UpstreamIntention,
    compete,
)


def _prediction(
    target: str,
    calibrated: float,
    *,
    raw: float | None = None,
    available: bool = True,
    cycle: int = 7,
    basis: str = "broadcast-final",
    adapter: str = "adapter-digest",
    prediction_id: str | None = None,
) -> PredictionSignal:
    return PredictionSignal(
        target=target,
        available=available,
        source="world_model",
        prediction_id=prediction_id or f"prediction-{target}",
        basis_broadcast_id=basis,
        cycle=cycle,
        raw_probability=calibrated if raw is None else raw,
        calibrated_probability=calibrated,
        adapter_digest=adapter,
    )


def _context(**changes: object) -> CompetitionContext:
    values: dict[str, object] = {
        "final_cycle": 7,
        "final_broadcast_id": "broadcast-final",
        "runtime_identity": "sha256:" + ("a" * 64),
        "adapter_digest": "adapter-digest",
        "language_manifest_digest": "manifest-digest",
        "language_response_digest": "response-digest",
        "predictions": (
            _prediction("action_effect", 0.65),
            _prediction("tool_outcome", 0.70),
            _prediction("next_observation", 0.55),
        ),
        "affect": AffectSnapshot(
            available=True,
            valence=0.1,
            arousal=0.2,
            controllability=0.8,
            needs=NeedSnapshot(epistemic_coherence=0.4, competence=0.3),
            source="affect_specialist",
            basis_broadcast_id="broadcast-final",
        ),
        "upstream_intention": UpstreamIntention(
            available=True,
            action="act",
            proposal_id="upstream-1",
            specialist="planning",
            source="global_workspace",
            basis_broadcast_id="broadcast-final",
            cycle=7,
        ),
    }
    values.update(changes)
    return CompetitionContext(**values)  # type: ignore[arg-type]


def _score_projection(context: CompetitionContext) -> list[tuple[object, ...]]:
    result = compete(context, [ActionCandidate.tool("inspect", {"item": "x"}, risk=0.1)])
    return [
        (
            row.action_digest,
            row.eligible,
            row.total_points,
            row.prediction_points,
            row.need_points,
            row.alignment_points,
            row.risk_penalty_points,
            row.adjusted_risk,
        )
        for row in result.rankings
    ]


def test_replay_and_nested_argument_key_permutations_are_identical() -> None:
    first = ActionCandidate.tool(
        "inspect",
        {"z": [3, {"beta": 2, "alpha": 1}], "a": {"right": False, "left": True}},
        risk=0.1,
        capabilities=("filesystem_read", "external_network"),
    )
    permuted = ActionCandidate.tool(
        "inspect",
        {"a": {"left": True, "right": False}, "z": [3, {"alpha": 1, "beta": 2}]},
        risk=0.1,
        capabilities=("external_network", "filesystem_read"),
    )
    context = _context()
    permuted_context = replace(context, predictions=tuple(reversed(context.predictions)))

    decision = compete(context, [first])
    replay = compete(permuted_context, [permuted])

    assert decision.to_dict() == replay.to_dict()
    assert decision.context_digest == replay.context_digest


def test_ties_are_order_independent_and_duplicate_actions_merge_conservatively() -> None:
    context = _context(
        predictions=(),
        prediction_channel_available=False,
        affect=AffectSnapshot.unavailable(),
        upstream_intention=UpstreamIntention.unavailable(),
    )
    first = ActionCandidate.tool("alpha", {"x": 1}, provider_call_id="call-z")
    first_duplicate = ActionCandidate.tool(
        "alpha",
        {"x": 1},
        risk=0.2,
        provider_call_id="call-a",
        provider_confidence=0.99,
    )
    second = ActionCandidate.tool("beta", {"x": 1}, risk=0.2, provider_call_id="call-b")

    forward = compete(context, [first, second, first_duplicate])
    reverse = compete(context, [first_duplicate, second, first])

    assert forward.to_dict() == reverse.to_dict()
    assert len(forward.rankings) == 4  # two unique tools plus respond and wait
    alpha = next(row for row in forward.rankings if row.name == "alpha")
    assert alpha.effective_risk == 0.2
    assert alpha.provider_call_ids == ("call-a", "call-z")


@pytest.mark.parametrize(
    "build",
    [
        lambda: ActionCandidate.tool("bad", {}, risk=float("nan")),
        lambda: _prediction("tool_outcome", float("nan")),
        lambda: AffectSnapshot(available=False, valence=float("nan")),
        lambda: _context(risk_limit=float("nan")),
    ],
)
def test_non_finite_values_are_rejected(build: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        build()  # type: ignore[operator]


def test_lesioned_channels_are_ignored_even_when_their_contents_are_poisoned() -> None:
    lesions = LesionMask(prediction=True, affect=True, broadcast=True)
    poison_high = _context(
        lesions=lesions,
        predictions=(
            _prediction(
                "tool_outcome",
                1.0,
                cycle=999,
                basis="wrong-broadcast",
                adapter="wrong-adapter",
            ),
        ),
        affect=AffectSnapshot(
            available=True,
            valence=-1.0,
            arousal=1.0,
            controllability=0.0,
            needs=NeedSnapshot(epistemic_coherence=1.0, competence=1.0),
            source="poison",
            basis_broadcast_id="wrong-broadcast",
        ),
        upstream_intention=UpstreamIntention(
            available=True,
            action="wait",
            proposal_id="poison-high",
            specialist="planning",
            source="poison",
            basis_broadcast_id="wrong-broadcast",
            cycle=999,
        ),
    )
    poison_low = replace(
        poison_high,
        predictions=(
            _prediction(
                "tool_outcome",
                0.0,
                cycle=123,
                basis="other-wrong-broadcast",
                adapter="other-wrong-adapter",
            ),
        ),
        affect=AffectSnapshot(
            available=True,
            valence=1.0,
            arousal=0.0,
            controllability=1.0,
            needs=NeedSnapshot(epistemic_coherence=-1.0, competence=-1.0),
            source="different-poison",
            basis_broadcast_id="other-wrong-broadcast",
        ),
        upstream_intention=UpstreamIntention(
            available=True,
            action="act",
            proposal_id="poison-low",
            specialist="planning",
            source="different-poison",
            basis_broadcast_id="other-wrong-broadcast",
            cycle=123,
        ),
    )

    assert _score_projection(poison_high) == _score_projection(poison_low)


def test_unavailable_prediction_is_ignored_and_records_neutral_bootstrap_fallback() -> None:
    unavailable_low = _context(
        predictions=(
            _prediction(
                "tool_outcome",
                0.0,
                available=False,
                cycle=2,
                basis="stale",
                adapter="old-adapter",
            ),
        )
    )
    unavailable_high = replace(
        unavailable_low,
        predictions=(
            _prediction(
                "tool_outcome",
                1.0,
                available=False,
                cycle=999,
                basis="poison",
                adapter="poison",
            ),
        ),
    )
    tool = ActionCandidate.tool("inspect", {})

    low = next(row for row in compete(unavailable_low, [tool]).rankings if row.action_kind == "tool")
    high = next(row for row in compete(unavailable_high, [tool]).rankings if row.action_kind == "tool")

    assert low.prediction_points == high.prediction_points == 0
    assert low.prediction_neutral_fallback is high.prediction_neutral_fallback is True
    assert low.prediction_probability == high.prediction_probability == 0.5
    assert low.prediction_source == high.prediction_source == "neutral_fallback"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"cycle": 6}, "final_cycle"),
        ({"basis": "broadcast-old"}, "final_broadcast_id"),
        ({"adapter": "adapter-old"}, "adapter digest"),
    ],
)
def test_available_predictions_must_have_the_exact_final_cycle_basis(
    changes: dict[str, object],
    message: str,
) -> None:
    signal = _prediction("tool_outcome", 0.5, **changes)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        _context(predictions=(signal,))


def test_scoring_uses_calibrated_final_probability_not_raw_provider_value() -> None:
    tool = ActionCandidate.tool("inspect", {})
    raw_high = _context(predictions=(_prediction("tool_outcome", 0.75, raw=0.99),))
    raw_low = _context(predictions=(_prediction("tool_outcome", 0.75, raw=0.01),))

    high = next(row for row in compete(raw_high, [tool]).rankings if row.action_kind == "tool")
    low = next(row for row in compete(raw_low, [tool]).rankings if row.action_kind == "tool")

    assert high.prediction_points == low.prediction_points == 1_200
    assert high.prediction_basis_broadcast_id == "broadcast-final"
    assert high.prediction_id == "prediction-tool_outcome"


def test_provider_metadata_cannot_change_action_identity_or_score() -> None:
    context = _context(
        predictions=(_prediction("tool_outcome", 1.0),),
        affect=AffectSnapshot.unavailable(),
    )
    first = ActionCandidate.tool(
        "inspect",
        {"item": "x"},
        provider_call_id="provider-call-a",
        provider_rationale="The provider says this is wonderful.",
        provider_confidence=1.0,
    )
    second = ActionCandidate.tool(
        "inspect",
        {"item": "x"},
        provider_call_id="provider-call-z",
        provider_rationale="The provider says this is terrible.",
        provider_confidence=0.0,
    )

    left = compete(context, [first])
    right = compete(context, [second])
    left_score = next(row for row in left.rankings if row.action_kind == "tool")
    right_score = next(row for row in right.rankings if row.action_kind == "tool")

    assert left.selected_action_kind == right.selected_action_kind == "tool"
    assert left.selected_action_digest == right.selected_action_digest
    assert left_score.total_points == right_score.total_points
    assert left_score.action_digest == right_score.action_digest
    assert left.selected_provider_call_ids != right.selected_provider_call_ids


def test_hard_gates_dominate_a_higher_scoring_candidate() -> None:
    context = _context(
        predictions=(
            _prediction("tool_outcome", 1.0),
            _prediction("action_effect", 0.0),
        ),
        affect=AffectSnapshot(
            available=True,
            needs=NeedSnapshot(epistemic_coherence=1.0, competence=1.0),
            source="affect_specialist",
            basis_broadcast_id="broadcast-final",
        ),
    )
    blocked = ActionCandidate.tool(
        "dangerous",
        {},
        constraints=(ConstraintDisposition("operator-denied", satisfied=False),),
    )

    decision = compete(context, [blocked])
    blocked_score = next(row for row in decision.rankings if row.action_kind == "tool")

    assert blocked_score.total_points > decision.selected.total_points
    assert blocked_score.eligible is False
    assert blocked_score.ineligibility_reasons == ("constraint:operator-denied",)
    assert decision.selected.eligible is True
    assert decision.selected_action_kind != "tool"


def test_upstream_wait_is_an_independent_hard_gate() -> None:
    context = _context(
        upstream_intention=UpstreamIntention(
            available=True,
            action="wait",
            proposal_id="upstream-wait",
            specialist="planning",
            source="global_workspace",
            basis_broadcast_id="broadcast-final",
            cycle=7,
        )
    )

    result = compete(context, [ActionCandidate.tool("inspect", {})])

    assert result.selected_action_kind == "wait"
    assert all(
        "upstream_wait_gate" in row.ineligibility_reasons for row in result.rankings if row.action_kind != "wait"
    )
