from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from conscio.v3.language_specialist import (
    ChatAsyncAdapter,
    LanguageInterpretation,
    LanguageModelManifest,
    LanguageReasoning,
    LanguageResponse,
    LanguageSpecialist,
    LanguageToolProposal,
    MalformedLanguageResponse,
    ManifestCompatibilityError,
    SamplingPolicy,
)


class ScriptedClient:
    def __init__(self, *responses: dict[str, Any], mutate_request: bool = False) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.mutate_request = mutate_request

    async def chat_async(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        if self.mutate_request:
            messages[0]["content"] = "client mutation"
            kwargs["tools"] = [{"mutated": True}]
        return self.responses.pop(0)


def manifest(**overrides: Any) -> LanguageModelManifest:
    values: dict[str, Any] = {
        "provider": "local-openai",
        "endpoint_id": "gpu-cluster-a/v1",
        "model_id": "acme/lingua-8b-r2026-07-01",
        "model_revision": "8f3c18c62e7d4b09b7c18c4cd188e605a21f1c72",
        "sampling": SamplingPolicy(temperature=0.1, top_p=0.9, max_tokens=777, seed=19),
        "weight_digest": "a" * 64,
        "config_digest": "b" * 64,
    }
    values.update(overrides)
    return LanguageModelManifest(**values)


def specialist(client: ScriptedClient, model_manifest: LanguageModelManifest | None = None) -> LanguageSpecialist:
    model_manifest = model_manifest or manifest()
    return LanguageSpecialist.from_chat_async(
        client,
        manifest=model_manifest,
        provider=model_manifest.provider,
        endpoint_id=model_manifest.endpoint_id,
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("provider", "unknown"),
        ("endpoint_id", "default"),
        ("model_id", "<MODEL>"),
        ("model_revision", None),
        ("model_revision", "latest"),
        ("model_revision", "main"),
    ],
)
def test_primary_research_rejects_unpinned_and_placeholder_fields(field_name: str, value: object) -> None:
    with pytest.raises(ValueError, match="primary research requires a pinned"):
        manifest(**{field_name: value})


def test_comparison_manifest_can_record_an_unpinned_remote_baseline() -> None:
    baseline = manifest(
        provider="remote-comparison",
        endpoint_id="comparison-api",
        model_id="published-alias",
        model_revision=None,
        research_use="comparison_baseline",
        weight_digest=None,
        config_digest=None,
    )

    assert baseline.model_revision is None
    assert baseline.research_use == "comparison_baseline"


def test_manifest_has_canonical_strict_authenticated_serialization() -> None:
    original = manifest()
    payload = original.to_json()
    restored = LanguageModelManifest.from_json(payload, expected_digest=original.digest())

    assert restored == original
    assert payload == json.dumps(original.to_dict(), sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="not in canonical"):
        LanguageModelManifest.from_json("\n" + payload)

    tampered = json.loads(payload)
    tampered["model_revision"] = "4" * 40
    tampered_payload = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ManifestCompatibilityError, match="digest"):
        LanguageModelManifest.from_json(tampered_payload, expected_digest=original.digest())

    extra = json.loads(payload)
    extra["unrecorded"] = True
    with pytest.raises(ValueError, match="keys differ"):
        LanguageModelManifest.from_json(json.dumps(extra, sort_keys=True, separators=(",", ":")))


def test_restore_and_replay_require_exact_manifest_compatibility() -> None:
    current = manifest()
    current.assert_compatible(current.to_json(), digest=current.digest())
    changed_policy = manifest(sampling=SamplingPolicy(max_tokens=778))

    with pytest.raises(ManifestCompatibilityError, match="incompatible"):
        current.assert_compatible(changed_policy)
    with pytest.raises(ManifestCompatibilityError, match="digest"):
        current.assert_compatible(current.to_json(), digest="c" * 64)


def test_manifest_and_sampling_are_immutable_and_adapter_identity_is_checked() -> None:
    model_manifest = manifest()
    with pytest.raises(FrozenInstanceError):
        model_manifest.model_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        model_manifest.sampling.temperature = 1.0  # type: ignore[misc]

    client = ScriptedClient({"role": "assistant", "content": "unused"})
    with pytest.raises(ManifestCompatibilityError, match="provider/endpoint"):
        LanguageSpecialist(
            ChatAsyncAdapter(client=client, provider="other-provider", endpoint_id=model_manifest.endpoint_id),
            model_manifest,
        )


def test_typed_outputs_never_promote_generated_text_to_fact() -> None:
    client = ScriptedClient(
        {"role": "assistant", "content": "Likely a request for a forecast."},
        {"role": "assistant", "content": "Compare the two available paths."},
        {"role": "assistant", "content": "I remain uncertain."},
    )
    boundary = specialist(client)

    interpretation = asyncio.run(boundary.interpret([{"role": "user", "content": "What does this mean?"}]))
    reasoning = asyncio.run(boundary.reason([{"role": "user", "content": "Choose."}]))
    response = asyncio.run(
        boundary.respond(
            [{"role": "user", "content": "How certain are you?"}],
            epistemic_status="self_report",
        )
    )

    assert isinstance(interpretation, LanguageInterpretation)
    assert isinstance(reasoning, LanguageReasoning)
    assert isinstance(response, LanguageResponse)
    assert interpretation.epistemic_status == "idea"
    assert reasoning.epistemic_status == "idea"
    assert response.epistemic_status == "self_report"
    assert {interpretation.epistemic_status, reasoning.epistemic_status, response.epistemic_status} <= {
        "idea",
        "self_report",
    }
    with pytest.raises(ValueError, match="status"):
        asyncio.run(boundary.respond([], epistemic_status="observation"))  # type: ignore[arg-type]


def test_trace_is_the_exact_client_request_and_isolated_from_mutation() -> None:
    client = ScriptedClient(
        {"role": "assistant", "content": "draft"},
        mutate_request=True,
    )
    boundary = specialist(client)
    messages = [{"role": "user", "content": "original", "metadata": {"sequence": 3}}]
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "read data",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = asyncio.run(boundary.respond(messages, tool_schemas=schemas))
    sent_messages, sent_kwargs = client.calls[0]

    assert messages[0]["content"] == "original"
    assert schemas[0]["function"]["name"] == "lookup"
    assert result.trace.request() == {"messages": sent_messages, **sent_kwargs}
    assert result.trace.manifest_digest == boundary.manifest_digest
    assert result.trace.request()["messages"][0]["content"] == "original"
    assert result.trace.request()["tools"][0]["function"]["name"] == "lookup"
    first_copy = result.trace.request()
    first_copy["messages"][0]["content"] = "trace reader mutation"
    assert result.trace.request()["messages"][0]["content"] == "original"


def test_tool_calls_are_inert_typed_proposals_with_copy_isolation() -> None:
    execution_count = 0

    def dangerous_callable() -> None:
        nonlocal execution_count
        execution_count += 1

    client = ScriptedClient(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "dangerous", "arguments": '{"path":"/tmp/value","n":2}'},
                }
            ],
        }
    )
    boundary = specialist(client)
    result = asyncio.run(
        boundary.respond(
            [{"role": "user", "content": "propose it"}],
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "dangerous",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
    )

    assert execution_count == 0
    assert not hasattr(boundary, "execute")
    assert not hasattr(boundary, "chat_async")
    assert not hasattr(boundary, "set_goal")
    assert result.text == ""
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert isinstance(proposal, LanguageToolProposal)
    assert proposal.name == "dangerous"
    assert proposal.epistemic_status == "idea"
    assert proposal.arguments == {"n": 2, "path": "/tmp/value"}
    arguments_copy = proposal.arguments
    arguments_copy["path"] = "changed"
    assert proposal.arguments["path"] == "/tmp/value"

    with pytest.raises(ValueError, match="finite JSON values"):
        asyncio.run(
            boundary.respond(
                [{"role": "user", "content": "bad schema"}],
                tool_schemas=[
                    {
                        "type": "function",
                        "function": {"name": "dangerous"},
                        "executor": dangerous_callable,
                    }
                ],
            )
        )
    assert execution_count == 0


@pytest.mark.parametrize(
    "response",
    [
        {"role": "assistant", "content": "", "tool_calls": "not-a-list"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "system", "content": "wrong author"},
        {"role": "assistant", "content": "ok", "unrecorded": True},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "x", "arguments": "[]"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "not-advertised", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "x", "arguments": '{"a":1,"a":2}'},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "x", "arguments": '{"n":NaN}'},
                }
            ],
        },
    ],
)
def test_malformed_tool_and_response_shapes_are_rejected(response: dict[str, Any]) -> None:
    client = ScriptedClient(response)
    boundary = specialist(client)

    with pytest.raises(MalformedLanguageResponse) as raised:
        asyncio.run(
            boundary.respond(
                [{"role": "user", "content": "test"}],
                tool_schemas=[{"type": "function", "function": {"name": "x"}}],
            )
        )
    assert raised.value.trace is not None
    assert raised.value.trace.manifest_digest == boundary.manifest_digest


def test_tool_calls_without_supplied_schemas_are_rejected() -> None:
    boundary = specialist(
        ScriptedClient(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    }
                ],
            }
        )
    )

    with pytest.raises(MalformedLanguageResponse, match="no tool schemas"):
        asyncio.run(boundary.respond([{"role": "user", "content": "test"}]))


def test_specialist_replay_guard_checks_serialized_manifest_and_digest() -> None:
    current = manifest()
    boundary = specialist(ScriptedClient({"role": "assistant", "content": "unused"}), current)

    boundary.assert_replay_compatible(current.to_json(), current.digest())
    incompatible = manifest(model_revision="3" * 40)
    with pytest.raises(ManifestCompatibilityError):
        boundary.assert_replay_compatible(incompatible.to_json(), incompatible.digest())
