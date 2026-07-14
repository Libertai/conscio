from __future__ import annotations

import unittest

import pytest

from conscio.v3.language_bridge import LanguageSpecialistToolLoopBridge, trace_to_dict
from conscio.v3.language_specialist import (
    LanguageModelManifest,
    LanguageSpecialist,
    ManifestCompatibilityError,
    SamplingPolicy,
)


class ProposalClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    async def chat_async(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append((messages, dict(kwargs)))
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": '{"item":"x"}'},
                }
            ],
        }


def _bridge(client: ProposalClient) -> LanguageSpecialistToolLoopBridge:
    manifest = LanguageModelManifest(
        provider="local-openai-compatible",
        endpoint_id="research-node-a",
        model_id="open-model-7b",
        model_revision="weights-2026-07-14",
        weight_digest="1" * 64,
        sampling=SamplingPolicy(temperature=0.2, max_tokens=512, seed=9),
    )
    specialist = LanguageSpecialist.from_chat_async(
        client,
        manifest=manifest,
        provider=manifest.provider,
        endpoint_id=manifest.endpoint_id,
    )
    return LanguageSpecialistToolLoopBridge(specialist)


class LanguageBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_bridge_returns_only_inert_proposal_shape_and_authenticated_trace(
        self,
    ) -> None:
        client = ProposalClient()
        bridge = _bridge(client)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "inspect",
                    "description": "inspect",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = await bridge.chat_async(
            [{"role": "user", "content": "inspect x"}],
            tools=tools,
            temperature=0.2,
            max_tokens=512,
        )

        assert result["tool_calls"][0]["function"]["name"] == "inspect"
        assert len(client.calls) == 1
        traces = bridge.drain_traces()
        assert len(traces) == 1
        structured = trace_to_dict(traces[0])
        assert structured["manifest_digest"] == bridge.manifest_digest
        assert structured["request"]["model"] == bridge.model
        assert bridge.drain_traces() == ()

    async def test_bridge_rejects_sampling_or_model_override_before_client_call(
        self,
    ) -> None:
        client = ProposalClient()
        bridge = _bridge(client)

        with pytest.raises(ManifestCompatibilityError, match="temperature"):
            await bridge.chat_async([], temperature=0.3, max_tokens=512)
        with pytest.raises(ManifestCompatibilityError, match="model"):
            await bridge.chat_async([], model="rolling-model")

        assert not client.calls
