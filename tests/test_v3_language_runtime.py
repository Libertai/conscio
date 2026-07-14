from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from conscio.config import ResearchConfig, ServiceConfig
from conscio.core.cognition import InputEvent
from conscio.memory.store import MemoryStore
from conscio.service import ConscioService
from conscio.tools.registry import ToolRegistry
from conscio.v3.language_bridge import LanguageSpecialistToolLoopBridge
from conscio.v3.language_specialist import (
    LanguageModelManifest,
    LanguageSpecialist,
    SamplingPolicy,
)
from conscio.v3.runtime import V3CognitiveRuntime


class PinnedClient:
    model = "open-model-7b"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    async def chat_async(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append((messages, dict(kwargs)))
        return {"role": "assistant", "content": "bounded answer"}


async def _tool_result() -> dict[str, object]:
    return {"output": "not called", "error": False}


def make_bridge(client: PinnedClient) -> LanguageSpecialistToolLoopBridge:
    manifest = LanguageModelManifest(
        provider="local-openai-compatible",
        endpoint_id="research-node-a",
        model_id=client.model,
        model_revision="weights-2026-07-14",
        sampling=SamplingPolicy(temperature=0.4, max_tokens=2400, seed=19),
        weight_digest="1" * 64,
        config_digest="2" * 64,
    )
    specialist = LanguageSpecialist.from_chat_async(
        client,
        manifest=manifest,
        provider=manifest.provider,
        endpoint_id=manifest.endpoint_id,
    )
    return LanguageSpecialistToolLoopBridge(specialist)


def test_runtime_persists_pinned_language_calls_and_strict_workspace() -> None:
    asyncio.run(_runtime_persists_pinned_language_calls_and_strict_workspace())


async def _runtime_persists_pinned_language_calls_and_strict_workspace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        client = PinnedClient()
        bridge = make_bridge(client)
        memory = MemoryStore(db_path=os.path.join(tmp, "language-runtime.db"))
        tools = ToolRegistry()
        tools.register("workspace_safe", _tool_result, "Visible non-memory tool.")
        tools.register(
            "private_memory_read",
            _tool_result,
            "Direct private-memory read.",
            capabilities={"memory_read"},
        )
        tools.register(
            "private_memory_write",
            _tool_result,
            "Direct private-memory write.",
            capabilities={"memory_write"},
        )
        runtime = V3CognitiveRuntime(
            llm=bridge,
            memory=memory,
            tools=tools,
            cognitive_cycles=2,
            strict_recurrent_workspace=True,
        )
        await runtime.initialize()
        try:
            await memory.add_fact(
                "legacy retrieval must not bypass the recurrent workspace",
                origin="user",
            )
            result = await runtime.run_episode(InputEvent(content="Give me a bounded answer", source="user"))
        finally:
            await runtime.close()

    assert runtime.modules == []
    assert runtime.prompt_assembler.memory_enabled is False
    assert runtime.prompt_assembler.self_state_enabled is False
    assert "legacy retrieval must not bypass" not in result.model_context
    assert "RECENT_EPISODES\nnone" in result.model_context
    assert "RELEVANT_MEMORY\nnone" in result.model_context
    assert "self:" not in result.model_context
    advertised_tools = {schema["function"]["name"] for schema in client.calls[0][1]["tools"]}
    assert "workspace_safe" in advertised_tools
    assert "private_memory_read" not in advertised_tools
    assert "private_memory_write" not in advertised_tools
    # The parent registry remains intact for private recurrent components; the
    # language-facing executor receives only the narrowed view.
    assert {"private_memory_read", "private_memory_write"} <= set(runtime.tools.list_tools())

    event_types = [event["event_type"] for event in result.causal_trace]
    assert event_types.count("language_specialist_call") == 1
    assert event_types.index("language_specialist_call") < event_types.index("action_outcome")
    language_event = next(event for event in result.causal_trace if event["event_type"] == "language_specialist_call")
    assert language_event["payload"]["manifest_digest"] == bridge.manifest_digest
    assert result.exact_model_inputs[0]["model"] == client.model
    assert result.exact_model_inputs[0]["seed"] == 19
    checkpoint = result.causal_trace[-1]
    assert checkpoint["event_type"] == "checkpoint"
    assert checkpoint["model_input"]["language_calls"][0] == language_event["payload"]
    assert checkpoint["model_input"]["language_manifests"] == [bridge.manifest]


def test_service_builds_distinct_pinned_chat_and_autonomous_roles(
    tmp_path: Path,
) -> None:
    asyncio.run(_service_builds_distinct_pinned_chat_and_autonomous_roles(tmp_path))


async def _service_builds_distinct_pinned_chat_and_autonomous_roles(
    tmp_path: Path,
) -> None:
    config = ServiceConfig(
        home=tmp_path / "home",
        autonomous=False,
        llm_base_url="http://127.0.0.1:8080/v1",
        research=ResearchConfig(
            strict_recurrent_workspace=True,
            require_pinned_language_model=True,
            language_provider="local-openai-compatible",
            language_endpoint_id="research-node-a",
            language_model_revision="weights-2026-07-14",
            language_weight_digest="3" * 64,
            language_config_digest="4" * 64,
            language_seed=23,
        ),
    )

    service = ConscioService(config)
    try:
        assert len(service.language_bridges) == 2
        chat, autonomous = service.language_bridges
        assert service.runtime.chat_strategy.llm is chat
        assert service.runtime.autonomous_strategy.llm is autonomous
        assert chat.manifest_digest != autonomous.manifest_digest
        assert service.runtime.language_manifest_digests == tuple(
            sorted((chat.manifest_digest, autonomous.manifest_digest))
        )
        assert service.llm_fast is None
        assert service._goal_review_interval == 0
        assert service.consolidation.llm is None
        assert service.runtime.modules == []
    finally:
        await service.memory.close()


def test_pinned_research_config_requires_complete_digests_and_no_auxiliary_llm() -> None:
    research = ResearchConfig(
        require_pinned_language_model=True,
        language_provider="local-openai-compatible",
        language_endpoint_id="research-node-a",
        language_model_revision="weights-2026-07-14",
        language_weight_digest="5" * 64,
        language_config_digest="",
    )
    config = ServiceConfig(
        llm_base_url="http://127.0.0.1:8080/v1",
        research=research,
    )
    with pytest.raises(ValueError, match="language_config_digest"):
        config.validate()

    config.research = replace(research, language_config_digest="6" * 64)
    config.ablation = replace(config.ablation, llm_appraisal=True)
    with pytest.raises(ValueError, match="llm_appraisal"):
        config.validate()
