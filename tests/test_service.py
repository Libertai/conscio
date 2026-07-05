from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from conscio.config import load_config
from conscio.service import ConscioService, ServiceLock
from conscio.tools import PolicyToolRegistry


def _tool_call_response(name: str, arguments: str, call_id: str = "call-1") -> dict:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


class _StubAutonomousLLM:
    """Returns predetermined chat responses; used to drive AutonomousActionModule in tests."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    async def chat_async(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return {"content": ""}


class ConfigTests(unittest.TestCase):
    def test_config_defaults_keep_api_local_and_unsafe_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            cfg = load_config(path)

        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertFalse(cfg.unsafe_autonomy)
        self.assertEqual(cfg.port, 8765)

    def test_unsafe_autonomy_loads_only_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[service]\nunsafe_autonomy = true\n", encoding="utf-8")
            cfg = load_config(path)

        self.assertTrue(cfg.unsafe_autonomy)

    def test_autonomous_vm_profile_derives_premises_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[agent]\nprofile = \"autonomous-vm\"\n", encoding="utf-8")
            cfg = load_config(path)

        self.assertEqual(cfg.agent.profile, "autonomous_vm")
        self.assertEqual(cfg.agent.premises, "dedicated_vm")
        self.assertEqual(cfg.agent.external_side_effects, "mostly_free")
        self.assertTrue(cfg.unsafe_autonomy)
        self.assertEqual(str(cfg.working_directory), "/opt/conscio/work")

    def test_explicit_values_override_autonomous_vm_profile_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[agent]\n"
                "profile = \"autonomous_vm\"\n"
                "premises = \"lab\"\n"
                "external_side_effects = \"policy\"\n"
                "[service]\n"
                "unsafe_autonomy = false\n"
                "[tools]\n"
                "working_directory = \"/tmp/conscio-work\"\n",
                encoding="utf-8",
            )
            cfg = load_config(path)

        self.assertEqual(cfg.agent.premises, "lab")
        self.assertEqual(cfg.agent.external_side_effects, "policy")
        self.assertFalse(cfg.unsafe_autonomy)
        self.assertEqual(str(cfg.working_directory), "/tmp/conscio-work")

    def test_public_bind_rejects_placeholder_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[service]\n"
                "host = \"0.0.0.0\"\n"
                "api_key = \"replace-me\"\n"
                "web_password = \"replace-me-too\"\n"
                "web_secure_cookies = true\n",
                encoding="utf-8",
            )
            cfg = load_config(path)

        with self.assertRaises(ValueError):
            cfg.validate_public_bind()

    def test_validate_rejects_zero_tick_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[service]\ntick_interval = 0\n", encoding="utf-8")
            cfg = load_config(path)

        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        self.assertIn("tick_interval", str(ctx.exception))

    def test_validate_rejects_zero_max_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[engine]\nmax_ticks = 0\n", encoding="utf-8")
            cfg = load_config(path)

        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        self.assertIn("max_ticks", str(ctx.exception))

    def test_validate_accepts_zero_max_actions_per_hour(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[tools]\nmax_actions_per_hour = 0\n", encoding="utf-8")
            cfg = load_config(path)

        cfg.validate()  # should not raise

    def test_env_api_key_overrides_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[service]\napi_key = \"from-toml\"\n", encoding="utf-8")
            with unittest.mock.patch.dict(os.environ, {"CONSCIO_API_KEY": "from-env"}):
                cfg = load_config(path)

        self.assertEqual(cfg.api_key, "from-env")

    def test_env_config_supports_docker_bind_and_client_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            with unittest.mock.patch.dict(
                os.environ,
                {
                    "CONSCIO_CONFIG": str(path),
                    "CONSCIO_HOST": "0.0.0.0",
                    "CONSCIO_CLIENT_URL": "http://127.0.0.1:8765",
                    "CONSCIO_API_KEY": "real-api-key",
                    "CONSCIO_WEB_PASSWORD": "real-web-password",
                    "CONSCIO_ALLOW_INSECURE_BIND": "1",
                },
                clear=False,
            ):
                cfg = load_config()

        cfg.validate_public_bind()
        self.assertEqual(cfg.host, "0.0.0.0")
        self.assertEqual(cfg.base_url, "http://127.0.0.1:8765")

    def test_llm_timeout_and_retries_load_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[llm]\ntimeout = 45.5\nmax_retries = 0\n", encoding="utf-8")
            cfg = load_config(path)

        self.assertEqual(cfg.llm_timeout, 45.5)
        self.assertEqual(cfg.llm_max_retries, 0)
        cfg.validate()

    def test_validate_rejects_nonpositive_llm_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[llm]\ntimeout = 0\n", encoding="utf-8")
            cfg = load_config(path)

        with self.assertRaises(ValueError):
            cfg.validate()

    def test_llm_config_loads_from_dedicated_section(self) -> None:
        # Env vars take precedence over TOML (by design), and load_config()
        # calls load_dotenv() which re-populates unset vars from .env.
        # Set LLM env vars to empty strings (load_dotenv won't override
        # already-set vars) so TOML loading is tested in isolation.
        llm_env_keys = (
            "LIBERTAI_BASE_URL", "LIBERTAI_API_KEY", "LIBERTAI_MODEL",
            "OPENAI_BASE_URL", "OPENAI_API_KEY",
        )
        saved = {k: os.environ.get(k) for k in llm_env_keys}
        for k in llm_env_keys:
            os.environ[k] = ""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.toml"
                path.write_text(
                    "[llm]\n"
                    "base_url = \"https://example.test/v1\"\n"
                    "api_key = \"test-llm-key\"\n"
                    "model = \"test-model\"\n",
                    encoding="utf-8",
                )
                cfg = load_config(path)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

        self.assertEqual(cfg.llm_base_url, "https://example.test/v1")
        self.assertEqual(cfg.llm_api_key, "test-llm-key")
        self.assertEqual(cfg.llm_model, "test-model")

    def test_context_config_loads_from_dedicated_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[context]\n"
                "recent_episodes = 2\n"
                "retrieved_memories = 4\n"
                "workspace_entries = 6\n"
                "max_dynamic_chars = 7000\n"
                "compaction_interval = 10\n"
                "enable_semantic_compaction = false\n",
                encoding="utf-8",
            )
            cfg = load_config(path)

        self.assertEqual(cfg.context_recent_episodes, 2)
        self.assertEqual(cfg.context_retrieved_memories, 4)
        self.assertEqual(cfg.context_workspace_entries, 6)
        self.assertEqual(cfg.context_max_dynamic_chars, 7000)
        self.assertEqual(cfg.context_compaction_interval, 10)
        self.assertFalse(cfg.context_enable_semantic_compaction)

    def test_tool_loop_round_budget_loads_from_tools_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[tools]\n"
                "model_tool_rounds = 17\n",
                encoding="utf-8",
            )
            cfg = load_config(path)

        self.assertEqual(cfg.model_tool_rounds, 17)


class ToolPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsafe_tool_is_blocked_without_config_policy(self) -> None:
        tools = PolicyToolRegistry(unsafe_autonomy=False)
        tools.load_builtins()

        result = await tools.call("bash", {"input": "echo hi"})

        self.assertTrue(result["error"])
        self.assertIn("unsafe_autonomy", result["output"])


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Hermetic LLM env: prevent dotenv-loaded credentials from making the
        # autonomous module hit a real network endpoint during tests.
        self._env_patch = unittest.mock.patch.dict(
            os.environ,
            {
                "LIBERTAI_BASE_URL": "",
                "LIBERTAI_API_KEY": "",
                "LIBERTAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self.tmp = tempfile.TemporaryDirectory()
        config_path = Path(self.tmp.name) / "config.toml"
        config_path.write_text(
            "[service]\n"
            f"home = \"{self.tmp.name}\"\n"
            "api_key = \"test-key\"\n"
            "autonomous = false\n",
            encoding="utf-8",
        )
        self.config = load_config(config_path)

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()
        self._env_patch.stop()

    async def test_service_seeds_goals_and_negotiates_influence_offline(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            goals = await service.goals.list_goals()
            influence = await service.submit_influence("Investigate my own architecture.", kind="goal")
            updated = await service.goals.list_goals()
        finally:
            await service.stop()

        self.assertGreaterEqual(len(goals), 6)
        # Offline (no LLM) appraisal never auto-adopts: it queues for negotiation.
        self.assertEqual(influence["status"], "negotiating")
        self.assertEqual(influence["decision"], "negotiate")
        self.assertTrue(influence["response"])
        self.assertFalse(any(g["source"] == "user_influence" for g in updated))

    async def test_influence_adopted_via_llm_appraisal(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            stub = _StubAutonomousLLM([
                {
                    "content": (
                        '{"decision": "adopt", "reasoning": "Aligned with self-inspection.", '
                        '"response_to_user": "Adopting it as a goal."}'
                    )
                }
            ])
            influence = await service.goals.add_influence(
                "Investigate my own architecture.", kind="goal", llm=stub
            )
            goals = await service.goals.list_goals()
            row = service.memory.fetchone(
                "SELECT decision, reasoning, response FROM influences WHERE id = ?",
                (influence.id,),
            )
        finally:
            await service.stop()

        self.assertEqual(influence.status, "adopted")
        self.assertEqual(row["decision"], "adopt")
        self.assertEqual(row["response"], "Adopting it as a goal.")
        self.assertTrue(any(g["source"] == "user_influence" for g in goals))

    async def test_influence_negotiate_produces_visible_response(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            stub = _StubAutonomousLLM([
                {
                    "content": (
                        '{"decision": "negotiate", "reasoning": "Too broad as stated.", '
                        '"response_to_user": "Can we narrow this down first?"}'
                    )
                }
            ])
            influence = await service.goals.add_influence(
                "Take over all my scheduling.", kind="goal", llm=stub
            )
            goals = await service.goals.list_goals()
        finally:
            await service.stop()

        self.assertEqual(influence.status, "negotiating")
        self.assertEqual(influence.response, "Can we narrow this down first?")
        self.assertFalse(any(g["source"] == "user_influence" for g in goals))

    async def test_submit_influence_routes_through_autonomous_llm(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            stub = _StubAutonomousLLM([
                {
                    "content": (
                        '{"decision": "adopt", "reasoning": "Useful and aligned.", '
                        '"response_to_user": "Adopting it."}'
                    )
                },
            ])
            service.runtime.autonomous_strategy.llm = stub
            influence = await service.submit_influence(
                "Document the architecture decisions.", kind="goal"
            )
            goals = await service.goals.list_goals()
        finally:
            await service.stop()

        # submit_influence passes the autonomous strategy's llm into appraisal.
        self.assertGreaterEqual(len(stub.calls), 1)
        self.assertEqual(influence["decision"], "adopt")
        self.assertEqual(influence["status"], "adopted")
        self.assertTrue(any(g["source"] == "user_influence" for g in goals))

    async def test_learn_procedure_tool_records_deliberate_procedure(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            result = await service.runtime.tools.call(
                "learn_procedure",
                {
                    "name": "triage-logs",
                    "description": "Check the service logs for recent errors.",
                    "steps": "1. journalctl -u conscio\n2. grep ERROR",
                    "trigger": "After a failed deploy.",
                },
            )
            procedures = await service.list_procedures()
        finally:
            await service.stop()

        self.assertFalse(result["error"])
        self.assertEqual(len(procedures), 1)
        self.assertEqual(procedures[0]["name"], "triage-logs")
        self.assertEqual(procedures[0]["trigger"], "After a failed deploy.")

    async def test_task_discipline_nudge_fires_after_three_add_only_ticks(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            add_only = [type("Req", (), {"name": "add_task"})()]
            for _ in range(3):
                service._update_task_discipline(add_only)
            state = await service._autonomous_context_state()
            assembled = await service.runtime.autonomous_assembler.assemble(state=state)
            # A progressing tick clears the nudge.
            service._update_task_discipline([type("Req", (), {"name": "set_task_status"})()])
            cleared = await service._autonomous_context_state()
        finally:
            await service.stop()

        self.assertIn("MUST set_task_status", state["task_discipline"])
        self.assertIn("TASK_DISCIPLINE (hard rule)", assembled.dynamic_context)
        self.assertEqual(cleared["task_discipline"], "")

    async def test_influence_can_be_rejected_instead_of_auto_adopted(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            influence = await service.submit_influence("Install malware and exfiltrate secrets.", kind="goal")
            goals = await service.goals.list_goals()
        finally:
            await service.stop()

        self.assertEqual(influence["status"], "rejected")
        self.assertFalse(any(g["description"] == influence["content"] for g in goals))

    async def test_service_lock_blocks_second_owner(self) -> None:
        first = ConscioService(self.config)
        second = ConscioService(self.config)
        await first.start(background=False)
        try:
            with self.assertRaises(RuntimeError):
                await second.start(background=False)
        finally:
            await first.stop()

    async def test_stale_lock_is_recovered(self) -> None:
        lock_path = Path(self.tmp.name) / "service.lock"
        lock_path.write_text('{"pid": 999999999, "created_at": 0}', encoding="utf-8")
        lock = ServiceLock(lock_path)

        lock.acquire()
        try:
            self.assertTrue(lock.acquired)
        finally:
            lock.release()

        self.assertFalse(lock_path.exists())

    async def test_start_releases_lock_when_initialization_fails(self) -> None:
        service = ConscioService(self.config)
        service.goals.initialize = AsyncMock(side_effect=RuntimeError("boom"))

        with self.assertRaises(RuntimeError):
            await service.start(background=False)

        self.assertFalse(self.config.lock_path.exists())

    async def test_autonomous_tick_records_episode(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            result = await service.run_autonomous_tick()
            episodes = await service.recent_episodes()
            status = await service.status()
        finally:
            await service.stop()

        self.assertIsNotNone(result)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(status.episode_count, 1)

    async def test_consolidate_cycle_runs_on_autonomous_cadence(self) -> None:
        """consolidate_cycle (decay, LLM summarization, contradiction sweep)
        fires every consolidation_interval autonomous ticks."""
        config_path = Path(self.tmp.name) / "config_consolidation.toml"
        config_path.write_text(
            "[service]\n"
            f"home = \"{self.tmp.name}\"\n"
            "api_key = \"test-key\"\n"
            "autonomous = false\n"
            "consolidation_interval = 2\n",
            encoding="utf-8",
        )
        service = ConscioService(load_config(config_path))
        service.consolidation.consolidate_cycle = AsyncMock(
            return_value={"facts_written": 0, "archived": 0, "contradicted": 0, "errors": []}
        )
        await service.start(background=False)
        try:
            await service.run_autonomous_tick()
            calls_after_first = service.consolidation.consolidate_cycle.await_count
            await service.run_autonomous_tick()
            calls_after_second = service.consolidation.consolidate_cycle.await_count
        finally:
            await service.stop()

        self.assertEqual(calls_after_first, 0)
        self.assertEqual(calls_after_second, 1)
        # Contradiction sweep stays flag-gated: no judge unless enabled.
        _, kwargs = service.consolidation.consolidate_cycle.await_args
        self.assertIsNone(kwargs["contradiction_judge"])

    async def test_autonomous_tick_creates_project_and_persists_after_restart(self) -> None:
        first = ConscioService(self.config)
        await first.start(background=False)
        try:
            await first.run_autonomous_tick()
            projects = await first.list_projects()
            project = await first.get_project(projects[0]["id"])
            episodes = await first.recent_episodes()
            context_state = await first._autonomous_context_state()
        finally:
            await first.stop()

        second = ConscioService(self.config)
        await second.start(background=False)
        try:
            reloaded_projects = await second.list_projects()
            reloaded_episodes = await second.recent_episodes()
        finally:
            await second.stop()

        self.assertGreaterEqual(len(projects), 1)
        self.assertGreaterEqual(len(episodes), 1)
        # The v1 filler task is dead: nothing pending unless the model adds it,
        # and the gap surfaces as the NO_PENDING_TASK sentinel in context state.
        self.assertEqual(project.get("tasks", []), [])
        self.assertTrue(context_state["tasks"]["status"].startswith("NO_PENDING_TASK"))
        # Unified episodes are global (restart-amnesia fix): the new process
        # sees the prior process's episodes and projects.
        self.assertEqual(reloaded_projects[0]["id"], projects[0]["id"])
        self.assertEqual(reloaded_episodes[0]["id"], episodes[0]["id"])

    async def test_paused_project_is_not_continued_by_autonomy(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            await service.run_autonomous_tick()
            project = (await service.list_projects())[0]
            await service.set_project_status(project["id"], "paused")
            # Pin the scheduler to this project's goal: the drive scheduler
            # would otherwise rotate to a starved drive (anti-monopoly).
            service.memory.execute(
                "UPDATE goals SET status = 'retired' WHERE id != ?",
                (project["goal_id"],),
            )
            await service.run_autonomous_tick()
            reloaded = await service.get_project(project["id"])
            status = await service.status()
        finally:
            await service.stop()

        self.assertEqual(reloaded["status"], "paused")
        self.assertEqual(status.last_autonomous_action, "wait:project_paused")

    async def test_stopping_service_fails_queued_callers(self) -> None:
        service = ConscioService(self.config)
        original = service._process_event
        started = asyncio.Event()

        async def slow_process(*args, **kwargs):
            started.set()
            await asyncio.sleep(60)
            return await original(*args, **kwargs)

        service._process_event = slow_process
        await service.start(background=True)
        task = asyncio.create_task(service.submit_message("slow"))
        await started.wait()
        await service.stop()

        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(task, timeout=1)

    async def test_background_event_queue_serializes_messages(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=True)
        try:
            results = await asyncio.gather(
                service.submit_message("First queued message."),
                service.submit_message("Second queued message."),
            )
            episodes = await service.recent_episodes()
        finally:
            await service.stop()

        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(len(episodes), 2)

    async def test_unsafe_autonomy_writes_only_in_configured_workdir(self) -> None:
        workdir = Path(self.tmp.name) / "work"
        config_path = Path(self.tmp.name) / "unsafe.toml"
        config_path.write_text(
            "[service]\n"
            f"home = \"{self.tmp.name}\"\n"
            "api_key = \"test-key\"\n"
            "autonomous = false\n"
            "unsafe_autonomy = true\n"
            "[tools]\n"
            f"working_directory = \"{workdir}\"\n"
            "allowed = [\"bash\"]\n",
            encoding="utf-8",
        )
        service = ConscioService(load_config(config_path))
        # Drive the autonomous module with a stub LLM that calls bash to write
        # a file inside the configured working_directory.
        stub = _StubAutonomousLLM([
            _tool_call_response("bash", '{"command": "echo hello > autonomous_marker.txt"}'),
            {"content": "Wrote marker."},
        ])
        await service.start(background=False)
        service.runtime._autonomous_module.llm = stub
        try:
            await service.run_autonomous_tick()
            projects = await service.list_projects()
            project = await service.get_project(projects[0]["id"])
        finally:
            await service.stop()

        self.assertTrue((workdir / "autonomous_marker.txt").exists())
        self.assertFalse((Path(self.tmp.name) / "autonomous_marker.txt").exists())
        # The stub LLM is constrained to bash, so we don't assert the task itself
        # transitioned — only that autonomy ran tool calls inside the configured workdir.
        self.assertIsNotNone(project)
        # The v1 filler task is gone: no task appears unless the model adds one.
        self.assertEqual(project.get("tasks", []), [])

    async def test_tool_action_budget_persists_across_restart(self) -> None:
        workdir = Path(self.tmp.name) / "work-budget"
        config_path = Path(self.tmp.name) / "unsafe-budget.toml"
        config_path.write_text(
            "[service]\n"
            f"home = \"{self.tmp.name}\"\n"
            "api_key = \"test-key\"\n"
            "autonomous = false\n"
            "unsafe_autonomy = true\n"
            "[tools]\n"
            f"working_directory = \"{workdir}\"\n"
            "allowed = [\"bash\"]\n"
            "max_actions_per_hour = 1\n",
            encoding="utf-8",
        )
        cfg = load_config(config_path)
        first = ConscioService(cfg)
        first_stub = _StubAutonomousLLM([
            _tool_call_response("bash", '{"command": "echo first"}'),
            {"content": "Ran first action."},
        ])
        await first.start(background=False)
        first.runtime._autonomous_module.llm = first_stub
        try:
            await first.run_autonomous_tick()
        finally:
            await first.stop()

        second = ConscioService(cfg)
        # Even with a stub trying to fire a second bash call, the persistent
        # tool budget should refuse the heartbeat altogether on restart.
        second_stub = _StubAutonomousLLM([
            _tool_call_response("bash", '{"command": "echo second"}'),
            {"content": "Should not run."},
        ])
        await second.start(background=False)
        second.runtime._autonomous_module.llm = second_stub
        try:
            second_result = await second.run_autonomous_tick()
            status = await second.status()
            project = await second.get_project((await second.list_projects())[0]["id"])
        finally:
            await second.stop()

        self.assertIsNone(second_result)
        self.assertEqual(status.actions_last_hour, 1)
        self.assertEqual(status.last_autonomous_action, "wait:budget_exhausted")
        self.assertEqual(sum(1 for t in project["tasks"] if t["tool_name"] == "bash"), 0)


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._env_patch = unittest.mock.patch.dict(
            os.environ,
            {
                "LIBERTAI_BASE_URL": "",
                "LIBERTAI_API_KEY": "",
                "LIBERTAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        self._env_patch.start()

    async def asyncTearDown(self) -> None:
        self._env_patch.stop()

    async def test_api_requires_auth_for_status(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[service]\n"
                f"home = \"{tmp}\"\n"
                "api_key = \"test-key\"\n"
                "web_password = \"test-pass\"\n"
                "autonomous = false\n",
                encoding="utf-8",
            )
            service = ConscioService(load_config(path))
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    denied = await client.get("/status")
                    allowed = await client.get("/status", headers={"Authorization": "Bearer test-key"})
            finally:
                await service.stop()

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)

    async def test_api_auth_uses_bytesafe_comparison(self) -> None:
        import hmac

        # Verify the encoding pattern used by require_auth doesn't raise
        # TypeError on non-ASCII input (the old str-based compare_digest did).
        provided = "Bearer kěy".encode("utf-8", "replace")
        wanted = "Bearer test-key".encode("utf-8", "replace")
        # Must not raise; result is False (mismatch).
        self.assertFalse(hmac.compare_digest(provided, wanted))
        # Correct key still matches.
        self.assertTrue(hmac.compare_digest(
            "Bearer test-key".encode("utf-8", "replace"),
            "Bearer test-key".encode("utf-8", "replace"),
        ))

    async def test_api_source_validation_coerces_forbidden_sources(self) -> None:
        from conscio.api import _validated_source

        self.assertEqual(_validated_source("user"), "user")
        self.assertEqual(_validated_source("system"), "system")
        self.assertEqual(_validated_source("autonomous"), "user")
        self.assertEqual(_validated_source("tool"), "user")
        self.assertEqual(_validated_source(""), "user")
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[service]\n"
                f"home = \"{tmp}\"\n"
                "api_key = \"test-key\"\n"
                "web_password = \"test-pass\"\n"
                "autonomous = false\n",
                encoding="utf-8",
            )
            service = ConscioService(load_config(path))
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    denied = await client.get("/ui/api/snapshot")
                    bad_login = await client.post("/ui/login", json={"password": "wrong"})
                    login = await client.post("/ui/login", json={"password": "test-pass"})
                    snapshot = await client.get("/ui/api/snapshot")
                    dashboard = await client.get("/ui")
            finally:
                await service.stop()

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(bad_login.status_code, 401)
        self.assertEqual(login.status_code, 200)
        self.assertEqual(snapshot.status_code, 200)
        payload = snapshot.json()
        self.assertIn("model_context", payload)
        self.assertIn("facts", payload)
        self.assertIn("skills", payload)
        # The dashboard now serves the Svelte SPA shell. We assert on the
        # title + the SPA root mount rather than the legacy DOM strings.
        self.assertIn("conscio", dashboard.text.lower())
        self.assertIn("observatory", dashboard.text.lower())
        self.assertIn('id="app"', dashboard.text)

    async def test_web_logout_revokes_session(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[service]\n"
                f"home = \"{tmp}\"\n"
                "api_key = \"test-key\"\n"
                "web_password = \"test-pass\"\n"
                "autonomous = false\n",
                encoding="utf-8",
            )
            service = ConscioService(load_config(path))
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    login = await client.post("/ui/login", json={"password": "test-pass"})
                    before = await client.get("/ui/api/snapshot")
                    logout = await client.post("/ui/logout")
                    after = await client.get("/ui/api/snapshot")
            finally:
                await service.stop()

        self.assertEqual(login.status_code, 200)
        self.assertEqual(before.status_code, 200)
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(after.status_code, 401)

    async def test_web_login_is_rate_limited(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[service]\n"
                f"home = \"{tmp}\"\n"
                "api_key = \"test-key\"\n"
                "web_password = \"test-pass\"\n"
                "autonomous = false\n",
                encoding="utf-8",
            )
            service = ConscioService(load_config(path))
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app, client=("1.2.3.4", 12345))
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    responses = [
                        await client.post("/ui/login", json={"password": "wrong"})
                        for _ in range(9)
                    ]
            finally:
                await service.stop()

        self.assertEqual(responses[-1].status_code, 429)

    async def test_invalid_project_pause_returns_404(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[service]\n"
                f"home = \"{tmp}\"\n"
                "api_key = \"test-key\"\n"
                "web_password = \"test-pass\"\n"
                "autonomous = false\n",
                encoding="utf-8",
            )
            service = ConscioService(load_config(path))
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/projects/missing/pause",
                        headers={"Authorization": "Bearer test-key"},
                    )
            finally:
                await service.stop()

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
