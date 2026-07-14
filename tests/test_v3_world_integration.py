from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from conscio.config import ServiceConfig
from conscio.service import ConscioService


class V3WorldModelIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_training_does_not_mutate_the_live_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConscioService(ServiceConfig(home=Path(tmp), autonomous=False, api_key="test-key"))
            await service.start(acquire_lock=False, background=False)
            try:
                before = await service.world_model_status()
                result = await service.shadow_world_model_learning(promote=False, synthetic_episodes=64, seed=17)
                after = await service.world_model_status()
            finally:
                await service.stop()

        self.assertTrue(result["eligible"])
        self.assertFalse(result["promoted"])
        self.assertEqual(before["model_version"], after["model_version"])
        self.assertEqual(before["lineage_id"], after["lineage_id"])
        self.assertEqual(after["lineage_migrations"], 0)
        self.assertFalse(result["llm_training_reachable"])

    async def test_promoted_weights_migrate_lineage_and_restore_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ServiceConfig(home=Path(tmp), autonomous=False, api_key="test-key")
            first = ConscioService(config)
            await first.start(acquire_lock=False, background=False)
            try:
                before = await first.world_model_status()
                result = await first.shadow_world_model_learning(promote=True, synthetic_episodes=64, seed=17)
                promoted = await first.world_model_status()
                advanced_episode = await first.runtime.run_episode("advance the promoted recurrent lineage")
                advanced = await first.world_model_status()
                advanced_state = tuple(first.runtime.recurrent_core.deterministic)
            finally:
                await first.stop()

            second = ConscioService(config)
            await second.start(acquire_lock=False, background=False)
            try:
                restored = await second.world_model_status()
                restored_state = tuple(second.runtime.recurrent_core.deterministic)
                migration_events = second.memory.fetchall(
                    "SELECT * FROM cognitive_events WHERE event_type = 'model_lineage_migration'"
                )
            finally:
                await second.stop()

        self.assertTrue(result["eligible"])
        self.assertTrue(result["promoted"])
        self.assertNotEqual(before["model_version"], promoted["model_version"])
        self.assertNotEqual(before["lineage_id"], promoted["lineage_id"])
        self.assertEqual(promoted["lineage_migrations"], 1)
        self.assertEqual(restored["model_version"], promoted["model_version"])
        self.assertEqual(restored["lineage_id"], promoted["lineage_id"])
        self.assertEqual(advanced["checkpoint_id"], advanced_episode.checkpoint_reference)
        self.assertNotEqual(advanced["checkpoint_id"], promoted["checkpoint_id"])
        self.assertEqual(restored["checkpoint_id"], advanced["checkpoint_id"])
        self.assertEqual(restored_state, advanced_state)
        self.assertEqual(len(migration_events), 1)

    async def test_world_model_api_is_authenticated_and_shadow_only_by_default(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi/httpx are not installed")

        with tempfile.TemporaryDirectory() as tmp:
            service = ConscioService(ServiceConfig(home=Path(tmp), autonomous=False, api_key="test-key"))
            await service.start(acquire_lock=False, background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    denied = await client.get("/learning/world-model")
                    status = await client.get(
                        "/learning/world-model",
                        headers={"Authorization": "Bearer test-key"},
                    )
                    trained = await client.post(
                        "/learning/world-model-shadow",
                        headers={"Authorization": "Bearer test-key"},
                        json={"synthetic_episodes": 64, "seed": 17},
                    )
            finally:
                await service.stop()

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(trained.status_code, 200)
        self.assertFalse(trained.json()["promoted"])

    async def test_persistence_trial_forbids_world_model_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConscioService(
                ServiceConfig(
                    home=Path(tmp),
                    autonomous=False,
                    api_key="test-key",
                    persistence_trial_enabled=True,
                    persistence_trial_revision="test-revision",
                )
            )
            await service.start(acquire_lock=False, background=False)
            try:
                with self.assertRaisesRegex(RuntimeError, "persistence trial"):
                    await service.shadow_world_model_learning(promote=True, synthetic_episodes=64, seed=17)
            finally:
                await service.stop()


if __name__ == "__main__":
    unittest.main()
