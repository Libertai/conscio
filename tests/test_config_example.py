from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_regen_module():
    spec = importlib.util.spec_from_file_location(
        "regen_config_example", REPO_ROOT / "scripts" / "regen-config-example.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["regen_config_example"] = module
    spec.loader.exec_module(module)
    return module


class ConfigExampleDriftTests(unittest.TestCase):
    def test_committed_example_matches_generated(self) -> None:
        module = _load_regen_module()
        committed = (REPO_ROOT / "config.example.toml").read_text(encoding="utf-8")
        self.assertEqual(
            committed,
            module.render(),
            "config.example.toml is stale — run: uv run python scripts/regen-config-example.py",
        )
