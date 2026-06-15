from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from conscio.cli import _tools_allow, _tools_deny
from conscio.config import load_config, write_default_config


class ToolPolicyCliTests(unittest.TestCase):
    def test_tools_deny_and_allow_edit_active_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            write_default_config(path)
            with patch.dict(os.environ, {"CONSCIO_CONFIG": str(path)}, clear=False):
                _tools_deny(argparse.Namespace(names=["bash", "execute_code"]))
                denied = load_config(path)
                self.assertEqual(denied.denied_tools, ["bash", "execute_code"])
                self.assertEqual(denied.allowed_tools, [])

                _tools_allow(argparse.Namespace(names=["bash"]))
                allowed = load_config(path)

        self.assertEqual(allowed.allowed_tools, ["bash"])
        self.assertEqual(allowed.denied_tools, ["execute_code"])


if __name__ == "__main__":
    unittest.main()
