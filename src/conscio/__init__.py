from __future__ import annotations

from importlib import metadata

from conscio.core.agent import ConsciousAgent
from conscio.core.runtime import CognitiveRuntime
from conscio.service import ConscioService

try:
    __version__ = metadata.version("conscio-agent")
except metadata.PackageNotFoundError:  # source tree without an installed dist
    __version__ = "0.0.0+unknown"

__all__ = ["CognitiveRuntime", "ConsciousAgent", "ConscioService", "__version__"]
