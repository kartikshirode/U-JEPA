"""Shared pytest fixtures and skip helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(config, items):
    skip_network = pytest.mark.skip(reason="set U_JEPA_V2_RUN_NETWORK=1 to enable")
    skip_cuda = pytest.mark.skip(reason="no CUDA device detected")
    skip_slow = pytest.mark.skip(reason="set U_JEPA_V2_RUN_SLOW=1 to enable")

    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    run_network = _flag("U_JEPA_V2_RUN_NETWORK")
    run_slow = _flag("U_JEPA_V2_RUN_SLOW")

    for item in items:
        if "network" in item.keywords and not run_network:
            item.add_marker(skip_network)
        if "cuda" in item.keywords and not cuda_available:
            item.add_marker(skip_cuda)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
