"""Verify env detection picks the right platform and paths."""
import os
import sys
from pathlib import Path

from u_jepa.util import env as env_mod

def test_detect_returns_env_with_known_paths():
    e = env_mod.detect()
    assert e.name in {"kaggle", "local_win", "local_linux", "colab", "unknown"}
    assert e.repo_root.exists()
    assert (e.repo_root / "pyproject.toml").exists()

def test_repo_root_marker_walk():
    found = env_mod._find_repo_root_via_marker()
    assert (found / "pyproject.toml").exists()
    assert (found / "src" / "u_jepa").exists()

def test_local_windows_when_on_win32():
    if sys.platform != "win32":
        return
    e = env_mod.detect()
    assert e.name == "local_win"
    assert e.is_kaggle is False
    assert e.can_run_vllm is False
