"""Execution-environment detection and path resolution.

The same scripts run on the local Windows laptop (light tests only) and on
Kaggle T4 / P100 sessions (heavy training). This module hides the differences
so the rest of the code never branches on platform.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Env:
    name: str           # "kaggle", "local_win", "local_linux", "colab", "unknown"
    is_kaggle: bool
    repo_root: Path
    results_dir: Path
    checkpoint_dir: Path
    hf_cache_dir: Path
    can_run_vllm: bool

def detect() -> Env:
    """Detect the current environment and return resolved paths."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists():
        repo = _find_repo_root_kaggle()
        # HF cache MUST live outside /kaggle/working: that path has a 20 GB
        # quota and Qwen3-14B alone is ~28 GB. /tmp on Kaggle has ~50+ GB
        # ephemeral space; lost between sessions but big enough for a model.
        return Env(
            name="kaggle",
            is_kaggle=True,
            repo_root=repo,
            results_dir=Path("/kaggle/working/results"),
            checkpoint_dir=Path("/kaggle/working/checkpoints"),
            hf_cache_dir=Path("/tmp/hf_cache"),
            can_run_vllm=True,
        )
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        repo = _find_repo_root_via_marker()
        return Env(
            name="colab",
            is_kaggle=False,
            repo_root=repo,
            results_dir=repo / "results",
            checkpoint_dir=repo / "checkpoints",
            hf_cache_dir=Path("/content/hf_cache"),
            can_run_vllm=True,
        )
    if sys.platform == "win32":
        repo = _find_repo_root_via_marker()
        return Env(
            name="local_win",
            is_kaggle=False,
            repo_root=repo,
            results_dir=repo / "results",
            checkpoint_dir=repo / "checkpoints",
            hf_cache_dir=repo / "hf_cache",
            can_run_vllm=False,
        )
    if sys.platform.startswith("linux"):
        repo = _find_repo_root_via_marker()
        return Env(
            name="local_linux",
            is_kaggle=False,
            repo_root=repo,
            results_dir=repo / "results",
            checkpoint_dir=repo / "checkpoints",
            hf_cache_dir=repo / "hf_cache",
            can_run_vllm=True,
        )
    repo = _find_repo_root_via_marker()
    return Env(
        name="unknown",
        is_kaggle=False,
        repo_root=repo,
        results_dir=repo / "results",
        checkpoint_dir=repo / "checkpoints",
        hf_cache_dir=repo / "hf_cache",
        can_run_vllm=False,
    )

def _find_repo_root_via_marker() -> Path:
    """Walk up from this file until we see pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]

def _find_repo_root_kaggle() -> Path:
    """On Kaggle, the repo is usually cloned under /kaggle/working/<repo>."""
    working = Path("/kaggle/working")
    if working.exists():
        for child in working.iterdir():
            if (child / "pyproject.toml").exists():
                return child
    return _find_repo_root_via_marker()

def prepare(env: Env | None = None) -> Env:
    """Create directories and set HF cache env var. Call this at script start."""
    env = env or detect()
    env.results_dir.mkdir(parents=True, exist_ok=True)
    env.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(env.hf_cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(env.hf_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(env.hf_cache_dir))
    return env
