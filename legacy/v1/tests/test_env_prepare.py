"""env.prepare() side effects: dir creation and HF cache env var defaults."""
from __future__ import annotations
from pathlib import Path

import pytest

from u_jepa.util import env as env_mod


def test_prepare_creates_all_directories(tmp_path, monkeypatch):
    """Feed prepare() a custom Env pointing at tmp_path and verify it
    creates every directory it advertises."""
    fake = env_mod.Env(
        name="unknown",
        is_kaggle=False,
        repo_root=tmp_path,
        results_dir=tmp_path / "results" / "nested",
        checkpoint_dir=tmp_path / "ckpts",
        hf_cache_dir=tmp_path / "hf",
        can_run_vllm=False,
    )
    # Wipe any inherited HF_HOME so the assertion below is meaningful.
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    returned = env_mod.prepare(fake)
    assert returned is fake
    assert (tmp_path / "results" / "nested").is_dir()
    assert (tmp_path / "ckpts").is_dir()
    assert (tmp_path / "hf").is_dir()


def test_prepare_sets_hf_cache_env_vars_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    fake = env_mod.Env(
        name="unknown",
        is_kaggle=False,
        repo_root=tmp_path,
        results_dir=tmp_path / "r",
        checkpoint_dir=tmp_path / "c",
        hf_cache_dir=tmp_path / "h",
        can_run_vllm=False,
    )
    env_mod.prepare(fake)
    import os
    assert os.environ["HF_HOME"] == str(tmp_path / "h")
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "h")
    assert os.environ["TRANSFORMERS_CACHE"] == str(tmp_path / "h")


def test_prepare_respects_existing_hf_cache_env_vars(tmp_path, monkeypatch):
    """If the user already set HF_HOME we must not overwrite it
    (prepare uses setdefault, not assignment)."""
    pre = str(tmp_path / "user_set_cache")
    monkeypatch.setenv("HF_HOME", pre)
    monkeypatch.setenv("HF_HUB_CACHE", pre)
    monkeypatch.setenv("TRANSFORMERS_CACHE", pre)
    fake = env_mod.Env(
        name="unknown",
        is_kaggle=False,
        repo_root=tmp_path,
        results_dir=tmp_path / "r",
        checkpoint_dir=tmp_path / "c",
        hf_cache_dir=tmp_path / "h",
        can_run_vllm=False,
    )
    env_mod.prepare(fake)
    import os
    assert os.environ["HF_HOME"] == pre
    assert os.environ["HF_HUB_CACHE"] == pre
    assert os.environ["TRANSFORMERS_CACHE"] == pre


def test_prepare_default_arg_calls_detect(monkeypatch, tmp_path):
    """When called with no argument, prepare() must call detect()."""
    sentinel = env_mod.Env(
        name="unknown",
        is_kaggle=False,
        repo_root=tmp_path,
        results_dir=tmp_path / "r",
        checkpoint_dir=tmp_path / "c",
        hf_cache_dir=tmp_path / "h",
        can_run_vllm=False,
    )
    monkeypatch.setattr(env_mod, "detect", lambda: sentinel)
    returned = env_mod.prepare()
    assert returned is sentinel
    assert (tmp_path / "r").is_dir()
