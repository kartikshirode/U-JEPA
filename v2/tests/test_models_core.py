"""Tests for the core-model registry. The HF-download paths are network-gated."""

from __future__ import annotations

import pytest

from u_jepa_v2.models import core


def test_registry_has_both_cores():
    assert "gpt2-xl" in core.CORE_MODELS
    assert "qwen2.5-1.5b" in core.CORE_MODELS


def test_resolve_by_short_name():
    spec = core.resolve_core("gpt2-xl")
    assert spec.hf_id == "gpt2-xl"
    assert spec.family == "gpt2"


def test_resolve_by_hf_id():
    spec = core.resolve_core("Qwen/Qwen2.5-1.5B-Instruct")
    assert spec.short_name == "qwen2.5-1.5b"


def test_resolve_is_case_insensitive():
    spec = core.resolve_core("GPT2-XL")
    assert spec.short_name == "gpt2-xl"


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        core.resolve_core("not-a-model")


def test_freeze_disables_grads():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    core.freeze_(m)
    assert all(not p.requires_grad for p in m.parameters())


@pytest.mark.network
@pytest.mark.slow
def test_load_gpt2_xl_smoke(tmp_path):
    """Full HF load. Only runs when both U_JEPA_V2_RUN_NETWORK=1 and U_JEPA_V2_RUN_SLOW=1."""
    model, tok = core.load_core("gpt2-xl", dtype="fp32", device="cpu", cache_dir=str(tmp_path))
    assert tok.pad_token_id is not None
    assert all(not p.requires_grad for p in model.parameters())
