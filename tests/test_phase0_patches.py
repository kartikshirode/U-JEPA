"""Unit tests for the Phase 0 reproduction script helpers.

Covers the three monkey-patch helpers and the namespace builder in
scripts/01_repro_latentmas_gsm8k.py. The patches run at module import
time, so importing the script via importlib gives us a handle we can
call the helpers on directly.

All tests run on Windows with no GPU, no vllm, no autoawq.
"""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "01_repro_latentmas_gsm8k.py"


def _load_phase0_module():
    """Load scripts/01_repro_latentmas_gsm8k.py as a fresh module object."""
    spec = importlib.util.spec_from_file_location("_phase0_for_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def phase0():
    return _load_phase0_module()


def test_phase_cfg_has_required_keys(phase0):
    cfg = phase0.PHASE_CFG
    needed = {
        "method", "model_name", "task", "split", "prompt", "max_samples",
        "generate_bs", "latent_steps", "max_new_tokens", "temperature",
        "top_p", "use_vllm", "tensor_parallel_size", "gpu_memory_utilization",
        "device", "device2", "seed",
    }
    missing = needed - set(cfg)
    assert not missing, f"missing keys: {missing}"
    # generate_bs MUST be >= 2 per the docstring comment about the
    # vendored squeeze(0) bug; pin that contract here.
    assert cfg["generate_bs"] >= 2, "generate_bs<2 breaks vendored run_batch_vllm"
    assert cfg["method"] == "latent_mas"
    assert cfg["model_name"].lower().endswith("awq"), "Phase 0 expects AWQ build"


def test_build_namespace_round_trips_phase_cfg(phase0):
    ns = phase0.build_namespace()
    cfg = phase0.PHASE_CFG
    for k, v in cfg.items():
        assert getattr(ns, k) == v, f"key {k!r} mismatch: {getattr(ns, k)!r} vs {v!r}"


def test_vllm_patch_is_noop_when_vllm_absent(phase0, monkeypatch):
    """Without vllm installed (local Windows path), the patch must
    swallow the ImportError silently and return None. Re-running it
    should never raise."""
    # Make absolutely sure 'vllm' is not importable in this test.
    monkeypatch.setitem(sys.modules, "vllm", None)
    # Should not raise even though vllm import will now hard-fail.
    result = phase0._patch_vllm_max_model_len(default_max_model_len=4096)
    assert result is None


def test_vllm_patch_sets_defaults_on_fake_vllm(phase0, monkeypatch):
    """Install a fake `vllm` module with an LLM class whose __init__ records
    the kwargs it was called with. The patch should wrap that __init__,
    fill max_model_len via setdefault, and force max_num_seqs /
    enable_prefix_caching / enforce_eager regardless of caller input
    (override semantics, per the patch docstring about V0 engine bugs)."""
    fake_vllm = types.ModuleType("vllm")
    captured = {}

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = dict(kwargs)

    fake_vllm.LLM = _FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    phase0._patch_vllm_max_model_len(default_max_model_len=1234, max_num_seqs=2)

    # Call the (now patched) init through a fresh instance.
    fake_vllm.LLM("Qwen/whatever")
    assert captured["kwargs"].get("max_model_len") == 1234
    assert captured["kwargs"].get("max_num_seqs") == 2
    assert captured["kwargs"].get("enable_prefix_caching") is False
    assert captured["kwargs"].get("enforce_eager") is True
    assert captured["kwargs"].get("enable_prompt_embeds") is True

    # max_model_len is setdefault; caller-supplied value must win there.
    # max_num_seqs / prefix_caching / enforce_eager / enable_prompt_embeds are
    # overrides; caller cannot turn them off (vendored ModelWrapper omits
    # enable_prompt_embeds on the else-branch which we always take).
    captured.clear()
    fake_vllm.LLM(
        "Qwen/whatever",
        max_model_len=9999,
        max_num_seqs=64,
        enable_prefix_caching=True,
        enforce_eager=False,
        enable_prompt_embeds=False,
    )
    assert captured["kwargs"]["max_model_len"] == 9999
    assert captured["kwargs"]["max_num_seqs"] == 2
    assert captured["kwargs"]["enable_prefix_caching"] is False
    assert captured["kwargs"]["enforce_eager"] is True
    assert captured["kwargs"]["enable_prompt_embeds"] is True


def test_transformers_activations_patch_adds_missing_classes(phase0):
    """After import, the three GELU shims must exist on transformers.activations.
    Running the patch a second time must be idempotent (no clobber)."""
    import transformers.activations as act

    for name in ("PytorchGELUTanh", "NewGELUActivation", "GELUActivation"):
        assert hasattr(act, name), f"shim {name} missing after patch"
        cls = getattr(act, name)
        # Each shim must be callable and produce a tensor of the same shape.
        x = torch.randn(2, 3)
        y = cls()(x)
        assert y.shape == x.shape

    # Idempotency: re-applying should not raise or change the identity of
    # existing classes that the patch only adds when missing.
    before = {n: getattr(act, n) for n in ("PytorchGELUTanh", "NewGELUActivation", "GELUActivation")}
    phase0._patch_transformers_activations_for_autoawq()
    after = {n: getattr(act, n) for n in ("PytorchGELUTanh", "NewGELUActivation", "GELUActivation")}
    assert before == after, "second patch call mutated already-present attributes"


def test_latent_realign_patch_is_noop_when_vendored_models_absent(phase0, monkeypatch):
    """Without the vendored 'models' module available the patch must
    return cleanly. We force the ImportError by inserting a sentinel."""
    monkeypatch.setitem(sys.modules, "models", None)
    # On local Windows the vendored path is not on sys.path by default
    # outside of the script's own bootstrap, so this must not blow up.
    result = phase0._patch_latent_realign_to_cpu_build()
    assert result is None


def test_latent_realign_patched_build_computes_on_cpu(phase0, monkeypatch):
    """Install a fake vendored 'models' module with a stub ModelWrapper,
    apply the patch, then call the patched method on a tiny model and
    confirm the math: realign_matrix should be eye when latent_space_realign
    is False, and have the right shape when True. Also confirm it ran
    on CPU (no CUDA required)."""
    fake_models = types.ModuleType("models")

    class _StubWrapper:
        pass

    fake_models.ModelWrapper = _StubWrapper
    monkeypatch.setitem(sys.modules, "models", fake_models)

    phase0._patch_latent_realign_to_cpu_build()

    # Tiny model with input + output embeddings of matching hidden size.
    hidden = 8
    vocab = 16
    inp = nn.Embedding(vocab, hidden)
    out = nn.Linear(hidden, vocab, bias=False)

    class _TinyModel:
        def get_input_embeddings(self):
            return inp

        def get_output_embeddings(self):
            return out

    args_off = types.SimpleNamespace(latent_space_realign=False)
    args_on = types.SimpleNamespace(latent_space_realign=True)

    instance = _StubWrapper()
    realign_off, norm_off = _StubWrapper._build_latent_realign_matrix(
        instance, _TinyModel(), torch.device("cpu"), args_off
    )
    assert realign_off.shape == (hidden, hidden)
    # off => identity fallback
    assert torch.allclose(realign_off, torch.eye(hidden), atol=1e-5)
    assert norm_off.device.type == "cpu"

    realign_on, norm_on = _StubWrapper._build_latent_realign_matrix(
        instance, _TinyModel(), torch.device("cpu"), args_on
    )
    assert realign_on.shape == (hidden, hidden)
    # The matrix solved from a small random init is almost never identity.
    assert not torch.allclose(realign_on, torch.eye(hidden), atol=1e-3)


def test_latent_realign_patch_raises_on_missing_embeddings(phase0, monkeypatch):
    fake_models = types.ModuleType("models")

    class _StubWrapper:
        pass

    fake_models.ModelWrapper = _StubWrapper
    monkeypatch.setitem(sys.modules, "models", fake_models)
    phase0._patch_latent_realign_to_cpu_build()

    class _BrokenModel:
        def get_input_embeddings(self):
            return None

        def get_output_embeddings(self):
            return None

    instance = _StubWrapper()
    with pytest.raises(RuntimeError, match="embeddings not accessible"):
        _StubWrapper._build_latent_realign_matrix(
            instance, _BrokenModel(), torch.device("cpu"), types.SimpleNamespace()
        )
