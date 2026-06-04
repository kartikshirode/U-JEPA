"""Error-path tests for OrthogonalLoRABank: missing targets, duplicate
adds, activating unknown tasks, multi-module hook delta correctness."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank, _safe_key


class _StubBlock(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)


class _EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.something = nn.Linear(4, 4, bias=False)


def test_safe_key_replaces_dots():
    assert _safe_key("model.layers.0.q_proj") == "model__layers__0__q_proj"
    assert _safe_key("q_proj") == "q_proj"


def test_bank_raises_when_no_target_modules_match():
    with pytest.raises(RuntimeError, match="No target modules"):
        OrthogonalLoRABank(_EmptyModel(), rank=4, target_modules=("q_proj",))


def test_add_task_duplicate_raises():
    bank = OrthogonalLoRABank(_StubBlock(d=8), rank=2, target_modules=("q_proj",))
    bank.add_task("t1")
    with pytest.raises(ValueError, match="already exists"):
        bank.add_task("t1")


def test_activate_unknown_task_raises_keyerror():
    bank = OrthogonalLoRABank(_StubBlock(d=8), rank=2, target_modules=("q_proj",))
    with pytest.raises(KeyError):
        bank.activate("never_added")


def test_add_task_sets_active_to_new_task():
    bank = OrthogonalLoRABank(_StubBlock(d=8), rank=2, target_modules=("q_proj",))
    bank.add_task("a")
    assert bank.active == "a"
    bank.add_task("b")
    assert bank.active == "b"


def test_base_parameters_are_all_frozen():
    base = _StubBlock(d=8)
    # Force-enable grads so we can confirm the bank turned them off.
    for p in base.parameters():
        p.requires_grad = True
    bank = OrthogonalLoRABank(base, rank=2, target_modules=("q_proj",))
    for p in bank.base.parameters():
        assert p.requires_grad is False


def test_hook_delta_matches_forward_target_for_each_target_module():
    """With both q_proj and v_proj hooked, the post-hook output of the
    block must equal base(x) + delta_q(x) + delta_v(x)."""
    base = _StubBlock(d=16)
    bank = OrthogonalLoRABank(base, rank=2, target_modules=("q_proj", "v_proj"))
    bank.add_task("t")
    with torch.no_grad():
        for name in ("q_proj", "v_proj"):
            A, B = bank.adapter_matrices("t", name)
            A.fill_(0.1)
            B.fill_(0.2)
    x = torch.randn(2, 16)
    bank._active = None
    no_adapter = base(x)
    bank.activate("t")
    handles = bank.install_hooks()
    try:
        with_adapter = base(x)
    finally:
        for h in handles:
            h.remove()
    delta_q = bank.forward_target(x, "q_proj")
    delta_v = bank.forward_target(x, "v_proj")
    # Order of hook execution matches base.forward summing q and v outputs.
    expected = no_adapter + delta_q + delta_v
    assert torch.allclose(with_adapter, expected, atol=1e-5)


def test_scale_equals_alpha_over_rank():
    bank = OrthogonalLoRABank(
        _StubBlock(d=8), rank=4, target_modules=("q_proj",), alpha=32.0,
    )
    assert bank.scale == pytest.approx(32.0 / 4)
