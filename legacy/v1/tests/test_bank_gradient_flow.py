"""End-to-end gradient flow through the LoRA bank on a CPU stub model.

This locks down the single most load-bearing property of the Phase 1 training
step: backprop updates the active task's adapters (A and B) and never touches the
frozen base weights. If this breaks, the run can train for hours and learn
nothing, or worse, silently corrupt the base.

The existing test_phase1_fixes covers grad flow through a single q_proj hook.
Here we go a layer deeper: a multi-target block (q_proj and v_proj plus a
non-target o_proj), running the full base forward, and asserting per-task
isolation when more than one task is in the bank.
"""
import torch
import torch.nn as nn

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank


class _MiniBlock(nn.Module):
    """q_proj and v_proj are LoRA targets; o_proj is not. The forward runs all
    three so a broken hook on any target shows up at the output."""

    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.v_proj(x))


def _seed_b(bank, task):
    # B starts at zero so the delta (B @ A) is zero and no gradient would reach
    # A. Perturb B so the adapter actually contributes before testing grad flow.
    with torch.no_grad():
        for name in bank._target_dims:
            _, B = bank.adapter_matrices(task, name)
            B.add_(0.1 * torch.randn_like(B))


def test_backward_updates_adapters_not_base():
    base = _MiniBlock(16)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj", "v_proj"))
    bank.add_task("t0")
    bank.activate("t0")
    _seed_b(bank, "t0")

    handles = bank.install_hooks()
    try:
        x = torch.randn(3, 16)
        loss = base(x).pow(2).sum()
        loss.backward()
    finally:
        for h in handles:
            h.remove()

    for name in bank._target_dims:
        A, B = bank.adapter_matrices("t0", name)
        assert A.grad is not None, f"{name}.A got no grad"
        assert B.grad is not None, f"{name}.B got no grad"
        assert A.grad.abs().sum().item() > 0, f"{name}.A grad is all zero"
        assert B.grad.abs().sum().item() > 0, f"{name}.B grad is all zero"

    # Frozen base weights must never receive a gradient.
    for pname, p in base.named_parameters():
        assert p.grad is None, f"frozen base param {pname} received a gradient"


def test_only_active_task_adapters_get_grad():
    base = _MiniBlock(16)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj", "v_proj"))
    bank.add_task("t0")
    bank.add_task("t1")
    bank.activate("t1")
    _seed_b(bank, "t1")

    handles = bank.install_hooks()
    try:
        base(torch.randn(2, 16)).pow(2).sum().backward()
    finally:
        for h in handles:
            h.remove()

    # Active task gets grads.
    for name in bank._target_dims:
        A, B = bank.adapter_matrices("t1", name)
        assert A.grad is not None
        assert B.grad is not None
    # Inactive task never participated in the forward, so no grad.
    for name in bank._target_dims:
        A, B = bank.adapter_matrices("t0", name)
        assert A.grad is None
        assert B.grad is None


def test_base_params_have_requires_grad_false():
    # The bank is responsible for freezing the base on construction.
    base = _MiniBlock(16)
    OrthogonalLoRABank(base, rank=4, target_modules=("q_proj", "v_proj"))
    for p in base.parameters():
        assert p.requires_grad is False
