"""OrthogonalLoRABank: per-task hot-swappable LoRA stack on a frozen base."""
import torch
import torch.nn as nn
import pytest

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank


class _StubLinear(nn.Module):
    """Tiny stand-in for a Qwen layer with q_proj and v_proj modules."""
    def __init__(self, d=64):
        super().__init__()
        self.d = d
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)


def test_add_task_creates_low_rank_adapter():
    base = _StubLinear(d=64)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj", "v_proj"))
    bank.add_task("fomc")
    assert "fomc" in bank.adapters
    a, b = bank.adapter_matrices("fomc", "q_proj")
    assert a.shape == (4, 64)
    assert b.shape == (64, 4)


def test_hot_swap_returns_different_deltas():
    base = _StubLinear(d=64)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("a")
    bank.add_task("b")
    # Asymmetric fills so B@A products differ between adapters
    # (filling (0.5,0.5) vs (-0.5,-0.5) gives the same delta because (-a)(-b)=ab)
    with torch.no_grad():
        A_a, B_a = bank.adapter_matrices("a", "q_proj")
        A_b, B_b = bank.adapter_matrices("b", "q_proj")
        A_a.fill_(0.3); B_a.fill_(0.4)
        A_b.fill_(0.7); B_b.fill_(0.9)
    x = torch.randn(2, 64)
    bank.activate("a")
    ya = bank.forward_target(x, "q_proj")
    bank.activate("b")
    yb = bank.forward_target(x, "q_proj")
    assert not torch.allclose(ya, yb)


def test_total_adapter_param_count_under_10M_for_qwen3_14b():
    """Qwen3-14B hidden_size = 5120. 4 domains x rank 16 x q_proj+v_proj."""
    base = _StubLinear(d=5120)
    bank = OrthogonalLoRABank(base, rank=16, target_modules=("q_proj", "v_proj"))
    for t in range(4):
        bank.add_task(f"d{t}")
    total = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    # 4 tasks * 2 modules * (16*5120 + 5120*16) = 1,310,720
    assert total < 10_000_000, f"too many trainable params: {total}"
    assert total > 1_000_000, f"suspiciously few: {total}"


def test_no_active_adapter_returns_zero_delta():
    base = _StubLinear(d=32)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("t1")
    bank._active = None
    x = torch.randn(2, 32)
    out = bank.forward_target(x, "q_proj")
    assert torch.allclose(out, torch.zeros_like(out))


def test_b_matrix_init_zero_so_adapter_starts_at_identity():
    base = _StubLinear(d=32)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("t1")
    _, b = bank.adapter_matrices("t1", "q_proj")
    assert torch.allclose(b, torch.zeros_like(b))
