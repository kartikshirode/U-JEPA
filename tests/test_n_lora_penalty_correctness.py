"""Correctness of the N-LoRA orthogonality penalty over a whole bank.

test_n_lora_loss already covers the single-module n_lora_penalty. This file
drives n_lora_penalty_over_bank end to end on a CPU stub bank, which is the
function the training loop actually calls. If the penalty is near zero for
overlapping adapters, or large for orthogonal ones, the forgetting story falls
apart. We also confirm previous-task A is detached so the penalty only pushes the
current task, never rewrites a finished one.
"""
import torch
import torch.nn as nn

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.continual.n_lora_loss import n_lora_penalty_over_bank


class _MiniBlock(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad = False


def _bank(d=8, rank=4):
    base = _MiniBlock(d)
    return OrthogonalLoRABank(base, rank=rank, target_modules=("q_proj",))


def _set_A(bank, task, value):
    with torch.no_grad():
        for name in bank._target_dims:
            A, _ = bank.adapter_matrices(task, name)
            A.copy_(value)


def test_penalty_large_when_a_matrices_identical():
    bank = _bank()
    bank.add_task("t0")
    bank.add_task("t1")
    a = torch.randn(bank.rank, 8)
    _set_A(bank, "t0", a)
    _set_A(bank, "t1", a)  # identical -> heavy overlap
    pen = n_lora_penalty_over_bank(bank, current_task="t1", prev_tasks=["t0"],
                                   collision_weight=0.0)
    assert pen.item() > 1.0


def test_penalty_near_zero_when_a_matrices_orthogonal():
    # rank x in_dim A matrices living in disjoint coordinate blocks.
    bank = _bank(d=8, rank=2)
    bank.add_task("t0")
    bank.add_task("t1")
    prev = torch.zeros(2, 8)
    prev[0, 0] = 1.0
    prev[1, 1] = 1.0
    cur = torch.zeros(2, 8)
    cur[0, 4] = 1.0
    cur[1, 5] = 1.0
    _set_A(bank, "t0", prev)
    _set_A(bank, "t1", cur)
    pen = n_lora_penalty_over_bank(bank, current_task="t1", prev_tasks=["t0"],
                                   collision_weight=0.0)
    assert pen.item() < 1e-8


def test_penalty_grows_with_more_prev_tasks():
    bank = _bank(d=8, rank=2)
    bank.add_task("t0")
    bank.add_task("t1")
    bank.add_task("t2")
    a = torch.ones(2, 8) * 0.3
    _set_A(bank, "t0", a)
    _set_A(bank, "t1", a)
    _set_A(bank, "t2", a)
    one = n_lora_penalty_over_bank(bank, "t2", ["t0"], collision_weight=0.0)
    two = n_lora_penalty_over_bank(bank, "t2", ["t0", "t1"], collision_weight=0.0)
    assert two.item() > one.item()
    assert torch.isclose(two, one * 2, atol=1e-5)


def test_prev_task_a_is_detached():
    bank = _bank()
    bank.add_task("t0")
    bank.add_task("t1")
    a_prev, _ = bank.adapter_matrices("t0", "q_proj")
    a_cur, _ = bank.adapter_matrices("t1", "q_proj")
    # Make sure both currently require grad.
    assert a_prev.requires_grad and a_cur.requires_grad
    a_prev.grad = None
    a_cur.grad = None
    pen = n_lora_penalty_over_bank(bank, "t1", ["t0"], collision_weight=0.0)
    pen.backward()
    # Current task receives gradient...
    assert a_cur.grad is not None
    assert a_cur.grad.abs().sum().item() > 0
    # ...previous task does NOT (it was detached inside the penalty).
    assert a_prev.grad is None


def test_empty_prevs_returns_zero_scalar_on_param_device():
    bank = _bank()
    bank.add_task("t0")
    pen = n_lora_penalty_over_bank(bank, "t0", [], collision_weight=0.0)
    assert pen.item() == 0.0
    # Must be a real 0-d tensor (so total = ce + w * pen stays type-safe).
    assert pen.dim() == 0
