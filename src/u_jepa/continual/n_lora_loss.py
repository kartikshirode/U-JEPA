"""N-LoRA + O-LoRA composite penalty on the A matrices of sequential adapters.

References:
  O-LoRA (Wang et al., EMNLP 2023 Findings) - orthogonality term
    ||A_curr @ A_prev.T||_F^2
  N-LoRA (Yang et al., COLING 2025) - add a non-collision term
    sum |A_curr| * |A_prev|
  encouraging adapters not to use the same parameter indices even if
  orthogonal in direction.

A_prev tensors are always detached: gradient only flows through the
current task's adapter, never back into frozen past-task adapters.
"""
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn


def n_lora_penalty(
    A_curr: torch.Tensor,
    A_prevs: Sequence[torch.Tensor],
    collision_weight: float = 0.01,
) -> torch.Tensor:
    """Composite penalty for one target module.

    Args:
        A_curr: (rank, in_dim) - the current task's A matrix
        A_prevs: list of (rank, in_dim) frozen A matrices from past tasks
        collision_weight: lambda on the N-LoRA non-collision term

    Returns:
        scalar tensor (on A_curr's device/dtype)
    """
    if not A_prevs:
        return A_curr.new_zeros(())
    loss = A_curr.new_zeros(())
    A_curr_abs = A_curr.abs() if collision_weight > 0 else None
    for A_prev in A_prevs:
        A_prev = A_prev.detach()
        # O-LoRA: Frobenius norm of the rank x rank inner-product matrix
        inner = A_curr @ A_prev.T
        loss = loss + inner.pow(2).sum()
        if collision_weight > 0:
            loss = loss + collision_weight * (A_curr_abs * A_prev.abs()).sum()
    return loss


def collect_a_matrices(
    bank,
    task_ids: Sequence[str],
    module_name: str,
) -> list[torch.Tensor]:
    """Convenience: pull A matrices for a list of tasks for one target module."""
    return [bank.adapter_matrices(t, module_name)[0] for t in task_ids]


def n_lora_penalty_over_bank(
    bank,
    current_task: str,
    prev_tasks: Sequence[str],
    collision_weight: float = 0.01,
) -> torch.Tensor:
    """Sum the penalty over every target module in a bank.

    The standard loss term to add to a training step.
    """
    if not prev_tasks:
        any_A, _ = next(iter(bank.adapters[current_task].values())), None
        return torch.zeros((), device=any_A.device, dtype=any_A.dtype)
    total = None
    for module_name in bank._target_dims:
        A_curr, _ = bank.adapter_matrices(current_task, module_name)
        A_prevs = collect_a_matrices(bank, prev_tasks, module_name)
        term = n_lora_penalty(A_curr, A_prevs, collision_weight=collision_weight)
        total = term if total is None else total + term
    return total
