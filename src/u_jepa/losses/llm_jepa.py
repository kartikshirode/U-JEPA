"""LLM-JEPA auxiliary loss (Huang, LeCun, Balestriero 2025).

Pairs two views of the same example (in our case the NL question and the SQL
answer for Spider) and pushes the predictor of view-A's pooled hidden state
toward view-B's pooled hidden state. View-B is detached so the gradient only
flows through the LoRA adapter and the predictor, never through the frozen
base going forward on view-B.

TiedPredictor is the [PRED] mechanism collapsed into one shared Linear applied
k times. The shared weight is the "tied" part; we trade architectural fidelity
for a tiny parameter count and a tiny memory footprint (one DxD matrix).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedPredictor(nn.Module):
    def __init__(self, hidden: int, k_tokens: int = 3):
        super().__init__()
        if k_tokens < 1:
            raise ValueError(f"k_tokens must be >= 1, got {k_tokens}")
        self.k = k_tokens
        self.proj = nn.Linear(hidden, hidden, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for _ in range(self.k):
            h = self.proj(h)
        return h


def llm_jepa_loss(
    predictor: TiedPredictor,
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """h_a, h_b: (B, D) pooled hidden states of view A and view B.

    Returns a scalar loss. h_b is detached so the target side never gets a
    gradient; this is how JEPA-family methods avoid the trivial collapse of
    making both representations equal to a constant.
    """
    h_b = h_b.detach()
    pred_b = predictor(h_a)
    if metric == "cosine":
        return (1.0 - F.cosine_similarity(pred_b, h_b, dim=-1)).mean()
    if metric == "mse":
        return F.mse_loss(pred_b, h_b)
    raise ValueError(f"unknown metric {metric!r}")
