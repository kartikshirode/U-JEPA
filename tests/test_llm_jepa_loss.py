"""LLM-JEPA loss: shape, alignment, divergence, gradient flow."""
import pytest
import torch

from u_jepa.losses.llm_jepa import TiedPredictor, llm_jepa_loss


def test_predictor_preserves_shape():
    pred = TiedPredictor(hidden=64, k_tokens=3)
    h = torch.randn(2, 64)
    assert pred(h).shape == h.shape


def test_predictor_rejects_zero_k():
    with pytest.raises(ValueError):
        TiedPredictor(hidden=8, k_tokens=0)


def test_aligned_views_drive_loss_to_zero():
    torch.manual_seed(0)
    pred = TiedPredictor(hidden=32, k_tokens=1)
    pred.proj.weight.data.copy_(torch.eye(32))
    pred.proj.bias.data.zero_()
    h = torch.randn(4, 32)
    loss = llm_jepa_loss(pred, h, h.clone(), metric="cosine")
    assert loss.item() < 1e-5


def test_misaligned_views_yield_higher_loss_than_aligned():
    torch.manual_seed(0)
    pred = TiedPredictor(hidden=16, k_tokens=2)
    h_a = torch.randn(8, 16)
    h_b_aligned = h_a.clone()
    h_b_random = torch.randn(8, 16)
    aligned = llm_jepa_loss(pred, h_a, h_b_aligned, metric="mse").item()
    random_pair = llm_jepa_loss(pred, h_a, h_b_random, metric="mse").item()
    assert random_pair > aligned


def test_target_side_is_detached():
    pred = TiedPredictor(hidden=8, k_tokens=1)
    h_a = torch.randn(2, 8, requires_grad=True)
    h_b = torch.randn(2, 8, requires_grad=True)
    loss = llm_jepa_loss(pred, h_a, h_b)
    loss.backward()
    assert h_a.grad is not None and h_a.grad.abs().sum() > 0
    assert h_b.grad is None, (
        f"target side must have no grad; got grad of magnitude "
        f"{h_b.grad.abs().sum().item() if h_b.grad is not None else 0}"
    )


def test_unknown_metric_raises():
    pred = TiedPredictor(hidden=4)
    with pytest.raises(ValueError):
        llm_jepa_loss(pred, torch.randn(1, 4), torch.randn(1, 4), metric="kl")
