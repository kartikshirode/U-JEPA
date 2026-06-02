"""SIGReg loss: shape contract, ordering between normal vs degenerate,
fallback path runs without the vendored package."""
import pytest
import torch

from u_jepa.losses import sigreg as sig_mod
from u_jepa.losses.sigreg import (
    _fallback_sliced_epps_pulley,
    sigreg_loss,
    using_lejepa,
)


def test_rejects_wrong_rank():
    with pytest.raises(ValueError):
        sigreg_loss(torch.randn(8))


def test_normal_data_scores_lower_than_constant_data():
    torch.manual_seed(0)
    normal = torch.randn(512, 32)
    # All samples project to nearly the same value: pathologically non-normal
    constant_dir = torch.zeros(512, 32)
    constant_dir[:, 0] = 1.0
    a = sigreg_loss(normal, num_slices=128).item()
    b = sigreg_loss(constant_dir, num_slices=128).item()
    assert b > a, f"expected degenerate > normal, got {b=} {a=}"


def test_fallback_runs_independently_of_lejepa():
    torch.manual_seed(0)
    h = torch.randn(256, 16)
    loss = _fallback_sliced_epps_pulley(h, num_slices=64)
    assert torch.is_tensor(loss) and loss.dim() == 0
    assert loss.item() >= 0.0


def test_fallback_orders_normal_below_degenerate():
    torch.manual_seed(0)
    normal = torch.randn(512, 16)
    constant_dir = torch.zeros(512, 16)
    constant_dir[:, 0] = 1.0
    a = _fallback_sliced_epps_pulley(normal, num_slices=64).item()
    b = _fallback_sliced_epps_pulley(constant_dir, num_slices=64).item()
    assert b > a


def test_using_lejepa_returns_bool():
    assert isinstance(using_lejepa(), bool)


def test_loss_is_differentiable():
    h = torch.randn(64, 8, requires_grad=True)
    loss = sigreg_loss(h, num_slices=32)
    loss.backward()
    assert h.grad is not None
