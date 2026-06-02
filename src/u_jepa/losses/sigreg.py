"""SIGReg loss: SIGReg / sliced Epps-Pulley statistic over pooled embeddings.

Primary path imports the vendored lejepa package (SlicingUnivariateTest wrapping
an Epps-Pulley univariate test). Fallback path computes a self-contained sliced
Epps-Pulley over random unit projections; cheaper, less rigorous, but enough to
keep the training loop alive if the vendored import breaks (path issues, lejepa
upstream rename, etc).

The statistic is large when embeddings are degenerate (constant directions, low
rank), small when they look isotropic-normal in projection. We treat it as a
regularizer added to the training loss with a small weight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_VENDORED = Path(__file__).resolve().parents[3] / "vendored" / "lejepa"
if _VENDORED.exists() and str(_VENDORED) not in sys.path:
    sys.path.insert(0, str(_VENDORED))

try:
    from lejepa.multivariate import SlicingUnivariateTest  # type: ignore
    from lejepa.univariate import EppsPulley  # type: ignore
    _HAS_LEJEPA = True
except Exception:
    _HAS_LEJEPA = False

# One-time backend announcement on import so logs always show which path is
# active without the caller having to remember to print it themselves.
print(
    f"[sigreg] backend={'lejepa' if _HAS_LEJEPA else 'fallback'}",
    file=sys.stderr,
    flush=True,
)

# Cache built SlicingUnivariateTest modules so we do not rebuild them per
# training step. Keyed on (num_slices, device.type, device.index) because
# the .to(device) call is the only state that has to follow the input.
_SIG_CACHE: dict = {}


def _get_lejepa_module(num_slices: int, device: "torch.device"):
    key = (num_slices, device.type, device.index)
    mod = _SIG_CACHE.get(key)
    if mod is None:
        univ = EppsPulley(t_max=3.0, n_points=17)
        mod = SlicingUnivariateTest(
            univariate_test=univ, num_slices=num_slices, reduction="mean",
        ).to(device)
        _SIG_CACHE[key] = mod
    return mod


def _fallback_sliced_epps_pulley(
    x: torch.Tensor, num_slices: int, n_points: int = 17
) -> torch.Tensor:
    """Self-contained sliced Epps-Pulley against a unit normal.

    Projects x onto `num_slices` random unit vectors, standardizes each
    projection, and compares its empirical characteristic function against
    the standard normal's at `n_points` frequencies between 0 and 3.
    """
    B, D = x.shape
    dirs = torch.randn(num_slices, D, device=x.device, dtype=x.dtype)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    projected = x @ dirs.T  # (B, num_slices)
    mean = projected.mean(dim=0, keepdim=True)
    std = projected.std(dim=0, keepdim=True) + 1e-6
    z = (projected - mean) / std
    t = torch.linspace(0.0, 3.0, n_points, device=x.device, dtype=x.dtype)
    # phi(t) = exp(-t^2/2) is the characteristic function of N(0,1) (real part)
    phi = torch.exp(-0.5 * t.pow(2))
    xt = z.unsqueeze(-1) * t  # (B, num_slices, n_points)
    cos_mean = xt.cos().mean(dim=0)  # (num_slices, n_points)
    sin_mean = xt.sin().mean(dim=0)
    err = (cos_mean - phi).pow(2) + sin_mean.pow(2)
    return err.mean()


def sigreg_loss(
    embeddings: torch.Tensor, num_slices: int = 128
) -> torch.Tensor:
    """embeddings: (B, D) pooled representations.

    Returns a non-negative scalar. Pass as an additive regularizer with a small
    weight (e.g. 0.1) alongside the main CE / JEPA loss.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"expected (B, D), got shape {tuple(embeddings.shape)}")
    if _HAS_LEJEPA:
        sig = _get_lejepa_module(num_slices, embeddings.device)
        # lejepa returns an N-scaled statistic (summed over the batch), while
        # the fallback returns a per-sample mean. Normalize so callers see
        # comparable magnitudes regardless of backend.
        return sig(embeddings) / max(1, embeddings.shape[0])
    return _fallback_sliced_epps_pulley(embeddings, num_slices=num_slices)


def using_lejepa() -> bool:
    """Expose which backend is live so callers and tests can log it."""
    return _HAS_LEJEPA
