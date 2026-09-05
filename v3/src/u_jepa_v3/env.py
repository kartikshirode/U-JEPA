"""Environment detection for v3.

v1 and v2 both hardcoded fp16 because the only GPU was a Kaggle T4, which is
Turing and has no native bf16. The H200 is Hopper. Hardcoding again would throw
away numerical headroom silently, so dtype is derived from compute capability
and only overridden on purpose.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

VALID_DTYPES = ("fp16", "bf16", "fp32")


def has_native_bf16(capability: tuple[int, int] | None) -> bool:
    """Ampere (8.0) and later have bf16 tensor cores. Turing (7.5) does not."""
    if capability is None:
        return False
    return capability[0] >= 8


def preferred_dtype_str(capability: tuple[int, int] | None = None) -> str:
    """Dtype to load models in. CPU-only means fp32."""
    override = os.environ.get("U_JEPA_V3_DTYPE")
    if override:
        if override not in VALID_DTYPES:
            raise ValueError(f"U_JEPA_V3_DTYPE={override!r} not in {VALID_DTYPES}")
        return override
    if capability is None:
        return "fp32"
    return "bf16" if has_native_bf16(capability) else "fp16"


def device_capability(index: int = 0) -> tuple[int, int] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(index)


def run_root() -> Path:
    raw = os.environ.get("U_JEPA_V3_RUN_DIR")
    root = Path(raw) if raw else Path.cwd() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class EnvSummary:
    python: str
    torch: str
    cuda_available: bool
    device_count: int
    capability: tuple[int, int] | None
    dtype: str
    run_root: str

    def as_dict(self) -> dict:
        return asdict(self)


def summarize() -> EnvSummary:
    try:
        import torch
        torch_version, cuda = torch.__version__, torch.cuda.is_available()
        count = torch.cuda.device_count() if cuda else 0
    except ImportError:
        torch_version, cuda, count = "not installed", False, 0
    cap = device_capability()
    return EnvSummary(
        python=sys.version.split()[0],
        torch=torch_version,
        cuda_available=cuda,
        device_count=count,
        capability=cap,
        dtype=preferred_dtype_str(cap),
        run_root=str(run_root()),
    )
