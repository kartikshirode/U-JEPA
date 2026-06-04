"""Runtime environment detection.

The build plan pins fp16 on T4 (no native bf16). Local machines may have bf16,
but for portability across Kaggle and laptop we default to fp16 unless the
caller asks otherwise.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, asdict
from typing import Literal


Runtime = Literal["kaggle", "colab", "local"]
DeviceStr = Literal["cuda", "cpu"]


def detect_runtime() -> Runtime:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or "google.colab" in os.environ.get("PYTHONPATH", ""):
        return "colab"
    return "local"


def detect_device() -> DeviceStr:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name() -> str | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except ImportError:
        return None


def gpu_vram_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except ImportError:
        return None


def has_native_bf16() -> bool:
    """T4 (Turing) returns False. Ampere+ returns True."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8
    except ImportError:
        return False


def preferred_dtype_str() -> str:
    """fp16 by default; bf16 only if hardware supports it AND the caller opts in.

    The plan locks fp16 for parity with Kaggle T4. This returns the safe default.
    """
    return "fp16"


@dataclass
class EnvSummary:
    runtime: Runtime
    device: DeviceStr
    gpu_name: str | None
    gpu_vram_gb: float | None
    native_bf16: bool
    python: str
    platform: str

    def as_dict(self) -> dict:
        return asdict(self)


def summary() -> EnvSummary:
    return EnvSummary(
        runtime=detect_runtime(),
        device=detect_device(),
        gpu_name=gpu_name(),
        gpu_vram_gb=gpu_vram_gb(),
        native_bf16=has_native_bf16(),
        python=platform.python_version(),
        platform=platform.platform(),
    )
