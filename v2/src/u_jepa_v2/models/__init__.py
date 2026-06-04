"""Frozen-core model loaders.

The build plan (F4) calls for GPT-2-XL for the editing-literature baselines
and Qwen2.5-1.5B-Instruct as the real system core. Both stay frozen.
"""

from .core import (
    CORE_MODELS,
    CoreSpec,
    freeze_,
    load_core,
    resolve_core,
)

__all__ = ["CORE_MODELS", "CoreSpec", "freeze_", "load_core", "resolve_core"]
