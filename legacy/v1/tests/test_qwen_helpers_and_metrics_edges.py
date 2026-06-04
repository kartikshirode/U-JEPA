"""Tests for qwen_base helpers (vram counter, target module name finder)
and metric edge cases (empty matrices, single-task forward transfer)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from u_jepa.eval.metrics import (
    average_accuracy,
    average_forgetting,
    backward_transfer,
    forward_transfer,
)
from u_jepa.models.qwen_base import qwen_vram_usage_mb, target_module_names


class _MiniQwenLike(nn.Module):
    """Looks just enough like a Qwen attention stack for target_module_names
    and the vram counter. Two layers, each with q_proj/k_proj/v_proj/o_proj."""

    def __init__(self, hidden=32, n_layers=2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            block = nn.Module()
            block.q_proj = nn.Linear(hidden, hidden, bias=False)
            block.k_proj = nn.Linear(hidden, hidden, bias=False)
            block.v_proj = nn.Linear(hidden, hidden, bias=False)
            block.o_proj = nn.Linear(hidden, hidden, bias=False)
            layers.append(block)
        self.layers = nn.ModuleList(layers)


def test_target_module_names_finds_q_and_v_only_by_default():
    model = _MiniQwenLike(hidden=16, n_layers=3)
    names = target_module_names(model)
    # 3 layers * (q_proj + v_proj) = 6 matches
    assert len(names) == 6
    assert all(n.endswith("q_proj") or n.endswith("v_proj") for n in names)
    assert not any(n.endswith("k_proj") or n.endswith("o_proj") for n in names)


def test_target_module_names_respects_custom_target_short_names():
    model = _MiniQwenLike(hidden=8, n_layers=2)
    names = target_module_names(model, target_short_names=("k_proj", "o_proj"))
    assert len(names) == 4
    assert all(n.endswith("k_proj") or n.endswith("o_proj") for n in names)


def test_target_module_names_skips_non_linear_modules():
    """If a target-short-named attribute is NOT an nn.Linear (e.g. swapped
    out for a quantized 4bit module), the helper must not return it.
    Stand-in: an Identity layer named q_proj."""

    class _Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Identity()
            self.v_proj = nn.Linear(8, 8, bias=False)

    matches = target_module_names(_Weird())
    assert matches == ["v_proj"]


def test_qwen_vram_usage_mb_is_close_to_hand_count():
    # A single 1024 x 1024 fp32 linear is 4 MB.
    m = nn.Linear(1024, 1024, bias=False)
    mb = qwen_vram_usage_mb(m)
    # 1024 * 1024 * 4 bytes = 4 MiB.
    assert mb == 4


def test_qwen_vram_usage_includes_buffers():
    class _WithBuf(nn.Module):
        def __init__(self):
            super().__init__()
            # 256 * 256 * 4 bytes = 256 KiB so under 1 MiB.
            self.register_buffer("buf", torch.zeros(256, 256))
            # 512 * 512 * 4 = 1 MiB
            self.lin = nn.Linear(512, 512, bias=False)

    mb = qwen_vram_usage_mb(_WithBuf())
    assert mb == 1  # only the lin layer reaches 1 MiB


def test_average_accuracy_empty_matrix_returns_zero():
    assert average_accuracy([]) == 0.0


def test_forward_transfer_zero_when_single_task():
    assert forward_transfer([[0.5]], [0.0]) == 0.0


def test_average_forgetting_zero_when_single_task():
    assert average_forgetting([[0.5]]) == 0.0
