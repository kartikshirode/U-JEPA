"""Tests for env detection.

These do not require GPU or network. They just exercise the public surface so
a refactor that breaks the contract gets caught.
"""

from __future__ import annotations

from u_jepa_v2 import env


def test_runtime_returns_known_value():
    assert env.detect_runtime() in {"kaggle", "colab", "local"}


def test_device_returns_known_value():
    assert env.detect_device() in {"cuda", "cpu"}


def test_preferred_dtype_is_fp16():
    assert env.preferred_dtype_str() == "fp16"


def test_summary_roundtrips_to_dict():
    s = env.summary()
    d = s.as_dict()
    for key in ("runtime", "device", "gpu_name", "gpu_vram_gb", "native_bf16", "python", "platform"):
        assert key in d


def test_native_bf16_flag_consistent_with_device():
    if env.detect_device() == "cpu":
        assert env.has_native_bf16() is False
