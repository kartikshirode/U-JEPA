"""Atomic torch.save / torch.load roundtrip."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from u_jepa_v2 import persistence as P


class _NotATensor:
    """Module-level so pickle can reference it by qualname."""


def test_save_torch_roundtrips(tmp_path: Path):
    state = {"key": torch.arange(6, dtype=torch.float32).reshape(2, 3), "scalar": torch.tensor(7)}
    path = tmp_path / "memory.pt"
    P.save_torch(path, state)
    loaded = P.load_torch(path)
    assert torch.equal(loaded["key"], state["key"])
    assert torch.equal(loaded["scalar"], state["scalar"])


def test_save_torch_atomic_on_failure(tmp_path: Path, monkeypatch):
    path = tmp_path / "memory.pt"
    P.save_torch(path, {"v": torch.zeros(3)})

    def boom(src, dst):
        raise OSError("simulated crash")

    monkeypatch.setattr(P.os, "replace", boom)
    with pytest.raises(OSError):
        P.save_torch(path, {"v": torch.ones(3)})

    leftovers = [p for p in path.parent.iterdir() if p.name.endswith(".pt.tmp")]
    assert leftovers == [], f"leftover temp files: {leftovers}"


def test_load_torch_rejects_arbitrary_objects(tmp_path: Path):
    """weights_only=True must refuse payloads that need full pickle execution."""
    path = tmp_path / "sneaky.pt"
    torch.save({"x": _NotATensor()}, path)
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        P.load_torch(path)
