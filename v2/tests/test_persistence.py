"""Tests for atomic JSON persistence + RunState.

Tensor save/load is tested separately under tests/test_persistence_torch.py
(skipped without torch import being trivial here; torch is a hard dep, so it
should always import in dev).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from u_jepa_v2 import persistence as P


def test_save_and_load_json_roundtrip(tmp_path: Path):
    payload = {"phase": "0", "step": 12, "thresholds": {"perplexity": 5.5}}
    path = tmp_path / "state.json"
    P.save_json(path, payload)
    loaded = P.load_json(path)
    assert loaded == payload


def test_save_json_writes_atomically(tmp_path: Path, monkeypatch):
    """If the writer crashes mid-way the old file must still be readable."""
    path = tmp_path / "state.json"
    P.save_json(path, {"v": 1})

    real_replace = P.os.replace

    def crashing_replace(src, dst):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(P.os, "replace", crashing_replace)

    with pytest.raises(RuntimeError):
        P.save_json(path, {"v": 2})

    monkeypatch.setattr(P.os, "replace", real_replace)
    assert P.load_json(path) == {"v": 1}
    tmp_leftovers = [p for p in path.parent.iterdir() if p.name.startswith(path.name + ".") and p.name.endswith(".tmp")]
    assert tmp_leftovers == [], f"leftover temp files: {tmp_leftovers}"


def test_save_json_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "a" / "b" / "c.json"
    P.save_json(path, [1, 2, 3])
    assert path.exists()


def test_dataclass_is_serialisable(tmp_path: Path):
    @dataclass
    class Cfg:
        name: str
        n: int

    state = P.RunState(phase="0", step=3, extras={"cfg": Cfg("x", 4)})
    path = tmp_path / "rs.json"
    P.save_json(path, state)
    raw = json.loads(path.read_text())
    assert raw["phase"] == "0"
    assert raw["step"] == 3
    assert raw["extras"]["cfg"] == {"name": "x", "n": 4}


def test_to_jsonable_numpy_scalars_roundtrip(tmp_path: Path):
    np = pytest.importorskip("numpy")
    payload = {"f": np.float32(0.55), "i": np.int64(3), "b": np.bool_(True)}
    path = tmp_path / "np.json"
    P.save_json(path, payload)
    loaded = P.load_json(path)
    assert loaded["f"] == pytest.approx(0.55)
    assert loaded["i"] == 3
    assert loaded["b"] is True


def test_to_jsonable_path_becomes_string(tmp_path: Path):
    path = tmp_path / "p.json"
    P.save_json(path, {"where": Path("a") / "b"})
    assert P.load_json(path)["where"] == str(Path("a") / "b")


def test_to_jsonable_unknown_type_raises(tmp_path: Path):
    class Weird:
        pass

    with pytest.raises(TypeError):
        P.save_json(tmp_path / "w.json", {"x": Weird()})


def test_kaggle_working_dir_defaults_to_local_checkpoints(monkeypatch):
    monkeypatch.delenv("U_JEPA_V2_CKPT_DIR", raising=False)
    p = P.kaggle_working_dir()
    assert isinstance(p, Path)
    assert p.name in {"working", "checkpoints"}
    assert p.is_absolute()


def test_kaggle_working_dir_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("U_JEPA_V2_CKPT_DIR", str(tmp_path / "ckpt"))
    assert P.kaggle_working_dir() == (tmp_path / "ckpt").resolve()


def test_sha256_matches_known_value(tmp_path: Path):
    f = tmp_path / "hello.txt"
    f.write_bytes(b"hello")
    assert P.sha256_of_file(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_checkpoint_timer_accumulates():
    t = P.CheckpointTimer()
    with t:
        time.sleep(0.01)
    first = t.elapsed
    assert first > 0
    with t:
        time.sleep(0.01)
    assert t.elapsed > first
