"""Atomic save/load for the things that must survive a 9h-session kill.

The frozen model weights do not need saving. Per the build plan, the things
that DO need to persist are: memory store, codebook entries, probe sets,
thresholds, results logs, and a tiny RunState (current phase, step, seed,
elapsed wallclock). Keep it boring; trust the filesystem; checksums optional.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(obj, np.generic):
        # np.float32 etc. are not Python floats; .item() converts losslessly.
        return obj.item()
    raise TypeError(
        f"save_json cannot serialise {type(obj).__name__!s}: {obj!r}. "
        "Convert it to a native type before persisting."
    )


def atomic_write_text(path: Path, text: str) -> None:
    """Write text via temp file + os.replace so a crash leaves the old file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            # fsync before replace: os.replace alone orders the rename, but a
            # host crash could still land an empty file without this.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    text = json.dumps(_to_jsonable(data), indent=indent, sort_keys=True, ensure_ascii=False)
    atomic_write_text(path, text + "\n")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch(path: str | Path, state: Mapping[str, Any]) -> Path:
    """Atomic torch.save. Used for tensors (e.g. memory keys/values, codebook)."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pt.tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(dict(state), tmp_name)
        # rb+ because Windows refuses fsync on a read-only handle.
        with open(tmp_name, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def load_torch(path: str | Path) -> dict[str, Any]:
    import torch
    # weights_only=True: saved state is tensors-only by design (see save_torch),
    # and untrusted .pt files must never get full pickle execution.
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def sha256_of_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass
class RunState:
    """Minimal resume state. Update + save_json after each meaningful step."""
    phase: str
    step: int = 0
    seed: int = 0
    elapsed_seconds: float = 0.0
    last_checkpoint: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def kaggle_working_dir() -> Path:
    """Checkpoint root. U_JEPA_V2_CKPT_DIR wins if set; /kaggle/working on
    Kaggle; locally <project root>/checkpoints, where the root is the first
    ancestor of this file containing .git or pyproject.toml.
    """
    override = os.environ.get("U_JEPA_V2_CKPT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent / "checkpoints"
    # No marker found (e.g. installed into site-packages): cwd-relative is the
    # last resort, resolved so the chosen path is at least absolute and loggable.
    return Path("checkpoints").resolve()


def kaggle_input_path(dataset_slug: str) -> Path | None:
    """Resolves /kaggle/input/<dataset_slug> if mounted, else None.

    Used to pick up a previously-saved RunState/state dataset from a prior session.
    """
    candidate = Path("/kaggle/input") / dataset_slug
    return candidate if candidate.exists() else None


class CheckpointTimer:
    """Tiny context manager so we can attribute wallclock to phases."""

    def __init__(self) -> None:
        self.start: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "CheckpointTimer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.start is not None:
            self.elapsed += time.perf_counter() - self.start
            self.start = None
