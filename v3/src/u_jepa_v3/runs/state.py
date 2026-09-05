"""Per-cell state. Cells are atomic; there is no mid-cell resume.

An earlier design saved a counter and resumed by skipping that many candidates.
Weights, editor normalization state and RNG state were never saved, so the
resumed process built a fresh editor and continued from the base model while
believing 40,000 edits had landed. Every number after a restart was wrong and
nothing in the output said so.

So a cell either finishes or is rerun from zero, and only `finished` cells are
skipped. Checkpoints are kept because they are the time series RQ1 reports, not
because anything resumes from them.

Writes go to a temp file in the same directory and are then renamed, which is
atomic on one filesystem. Serialisation happens before the temp file is opened
so an unserialisable payload cannot destroy the previous good state.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunState:
    cell_id: str
    checkpoints: list[dict] = field(default_factory=list)
    finished: bool = False
    meta: dict = field(default_factory=dict)


def save(state: RunState, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(state), indent=2)  # raises before we touch disk
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load(path: str | Path) -> RunState:
    return RunState(**json.loads(Path(path).read_text(encoding="utf-8")))


def is_finished(path: str | Path) -> bool:
    """True only for a readable state file whose cell ran to completion."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        return load(path).finished
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
