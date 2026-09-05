"""Grid expansion and idempotent sharding across nodes.

Topology of the 4 H200s is unconfirmed, so nothing here does collectives. Each
node takes an interleaved slice and writes one JSON per cell. Interleaved rather
than contiguous, because an unbalanced grid (3 editors by 5 seeds) would
otherwise pile the expensive cells onto one node.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from .state import is_finished


@dataclass(frozen=True)
class Cell:
    params: dict

    @property
    def cell_id(self) -> str:
        canonical = json.dumps(self.params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def expand(grid: dict[str, list]) -> list[Cell]:
    keys = sorted(grid)
    return [Cell(dict(zip(keys, v))) for v in itertools.product(*(grid[k] for k in keys))]


def shard(cells: list[Cell], node: int, of: int) -> list[Cell]:
    if of < 1:
        raise ValueError(f"of must be >= 1, got {of}")
    if not 0 <= node < of:
        raise ValueError(f"node must be in [0, {of}), got {node}")
    return [c for i, c in enumerate(cells) if i % of == node]


def pending(cells: list[Cell], out_dir: str | Path) -> list[Cell]:
    out_dir = Path(out_dir)
    return [c for c in cells if not is_finished(out_dir / f"{c.cell_id}.json")]
