"""CLI: run this node's share of a grid, skipping finished cells.

    python -m u_jepa_v3.runs.worker --grid grids/rq1.json --out runs/rq1 \
        --node 0 --of 4

The runner is injected rather than imported so the sharding logic stays
testable on CPU and the RQ1 driver stays out of it. `--dry-run` reports the
pending count and exits, which is how you check a shard split before spending
GPU hours on it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

from .grid import Cell, expand, pending, shard
from .state import RunState, save


def run_cell(cell: Cell, out_dir: str | Path, runner: Callable[[dict], RunState]) -> Path:
    """Run one cell to completion and write its state. Never raises.

    A cell that dies is written unfinished with its error recorded, so the next
    pass picks it up and reruns it from zero. There is no partial resume; see
    the note in runs/state.py.
    """
    path = Path(out_dir) / f"{cell.cell_id}.json"
    try:
        state = runner(dict(cell.params))
        state.cell_id = cell.cell_id
        state.meta = {**state.meta, "params": dict(cell.params)}
    except Exception as exc:
        state = RunState(
            cell_id=cell.cell_id,
            finished=False,
            meta={"params": dict(cell.params), "error": f"{type(exc).__name__}: {exc}"},
        )
    save(state, path)
    return path


def _default_runner(params: dict) -> RunState:
    """Build the RQ1 cell from its params and run it.

    Imported lazily so `--dry-run` works without torch, transformers or
    easyeditor installed.
    """
    from ..experiments.rq1_survival import run_cell_from_params

    return run_cell_from_params(params)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="u_jepa_v3.runs.worker")
    parser.add_argument("--grid", required=True, help="JSON file mapping dimension to list")
    parser.add_argument("--out", required=True, help="directory for per-cell state files")
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--of", type=int, required=True)
    parser.add_argument("--device", default=None,
                        help="CUDA device index for this node; defaults to --node")
    parser.add_argument("--dry-run", action="store_true", help="report pending cells and exit")
    args = parser.parse_args(argv)

    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    mine = shard(expand(grid), node=args.node, of=args.of)
    todo = pending(mine, args.out)
    print(f"node {args.node}/{args.of}: {len(mine)} assigned, {len(todo)} pending")

    if args.dry_run:
        return 0

    # One process per node, one GPU per process. No collectives, so this holds
    # whatever the topology turns out to be.
    device = args.device if args.device is not None else str(args.node)
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    for index, cell in enumerate(todo, start=1):
        print(f"  [{index}/{len(todo)}] {cell.cell_id} {cell.params}", flush=True)
        path = run_cell(cell, args.out, _default_runner)
        print(f"      wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
