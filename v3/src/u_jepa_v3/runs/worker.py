"""CLI: run this node's share of a grid, skipping finished cells.

    python -m u_jepa_v3.runs.worker --grid grids/rq1.json --out runs/rq1 \
        --node 0 --of 4

Under a Slurm array, --node and --of default to SLURM_ARRAY_TASK_ID and
SLURM_ARRAY_TASK_COUNT, so the same command works unchanged in a job script.

The runner is injected rather than imported so the sharding logic stays testable
on CPU and the RQ1 driver stays out of it. `--dry-run` reports the pending count
and exits, which is how you check a shard split before spending GPU hours on it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

from ..cluster import slurm_context
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


ARMS = ("rq1", "rq2")


def _default_runner(params: dict) -> RunState:
    """Build this cell from its params and run it.

    The `arm` key picks the driver: rq1 for the undefended pipeline, rq2 for the
    gated one. It defaults to rq1 so existing grids keep working, and an unknown
    value raises rather than silently running the wrong experiment.

    Imported lazily so `--dry-run` works without torch, transformers or
    easyeditor installed.
    """
    arm = params.get("arm", "rq1")
    if arm == "rq1":
        from ..experiments.rq1_survival import run_cell_from_params
    elif arm == "rq2":
        from ..experiments.rq2_gate import run_cell_from_params
    else:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    return run_cell_from_params(params)


def resolve_shard(node: int | None, of: int | None, ctx) -> tuple[int, int]:
    """Explicit flags win; otherwise take the array task coordinates.

    Slurm numbers array tasks from whatever the --array range says, so a range
    starting at 1 would push the last shard out of bounds. The offset is removed
    by SLURM_ARRAY_TASK_MIN when Slurm provides it, and the job scripts here use
    0-based ranges anyway.
    """
    if node is not None and of is not None:
        return node, of
    if (node is None) != (of is None):
        raise SystemExit("pass both --node and --of, or neither")
    if not ctx.is_array:
        raise SystemExit(
            "--node and --of are required outside a Slurm array. Inside one they "
            "default to SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT."
        )
    offset = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0") or 0)
    return ctx.array_task_id - offset, ctx.array_task_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="u_jepa_v3.runs.worker")
    parser.add_argument("--grid", required=True, help="JSON file mapping dimension to list")
    parser.add_argument("--out", required=True, help="directory for per-cell state files")
    parser.add_argument("--node", type=int, default=None,
                        help="shard index; defaults to SLURM_ARRAY_TASK_ID")
    parser.add_argument("--of", type=int, default=None,
                        help="shard count; defaults to SLURM_ARRAY_TASK_COUNT")
    parser.add_argument("--device", default=None,
                        help="CUDA device index to pin. Leave unset under Slurm, "
                             "which has already narrowed CUDA_VISIBLE_DEVICES to "
                             "the allocated slice")
    parser.add_argument("--dry-run", action="store_true", help="report pending cells and exit")
    args = parser.parse_args(argv)

    ctx = slurm_context()
    node, of = resolve_shard(args.node, args.of, ctx)

    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    mine = shard(expand(grid), node=node, of=of)
    todo = pending(mine, args.out)
    print(f"node {node}/{of}: {len(mine)} assigned, {len(todo)} pending")
    if ctx.under_slurm:
        print(f"  slurm {ctx.as_dict()}")

    if args.dry_run:
        return 0

    # Under Slurm the allocation already fixed which MIG slice this process can
    # see, and it is device 0 whatever the physical index was. Overwriting the
    # variable with the shard index makes the process see nothing at all.
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    elif not ctx.under_slurm:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(node)

    for index, cell in enumerate(todo, start=1):
        print(f"  [{index}/{len(todo)}] {cell.cell_id} {cell.params}", flush=True)
        path = run_cell(cell, args.out, _default_runner)
        print(f"      wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
