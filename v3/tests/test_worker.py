"""Shard resolution under Slurm, and picking the right experiment driver."""
from __future__ import annotations

import json

import pytest

from u_jepa_v3.cluster import slurm_context
from u_jepa_v3.runs import worker


def ctx(**env):
    return slurm_context(env)


def test_explicit_flags_win_over_the_array():
    resolved = worker.resolve_shard(2, 8, ctx(SLURM_JOB_ID="1",
                                              SLURM_ARRAY_TASK_ID="5",
                                              SLURM_ARRAY_TASK_COUNT="14"))
    assert resolved == (2, 8)


def test_the_array_supplies_the_shard_when_the_flags_are_absent():
    resolved = worker.resolve_shard(None, None, ctx(SLURM_JOB_ID="1",
                                                    SLURM_ARRAY_TASK_ID="5",
                                                    SLURM_ARRAY_TASK_COUNT="14"))
    assert resolved == (5, 14)


def test_a_one_based_array_is_shifted_back_into_range(monkeypatch):
    """An --array=1-14 range would otherwise put the last shard out of bounds."""
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "1")
    resolved = worker.resolve_shard(None, None, ctx(SLURM_JOB_ID="1",
                                                    SLURM_ARRAY_TASK_ID="14",
                                                    SLURM_ARRAY_TASK_COUNT="14"))
    assert resolved == (13, 14)


def test_half_a_shard_specification_is_refused():
    with pytest.raises(SystemExit, match="both --node and --of"):
        worker.resolve_shard(1, None, ctx())


def test_outside_an_array_the_flags_are_required():
    with pytest.raises(SystemExit, match="required outside a Slurm array"):
        worker.resolve_shard(None, None, ctx())


def test_an_unknown_arm_raises_instead_of_running_the_wrong_experiment():
    with pytest.raises(ValueError, match="unknown arm"):
        worker._default_runner({"arm": "rq9"})


def test_a_dry_run_reports_the_split_without_touching_a_gpu(tmp_path, capsys):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"editor": ["a", "b", "c"], "seed": [0, 1]}),
                    encoding="utf-8")
    code = worker.main(["--grid", str(grid), "--out", str(tmp_path / "out"),
                        "--node", "0", "--of", "3", "--dry-run"])
    assert code == 0
    assert "node 0/3: 2 assigned, 2 pending" in capsys.readouterr().out


def test_a_dry_run_does_not_set_the_visible_devices(tmp_path, monkeypatch):
    """Under Slurm that variable is the allocation, and overwriting it blinds the job."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"seed": [0]}), encoding="utf-8")
    worker.main(["--grid", str(grid), "--out", str(tmp_path / "out"),
                 "--node", "2", "--of", "4", "--dry-run"])
    import os

    assert "CUDA_VISIBLE_DEVICES" not in os.environ
