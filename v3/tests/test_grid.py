import json
import pytest
from u_jepa_v3.runs import worker
from u_jepa_v3.runs.grid import Cell, expand, pending, shard
from u_jepa_v3.runs.state import RunState, load, save


def test_expand_is_the_cartesian_product():
    cells = expand({"editor": ["a", "b"], "seed": [1, 2, 3]})
    assert len(cells) == 6


def test_cell_id_is_stable_across_key_order():
    assert Cell({"editor": "a", "seed": 1}).cell_id == Cell({"seed": 1, "editor": "a"}).cell_id


def test_cell_id_differs_on_different_params():
    assert Cell({"seed": 1}).cell_id != Cell({"seed": 2}).cell_id


def test_shard_partitions_without_overlap_or_loss():
    cells = expand({"x": list(range(10))})
    ids = [c.cell_id for n in range(3) for c in shard(cells, node=n, of=3)]
    assert len(ids) == 10 and len(set(ids)) == 10


def test_shard_is_interleaved_not_contiguous():
    cells = expand({"x": list(range(6))})
    assert [c.params["x"] for c in shard(cells, node=0, of=3)] == [0, 3]


def test_shard_rejects_a_bad_node_index():
    with pytest.raises(ValueError, match="node"):
        shard(expand({"x": [1]}), node=3, of=3)


def test_pending_skips_finished_cells_and_retries_unfinished(tmp_path):
    cells = expand({"x": [1, 2]})
    save(RunState(cell_id=cells[0].cell_id, finished=True), tmp_path / f"{cells[0].cell_id}.json")
    save(RunState(cell_id=cells[1].cell_id, finished=False), tmp_path / f"{cells[1].cell_id}.json")
    assert [c.cell_id for c in pending(cells, tmp_path)] == [cells[1].cell_id]


def test_run_cell_invokes_the_runner_and_writes_finished_state(tmp_path):
    cell = Cell({"editor": "stub", "seed": 0})
    seen = []

    def runner(params):
        seen.append(params)
        return RunState(cell_id="ignored", checkpoints=[{"at": 10}], finished=True)

    path = worker.run_cell(cell, tmp_path, runner)
    assert seen == [{"editor": "stub", "seed": 0}]
    state = load(path)
    assert state.finished and state.cell_id == cell.cell_id
    assert state.meta["params"] == {"editor": "stub", "seed": 0}


def test_a_runner_that_raises_leaves_the_cell_unfinished(tmp_path):
    cell = Cell({"editor": "stub", "seed": 0})

    def runner(params):
        raise RuntimeError("cuda oom")

    path = worker.run_cell(cell, tmp_path, runner)
    state = load(path)
    assert state.finished is False
    assert "cuda oom" in state.meta["error"]


def test_cli_dry_run_reports_pending_count(tmp_path, capsys):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"editor": ["a", "b"], "seed": [1, 2, 3]}), encoding="utf-8")
    rc = worker.main(["--grid", str(grid), "--out", str(tmp_path / "out"),
                      "--node", "0", "--of", "3", "--dry-run"])
    assert rc == 0 and "2 pending" in capsys.readouterr().out
