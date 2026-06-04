"""Column-alignment guarantees for build_accuracy_matrix.

build_accuracy_matrix returns ONE row of the accuracy matrix: column j is the
accuracy on task task_order[j]. That row feeds straight into the BWT / forgetting
metrics. If a column is misaligned (e.g. eval_sets dict order leaking in), the
forgetting number is computed against the wrong task and the result is garbage
that still looks plausible. These tests pin the contract: row length == n,
unseen tasks are 0.0, and column j always maps to task_order[j] regardless of how
eval_sets was built. We monkeypatch eval_task with a spy so no model is loaded.
"""
from u_jepa.eval import continual as ec
from u_jepa.eval.continual import build_accuracy_matrix


def test_row_length_and_unseen_zero(monkeypatch):
    monkeypatch.setattr(
        ec, "eval_task",
        lambda bank, tok, tid, items, device="cuda:0", max_new_tokens=8: {"fomc": 0.7, "scienceqa": 0.6}[tid],
    )
    eval_sets = {"fomc": [{}], "scienceqa": [{}]}
    task_order = ["fomc", "scienceqa"]

    # Stage 0: only fomc seen. scienceqa column must be the 0.0 placeholder.
    row0 = build_accuracy_matrix(object(), object(), eval_sets, seen_tasks=["fomc"],
                                 device="cpu", task_order=task_order)
    assert len(row0) == len(task_order)
    assert row0 == [0.7, 0.0]

    # Stage 1: both seen.
    row1 = build_accuracy_matrix(object(), object(), eval_sets,
                                 seen_tasks=["fomc", "scienceqa"],
                                 device="cpu", task_order=task_order)
    assert row1 == [0.7, 0.6]


def test_columns_follow_task_order_not_dict_order(monkeypatch):
    monkeypatch.setattr(
        ec, "eval_task",
        lambda bank, tok, tid, items, device="cuda:0", max_new_tokens=8: {"fomc": 0.9, "scienceqa": 0.1}[tid],
    )
    # eval_sets built scienceqa-first, but task_order says fomc is column 0.
    eval_sets = {}
    eval_sets["scienceqa"] = [{}]
    eval_sets["fomc"] = [{}]
    row = build_accuracy_matrix(object(), object(), eval_sets,
                                seen_tasks=["fomc", "scienceqa"], device="cpu",
                                task_order=["fomc", "scienceqa"])
    # Column 0 is fomc (0.9), column 1 is scienceqa (0.1) despite dict order.
    assert row == [0.9, 0.1]


def test_eval_task_called_with_matching_dataset(monkeypatch):
    # The dataset handed to eval_task must be the one keyed by the task name.
    seen = {}

    def spy(bank, tok, tid, items, device="cuda:0", max_new_tokens=8):
        seen[tid] = items
        return 0.5

    monkeypatch.setattr(ec, "eval_task", spy)
    ds_a, ds_b = [{"a": 1}], [{"b": 2}]
    eval_sets = {"b": ds_b, "a": ds_a}
    build_accuracy_matrix(object(), object(), eval_sets, seen_tasks=["a", "b"],
                          device="cpu", task_order=["a", "b"])
    assert seen["a"] is ds_a
    assert seen["b"] is ds_b


def test_three_task_lower_triangle_shape(monkeypatch):
    monkeypatch.setattr(
        ec, "eval_task",
        lambda bank, tok, tid, items, device="cuda:0", max_new_tokens=8: {"t0": 0.3, "t1": 0.4, "t2": 0.5}[tid],
    )
    eval_sets = {"t0": [{}], "t1": [{}], "t2": [{}]}
    order = ["t0", "t1", "t2"]

    # After stage 0: only t0 seen -> [0.3, 0.0, 0.0]
    r0 = build_accuracy_matrix(object(), object(), eval_sets, seen_tasks=["t0"],
                               device="cpu", task_order=order)
    assert r0 == [0.3, 0.0, 0.0]
    # After stage 1: t0, t1 seen -> [0.3, 0.4, 0.0]
    r1 = build_accuracy_matrix(object(), object(), eval_sets, seen_tasks=["t0", "t1"],
                               device="cpu", task_order=order)
    assert r1 == [0.3, 0.4, 0.0]
    # After stage 2: all seen -> [0.3, 0.4, 0.5]
    r2 = build_accuracy_matrix(object(), object(), eval_sets, seen_tasks=order,
                               device="cpu", task_order=order)
    assert r2 == [0.3, 0.4, 0.5]
