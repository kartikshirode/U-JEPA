"""Hand-computed checks of BWT and forgetting on small accuracy matrices.

These complement test_metrics.py with the exact Phase 1 scenario: a model learns
task 0, then task 1, and task 0 accuracy drops. The numbers here are worked out
by hand so a refactor of the metric formulas gets caught. A[i][j] is accuracy on
task j after training through task i.
"""
import pytest

from u_jepa.eval.metrics import backward_transfer, average_forgetting


def test_task0_drops_after_task1_bwt_and_forgetting():
    # Stage 0 (after task 0): task0 = 0.80, task1 unseen = 0.0
    # Stage 1 (after task 1): task0 = 0.50 (forgot), task1 = 0.90
    A = [
        [0.80, 0.00],
        [0.50, 0.90],
    ]
    # BWT = A[1][0] - A[0][0] = 0.50 - 0.80 = -0.30
    assert backward_transfer(A) == pytest.approx(-0.30)
    # Forgetting task0 = max past (A[0][0]=0.80, A[1][0]=0.50) - final 0.50 = 0.30
    assert average_forgetting(A) == pytest.approx(0.30)


def test_catastrophic_forgetting_task0_to_near_zero():
    # The pivot-trigger case: task 0 collapses after task 1.
    A = [
        [0.85, 0.00],
        [0.02, 0.88],
    ]
    assert backward_transfer(A) == pytest.approx(0.02 - 0.85)
    assert average_forgetting(A) == pytest.approx(0.85 - 0.02)


def test_no_forgetting_is_zero_with_neutral_bwt():
    # task 0 fully retained after task 1.
    A = [
        [0.80, 0.00],
        [0.80, 0.70],
    ]
    assert backward_transfer(A) == pytest.approx(0.0)
    assert average_forgetting(A) == pytest.approx(0.0)


def test_three_task_forgetting_uses_max_past_accuracy():
    # task0 over stages: 0.9 (s0), 0.7 (s1), 0.4 (s2)
    # task1 over stages:      -    0.8 (s1), 0.5 (s2)
    A = [
        [0.9, 0.0, 0.0],
        [0.7, 0.8, 0.0],
        [0.4, 0.5, 0.6],
    ]
    # Forgetting task0 = max(0.9, 0.7, 0.4) - 0.4 = 0.5
    # Forgetting task1 = max(0.8, 0.5)      - 0.5 = 0.3
    # average = (0.5 + 0.3) / 2 = 0.4
    assert average_forgetting(A) == pytest.approx(0.4)
    # BWT = mean(A[2][0]-A[0][0], A[2][1]-A[1][1]) = ((0.4-0.9)+(0.5-0.8))/2 = -0.4
    assert backward_transfer(A) == pytest.approx(-0.4)


def test_single_task_matrix_is_neutral():
    # With one task there is nothing to forget.
    assert backward_transfer([[0.7]]) == 0.0
    assert average_forgetting([[0.7]]) == 0.0
