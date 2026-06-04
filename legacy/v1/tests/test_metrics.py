"""BWT, forgetting, average accuracy, and forward transfer."""
import pytest
from u_jepa.eval.metrics import (
    backward_transfer,
    average_forgetting,
    average_accuracy,
    forward_transfer,
)


def test_bwt_zero_when_single_task():
    assert backward_transfer([[0.8]]) == 0.0


def test_bwt_zero_when_final_matches_diagonal():
    A = [[0.8, 0.0], [0.8, 0.7]]
    assert backward_transfer(A) == 0.0


def test_bwt_negative_when_forgetting():
    A = [[0.8, 0.0], [0.6, 0.7]]
    assert round(backward_transfer(A), 3) == -0.2


def test_bwt_positive_when_backward_help():
    A = [[0.5, 0.0], [0.7, 0.6]]
    assert round(backward_transfer(A), 3) == 0.2


def test_average_forgetting_zero_when_no_drop():
    A = [[0.8, 0.0], [0.8, 0.7]]
    assert average_forgetting(A) == 0.0


def test_average_forgetting_uses_peak_not_diagonal():
    """Peak is max over k>=i, not just A[i][i]."""
    # task 0 peaked at A[1][0]=0.9 (mid-training), dropped to 0.5 at end
    A = [[0.8, 0.0, 0.0],
         [0.9, 0.7, 0.0],
         [0.5, 0.6, 0.6]]
    f = average_forgetting(A)
    # task 0 forgetting: peak 0.9 - final 0.5 = 0.4
    # task 1 forgetting: peak 0.7 - final 0.6 = 0.1
    assert round(f, 3) == round((0.4 + 0.1) / 2, 3)


def test_average_accuracy_is_final_row_mean():
    A = [[0.8, 0.0], [0.5, 0.7]]
    assert average_accuracy(A) == pytest.approx(0.6)


def test_forward_transfer_above_random_baseline():
    A = [[0.8, 0.0], [0.5, 0.6]]
    baseline = [0.0, 0.3]
    # FWT = (A[0][1] - b[1]) / 1 = (0.0 - 0.3) = -0.3
    assert round(forward_transfer(A, baseline), 3) == -0.3
