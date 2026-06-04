"""Continual-learning metrics following Lopez-Paz and Ranzato (2017).

Convention: A[i][j] = accuracy on task j after training through task i.
A is an upper-triangular-ish matrix - entries with j > i are typically 0
(task j has not been seen yet) but we do not enforce that.
"""
from __future__ import annotations
from typing import Sequence


def backward_transfer(A: Sequence[Sequence[float]]) -> float:
    """BWT = (1/(T-1)) * sum_{i<T-1} (A[T-1][i] - A[i][i]).

    Positive: training on later tasks helped earlier ones.
    Zero: no effect.
    Negative: forgetting.
    """
    T = len(A)
    if T < 2:
        return 0.0
    return sum(A[T - 1][i] - A[i][i] for i in range(T - 1)) / (T - 1)


def average_forgetting(A: Sequence[Sequence[float]]) -> float:
    """Forgetting on task i = max_{k>=i, k<T} A[k][i] - A[T-1][i].

    Averaged over the first T-1 tasks.
    """
    T = len(A)
    if T < 2:
        return 0.0
    total = 0.0
    for i in range(T - 1):
        peak = max(A[k][i] for k in range(i, T))
        total += peak - A[T - 1][i]
    return total / (T - 1)


def average_accuracy(A: Sequence[Sequence[float]]) -> float:
    """A_T = (1/T) * sum_i A[T-1][i]. Final-row mean."""
    T = len(A)
    if T == 0:
        return 0.0
    return sum(A[T - 1]) / T


def forward_transfer(A: Sequence[Sequence[float]], b: Sequence[float]) -> float:
    """FWT = (1/(T-1)) * sum_{i>0} (A[i-1][i] - b[i]).

    b: baseline accuracies on each task before any training (random init).
    Positive: learning task i-1 helped on task i above the random baseline.
    """
    T = len(A)
    if T < 2:
        return 0.0
    return sum(A[i - 1][i] - b[i] for i in range(1, T)) / (T - 1)
