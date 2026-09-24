"""Confusion-point extraction (CLAUDE.md Sec. 7.4).

The rule is asymmetric by classifier and deliberately so: the SVM uses maximal
confusion, the MLP and CNN the first point where P(deloc) drops below 0.5,
because they jump rather than crossing over gradually.

Each rule reads ``P(delocalized)`` sampled along a generic ascending ``axis``,
which may be the randomness strength (Paper A) or the time step (the
time-resolved use).
"""

from __future__ import annotations

import numpy as np


def first_below_half(axis: np.ndarray, p_deloc: np.ndarray) -> float | None:
    """MLP/CNN rule: first axis point with ``P(deloc) < 0.5``, else None."""
    axis, p = _check(axis, p_deloc)
    below = np.flatnonzero(p < 0.5)
    return None if below.size == 0 else float(axis[below[0]])


def sustained_below_half(axis: np.ndarray, p_deloc: np.ndarray) -> float | None:
    """First point below 0.5 that the curve never comes back up from.

    Not the Sec. 7.4 rule -- a robustness check on it, since with finitely many
    realisations P(deloc) can dip below 0.5 once before the transition proper.
    Quote the two together when they disagree.
    """
    axis, p = _check(axis, p_deloc)
    above = np.flatnonzero(p >= 0.5)
    if above.size == 0:
        return float(axis[0])
    last_above = int(above[-1])
    return None if last_above == p.size - 1 else float(axis[last_above + 1])


def max_confusion(axis: np.ndarray, p_deloc: np.ndarray) -> float:
    """SVM rule: where the two class probabilities are equal, i.e. min |P - 0.5|.

    Do not apply this one to an MLP or CNN.
    """
    axis, p = _check(axis, p_deloc)
    return float(axis[int(np.argmin(np.abs(p - 0.5)))])


def onset(axis: np.ndarray, score: np.ndarray, threshold: float = 0.95) -> float | None:
    """First axis point where ``score`` reaches ``threshold`` and stays there.

    Used with time on the axis and test accuracy as the score. Requiring the
    threshold to hold for the rest of the axis keeps one lucky slice from
    setting the answer.
    """
    axis, s = _check(axis, score)
    ok = s >= threshold
    if not ok.any():
        return None
    failed = np.flatnonzero(~ok)
    start = 0 if failed.size == 0 else int(failed[-1]) + 1
    return None if start >= s.size else float(axis[start])


def _check(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if axis.shape != values.shape:
        raise ValueError(f"axis {axis.shape} and values {values.shape} must match")
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"axis must be a non-empty 1-D array, got {axis.shape}")
    if np.any(np.diff(axis) < 0):
        raise ValueError("axis must be sorted ascending")
    return axis, values
