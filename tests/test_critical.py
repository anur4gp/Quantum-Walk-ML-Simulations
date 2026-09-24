"""Critical-point extraction rules (CLAUDE.md Sec. 7.4).

The asymmetry is what these lock down: MLP/CNN take the first crossing of 0.5,
the SVM takes maximal confusion. Letting one stand in for the other would be a
silent method swap.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.critical import (
    first_below_half,
    max_confusion,
    onset,
    sustained_below_half,
)

AXIS = np.linspace(0.0, 1.0, 11)


def test_first_below_half_takes_the_first_crossing() -> None:
    p = np.array([1.0, 1.0, 1.0, 0.9, 0.6, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0])
    assert first_below_half(AXIS, p) == pytest.approx(0.5)


def test_first_below_half_is_not_the_midpoint() -> None:
    """A gradual crossover must still report the first point, not the centre."""
    p = np.linspace(1.0, 0.0, 11)
    assert first_below_half(AXIS, p) == pytest.approx(0.6)
    assert max_confusion(AXIS, p) == pytest.approx(0.5)


def test_no_crossing_returns_none() -> None:
    assert first_below_half(AXIS, np.full(11, 0.8)) is None
    assert sustained_below_half(AXIS, np.full(11, 0.8)) is None


def test_always_below_returns_the_first_axis_point() -> None:
    assert sustained_below_half(AXIS, np.full(11, 0.2)) == pytest.approx(0.0)


def test_sustained_ignores_a_single_early_dip() -> None:
    """A noisy dip sets the plain rule but not the robustness check."""
    p = np.array([1.0, 1.0, 0.4, 0.9, 0.8, 0.7, 0.6, 0.3, 0.1, 0.0, 0.0])
    assert first_below_half(AXIS, p) == pytest.approx(0.2)
    assert sustained_below_half(AXIS, p) == pytest.approx(0.7)


def test_max_confusion_finds_the_equal_probability_point() -> None:
    p = np.array([1.0, 0.95, 0.9, 0.8, 0.7, 0.51, 0.2, 0.1, 0.0, 0.0, 0.0])
    assert max_confusion(AXIS, p) == pytest.approx(0.5)


def test_onset_requires_the_threshold_to_hold() -> None:
    score = np.array([0.5, 0.6, 0.97, 0.8, 0.96, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert onset(AXIS, score, 0.95) == pytest.approx(0.4)


def test_onset_never_reached() -> None:
    assert onset(AXIS, np.full(11, 0.5), 0.95) is None


def test_onset_from_the_first_point() -> None:
    assert onset(AXIS, np.ones(11), 0.95) == pytest.approx(0.0)


def test_onset_with_a_boolean_gate() -> None:
    """Threshold 1.0 on a 0/1 array is 'first index from which all are true'."""
    gate = np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    assert onset(AXIS, gate, 1.0) == pytest.approx(0.3)


@pytest.mark.parametrize(
    "axis,values",
    [
        (np.array([0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([]), np.array([])),
        (np.array([1.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0])),
    ],
)
def test_malformed_input_is_rejected(axis, values) -> None:
    with pytest.raises(ValueError):
        first_below_half(axis, values)
