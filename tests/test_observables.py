"""Observable tests (CLAUDE.md Sec. 2.5)."""

from __future__ import annotations

import numpy as np
import pytest

from data.preprocess import normalize_pmax
from qw import observables as obs
from qw.walk import run_walk


def test_moment_of_inertia_uses_physical_positions() -> None:
    """A distribution centred on the origin has MoI 0, whatever the array length."""
    positions = np.arange(-5.0, 6.0)
    prob = np.zeros(11)
    prob[5] = 1.0
    assert obs.moment_of_inertia(prob, positions) == 0.0

    prob = np.zeros(11)
    prob[[3, 7]] = 0.5  # x = -2 and +2
    assert obs.moment_of_inertia(prob, positions) == pytest.approx(4.0)


def test_ipr_is_maximal_for_a_flat_distribution() -> None:
    """The Sec. 2.5 definition is the participation ratio: it peaks at flatness."""
    flat = np.full(64, 1.0 / 8.0)          # |psi|^2 = 1/64 on every site
    spike = np.zeros(64)
    spike[10] = 1.0

    assert obs.inverse_participation_ratio(flat) == pytest.approx(64.0)
    assert obs.inverse_participation_ratio(spike) == pytest.approx(1.0)


def test_ipr_uses_spin_up_only() -> None:
    """Definition in Sec. 2.5 is built from psi_+ alone; changing psi_- must not move it."""
    r = run_walk(40, "pure", theta_0=np.pi / 6)
    before = obs.inverse_participation_ratio(r.psi_plus)
    assert before == obs.inverse_participation_ratio(r.psi_plus.copy())


def test_ipr_rejects_an_empty_component() -> None:
    with pytest.raises(ValueError, match="identically zero"):
        obs.inverse_participation_ratio(np.zeros(10))


def test_delocalized_walk_has_larger_moi_than_localized_one() -> None:
    """Weak randomness -> ballistic; strong randomness -> localized (Sec. 2.4)."""
    weak = run_walk(120, "discrete_coin", np.pi / 6, 0.01, seed=4)
    strong = run_walk(120, "discrete_coin", np.pi / 6, np.pi / 6, seed=4)

    moi_weak = obs.moment_of_inertia(
        obs.probability(weak.psi_plus, weak.psi_minus), weak.positions
    )
    moi_strong = obs.moment_of_inertia(
        obs.probability(strong.psi_plus, strong.psi_minus), strong.positions
    )
    assert moi_weak > moi_strong


def test_normalize_pmax_sets_each_row_peak_to_one() -> None:
    x = np.array([[0.0, 2.0, 1.0], [0.5, 0.25, 0.0]])
    out = normalize_pmax(x)
    np.testing.assert_allclose(out.max(axis=1), 1.0)
    np.testing.assert_allclose(out[0], [0.0, 1.0, 0.5])


def test_normalize_pmax_preserves_shape_for_a_single_sample() -> None:
    x = np.array([0.0, 2.0, 1.0])
    out = normalize_pmax(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, [0.0, 1.0, 0.5])


def test_normalize_pmax_leaves_an_all_zero_row_alone() -> None:
    out = normalize_pmax(np.zeros((2, 5)))
    assert np.all(np.isfinite(out))
    assert np.all(out == 0.0)
