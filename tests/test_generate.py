"""Dataset generation tests (CLAUDE.md Sec. 6, the data contract)."""

from __future__ import annotations

import numpy as np
import pytest

from data.generate import DELOCALIZED, LOCALIZED, make_dataset
from qw.observables import probability
from qw.walk import run_walk

DELOC = (0.0, 0.05)
LOC = (0.45, 0.52)


def test_shapes_labels_and_balance() -> None:
    ds = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=40, seed=0)
    assert ds.X.shape == (40, 2 * 30 + 1)
    assert ds.X.dtype == np.float64
    assert set(np.unique(ds.y)) == {DELOCALIZED, LOCALIZED}
    assert np.sum(ds.y == DELOCALIZED) == np.sum(ds.y == LOCALIZED) == 20


def test_labels_are_never_flipped() -> None:
    """Weak randomness -> 0, strong randomness -> 1."""
    ds = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=40, seed=0)
    assert ds.control_values[ds.y == DELOCALIZED].max() < ds.control_values[ds.y == LOCALIZED].min()


def test_rows_carry_physical_normalisation() -> None:
    """Samples leave here summing to 1; P_max=1 is the classifier's job."""
    ds = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=20, seed=0)
    np.testing.assert_allclose(ds.X.sum(axis=1), 1.0, atol=1e-12)


def test_metadata_reproduces_any_individual_sample() -> None:
    """Sec. 6 requires channel/theta_0/N/control_value/seed to travel along."""
    ds = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=10, seed=0)
    i = 3
    r = run_walk(
        ds.n,
        ds.channel,
        theta_0=ds.theta_0,
        control_value=float(ds.control_values[i]),
        seed=int(ds.seeds[i]),
        parity=ds.parity,
    )
    np.testing.assert_array_equal(probability(r.psi_plus, r.psi_minus), ds.X[i])


def test_master_seed_reproduces_the_whole_dataset() -> None:
    a = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=20, seed=7)
    b = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=20, seed=7)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.seeds, b.seeds)


def test_different_master_seeds_differ() -> None:
    a = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=20, seed=7)
    b = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=20, seed=8)
    assert not np.array_equal(a.X, b.X)


def test_samples_within_a_class_are_independent_realisations() -> None:
    ds = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=20, seed=0)
    assert len(set(ds.seeds.tolist())) == len(ds.seeds)
    assert not np.array_equal(ds.X[0], ds.X[1])


def test_windows_are_recorded() -> None:
    ds = make_dataset(30, "discrete_coin", DELOC, LOC, n_samples=10, seed=0)
    assert ds.window_delocalized == DELOC
    assert ds.window_localized == LOC


def test_pure_channel_is_rejected() -> None:
    with pytest.raises(ValueError, match="no control parameter"):
        make_dataset(30, "pure", DELOC, LOC, n_samples=10)


def test_malformed_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="low <= high"):
        make_dataset(30, "discrete_coin", (0.5, 0.1), LOC, n_samples=10)


@pytest.mark.parametrize(
    "channel,deloc,loc",
    [
        ("discrete_coin", (0.0, 0.05), (0.5, 0.9)),          # delta_theta > theta_0
        ("random_translation", (0.0, 0.02), (0.45, 0.52)),   # p_r > 0.5
        ("continuous_coin", (-0.1, 0.0), (0.4, 0.5)),        # negative delta_theta_max
    ],
)
def test_out_of_range_control_window_is_rejected(channel, deloc, loc) -> None:
    """Sec. 2.3 fixes each channel's control range; silently leaving it is a bug."""
    with pytest.raises(ValueError, match="outside the range"):
        make_dataset(20, channel, deloc, loc, n_samples=10, theta_0=np.pi / 6)


def test_control_window_at_the_range_endpoints_is_allowed() -> None:
    make_dataset(20, "random_translation", (0.0, 0.0), (0.5, 0.5), n_samples=4)
    make_dataset(20, "discrete_coin", (0.0, 0.0), (np.pi / 6, np.pi / 6), n_samples=4)
