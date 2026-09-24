"""Full space-time evolution: trajectory driver, dataset, normalisation.

These lock down that the physics did not change when the sample became a
trajectory: the final slice is bitwise :func:`run_walk`'s output, unitarity
holds at every recorded step, and the same seed gives the same arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.evolution import make_evolution_dataset, make_probe_evolutions
from data.generate import DELOCALIZED, LOCALIZED, make_dataset
from data.preprocess import normalize_pmax, normalize_pmax_evolution
from qw.observables import (
    moment_of_inertia,
    moment_of_inertia_series,
    probability,
)
from qw.walk import lattice_positions, run_walk, run_walk_trajectory

TOL = 1e-12

# (channel, a control value in the interesting range)
CHANNEL_CASES = [
    ("pure", 0.0),
    ("discrete_coin", np.pi / 12),
    ("continuous_coin", np.pi / 8),
    ("random_translation", 0.3),
]

DELOC = (0.0, 0.05)
LOC = (0.45, 0.52)


# --- the trajectory is the same walk -----------------------------------------


@pytest.mark.parametrize("channel,control", CHANNEL_CASES)
def test_final_slice_matches_run_walk(channel: str, control: float) -> None:
    """Bitwise, not approximately: same seed, same schedule, same amplitudes."""
    kwargs = dict(theta_0=np.pi / 6, control_value=control, seed=11)
    snapshot = run_walk(30, channel, **kwargs)
    traj = run_walk_trajectory(30, channel, **kwargs)

    assert np.array_equal(traj.final.psi_plus, snapshot.psi_plus)
    assert np.array_equal(traj.final.psi_minus, snapshot.psi_minus)
    assert traj.final.n_steps == snapshot.n_steps


@pytest.mark.parametrize("channel,control", CHANNEL_CASES)
@pytest.mark.parametrize("parity", ["odd", "even"])
def test_probability_conserved_at_every_recorded_step(
    channel: str, control: float, parity: str
) -> None:
    traj = run_walk_trajectory(
        25, channel, control_value=control, seed=3, parity=parity, include_initial=True
    )
    totals = traj.probability.sum(axis=1)
    assert np.allclose(totals, 1.0, atol=TOL)


@pytest.mark.parametrize("channel,control", CHANNEL_CASES)
def test_deterministic(channel: str, control: float) -> None:
    a = run_walk_trajectory(20, channel, control_value=control, seed=5)
    b = run_walk_trajectory(20, channel, control_value=control, seed=5)
    assert np.array_equal(a.psi_t, b.psi_t)


def test_zero_p_r_reproduces_the_pure_trajectory() -> None:
    """P_r = 0 never applies T^-1, so the whole history must match the pure walk."""
    pure = run_walk_trajectory(20, "pure")
    rt = run_walk_trajectory(20, "random_translation", control_value=0.0, seed=99)
    assert np.array_equal(rt.psi_t, pure.psi_t)


# --- what gets recorded ------------------------------------------------------


def test_times_and_shape() -> None:
    traj = run_walk_trajectory(15, "pure")
    assert traj.psi_t.shape == (15, 2 * 15 + 1, 2)
    assert np.array_equal(traj.times, np.arange(1, 16))
    assert traj.psi_t.dtype == np.complex128


def test_include_initial_adds_the_undisturbed_state() -> None:
    traj = run_walk_trajectory(15, "pure", include_initial=True)
    assert traj.times[0] == 0
    assert np.isclose(np.abs(traj.psi_t[0]).max(), 1 / np.sqrt(2))


@pytest.mark.parametrize("stride", [1, 2, 5])
def test_record_every_subsamples_without_changing_the_walk(stride: int) -> None:
    """A strided trajectory is a subset of the full one, not a different walk."""
    full = run_walk_trajectory(20, "discrete_coin", control_value=0.2, seed=8)
    strided = run_walk_trajectory(
        20, "discrete_coin", control_value=0.2, seed=8, record_every=stride
    )
    keep = np.isin(full.times, strided.times)
    assert np.array_equal(strided.psi_t, full.psi_t[keep])


def test_record_every_rejects_zero() -> None:
    with pytest.raises(ValueError, match="record_every"):
        run_walk_trajectory(10, "pure", record_every=0)


def test_n_steps_beyond_the_boundary_is_rejected() -> None:
    with pytest.raises(ValueError, match="boundary-free"):
        run_walk_trajectory(10, "pure", n_steps=11)


# --- dataset -----------------------------------------------------------------


def test_dataset_shapes_labels_and_balance() -> None:
    ds = make_evolution_dataset(20, "discrete_coin", DELOC, LOC, n_samples=20, seed=0)
    assert ds.P.shape == (20, 20, 2 * 20 + 1)
    assert ds.P.dtype == np.float64
    assert set(np.unique(ds.y)) == {DELOCALIZED, LOCALIZED}
    assert np.sum(ds.y == DELOCALIZED) == np.sum(ds.y == LOCALIZED) == 10


def test_every_slice_is_a_normalised_distribution() -> None:
    ds = make_evolution_dataset(20, "continuous_coin", DELOC, LOC, n_samples=10, seed=0)
    assert np.allclose(ds.P.sum(axis=2), 1.0, atol=TOL)


def test_flat_is_reversible() -> None:
    ds = make_evolution_dataset(15, "discrete_coin", DELOC, LOC, n_samples=8, seed=0)
    flat = ds.flat()
    assert flat.shape == (8, ds.n_times * ds.n_sites)
    assert np.array_equal(flat.reshape(8, ds.n_times, ds.n_sites), ds.P)


def test_final_slice_matches_the_phase_one_dataset() -> None:
    """The snapshot and evolution builders must agree walk for walk.

    Same seed, same windows, same sample count -> the last time slice of the
    evolution dataset is exactly the Phase-1 feature matrix. This is what lets
    a snapshot baseline and a full-evolution model be compared on one run.
    """
    kwargs = dict(n_samples=12, seed=4)
    snap = make_dataset(18, "discrete_coin", DELOC, LOC, **kwargs)
    ev = make_evolution_dataset(18, "discrete_coin", DELOC, LOC, **kwargs)
    assert np.array_equal(ev.at_time(-1), snap.X)
    assert np.array_equal(ev.y, snap.y)
    assert np.array_equal(ev.seeds, snap.seeds)


def test_sample_is_reproducible_from_its_metadata() -> None:
    ds = make_evolution_dataset(15, "random_translation", (0.0, 0.02), (0.4, 0.5),
                                n_samples=6, seed=2)
    k = 3
    redone = run_walk_trajectory(
        15,
        "random_translation",
        theta_0=ds.theta_0,
        control_value=float(ds.control_values[k]),
        seed=int(ds.seeds[k]),
    )
    assert np.array_equal(redone.probability, ds.P[k])


def test_pure_channel_is_rejected() -> None:
    with pytest.raises(ValueError, match="no control parameter"):
        make_evolution_dataset(10, "pure", DELOC, LOC, n_samples=4)


def test_control_window_outside_the_channel_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="mirror-symmetric"):
        make_evolution_dataset(10, "random_translation", (0.0, 0.02), (0.6, 0.9),
                               n_samples=4)


# --- probes ------------------------------------------------------------------


def test_probe_shape_and_ordering() -> None:
    controls = np.linspace(0.0, 0.5, 5)
    probe = make_probe_evolutions(12, "random_translation", controls,
                                  n_realizations=3, seed=1)
    assert probe.P.shape == (5, 3, 12, 2 * 12 + 1)
    assert np.array_equal(probe.control_values, controls)
    assert np.allclose(probe.P.sum(axis=3), 1.0, atol=TOL)


def test_probes_do_not_reuse_training_seeds() -> None:
    """Default probe seed differs from the default dataset seed, on purpose."""
    ds = make_evolution_dataset(12, "discrete_coin", DELOC, LOC, n_samples=6, seed=0)
    probe = make_probe_evolutions(12, "discrete_coin", np.array([0.1, 0.2]),
                                  n_realizations=3, seed=1)
    assert not set(ds.seeds.tolist()) & set(probe.seeds.reshape(-1).tolist())


# --- normalisation -----------------------------------------------------------


def test_slice_mode_normalises_each_time_step() -> None:
    ds = make_evolution_dataset(15, "discrete_coin", DELOC, LOC, n_samples=6, seed=0)
    out = normalize_pmax_evolution(ds.P, "slice")
    assert np.allclose(out.max(axis=2), 1.0)


def test_slice_mode_agrees_with_the_phase_one_normaliser() -> None:
    """One time slice, normalised either way, must come out identical."""
    ds = make_evolution_dataset(15, "discrete_coin", DELOC, LOC, n_samples=6, seed=0)
    assert np.allclose(
        normalize_pmax_evolution(ds.P, "slice")[:, 7, :],
        normalize_pmax(ds.P[:, 7, :]),
    )


def test_global_mode_normalises_the_whole_image() -> None:
    ds = make_evolution_dataset(15, "discrete_coin", DELOC, LOC, n_samples=6, seed=0)
    out = normalize_pmax_evolution(ds.P, "global")
    assert np.allclose(out.max(axis=(1, 2)), 1.0)


def test_none_mode_keeps_the_physical_normalisation() -> None:
    ds = make_evolution_dataset(15, "discrete_coin", DELOC, LOC, n_samples=6, seed=0)
    out = normalize_pmax_evolution(ds.P, "none")
    assert np.array_equal(out, ds.P)
    assert np.allclose(out.sum(axis=2), 1.0, atol=TOL)


def test_normalisation_handles_a_probe_shaped_array() -> None:
    probe = make_probe_evolutions(10, "discrete_coin", np.array([0.1, 0.3]),
                                  n_realizations=2, seed=1)
    out = normalize_pmax_evolution(probe.P, "slice")
    assert out.shape == probe.P.shape
    assert np.allclose(out.max(axis=3), 1.0)


def test_unknown_normalisation_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        normalize_pmax_evolution(np.ones((2, 3, 5)), "peak")


# --- observables over a trajectory -------------------------------------------


def test_moi_series_matches_the_scalar_definition() -> None:
    traj = run_walk_trajectory(20, "discrete_coin", control_value=0.2, seed=6)
    positions = lattice_positions(20)
    series = moment_of_inertia_series(traj.probability, positions)
    assert series.shape == (20,)
    for t in (0, 9, 19):
        assert np.isclose(series[t], moment_of_inertia(traj.probability[t], positions))


def test_pure_walk_moi_is_ballistic() -> None:
    """MoI ~ t^2 for the pure walk -- the free correctness check on a trajectory."""
    traj = run_walk_trajectory(120, "pure", theta_0=np.pi / 4)
    positions = lattice_positions(120)
    moi = moment_of_inertia_series(traj.probability, positions)
    t = traj.times.astype(float)
    slope = np.polyfit(np.log(t[20:]), np.log(moi[20:]), 1)[0]
    assert abs(slope - 2.0) < 0.02


def test_probability_property_matches_observables() -> None:
    traj = run_walk_trajectory(15, "continuous_coin", control_value=0.3, seed=1)
    direct = probability(traj.psi_t[:, :, 0], traj.psi_t[:, :, 1])
    assert np.array_equal(traj.probability, direct)
