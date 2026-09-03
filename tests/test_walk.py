"""Walk-level tests: unitarity, known limits, determinism, symmetry.

CLAUDE.md Sec. 8, "Testing".
"""

from __future__ import annotations

import numpy as np
import pytest

from qw import observables as obs
from qw import operators as ops
from qw.randomness import CHANNELS, make_schedule
from qw.walk import evolve, initial_state, lattice_positions, max_steps, run_walk

TOL = 1e-12

# (channel, a control value in the interesting range)
CHANNEL_CASES = [
    ("pure", 0.0),
    ("discrete_coin", np.pi / 12),
    ("continuous_coin", np.pi / 8),
    ("random_translation", 0.3),
]


# --- unitarity ---------------------------------------------------------------


@pytest.mark.parametrize("channel,control", CHANNEL_CASES)
@pytest.mark.parametrize("parity", ["odd", "even"])
def test_probability_conserved_at_every_step(
    channel: str, control: float, parity: str
) -> None:
    """sum_x P(x) = 1 to 1e-12 for every variant, checked after every step."""
    n = 40
    steps = max_steps(n, parity)
    schedule = make_schedule(
        channel, steps, np.pi / 6, control, np.random.default_rng(7)
    )

    psi = initial_state(n, parity)
    assert abs(obs.total_probability(psi[:, 0], psi[:, 1]) - 1.0) < TOL

    for theta, use_inv in zip(schedule.thetas, schedule.inverse_translation):
        psi = ops.apply_coin(psi, ops.coin(theta))
        psi = ops.translate_inverse(psi) if use_inv else ops.translate(psi)
        total = obs.total_probability(psi[:, 0], psi[:, 1])
        assert abs(total - 1.0) < TOL, f"{channel}/{parity}: total = {total!r}"


def test_step_cap_is_exactly_tight() -> None:
    """At max_steps the walker can *touch* the outermost sites but loses nothing.

    The cap is what makes the open boundaries of `translate` harmless: after
    `max_steps` steps the support has just saturated the lattice, so no
    amplitude has been shifted off either end. One more step would leak.
    """
    for parity in ("odd", "even"):
        r = run_walk(40, "random_translation", control_value=0.5, seed=1, parity=parity)
        assert obs.total_probability(r.psi_plus, r.psi_minus) == pytest.approx(
            1.0, abs=TOL
        )


def test_n_steps_beyond_the_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds the boundary-free maximum"):
        run_walk(10, n_steps=11)


# --- pure Hadamard walk, N = 300 ---------------------------------------------


def test_hadamard_walk_has_two_symmetric_ballistic_peaks() -> None:
    """Pure walk, N = 300, theta = pi/4: the classic two-peak distribution.

    Reference figure: figures/Total_probabilities.png. The ballistic peaks of
    the Hadamard walk sit at x ~ +-N/sqrt(2) ~ +-212, matching
    figures/Spin_probabilities.png.
    """
    n = 300
    r = run_walk(n, "pure", theta_0=np.pi / 4)
    p = obs.probability(r.psi_plus, r.psi_minus)

    # mirror symmetry of the whole distribution
    np.testing.assert_allclose(p, p[::-1], atol=1e-14)

    left = r.positions[np.argmax(p[: n + 1])]
    right = r.positions[n + 1 + np.argmax(p[n + 1 :])]
    assert left == pytest.approx(-right)
    assert abs(right) == pytest.approx(n / np.sqrt(2), rel=0.05)

    # ballistic, not diffusive: MoI ~ N^2
    moi = obs.moment_of_inertia(p, r.positions)
    assert 0.2 < moi / n**2 < 1.0


# --- known limits in theta ---------------------------------------------------


def test_maximal_spreading_at_theta_multiple_of_pi() -> None:
    """theta = n*pi: the coin is (+-) the identity, so both spins run free.

    Reproduces the endpoints of figures/Peak_spin_updown_theta_vs_distance.png,
    where the peak displacement is maximal (~N) at theta = 0.
    """
    n = 60
    for theta in (0.0, np.pi, -np.pi):
        r = run_walk(n, "pure", theta_0=theta)
        p_up = np.abs(r.psi_plus) ** 2
        assert abs(r.positions[np.argmax(p_up)]) == pytest.approx(n)


@pytest.mark.parametrize("theta", [np.pi / 2, -np.pi / 2, 3 * np.pi / 2])
@pytest.mark.parametrize("n", [30, 31, 60, 61])
def test_minimal_spreading_at_odd_multiples_of_half_pi(theta: float, n: int) -> None:
    """theta = (2k+1)*pi/2: the coin swaps the spins, so the walker cannot spread.

    The coin is purely off-diagonal, so each step relabels |+> <-> |-> and the
    walker just oscillates between x = 0 and x = +-1 with period 2: MoI is 1
    after an odd number of steps and 0 after an even one. Either way it stays
    bounded by 1 instead of growing with N -- minimal spreading, the theta =
    +-pi/2 endpoints of figures/Peak_spin_updown_theta_vs_distance.png.
    """
    r = run_walk(n, "pure", theta_0=theta)
    p = obs.probability(r.psi_plus, r.psi_minus)
    moi = obs.moment_of_inertia(p, r.positions)
    assert moi == pytest.approx(float(n % 2), abs=1e-9)


# --- randomness channels -----------------------------------------------------


def test_random_translation_at_pr_zero_is_bitwise_identical_to_pure() -> None:
    """P_r = 0 never applies T^-1, so it must reproduce the pure walk exactly."""
    pure = run_walk(50, "pure", theta_0=np.pi / 6)
    rt = run_walk(50, "random_translation", theta_0=np.pi / 6, control_value=0.0, seed=3)

    np.testing.assert_array_equal(pure.psi_plus, rt.psi_plus)
    np.testing.assert_array_equal(pure.psi_minus, rt.psi_minus)


def test_discrete_coin_at_zero_delta_is_bitwise_identical_to_pure() -> None:
    pure = run_walk(50, "pure", theta_0=np.pi / 6)
    dc = run_walk(50, "discrete_coin", theta_0=np.pi / 6, control_value=0.0, seed=3)

    np.testing.assert_array_equal(pure.psi_plus, dc.psi_plus)
    np.testing.assert_array_equal(pure.psi_minus, dc.psi_minus)


def test_discrete_coin_draws_only_the_two_allowed_angles() -> None:
    s = make_schedule(
        "discrete_coin", 500, np.pi / 6, np.pi / 12, np.random.default_rng(0)
    )
    assert set(np.round(s.thetas, 12)) == {
        round(np.pi / 6 - np.pi / 12, 12),
        round(np.pi / 6 + np.pi / 12, 12),
    }
    # fair coin: both angles roughly equally often
    assert 0.4 < np.mean(s.thetas > np.pi / 6) < 0.6


def test_continuous_coin_support_is_one_sided() -> None:
    """Delta_theta(t) ~ Uniform(0, Delta_theta_M), not symmetric about theta_0."""
    theta_0, dmax = np.pi / 6, np.pi / 8
    s = make_schedule("continuous_coin", 2000, theta_0, dmax, np.random.default_rng(0))
    assert s.thetas.min() >= theta_0
    assert s.thetas.max() <= theta_0 + dmax
    assert np.mean(s.thetas) == pytest.approx(theta_0 + dmax / 2, rel=0.05)


def test_random_translation_inverse_fraction_matches_pr() -> None:
    s = make_schedule(
        "random_translation", 5000, np.pi / 6, 0.3, np.random.default_rng(0)
    )
    assert np.mean(s.inverse_translation) == pytest.approx(0.3, abs=0.02)


# --- determinism -------------------------------------------------------------


@pytest.mark.parametrize("channel,control", CHANNEL_CASES)
def test_same_seed_gives_identical_arrays(channel: str, control: float) -> None:
    a = run_walk(40, channel, control_value=control, seed=12345)
    b = run_walk(40, channel, control_value=control, seed=12345)
    np.testing.assert_array_equal(a.psi_plus, b.psi_plus)
    np.testing.assert_array_equal(a.psi_minus, b.psi_minus)


@pytest.mark.parametrize("channel,control", CHANNEL_CASES[1:])
def test_different_seeds_give_different_realisations(
    channel: str, control: float
) -> None:
    a = run_walk(40, channel, control_value=control, seed=1)
    b = run_walk(40, channel, control_value=control, seed=2)
    assert not np.array_equal(a.psi_plus, b.psi_plus)


# --- symmetry ----------------------------------------------------------------


@pytest.mark.parametrize("channel,control", CHANNEL_CASES)
def test_spin_totals_are_equal(channel: str, control: float) -> None:
    """Symmetric initial condition with phi1 = phi2 = pi/2 keeps P_+ and P_- balanced."""
    r = run_walk(60, channel, control_value=control, seed=9)
    total_up = float(np.sum(np.abs(r.psi_plus) ** 2))
    total_down = float(np.sum(np.abs(r.psi_minus) ** 2))
    assert total_up == pytest.approx(total_down, abs=1e-12)


# --- lattice bookkeeping -----------------------------------------------------


def test_odd_lattice_shape_and_centre() -> None:
    n = 25
    pos = lattice_positions(n, "odd")
    assert pos.size == 2 * n + 1
    assert pos[n] == 0.0
    assert max_steps(n, "odd") == n


def test_even_lattice_has_no_zero_site() -> None:
    n = 25
    pos = lattice_positions(n, "even")
    assert pos.size == 2 * n
    assert 0.0 not in pos
    assert max_steps(n, "even") == n - 1


def test_all_channels_are_runnable() -> None:
    for channel in CHANNELS:
        run_walk(20, channel, control_value=0.1, seed=0)
