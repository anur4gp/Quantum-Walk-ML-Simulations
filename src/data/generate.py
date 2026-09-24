"""Labelled sample generation for the two-class problem (CLAUDE.md Sec. 6).

One sample is the final-time distribution of one walk realisation, shape
``(2n+1,)`` on the Paper A odd lattice. Labels: 0 = delocalized (weak
randomness), 1 = localized (strong). Never flipped.

Samples leave here with the physical normalisation ``sum_x P(x) = 1``; the
``P_max = 1`` ML normalisation is the classifier's job, so one flag covers
training and inference (see :mod:`data.preprocess`).

No caching yet -- every call re-simulates (CLAUDE.md Sec. 6 wants ``data/*.npz``
keyed by a config hash).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qw.observables import probability
from qw.operators import THETA_0_DEFAULT
from qw.randomness import Channel
from qw.walk import Parity, run_walk

N_SAMPLES_DEFAULT: int = 1800  # Paper A's value (CLAUDE.md Sec. 6)

DELOCALIZED: int = 0
LOCALIZED: int = 1


@dataclass(frozen=True)
class Dataset:
    """Labelled distributions and the metadata that reproduces each one.

    ``X`` is ``(n_samples, n_sites)`` with rows summing to 1; row ``i`` is
    reproduced by ``run_walk(n, channel, theta_0, control_values[i],
    seed=seeds[i])``. The two windows are recorded because they are a result,
    not an implementation detail (CLAUDE.md Sec. 6).
    """

    X: np.ndarray
    y: np.ndarray
    control_values: np.ndarray
    seeds: np.ndarray
    channel: Channel
    theta_0: float
    n: int
    parity: Parity
    window_delocalized: tuple[float, float]
    window_localized: tuple[float, float]

    def __len__(self) -> int:
        return int(self.y.size)

    @property
    def n_sites(self) -> int:
        return int(self.X.shape[1])


def make_dataset(
    n: int,
    channel: Channel,
    window_delocalized: tuple[float, float],
    window_localized: tuple[float, float],
    *,
    n_samples: int = N_SAMPLES_DEFAULT,
    theta_0: float = THETA_0_DEFAULT,
    seed: int = 0,
    parity: Parity = "odd",
) -> Dataset:
    """Balanced two-class dataset from the two extreme regimes.

    Half the samples draw the control parameter uniformly from
    ``window_delocalized`` (label 0), half from ``window_localized`` (label 1).
    The windows have no defaults: choosing them is a research decision
    (CLAUDE.md Sec. 6, Sec. 9). ``n_samples`` is rounded down to even, and
    per-sample seeds are spawned from ``seed``.
    """
    if channel == "pure":
        raise ValueError("channel 'pure' has no control parameter to sweep")
    for name, window in (
        ("window_delocalized", window_delocalized),
        ("window_localized", window_localized),
    ):
        if len(window) != 2 or window[0] > window[1]:
            raise ValueError(f"{name} must be (low, high) with low <= high, got {window}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")

    _check_control_range(channel, window_delocalized, window_localized, theta_0)

    per_class = n_samples // 2
    ss = np.random.SeedSequence(seed)
    draw_rng = np.random.default_rng(ss.spawn(1)[0])

    controls = np.concatenate(
        [
            draw_rng.uniform(*window_delocalized, size=per_class),
            draw_rng.uniform(*window_localized, size=per_class),
        ]
    )
    labels = np.concatenate(
        [
            np.full(per_class, DELOCALIZED, dtype=int),
            np.full(per_class, LOCALIZED, dtype=int),
        ]
    )
    seeds = np.array(
        [int(s.generate_state(1)[0]) for s in ss.spawn(2 * per_class)], dtype=np.int64
    )

    rows = []
    for control_value, sample_seed in zip(controls, seeds):
        result = run_walk(
            n,
            channel,
            theta_0=theta_0,
            control_value=float(control_value),
            seed=int(sample_seed),
            parity=parity,
        )
        rows.append(probability(result.psi_plus, result.psi_minus))

    return Dataset(
        X=np.asarray(rows, dtype=np.float64),
        y=labels,
        control_values=controls,
        seeds=seeds,
        channel=channel,
        theta_0=float(theta_0),
        n=n,
        parity=parity,
        window_delocalized=tuple(map(float, window_delocalized)),
        window_localized=tuple(map(float, window_localized)),
    )


def _check_control_range(
    channel: Channel,
    window_delocalized: tuple[float, float],
    window_localized: tuple[float, float],
    theta_0: float,
) -> None:
    """Reject control values outside the channel's defined range (CLAUDE.md Sec. 2.3)."""
    low = min(window_delocalized[0], window_localized[0])
    high = max(window_delocalized[1], window_localized[1])

    limits: dict[Channel, tuple[float, float, str]] = {
        # Beyond theta_0 one of the two discrete coin angles goes negative.
        "discrete_coin": (0.0, theta_0, "delta_theta must satisfy 0 <= x <= theta_0"),
        "continuous_coin": (0.0, np.inf, "delta_theta_max must satisfy x >= 0"),
        # Above 0.5 a window silently duplicates one below it, by mirror symmetry.
        "random_translation": (
            0.0,
            0.5,
            "p_r must satisfy 0 <= x <= 0.5; above 0.5 the walk is "
            "mirror-symmetric to 1 - p_r by parity",
        ),
    }
    lo, hi, message = limits[channel]
    if low < lo or high > hi:
        raise ValueError(
            f"control window [{low}, {high}] is outside the range for "
            f"channel {channel!r}: {message}"
        )
