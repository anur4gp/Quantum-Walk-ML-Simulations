"""Space-time samples: the full evolution P(x, t) of a walk realisation.

A departure from the Phase-1 data contract, where one sample is the final-time
distribution alone (CLAUDE.md Sec. 6). Here one sample is
``P[k, t, x]``, shape ``(n_times, 2n+1)``, with every slice still carrying the
physical ``sum_x P(x, t) = 1``. The physics is unchanged; only what is kept
from each run differs.

Two consequences: the representation is our extension, not Paper A's, so
nothing from it compares directly against Paper A Table I; and the slices of
one evolution share a single realisation of the randomness, so they are not
independent samples.

Labels are unchanged: 0 = delocalized, 1 = localized. No caching, as in
:mod:`data.generate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qw.operators import THETA_0_DEFAULT
from qw.randomness import Channel
from qw.walk import Parity, run_walk_trajectory

from .generate import DELOCALIZED, LOCALIZED, N_SAMPLES_DEFAULT, _check_control_range


@dataclass(frozen=True)
class EvolutionDataset:
    """Labelled space-time distributions, ``P`` of shape ``(n_samples, n_times, n_sites)``.

    Sample ``k`` is reproduced by ``run_walk_trajectory(n, channel, theta_0,
    control_values[k], seed=seeds[k], record_every=record_every)``.
    """

    P: np.ndarray
    y: np.ndarray
    times: np.ndarray
    control_values: np.ndarray
    seeds: np.ndarray
    channel: Channel
    theta_0: float
    n: int
    parity: Parity
    n_steps: int
    record_every: int
    window_delocalized: tuple[float, float]
    window_localized: tuple[float, float]

    def __len__(self) -> int:
        return int(self.y.size)

    @property
    def n_times(self) -> int:
        return int(self.P.shape[1])

    @property
    def n_sites(self) -> int:
        return int(self.P.shape[2])

    def at_time(self, index: int) -> np.ndarray:
        """The ``(n_samples, n_sites)`` snapshot at recorded slice ``index``.

        This is the ordinary Phase-1 feature matrix, so a snapshot model and a
        full-evolution model can be trained on the same walks.
        """
        return self.P[:, index, :]

    def flat(self) -> np.ndarray:
        """The evolution as ``(n_samples, n_times * n_sites)`` features, row-major."""
        return self.P.reshape(len(self), -1)


@dataclass(frozen=True)
class ProbeSet:
    """Unlabelled evolutions on a grid of control values, for the sweep.

    ``P`` is ``(n_controls, n_realizations, n_times, n_sites)``. Unlabelled on
    purpose: labelling intermediate randomness would presuppose the answer.
    """

    P: np.ndarray
    control_values: np.ndarray
    times: np.ndarray
    seeds: np.ndarray
    channel: Channel
    theta_0: float
    n: int
    parity: Parity
    n_steps: int
    record_every: int

    @property
    def n_controls(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_realizations(self) -> int:
        return int(self.P.shape[1])


def _spawn_seeds(ss: np.random.SeedSequence, count: int) -> np.ndarray:
    """One independent, reproducible int seed per realisation."""
    return np.array(
        [int(child.generate_state(1)[0]) for child in ss.spawn(count)], dtype=np.int64
    )


def _simulate(
    n: int,
    channel: Channel,
    controls: np.ndarray,
    seeds: np.ndarray,
    theta_0: float,
    parity: Parity,
    n_steps: int | None,
    record_every: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run one trajectory per (control, seed) pair; return (P, times, n_steps)."""
    rows = []
    times: np.ndarray | None = None
    steps = 0
    for control_value, sample_seed in zip(controls, seeds):
        traj = run_walk_trajectory(
            n,
            channel,
            theta_0=theta_0,
            control_value=float(control_value),
            seed=int(sample_seed),
            parity=parity,
            n_steps=n_steps,
            record_every=record_every,
        )
        rows.append(traj.probability)
        times = traj.times
        steps = traj.n_steps
    assert times is not None
    return np.asarray(rows, dtype=np.float64), times, steps


def make_evolution_dataset(
    n: int,
    channel: Channel,
    window_delocalized: tuple[float, float],
    window_localized: tuple[float, float],
    *,
    n_samples: int = N_SAMPLES_DEFAULT,
    theta_0: float = THETA_0_DEFAULT,
    seed: int = 0,
    parity: Parity = "odd",
    n_steps: int | None = None,
    record_every: int = 1,
) -> EvolutionDataset:
    """Balanced two-class dataset of full evolutions.

    Labelling is exactly :func:`data.generate.make_dataset`'s; only the whole
    trajectory is kept. ``record_every`` trades time resolution for feature
    count. Memory is ``n_samples * n_times * n_sites * 8`` bytes -- about
    185 MB at n=80 with 1800 samples.
    """
    if channel == "pure":
        raise ValueError("channel 'pure' has no control parameter to sweep")
    for name, window in (
        ("window_delocalized", window_delocalized),
        ("window_localized", window_localized),
    ):
        if len(window) != 2 or window[0] > window[1]:
            raise ValueError(
                f"{name} must be (low, high) with low <= high, got {window}"
            )
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
    seeds = _spawn_seeds(ss, 2 * per_class)

    p, times, steps = _simulate(
        n, channel, controls, seeds, theta_0, parity, n_steps, record_every
    )

    return EvolutionDataset(
        P=p,
        y=labels,
        times=times,
        control_values=controls,
        seeds=seeds,
        channel=channel,
        theta_0=float(theta_0),
        n=n,
        parity=parity,
        n_steps=steps,
        record_every=record_every,
        window_delocalized=tuple(map(float, window_delocalized)),
        window_localized=tuple(map(float, window_localized)),
    )


def make_probe_evolutions(
    n: int,
    channel: Channel,
    control_values: np.ndarray,
    *,
    n_realizations: int = 20,
    theta_0: float = THETA_0_DEFAULT,
    seed: int = 1,
    parity: Parity = "odd",
    n_steps: int | None = None,
    record_every: int = 1,
) -> ProbeSet:
    """``n_realizations`` evolutions at each of ``control_values``.

    ``seed`` defaults to 1, not 0, so probes never reuse the training walks.
    """
    if channel == "pure":
        raise ValueError("channel 'pure' has no control parameter to sweep")
    controls = np.asarray(control_values, dtype=np.float64)
    if controls.ndim != 1 or controls.size == 0:
        raise ValueError(
            f"control_values must be a non-empty 1-D array, got {controls.shape}"
        )
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be at least 1, got {n_realizations}")

    span = (float(controls.min()), float(controls.max()))
    _check_control_range(channel, span, span, theta_0)

    ss = np.random.SeedSequence(seed)
    seeds = _spawn_seeds(ss, controls.size * n_realizations).reshape(
        controls.size, n_realizations
    )
    flat_controls = np.repeat(controls, n_realizations)

    p, times, steps = _simulate(
        n,
        channel,
        flat_controls,
        seeds.reshape(-1),
        theta_0,
        parity,
        n_steps,
        record_every,
    )

    return ProbeSet(
        P=p.reshape(controls.size, n_realizations, p.shape[1], p.shape[2]),
        control_values=controls,
        times=times,
        seeds=seeds,
        channel=channel,
        theta_0=float(theta_0),
        n=n,
        parity=parity,
        n_steps=steps,
        record_every=record_every,
    )
