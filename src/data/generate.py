"""Sample generation and labelling for the two-class problem.

Data contract: CLAUDE.md Sec. 6.

* One sample = the final-time probability distribution of one walk realisation,
  shape ``(2N+1,)`` float64 on the Paper A odd lattice.
* Labels: ``0 = delocalized`` (weak randomness), ``1 = localized`` (strong).
  Never flipped.
* Metadata travels with every sample: ``channel``, ``theta_0``, ``n``,
  ``control_value``, ``seed``.
* Default 1800 samples (Paper A found estimates stable from ~2000; more gave
  no visible improvement).

Samples are returned with the *physical* normalisation ``sum_x P(x) = 1``.
The ``P_max = 1`` ML normalisation is applied by the classifier, not here, so
that a single toggle covers both training and inference
(see :mod:`data.preprocess`).

NOT YET IMPLEMENTED: config-hash caching to ``data/*.npz`` with logged cache
hits and misses. Every call below re-simulates from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qw.observables import probability
from qw.operators import THETA_0_DEFAULT
from qw.randomness import Channel
from qw.walk import Parity, run_walk

#: Paper A's sample count (CLAUDE.md Sec. 6).
N_SAMPLES_DEFAULT: int = 1800

DELOCALIZED: int = 0
LOCALIZED: int = 1


@dataclass(frozen=True)
class Dataset:
    """Labelled distributions plus the metadata needed to reproduce each one.

    Attributes
    ----------
    X
        Shape ``(n_samples, n_sites)`` float64. Each row sums to 1.
    y
        Shape ``(n_samples,)`` int. ``0 = delocalized``, ``1 = localized``.
    control_values
        The randomness strength each sample was drawn at.
    seeds
        Per-sample seed. ``run_walk(n, channel, theta_0, control_values[i],
        seed=seeds[i])`` reproduces row ``i`` exactly.
    window_delocalized, window_localized
        The ``(low, high)`` control-parameter ranges the two classes were drawn
        from. These are a *result*, not an implementation detail -- record them
        with anything derived from this dataset (CLAUDE.md Sec. 6).
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
    """Simulate a balanced two-class dataset from the two extreme regimes.

    Half the samples are drawn with the control parameter uniform in
    ``window_delocalized`` (label 0), half in ``window_localized`` (label 1).

    The windows have no defaults on purpose. Too narrow and the classes carry
    too little information about the configurations; too wide and
    transition-regime data leaks into training and corrupts the critical-value
    estimate. Choosing them is a research decision (CLAUDE.md Sec. 6, Sec. 9).

    Parameters
    ----------
    n
        Lattice parameter: ``2n+1`` sites on the odd (Paper A) lattice.
    channel
        Randomness channel. ``"pure"`` is rejected -- it has no control
        parameter, so it cannot produce two classes.
    window_delocalized, window_localized
        ``(low, high)`` control-parameter ranges, inclusive of ``low``.
    n_samples
        Total samples across both classes. Rounded down to even.
    seed
        Master seed. Per-sample seeds are spawned from it, so the whole
        dataset is reproducible from this one integer.

    Returns
    -------
    Dataset
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
    # One independent, reproducible seed per realisation.
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
    """Reject control values outside the range the channel is defined on.

    Ranges are from CLAUDE.md Sec. 2.3:

    * ``discrete_coin``   -- ``0 <= delta_theta <= theta_0``. Beyond ``theta_0``
      one of the two coin angles goes negative, which is a different walk.
    * ``continuous_coin`` -- ``delta_theta_max >= 0``.
    * ``random_translation`` -- ``0 <= p_r <= 0.5``. Above 0.5 the walk is
      mirror-symmetric to ``1 - p_r`` by parity, so a window there silently
      duplicates one below it rather than extending the scan.
    """
    low = min(window_delocalized[0], window_localized[0])
    high = max(window_delocalized[1], window_localized[1])

    limits: dict[Channel, tuple[float, float, str]] = {
        "discrete_coin": (0.0, theta_0, "delta_theta must satisfy 0 <= x <= theta_0"),
        "continuous_coin": (0.0, np.inf, "delta_theta_max must satisfy x >= 0"),
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
