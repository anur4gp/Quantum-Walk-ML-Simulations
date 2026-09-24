"""The three channels of classical randomness (Paper A, Sec. II B).

A channel produces a *schedule*: the coin angle at each time step, and whether
that step uses the inverse translation. Drawing the whole schedule up front
keeps the evolution loop deterministic and the realisation inspectable.

All three channels are temporal -- redrawn every step, uniform in space. Every
function takes an explicit ``rng``; no module-level ``np.random``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .operators import THETA_0_DEFAULT

Channel = Literal["pure", "discrete_coin", "continuous_coin", "random_translation"]

CHANNELS: tuple[Channel, ...] = (
    "pure",
    "discrete_coin",
    "continuous_coin",
    "random_translation",
)

CONTROL_PARAMETER: dict[Channel, str] = {
    "pure": "",
    "discrete_coin": "delta_theta",
    "continuous_coin": "delta_theta_max",
    "random_translation": "p_r",
}


@dataclass(frozen=True)
class Schedule:
    """One realisation of the classical randomness.

    ``thetas`` is the coin angle per step; ``inverse_translation`` is True
    where ``T^-1`` replaces ``T``. Both are 1-D, one entry per time step.
    """

    thetas: np.ndarray
    inverse_translation: np.ndarray
    channel: Channel
    control_value: float

    def __post_init__(self) -> None:
        if self.thetas.shape != self.inverse_translation.shape:
            raise ValueError(
                f"thetas {self.thetas.shape} and inverse_translation "
                f"{self.inverse_translation.shape} must have the same shape"
            )
        if self.thetas.ndim != 1:
            raise ValueError("schedule arrays must be 1-D, one entry per time step")

    @property
    def n_steps(self) -> int:
        return int(self.thetas.shape[0])


def pure(n_steps: int, theta_0: float = THETA_0_DEFAULT) -> Schedule:
    """No randomness: fixed coin, conventional translation. Draws nothing."""
    return Schedule(
        thetas=np.full(n_steps, float(theta_0)),
        inverse_translation=np.zeros(n_steps, dtype=bool),
        channel="pure",
        control_value=0.0,
    )


def discrete_coin(
    n_steps: int,
    theta_0: float,
    delta_theta: float,
    rng: np.random.Generator,
) -> Schedule:
    r"""Discrete random rotation (Paper A, Sec. II B 1), "Jittered".

    A fair classical coin picks :math:`\theta_0\pm\Delta\theta` each step.
    Control parameter ``delta_theta``, scanned over ``[0, theta_0]``.
    """
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_steps)
    return Schedule(
        thetas=theta_0 + signs * delta_theta,
        inverse_translation=np.zeros(n_steps, dtype=bool),
        channel="discrete_coin",
        control_value=float(delta_theta),
    )


def continuous_coin(
    n_steps: int,
    theta_0: float,
    delta_theta_max: float,
    rng: np.random.Generator,
) -> Schedule:
    r"""Continuous random rotation (Paper A, Sec. II B 2), "Uniform Jittered".

    :math:`\theta(t)=\theta_0+\Delta\theta(t)`, with
    :math:`\Delta\theta(t)\sim\mathrm{Uniform}(0,\Delta\theta_M)` redrawn each
    step. The support is one-sided, not symmetric about ``theta_0``.
    """
    return Schedule(
        thetas=theta_0 + rng.uniform(0.0, delta_theta_max, size=n_steps),
        inverse_translation=np.zeros(n_steps, dtype=bool),
        channel="continuous_coin",
        control_value=float(delta_theta_max),
    )


def random_translation(
    n_steps: int,
    theta_0: float,
    p_r: float,
    rng: np.random.Generator,
) -> Schedule:
    r"""Random translation (Paper A, Sec. II B 3), "Random Transform".

    Fixed coin; :math:`\hat T^{-1}` replaces :math:`\hat T` with probability
    ``p_r``. Control parameter :math:`P_r\in[0,0.5]` -- above 0.5 the walk is
    the mirror image of ``1 - p_r``.
    """
    return Schedule(
        thetas=np.full(n_steps, float(theta_0)),
        inverse_translation=rng.random(n_steps) < p_r,
        channel="random_translation",
        control_value=float(p_r),
    )


def make_schedule(
    channel: Channel,
    n_steps: int,
    theta_0: float = THETA_0_DEFAULT,
    control_value: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Schedule:
    """Dispatch to ``channel``. ``rng`` is required for everything but ``pure``."""
    if channel == "pure":
        return pure(n_steps, theta_0)
    if rng is None:
        raise ValueError(f"channel {channel!r} draws randomness and needs an rng")
    if channel == "discrete_coin":
        return discrete_coin(n_steps, theta_0, control_value, rng)
    if channel == "continuous_coin":
        return continuous_coin(n_steps, theta_0, control_value, rng)
    if channel == "random_translation":
        return random_translation(n_steps, theta_0, control_value, rng)
    raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")
