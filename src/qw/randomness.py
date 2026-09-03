"""The three channels of classical randomness (Paper A, Sec. II B).

Each channel is realised as a *schedule*: a per-time-step list of coin angles
and a per-time-step boolean saying whether the inverse translation is used.
Producing the whole schedule up front (rather than drawing inside the evolution
loop) keeps the walk driver deterministic given a schedule, makes a realisation
inspectable and cacheable, and keeps the random draws in one place.

All three channels here are *temporal*: the randomness is redrawn every time
step and is uniform in space. Paper B's spatially-dependent randomness (drawn
once per site, frozen in time) is not part of the current phase; it would enter
as a different schedule type consumed by the same driver, so nothing here
forecloses it.

Every function takes an explicit ``rng``; no module-level ``np.random`` use.
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

#: Control parameter carried by each channel (CLAUDE.md Sec. 2.3 / Glossary).
CONTROL_PARAMETER: dict[Channel, str] = {
    "pure": "",
    "discrete_coin": "delta_theta",
    "continuous_coin": "delta_theta_max",
    "random_translation": "p_r",
}


@dataclass(frozen=True)
class Schedule:
    """One realisation of the classical randomness, for ``n_steps`` time steps.

    Attributes
    ----------
    thetas
        Shape ``(n_steps,)`` float64. Coin angle used at each time step.
    inverse_translation
        Shape ``(n_steps,)`` bool. ``True`` where :math:`\\hat T^{-1}` replaces
        :math:`\\hat T` at that step.
    channel
        Which randomness channel produced this schedule.
    control_value
        The value of that channel's control parameter (``0.0`` for ``pure``).
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
    """No classical randomness: a fixed coin and the conventional translation.

    Takes no ``rng`` because nothing is drawn. This is the control case
    (``pure`` in CLAUDE.md Sec. 4).
    """
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
    r"""Discrete random rotation (Paper A, Sec. II B 1) -- "Jittered".

    Two coin operators with :math:`\theta_{1,2}=\theta_0\pm\Delta\theta`. At
    each time step a *fair* classical coin picks one, ``P = 1/2`` each. The
    fairness is part of the channel definition and is deliberately not exposed
    as a parameter.

    Control parameter: ``delta_theta``, scanned over
    :math:`0\le\Delta\theta\le\theta_0`.
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
    r"""Continuous random rotation (Paper A, Sec. II B 2) -- "Uniform Jittered".

    :math:`\theta(t)=\theta_0+\Delta\theta(t)` with
    :math:`\Delta\theta(t)\sim\mathrm{Uniform}(0,\Delta\theta_M)`, redrawn each
    step. Note the support is one-sided ``[0, delta_theta_max)``, not
    symmetric about ``theta_0``.

    Control parameter: ``delta_theta_max``.
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
    r"""Random translation (Paper A, Sec. II B 3) -- "Random Transform".

    One fixed coin. At each step :math:`\hat T^{-1}` is applied with
    probability ``p_r`` instead of :math:`\hat T`.

    Control parameter: :math:`P_r\in[0,0.5]`. Above ``0.5`` the walk is
    mirror-symmetric to ``1 - p_r`` by parity, so the scan stops there.
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
    """Dispatch to the channel named by ``channel``.

    ``rng`` is required for every channel except ``"pure"``.
    """
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
