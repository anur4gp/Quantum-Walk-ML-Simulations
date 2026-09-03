"""Evolution driver for the discrete-time quantum walk.

Lattice parity is an explicit parameter, never hardcoded (CLAUDE.md Sec. 2.2):

===============  ==========================  ==================================
                 ``parity="odd"`` (Paper A)  ``parity="even"`` (Paper B)
===============  ==========================  ==================================
sites            ``2N+1``, centred on x=0    ``2N``, no x=0
positions        ``-N ... N``                ``-N ... -1, 1 ... N``
initial state    ``(|0,+> + |0,->)/sqrt(2)`` ``sum_{x=+-1, s=+-} |x,s> / 2``
max steps        ``N``                       ``N-1`` (conventional/symmetric)
===============  ==========================  ==================================

The current ML work uses the Paper A odd lattice with the conventional walk.
The even lattice exists here because Paper B's symmetric and split-step
translations are not unitary on a lattice with a central x=0 site; those two
translation variants are themselves not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import operators as ops
from .randomness import Channel, Schedule, make_schedule

Parity = Literal["odd", "even"]


def lattice_positions(n: int, parity: Parity = "odd") -> np.ndarray:
    """Physical position ``x`` of each array index, as a float64 array.

    ``odd`` -> ``2n+1`` sites, ``-n ... n``.
    ``even`` -> ``2n`` sites, ``-n ... -1, 1 ... n`` (no zero site).
    """
    if parity == "odd":
        return np.arange(-n, n + 1, dtype=np.float64)
    if parity == "even":
        return np.concatenate(
            [np.arange(-n, 0, dtype=np.float64), np.arange(1, n + 1, dtype=np.float64)]
        )
    raise ValueError(f"parity must be 'odd' or 'even', got {parity!r}")


def max_steps(n: int, parity: Parity = "odd") -> int:
    """Largest step count for which no amplitude reaches the boundary.

    CLAUDE.md Sec. 2.2: ``N`` on the odd lattice, ``N-1`` on the even lattice
    for the conventional and symmetric walks.
    """
    return n if parity == "odd" else n - 1


def initial_state(n: int, parity: Parity = "odd") -> np.ndarray:
    """Symmetric initial state for the chosen lattice parity.

    Odd lattice (Paper A): ``(|0,+> + |0,->)/sqrt(2)``.
    Even lattice (Paper B): equal weight ``1/2`` on ``x = +-1``, both spins.
    """
    positions = lattice_positions(n, parity)
    psi = np.zeros((positions.size, 2), dtype=np.complex128)
    if parity == "odd":
        psi[n, :] = 1.0 / np.sqrt(2.0)
    else:
        psi[positions == -1, :] = 0.5
        psi[positions == 1, :] = 0.5
    return psi


@dataclass(frozen=True)
class WalkResult:
    """Final state of one walk realisation, plus the metadata it was run with.

    ``psi_plus`` / ``psi_minus`` are the complex amplitudes of the ``|x,+>``
    and ``|x,->`` components; ``positions`` gives the physical ``x`` of each
    entry. The metadata fields are the ones the data contract (CLAUDE.md
    Sec. 6) requires to travel with every sample.
    """

    psi_plus: np.ndarray
    psi_minus: np.ndarray
    positions: np.ndarray
    n_steps: int
    parity: Parity
    channel: Channel
    theta_0: float
    control_value: float
    seed: int | None

    @property
    def psi(self) -> np.ndarray:
        """The state as a single ``(L, 2)`` array, columns ordered ``(+, -)``."""
        return np.stack([self.psi_plus, self.psi_minus], axis=1)


def evolve(psi: np.ndarray, schedule: Schedule, phi1: float, phi2: float) -> np.ndarray:
    r"""Apply ``(T C)^n`` for the coin angles and translations in ``schedule``.

    One Python loop over *time steps*; the lattice is handled vectorised.
    Evolution is :math:`|\psi(t)\rangle=(\hat T\hat C)^N|\psi_0\rangle`
    (CLAUDE.md Sec. 2.1) -- coin first, then translation.
    """
    for theta, use_inverse in zip(schedule.thetas, schedule.inverse_translation):
        psi = ops.apply_coin(psi, ops.coin(theta, phi1, phi2))
        psi = ops.translate_inverse(psi) if use_inverse else ops.translate(psi)
    return psi


def run_walk(
    n: int,
    channel: Channel = "pure",
    theta_0: float = ops.THETA_0_DEFAULT,
    control_value: float = 0.0,
    *,
    seed: int | None = None,
    parity: Parity = "odd",
    n_steps: int | None = None,
    phi1: float = ops.PHI_DEFAULT,
    phi2: float = ops.PHI_DEFAULT,
) -> WalkResult:
    """Run one realisation of the walk and return its final state.

    Parameters
    ----------
    n
        Lattice parameter: ``2n+1`` sites (odd) or ``2n`` sites (even).
    channel
        Randomness channel, see :mod:`qw.randomness`.
    theta_0
        Base coin angle. Default ``pi/6`` (Paper A).
    control_value
        The channel's control parameter: ``delta_theta``,
        ``delta_theta_max`` or ``p_r``. Ignored for ``"pure"``.
    seed
        Seed for this realisation's classical randomness. ``None`` is allowed
        only for ``"pure"``, which draws nothing.
    parity
        Lattice parity, see module docstring.
    n_steps
        Number of time steps. Defaults to :func:`max_steps`, the largest value
        that keeps the walker off the boundary.
    phi1, phi2
        Coin phases. Both default to ``pi/2``, as in both papers.
    """
    steps = max_steps(n, parity) if n_steps is None else n_steps
    if steps > max_steps(n, parity):
        raise ValueError(
            f"n_steps={steps} exceeds the boundary-free maximum "
            f"{max_steps(n, parity)} for n={n}, parity={parity!r}"
        )

    rng = None if seed is None else np.random.default_rng(seed)
    schedule = make_schedule(channel, steps, theta_0, control_value, rng)
    psi = evolve(initial_state(n, parity), schedule, phi1, phi2)

    return WalkResult(
        psi_plus=psi[:, ops.SPIN_UP],
        psi_minus=psi[:, ops.SPIN_DOWN],
        positions=lattice_positions(n, parity),
        n_steps=steps,
        parity=parity,
        channel=channel,
        theta_0=float(theta_0),
        control_value=float(control_value),
        seed=seed,
    )
