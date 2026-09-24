"""Evolution driver: |psi(t)> = (T C)^N |psi_0> (CLAUDE.md Sec. 2.1).

Lattice parity is an explicit parameter, never hardcoded (CLAUDE.md Sec. 2.2):

    odd  (Paper A): 2n+1 sites, -n..n, start (|0,+> + |0,->)/sqrt(2), n steps
    even (Paper B): 2n sites, no x=0, start 1/2 on x=+-1 both spins, n-1 steps

The ML work uses the odd lattice with the conventional walk. The even lattice
is here because Paper B's symmetric and split-step translations are not
unitary on a lattice with a central site; those translations are not
implemented yet.

``run_walk`` keeps only the final state; ``run_walk_trajectory`` keeps the
whole history P(x, t). Same driver, same schedule, same seed -- they differ
only in what is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import operators as ops
from .randomness import Channel, Schedule, make_schedule

Parity = Literal["odd", "even"]


def lattice_positions(n: int, parity: Parity = "odd") -> np.ndarray:
    """Physical position x of each array index: ``-n..n`` (odd), no zero (even)."""
    if parity == "odd":
        return np.arange(-n, n + 1, dtype=np.float64)
    if parity == "even":
        return np.concatenate(
            [np.arange(-n, 0, dtype=np.float64), np.arange(1, n + 1, dtype=np.float64)]
        )
    raise ValueError(f"parity must be 'odd' or 'even', got {parity!r}")


def max_steps(n: int, parity: Parity = "odd") -> int:
    """Largest step count with no amplitude at the boundary (CLAUDE.md Sec. 2.2)."""
    return n if parity == "odd" else n - 1


def initial_state(n: int, parity: Parity = "odd") -> np.ndarray:
    """Symmetric initial state for the chosen parity (CLAUDE.md Sec. 2.2)."""
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
    """Final state of one realisation, plus the metadata of CLAUDE.md Sec. 6."""

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
        """The state as one ``(L, 2)`` array, columns ``(+, -)``."""
        return np.stack([self.psi_plus, self.psi_minus], axis=1)


def evolve(psi: np.ndarray, schedule: Schedule, phi1: float, phi2: float) -> np.ndarray:
    """Apply coin then translation once per entry in ``schedule``."""
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
    """Run one realisation and return its final state.

    ``control_value`` is the channel's control parameter (delta_theta,
    delta_theta_max or p_r) and is ignored for ``"pure"``. ``seed`` may be None
    only for ``"pure"``, which draws nothing. ``n_steps`` defaults to
    :func:`max_steps`.
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


@dataclass(frozen=True)
class TrajectoryResult:
    """Whole evolution of one realisation. ``psi_t`` is ``(n_recorded, L, 2)``."""

    psi_t: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    n_steps: int
    parity: Parity
    channel: Channel
    theta_0: float
    control_value: float
    seed: int | None

    @property
    def probability(self) -> np.ndarray:
        """P(x, t), shape ``(n_recorded, L)``. Each row sums to 1."""
        from .observables import probability

        return probability(self.psi_t[..., ops.SPIN_UP], self.psi_t[..., ops.SPIN_DOWN])

    @property
    def final(self) -> WalkResult:
        """The last recorded slice as an ordinary :class:`WalkResult`."""
        return WalkResult(
            psi_plus=self.psi_t[-1, :, ops.SPIN_UP],
            psi_minus=self.psi_t[-1, :, ops.SPIN_DOWN],
            positions=self.positions,
            n_steps=int(self.times[-1]),
            parity=self.parity,
            channel=self.channel,
            theta_0=self.theta_0,
            control_value=self.control_value,
            seed=self.seed,
        )


def evolve_trajectory(
    psi: np.ndarray,
    schedule: Schedule,
    phi1: float,
    phi2: float,
    *,
    record_every: int = 1,
    include_initial: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """As :func:`evolve`, but returns ``(psi_t, times)`` with the kept states.

    Unrecorded steps are still evolved, just not stored. ``include_initial``
    adds the t=0 slice, which is the same for every realisation and so carries
    no label information.
    """
    if record_every < 1:
        raise ValueError(f"record_every must be >= 1, got {record_every}")

    frames: list[np.ndarray] = []
    times: list[int] = []
    if include_initial:
        frames.append(psi.copy())
        times.append(0)

    for step, (theta, use_inverse) in enumerate(
        zip(schedule.thetas, schedule.inverse_translation), start=1
    ):
        psi = ops.apply_coin(psi, ops.coin(theta, phi1, phi2))
        psi = ops.translate_inverse(psi) if use_inverse else ops.translate(psi)
        if step % record_every == 0:
            frames.append(psi.copy())
            times.append(step)

    if not frames:
        raise ValueError(
            f"record_every={record_every} recorded nothing from "
            f"{schedule.n_steps} steps"
        )
    return np.asarray(frames, dtype=np.complex128), np.asarray(times, dtype=int)


def run_walk_trajectory(
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
    record_every: int = 1,
    include_initial: bool = False,
) -> TrajectoryResult:
    """Run one realisation and keep its whole evolution.

    Same schedule and same final state as :func:`run_walk` at the same seed.
    """
    steps = max_steps(n, parity) if n_steps is None else n_steps
    if steps > max_steps(n, parity):
        raise ValueError(
            f"n_steps={steps} exceeds the boundary-free maximum "
            f"{max_steps(n, parity)} for n={n}, parity={parity!r}"
        )

    rng = None if seed is None else np.random.default_rng(seed)
    schedule = make_schedule(channel, steps, theta_0, control_value, rng)
    psi_t, times = evolve_trajectory(
        initial_state(n, parity),
        schedule,
        phi1,
        phi2,
        record_every=record_every,
        include_initial=include_initial,
    )

    return TrajectoryResult(
        psi_t=psi_t,
        times=times,
        positions=lattice_positions(n, parity),
        n_steps=steps,
        parity=parity,
        channel=channel,
        theta_0=float(theta_0),
        control_value=float(control_value),
        seed=seed,
    )
