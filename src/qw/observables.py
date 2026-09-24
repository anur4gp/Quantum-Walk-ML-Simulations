"""Manual baseline observables: P(x), MoI, IPR (Paper A, Sec. II C; CLAUDE.md Sec. 2.5).

Plain arrays in, numbers out -- nothing here imports the walk driver. Paper B's
entanglement quantities (CLAUDE.md Sec. 2.6) are out of scope for this phase.
"""

from __future__ import annotations

import numpy as np


def probability(psi_plus: np.ndarray, psi_minus: np.ndarray) -> np.ndarray:
    r""":math:`P(x)=|\psi_+(x)|^2+|\psi_-(x)|^2`. One row of this is one ML sample."""
    return (np.abs(psi_plus) ** 2 + np.abs(psi_minus) ** 2).astype(np.float64)


def total_probability(psi_plus: np.ndarray, psi_minus: np.ndarray) -> float:
    r""":math:`\sum_x P(x)`, which must stay at 1 to ~1e-12 (unitarity tests)."""
    return float(probability(psi_plus, psi_minus).sum())


def moment_of_inertia(prob: np.ndarray, positions: np.ndarray) -> float:
    r""":math:`\mathrm{MoI}=\sum_x x^2P(x)`. ``positions`` are physical x, not indices."""
    if prob.shape != positions.shape:
        raise ValueError(
            f"prob {prob.shape} and positions {positions.shape} must match"
        )
    return float(np.sum(positions**2 * prob))


def inverse_participation_ratio(psi_plus: np.ndarray) -> float:
    r""":math:`(\sum_x|\psi_+|^2)^2/\sum_x|\psi_+|^4` (CLAUDE.md Sec. 2.5).

    Spin-up component only, as written in the paper, and oriented so that it
    grows with flatness -- the critical value is its maximum. It cannot tell a
    one-peak distribution from a two-peak one.
    """
    p_up = np.abs(psi_plus) ** 2
    denominator = float(np.sum(p_up**2))
    if denominator == 0.0:
        raise ValueError("spin-up component is identically zero; IPR undefined")
    return float(np.sum(p_up) ** 2 / denominator)


def moment_of_inertia_series(prob: np.ndarray, positions: np.ndarray) -> np.ndarray:
    r"""Same :math:`\sum_x x^2P(x)` over a stack whose last axis is the lattice.

    ``prob`` may be ``(n_times, n_sites)`` or any higher-rank sweep; returns
    ``prob.shape[:-1]``.
    """
    prob = np.asarray(prob, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if prob.shape[-1] != positions.shape[-1] or positions.ndim != 1:
        raise ValueError(
            f"prob's last axis {prob.shape[-1:]} must match positions {positions.shape}"
        )
    return prob @ positions**2
