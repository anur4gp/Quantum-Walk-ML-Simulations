"""Coin and translation operators (Paper A, Sec. II A).

A state is a ``(L, 2)`` complex128 array: ``psi[i, 0] = <x_i,+|psi>``,
``psi[i, 1] = <x_i,-|psi>``. Index ``i`` maps to position ``x`` in
:mod:`qw.walk`, which owns the lattice parity.
"""

from __future__ import annotations

import numpy as np

# Both papers fix phi_1 = phi_2 = pi/2 and theta_0 = pi/6 (CLAUDE.md Sec. 2.1).
PHI_DEFAULT: float = np.pi / 2
THETA_0_DEFAULT: float = np.pi / 6

SPIN_UP: int = 0
SPIN_DOWN: int = 1


def coin(
    theta: float,
    phi1: float = PHI_DEFAULT,
    phi2: float = PHI_DEFAULT,
) -> np.ndarray:
    r"""The 2x2 coin operator :math:`\hat C(\theta,\phi_1,\phi_2)` (Paper A, Sec. II A).

    At the default phases this is ``[[cos, i sin], [i sin, cos]]``.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, np.exp(1j * phi1) * s],
            [np.exp(1j * phi2) * s, -np.exp(1j * (phi1 + phi2)) * c],
        ],
        dtype=np.complex128,
    )


def apply_coin(psi: np.ndarray, c: np.ndarray) -> np.ndarray:
    r"""Apply the coin at every site: :math:`\psi'_{x\sigma}=\sum_\tau C_{\sigma\tau}\psi_{x\tau}`."""
    # psi @ c.T multiplies the spinor from the left, as the matrix is written.
    return psi @ c.T


def translate(psi: np.ndarray) -> np.ndarray:
    r"""Conventional translation :math:`\hat T`: ``|x,+> -> |x+1,+>``, ``|x,-> -> |x-1,->``.

    Open boundaries: amplitude shifted off either end is dropped. The step cap
    in :func:`qw.walk.max_steps` keeps the walker off the boundary, so nothing
    is ever lost in practice.
    """
    out = np.zeros_like(psi)
    out[1:, SPIN_UP] = psi[:-1, SPIN_UP]
    out[:-1, SPIN_DOWN] = psi[1:, SPIN_DOWN]
    return out


def translate_inverse(psi: np.ndarray) -> np.ndarray:
    r""":math:`\hat T^{-1}`: ``|x,+> -> |x-1,+>``, ``|x,-> -> |x+1,->`` (Paper A, Sec. II B 3)."""
    out = np.zeros_like(psi)
    out[:-1, SPIN_UP] = psi[1:, SPIN_UP]
    out[1:, SPIN_DOWN] = psi[:-1, SPIN_DOWN]
    return out
