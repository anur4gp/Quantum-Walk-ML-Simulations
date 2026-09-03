"""Coin and translation operators for the discrete-time quantum walk.

Conventions follow Paper A (Phys. Rev. E 108, 035308), Sec. II A.

State representation
--------------------
A walk state is a ``(L, 2)`` ``complex128`` array ``psi``, where ``L`` is the
number of lattice sites and the two columns are the internal ("coin"/"spin")
components in the order ``(+, -)``:

    psi[i, 0] == <x_i, +|psi>
    psi[i, 1] == <x_i, -|psi>

The mapping from array index ``i`` to physical position ``x`` is owned by
:mod:`qw.walk` (it depends on the lattice parity), not by this module.
"""

from __future__ import annotations

import numpy as np

# Paper A and Paper B both fix phi_1 = phi_2 = pi/2 throughout (CLAUDE.md Sec. 2.1).
PHI_DEFAULT: float = np.pi / 2

# Paper A's default coin angle (CLAUDE.md Sec. 2.1).
THETA_0_DEFAULT: float = np.pi / 6

SPIN_UP: int = 0
SPIN_DOWN: int = 1


def coin(
    theta: float,
    phi1: float = PHI_DEFAULT,
    phi2: float = PHI_DEFAULT,
) -> np.ndarray:
    r"""Return the 2x2 coin (rotation) operator :math:`\hat C(\theta,\phi_1,\phi_2)`.

    Paper A, Sec. II A::

        C = [[      cos(theta),        e^{i phi1} sin(theta)],
             [e^{i phi2} sin(theta), -e^{i(phi1+phi2)} cos(theta)]]

    For the default ``phi1 = phi2 = pi/2`` this reduces to the symmetric

        C = [[cos(theta), i sin(theta)],
             [i sin(theta), cos(theta)]]

    Parameters
    ----------
    theta
        Coin angle, radians.
    phi1, phi2
        Coin phases, radians. Default ``pi/2`` for both, as in both papers.

    Returns
    -------
    ndarray
        Shape ``(2, 2)``, ``complex128``. Unitary for all real arguments.
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
    r"""Apply a coin operator to every site: :math:`\psi'_{x\sigma} = \sum_\tau C_{\sigma\tau}\psi_{x\tau}`.

    Parameters
    ----------
    psi
        State array, shape ``(L, 2)``.
    c
        Coin operator, shape ``(2, 2)``, e.g. from :func:`coin`.

    Returns
    -------
    ndarray
        New state array, shape ``(L, 2)``. The input is not modified.

    Notes
    -----
    Implemented as ``psi @ c.T`` so that ``c`` multiplies the spinor from the
    left, matching the matrix written in Paper A. For the default
    ``phi1 = phi2 = pi/2`` the coin is symmetric and ``c.T == c``.
    """
    return psi @ c.T


def translate(psi: np.ndarray) -> np.ndarray:
    r"""Conventional translation :math:`\hat T`.

    Paper A, Sec. II A: :math:`\hat T|\psi_x,+\rangle=|\psi_{x+1},+\rangle`,
    :math:`\hat T|\psi_x,-\rangle=|\psi_{x-1},-\rangle`.

    Open boundaries: amplitude shifted past either end of the array is
    discarded. On the Paper A lattice (``L = 2N+1`` sites, ``N`` steps from the
    centre) the walker never reaches the boundary, so no amplitude is lost and
    the operator is unitary on the reachable subspace.
    """
    out = np.zeros_like(psi)
    out[1:, SPIN_UP] = psi[:-1, SPIN_UP]
    out[:-1, SPIN_DOWN] = psi[1:, SPIN_DOWN]
    return out


def translate_inverse(psi: np.ndarray) -> np.ndarray:
    r"""Inverse translation :math:`\hat T^{-1}`.

    Paper A, Sec. II B 3: :math:`|\psi_x,+\rangle\to|\psi_{x-1},+\rangle`,
    :math:`|\psi_x,-\rangle\to|\psi_{x+1},-\rangle`. This is the operator
    applied with probability ``P_r`` in the random-translation channel.
    """
    out = np.zeros_like(psi)
    out[:-1, SPIN_UP] = psi[1:, SPIN_UP]
    out[1:, SPIN_DOWN] = psi[:-1, SPIN_DOWN]
    return out
