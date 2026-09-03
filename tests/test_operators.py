"""Operator-level tests (CLAUDE.md Sec. 8, "Testing")."""

from __future__ import annotations

import numpy as np
import pytest

from qw import operators as ops
from qw.walk import initial_state

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("theta", [0.0, np.pi / 6, np.pi / 4, np.pi / 2, 1.234])
@pytest.mark.parametrize("phi", [np.pi / 2, 0.3])
def test_coin_is_unitary(theta: float, phi: float) -> None:
    c = ops.coin(theta, phi, phi)
    np.testing.assert_allclose(c.conj().T @ c, np.eye(2), atol=1e-14)


def test_coin_reduces_to_paper_form_at_default_phases() -> None:
    """With phi1 = phi2 = pi/2 the coin is [[cos, i sin], [i sin, cos]]."""
    theta = 0.4
    expected = np.array(
        [[np.cos(theta), 1j * np.sin(theta)], [1j * np.sin(theta), np.cos(theta)]]
    )
    np.testing.assert_allclose(ops.coin(theta), expected, atol=1e-14)


def test_coin_is_symmetric_at_default_phases() -> None:
    """Justifies that legacy `psi @ C` and correct `psi @ C.T` agree by default."""
    c = ops.coin(0.7)
    np.testing.assert_allclose(c, c.T, atol=1e-15)


def test_translation_inverse_is_left_and_right_inverse() -> None:
    """T^-1 T = 1 and T T^-1 = 1 away from the boundary."""
    psi = np.zeros((11, 2), dtype=np.complex128)
    psi[2:9, :] = RNG.normal(size=(7, 2)) + 1j * RNG.normal(size=(7, 2))

    np.testing.assert_allclose(ops.translate_inverse(ops.translate(psi)), psi, atol=0)
    np.testing.assert_allclose(ops.translate(ops.translate_inverse(psi)), psi, atol=0)


def test_translate_moves_spins_in_opposite_directions() -> None:
    psi = initial_state(3)  # 7 sites, index 3 is x = 0
    out = ops.translate(psi)
    assert out[4, ops.SPIN_UP] == psi[3, ops.SPIN_UP]   # |+> moved to x = +1
    assert out[2, ops.SPIN_DOWN] == psi[3, ops.SPIN_DOWN]  # |-> moved to x = -1
    assert out[3, ops.SPIN_UP] == 0
    assert out[3, ops.SPIN_DOWN] == 0
