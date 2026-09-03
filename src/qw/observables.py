"""Manual (baseline) observables: P(x), moment of inertia, IPR.

These are the ground-truth comparators the ML classifiers are checked against
(Paper A, Sec. II C; CLAUDE.md Sec. 2.5).

Everything here takes plain arrays, so nothing downstream needs to import the
walk driver to compute an observable.

Paper B's entanglement quantities (entanglement entropy, the overlap proxy;
CLAUDE.md Sec. 2.6) are deliberately not implemented -- they are out of scope
for the current phase.
"""

from __future__ import annotations

import numpy as np


def probability(psi_plus: np.ndarray, psi_minus: np.ndarray) -> np.ndarray:
    r"""Position-resolved probability :math:`P(x)=|\psi_+(x)|^2+|\psi_-(x)|^2`.

    Returns a ``float64`` array with the same shape as the inputs. This is the
    quantity that becomes one ML sample (CLAUDE.md Sec. 6).
    """
    return (np.abs(psi_plus) ** 2 + np.abs(psi_minus) ** 2).astype(np.float64)


def total_probability(psi_plus: np.ndarray, psi_minus: np.ndarray) -> float:
    r""":math:`\sum_x P(x)`. Should stay at 1 to ~1e-12; used by the unitarity tests."""
    return float(probability(psi_plus, psi_minus).sum())


def moment_of_inertia(prob: np.ndarray, positions: np.ndarray) -> float:
    r"""Moment of inertia :math:`\mathrm{MoI}=\sum_x x^2 P(x)` (CLAUDE.md Sec. 2.5).

    ``positions`` must be *physical* positions (centred on the origin), not
    array indices. Delocalised walks give ``MoI ~ N^2``; the critical value is
    read off as the kink where MoI departs from that scaling.
    """
    if prob.shape != positions.shape:
        raise ValueError(
            f"prob {prob.shape} and positions {positions.shape} must match"
        )
    return float(np.sum(positions**2 * prob))


def inverse_participation_ratio(psi_plus: np.ndarray) -> float:
    r"""IPR as defined in CLAUDE.md Sec. 2.5 / Paper A, Sec. II C.

    .. math::
        \mathrm{IPR} = \frac{\left(\sum_x |\psi_+(x)|^2\right)^2}
                            {\sum_x |\psi_+(x)|^4}

    Note two things about this definition, both intentional:

    * It is built from the **spin-up component only**, not from the total
      ``P(x)``, exactly as written in the paper.
    * It is the *participation ratio* orientation: it grows as the
      distribution flattens, so the critical value is its **maximum**.

    IPR cannot distinguish a one-peak from a two-peak distribution -- it only
    flags flatness. Do not read more into it than that.
    """
    p_up = np.abs(psi_plus) ** 2
    denominator = float(np.sum(p_up**2))
    if denominator == 0.0:
        raise ValueError("spin-up component is identically zero; IPR undefined")
    return float(np.sum(p_up) ** 2 / denominator)
