"""Shared classifier interface (CLAUDE.md Sec. 7)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Classifier(Protocol):
    """Two-class classifier over probability distributions.

    ``X`` is ``(n_samples, n_sites)``; ``y`` is 0 = delocalized, 1 = localized.
    ``predict_proba`` must return genuine two-class probabilities -- the
    confusion-point method in Sec. 7.4 depends on it.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Shape ``(n_samples, 2)``, columns ``[P(deloc), P(loc)]``."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...
