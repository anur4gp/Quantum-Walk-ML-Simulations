"""Shared classifier interface (CLAUDE.md Sec. 7).

Every classifier -- SVM, MLP, CNN -- implements this same protocol so the
critical-point extraction in Phase 2 can treat them interchangeably.

``predict_proba`` returning genuine two-class probabilities is
non-negotiable: the whole confusion-point method depends on it. That is why
the SVM uses ``modified_huber`` loss rather than hinge, and why the CNN ends
in a softmax.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Classifier(Protocol):
    """Two-class classifier over probability distributions.

    ``X`` has shape ``(n_samples, n_sites)``; ``y`` has shape ``(n_samples,)``
    with ``0 = delocalized`` and ``1 = localized``.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on labelled distributions from the two extreme regimes."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, shape ``(n_samples, 2)``, columns ``[P(0), P(1)]``."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean accuracy on ``(X, y)``."""
        ...
