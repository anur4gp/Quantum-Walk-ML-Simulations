"""SVM classifier (Paper A, Sec. III B 1)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from data.preprocess import normalize_pmax


class SVMClassifier:
    """Linear SGDClassifier with modified Huber loss.

    Modified Huber, not hinge: hinge gives a hard true/false decision, this
    gives calibrated probabilities, which Sec. 7.4 needs. ``normalize``
    applies P_max = 1 to training and inference alike.
    """

    def __init__(
        self,
        *,
        normalize: bool = True,
        alpha: float = 1e-4,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int = 0,
    ) -> None:
        self.normalize = normalize
        self.random_state = random_state
        self._model = SGDClassifier(
            loss="modified_huber",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return normalize_pmax(arr) if self.normalize else arr

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on labelled distributions; ``y`` must contain both classes."""
        y = np.asarray(y)
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError(
                f"expected both classes 0 (delocalized) and 1 (localized), got {classes}"
            )
        self._model.fit(self._prepare(X), y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``(n_samples, 2)``, columns ``[P(deloc), P(loc)]``."""
        proba = self._model.predict_proba(self._prepare(X))
        # Checked, not assumed: a transposed column order would silently flip
        # every critical value downstream.
        if not np.array_equal(self._model.classes_, np.array([0, 1])):
            raise RuntimeError(
                f"unexpected class order {self._model.classes_}; "
                "columns must be [P(delocalized), P(localized)]"
            )
        return np.asarray(proba, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(self._prepare(X)))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self._model.score(self._prepare(X), np.asarray(y)))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance from the separating hyperplane."""
        return np.asarray(self._model.decision_function(self._prepare(X)))

    @property
    def weights(self) -> np.ndarray:
        """Weight per lattice site. The model is linear, so positive means localized."""
        return np.asarray(self._model.coef_).ravel()

    @property
    def intercept(self) -> float:
        return float(np.asarray(self._model.intercept_).ravel()[0])
