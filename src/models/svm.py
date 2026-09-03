from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from data.preprocess import normalize_pmax


class SVMClassifier:

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
            loss="modified_huber",  # Paper A, Sec. III B 1 -- gives probabilities
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    # -- internals ------------------------------------------------------------

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return normalize_pmax(arr) if self.normalize else arr

    # -- Classifier protocol --------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on labelled distributions from the two extreme regimes.

        ``y`` must contain both classes, with ``0 = delocalized`` and
        ``1 = localized``.
        """
        y = np.asarray(y)
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError(
                f"expected both classes 0 (delocalized) and 1 (localized), got {classes}"
            )
        self._model.fit(self._prepare(X), y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, shape ``(n_samples, 2)``, columns ``[P(0), P(1)]``.

        Column order is asserted against ``classes_`` rather than assumed, so a
        silently transposed confusion point is impossible.
        """
        proba = self._model.predict_proba(self._prepare(X))
        if not np.array_equal(self._model.classes_, np.array([0, 1])):
            raise RuntimeError(
                f"unexpected class order {self._model.classes_}; "
                "columns must be [P(delocalized), P(localized)]"
            )
        return np.asarray(proba, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard class labels, shape ``(n_samples,)``."""
        return np.asarray(self._model.predict(self._prepare(X)))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean accuracy on ``(X, y)``."""
        return float(self._model.score(self._prepare(X), np.asarray(y)))

    # -- inspection -----------------------------------------------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance from the separating hyperplane, shape ``(n_samples,)``."""
        return np.asarray(self._model.decision_function(self._prepare(X)))

    @property
    def weights(self) -> np.ndarray:
        """Learned weight per lattice site, shape ``(n_sites,)``.

        The model is linear, so this is directly interpretable: it is the
        spatial profile the classifier is actually keying on. Positive weight
        pushes a site's probability toward ``localized``.
        """
        return np.asarray(self._model.coef_).ravel()

    @property
    def intercept(self) -> float:
        return float(np.asarray(self._model.intercept_).ravel()[0])
