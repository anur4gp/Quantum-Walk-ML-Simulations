"""MLP classifier (Paper A, Sec. III B 2).

Architecture: input (sized to the lattice) -> 400 -> 200 -> 100 -> 50 -> 2,
alpha = 0.001. Paper A held these sizes fixed from N=80 to N=1000, so they are
not derived from ``n_sites`` here.

Two deviations to keep straight (both recorded in notes/project_reference.tex):
sklearn uses a single logistic output unit for a binary problem rather than
Paper A's two softmax neurons -- equivalent up to reparametrisation; and Paper
A does not pin the solver, learning rate, batch size or iteration count, so
those are sklearn defaults and are *our* choice, not paper values.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier as SkMLPClassifier

from data.preprocess import normalize_pmax

HIDDEN_LAYER_SIZES_PAPER_A: tuple[int, ...] = (400, 200, 100, 50)
ALPHA_PAPER_A: float = 0.001


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Two-class MLP over probability distributions (CLAUDE.md Sec. 7.2).

    ``X`` is ``(n_samples, n_sites)``; ``y`` is 0 = delocalized, 1 = localized.
    ``normalize`` applies P_max = 1 to training and inference alike.
    Subclassing the sklearn base classes makes this clonable, so ``normalize``
    can be grid-searched like any other hyperparameter.
    """

    def __init__(
        self,
        *,
        normalize: bool = True,
        hidden_layer_sizes: tuple[int, ...] = HIDDEN_LAYER_SIZES_PAPER_A,
        alpha: float = ALPHA_PAPER_A,
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        batch_size: int | str = "auto",
        max_iter: int = 200,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        validation_fraction: float = 0.1,
        tol: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        # Plain assignment only, no derived state: that is what keeps the
        # estimator clonable and therefore usable inside GridSearchCV.
        self.normalize = normalize
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.random_state = random_state

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return normalize_pmax(arr) if self.normalize else arr

    def _build(self) -> SkMLPClassifier:
        return SkMLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            alpha=self.alpha,
            solver=self.solver,
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            tol=self.tol,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        """Train on labelled distributions; ``y`` must contain both classes."""
        y = np.asarray(y)
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError(
                f"expected both classes 0 (delocalized) and 1 (localized), got {classes}"
            )
        self._model = self._build()
        self._model.fit(self._prepare(X), y)
        self.classes_ = self._model.classes_
        self.n_features_in_ = self._model.n_features_in_
        return self

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

    @property
    def loss_curve(self) -> np.ndarray:
        """Training loss per iteration."""
        return np.asarray(self._model.loss_curve_, dtype=np.float64)

    @property
    def n_iter(self) -> int:
        """Iterations run. Equal to ``max_iter`` means it did not converge."""
        return int(self._model.n_iter_)

    @property
    def input_weight_magnitude(self) -> np.ndarray:
        """Per-site l2 norm of the first-layer weights.

        A sensitivity, not a decision boundary: the MLP is nonlinear, so unlike
        ``SVMClassifier.weights`` this cannot be read directionally.
        """
        return np.linalg.norm(np.asarray(self._model.coefs_[0]), axis=1)


#: Grids for ``scripts/train_mlp.py --grid NAME``. "paper_a" is a single point,
#: so a tuned result always has Paper A's architecture to compare against;
#: "coarse" is the pair of axes Paper A searched. The rest are our exploration.
PARAM_GRIDS: dict[str, dict] = {
    "paper_a": {
        "hidden_layer_sizes": [HIDDEN_LAYER_SIZES_PAPER_A],
        "alpha": [ALPHA_PAPER_A],
    },
    "coarse": {
        "hidden_layer_sizes": [
            (100, 50),
            (200, 100, 50),
            HIDDEN_LAYER_SIZES_PAPER_A,
        ],
        "alpha": [1e-4, 1e-3, 1e-2],
    },
    "architecture": {
        "hidden_layer_sizes": [
            (50,),
            (100, 50),
            (200, 100, 50),
            HIDDEN_LAYER_SIZES_PAPER_A,
            (800, 400, 200, 100),
        ],
    },
    "regularization": {
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    "normalization": {
        "normalize": [True, False],
        "alpha": [1e-4, 1e-3, 1e-2],
    },
    "optimizer": {
        "solver": ["adam", "sgd"],
        "learning_rate_init": [1e-4, 1e-3, 1e-2],
    },
}
