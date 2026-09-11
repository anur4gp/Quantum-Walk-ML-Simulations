"""MLP classifier (Paper A, Sec. III B 2).

Architecture, from CLAUDE.md Sec. 7.2::

    input (auto-sized to the lattice) -> 400 -> 200 -> 100 -> 50 -> output 2

with regularisation ``alpha = 0.001``. The hidden sizes were kept fixed from
N=80 to N=1000 in Paper A; scaling them with lattice size gave negligible
improvement, so they are *not* derived from ``n_sites`` here either.

Two honest caveats about the mapping onto sklearn
-------------------------------------------------
1. **"Output 2 neurons."** Paper A describes a two-neuron output layer.
   ``sklearn``'s ``MLPClassifier`` uses a *single* logistic output unit for a
   binary problem, not two softmax units. The two are equivalent up to
   reparametrisation and ``predict_proba`` still returns the required
   ``(n, 2)``, but the network written to disk is not literally the one in the
   paper's figure. Flagged rather than papered over; if the difference ever
   matters, the CNN (Keras, genuine 2-neuron softmax) is the place to check it
   against.
2. **Optimiser settings.** Paper A specifies the layer sizes and ``alpha``.
   It does not (per CLAUDE.md) pin the solver, learning rate, batch size or
   iteration count, so those keep sklearn's defaults and are exposed as
   constructor arguments. Anything tuned here is *our* choice and must be
   recorded with any number quoted from it -- it is not a paper value.

Grid search
-----------
The class subclasses ``BaseEstimator``/``ClassifierMixin``, so it is directly
usable inside ``GridSearchCV`` -- including the ``normalize`` flag, which
becomes just another tunable hyperparameter. See ``PARAM_GRIDS`` and
``scripts/train_mlp.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier as SkMLPClassifier

from data.preprocess import normalize_pmax

#: Paper A's hidden layers (CLAUDE.md Sec. 7.2). Fixed, not lattice-scaled.
HIDDEN_LAYER_SIZES_PAPER_A: tuple[int, ...] = (400, 200, 100, 50)

#: Paper A's regularisation strength (CLAUDE.md Sec. 7.2).
ALPHA_PAPER_A: float = 0.001


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Two-class MLP over probability distributions (CLAUDE.md Sec. 7).

    ``X`` has shape ``(n_samples, n_sites)``; ``y`` is ``0 = delocalized``,
    ``1 = localized``.

    Parameters
    ----------
    normalize
        Apply the ``P_max = 1`` ML normalisation (Paper A, Sec. III B 4) to
        both training and inference data. Paper A found this substantially
        reduces variance for the MLP, so it is on by default -- but it is a
        flag, and it is a legitimate thing to grid-search over.
    hidden_layer_sizes, alpha
        Paper A values by default.
    solver, learning_rate_init, batch_size, max_iter, early_stopping,
    n_iter_no_change, validation_fraction, tol
        Not specified by Paper A. sklearn defaults unless you change them.
    random_state
        Seeds sklearn's weight initialisation and shuffling. Required for the
        reproducibility rule in CLAUDE.md Sec. 8.
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
        # Plain attribute assignment only, no derived state: this is what makes
        # the estimator clonable, and therefore usable inside GridSearchCV.
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

    # -- internals ------------------------------------------------------------

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

    # -- Classifier protocol --------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        """Train on labelled distributions from the two extreme regimes.

        ``y`` must contain both classes, with ``0 = delocalized`` and
        ``1 = localized``. Returns ``self`` so it composes with sklearn tooling
        (the ``Classifier`` protocol only requires that it trains).
        """
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

    @property
    def loss_curve(self) -> np.ndarray:
        """Training loss per iteration -- the quickest check that it converged."""
        return np.asarray(self._model.loss_curve_, dtype=np.float64)

    @property
    def n_iter(self) -> int:
        """Iterations actually run. Equal to ``max_iter`` means it did *not* converge."""
        return int(self._model.n_iter_)

    @property
    def input_weight_magnitude(self) -> np.ndarray:
        """Per-lattice-site :math:`\\ell_2` norm of the first-layer weights.

        Shape ``(n_sites,)``. Not a decision boundary -- an MLP is not linear,
        so unlike ``SVMClassifier.weights`` this cannot be read as "positive
        means localized". It only says which sites the first layer is
        *sensitive* to, which is still the fastest way to spot a network keying
        on the lattice edge or on a single site.
        """
        return np.linalg.norm(np.asarray(self._model.coefs_[0]), axis=1)


#: Named grids for ``GridSearchCV`` (see ``scripts/train_mlp.py --grid NAME``).
#:
#: ``"paper_a"`` is deliberately a single point: it pins Paper A's stated
#: architecture so a tuned result always has something to be compared against.
#: The others are exploration and are *our* choices, not paper values.
PARAM_GRIDS: dict[str, dict] = {
    "paper_a": {
        "hidden_layer_sizes": [HIDDEN_LAYER_SIZES_PAPER_A],
        "alpha": [ALPHA_PAPER_A],
    },
    # Paper A explored exactly these two axes with GridSearchCV (Sec. III B 2).
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
    # Does the P_max = 1 normalisation actually earn its place for the MLP?
    # Paper A says yes (variance reduction); this makes that checkable here.
    "normalization": {
        "normalize": [True, False],
        "alpha": [1e-4, 1e-3, 1e-2],
    },
    "optimizer": {
        "solver": ["adam", "sgd"],
        "learning_rate_init": [1e-4, 1e-3, 1e-2],
    },
}
