"""MLP classifier tests (CLAUDE.md Sec. 7.2).

Validation order follows CLAUDE.md Sec. 1: synthetic / trivially separable data
first, then real QW distributions from the far-delocalized and far-localized
regimes. The synthetic fixture is the same shape as the one in
``test_svm.py`` -- one central peak vs. two symmetric peaks -- so the two
classifiers are being asked the same question.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from data.generate import DELOCALIZED, LOCALIZED, make_dataset
from models.base import Classifier
from models.mlp import ALPHA_PAPER_A, HIDDEN_LAYER_SIZES_PAPER_A, PARAM_GRIDS, MLPClassifier

N_SITES = 121
CENTRE = N_SITES // 2
THETA_0 = np.pi / 6


def _gaussian(centre: float, width: float) -> np.ndarray:
    x = np.arange(N_SITES, dtype=np.float64)
    return np.exp(-((x - centre) ** 2) / (2 * width**2))


def synthetic_dataset(n_per_class: int = 60, seed: int = 0):
    """One central peak (localized) vs two symmetric peaks (delocalized)."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n_per_class):
        offset = rng.uniform(35, 45)
        two_peak = _gaussian(CENTRE - offset, 6.0) + _gaussian(CENTRE + offset, 6.0)
        rows.append(two_peak / two_peak.sum())
        labels.append(DELOCALIZED)

        one_peak = _gaussian(CENTRE, rng.uniform(6.0, 10.0))
        rows.append(one_peak / one_peak.sum())
        labels.append(LOCALIZED)
    return np.asarray(rows), np.asarray(labels)


def _fast() -> MLPClassifier:
    """A small net, so the tests stay quick. Architecture is checked separately."""
    return MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=0)


# --- interface ---------------------------------------------------------------


def test_satisfies_the_classifier_protocol() -> None:
    assert isinstance(MLPClassifier(), Classifier)


def test_paper_a_defaults() -> None:
    """The default architecture is Paper A's, not sklearn's (CLAUDE.md Sec. 7.2)."""
    clf = MLPClassifier()
    assert clf.hidden_layer_sizes == HIDDEN_LAYER_SIZES_PAPER_A == (400, 200, 100, 50)
    assert clf.alpha == ALPHA_PAPER_A == 0.001
    assert clf.normalize is True


def test_predict_proba_shape_and_normalisation() -> None:
    X, y = synthetic_dataset()
    clf = _fast().fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_predict_proba_column_order_is_deloc_then_loc() -> None:
    """Column 0 must be P(delocalized). A flipped column silently inverts Phase 2."""
    X, y = synthetic_dataset()
    clf = _fast().fit(X, y)
    proba = clf.predict_proba(X)

    assert proba[y == DELOCALIZED, DELOCALIZED].mean() > 0.5
    assert proba[y == LOCALIZED, LOCALIZED].mean() > 0.5
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))


def test_predict_agrees_with_predict_proba() -> None:
    X, y = synthetic_dataset()
    clf = _fast().fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), clf.predict_proba(X).argmax(axis=1))


def test_single_sample_is_accepted() -> None:
    """Phase 2 feeds distributions one control value at a time."""
    X, y = synthetic_dataset()
    clf = _fast().fit(X, y)
    assert clf.predict_proba(X[0]).shape == (1, 2)


def test_rejects_single_class_training_data() -> None:
    X, y = synthetic_dataset()
    with pytest.raises(ValueError, match="both classes"):
        _fast().fit(X[y == DELOCALIZED], y[y == DELOCALIZED])


# --- reproducibility ---------------------------------------------------------


def test_same_random_state_gives_identical_probabilities() -> None:
    """CLAUDE.md Sec. 8: every number must be regenerable from a config + seed."""
    X, y = synthetic_dataset()
    a = _fast().fit(X, y).predict_proba(X)
    b = _fast().fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_different_random_state_changes_the_fit() -> None:
    """Guards against random_state being silently ignored."""
    X, y = synthetic_dataset()
    a = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=20, random_state=0).fit(X, y)
    b = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=20, random_state=1).fit(X, y)
    assert not np.array_equal(a.loss_curve[:5], b.loss_curve[:5])


# --- normalization -----------------------------------------------------------


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_normalize_flag_changes_what_the_network_sees() -> None:
    """P_max = 1 must be a flag, not a hardcode (CLAUDE.md Sec. 6)."""
    X, y = synthetic_dataset()
    on = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=50, random_state=0,
                       normalize=True).fit(X, y)
    off = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=50, random_state=0,
                        normalize=False).fit(X, y)
    assert not np.allclose(on.predict_proba(X), off.predict_proba(X))


def test_normalization_is_applied_at_inference_too() -> None:
    """A rescaled distribution is physically the same sample under P_max = 1."""
    X, y = synthetic_dataset()
    clf = _fast().fit(X, y)
    np.testing.assert_allclose(
        clf.predict_proba(X[:5]), clf.predict_proba(X[:5] * 7.3), atol=1e-9
    )


# --- sklearn integration (the PI's GridSearchCV request) ---------------------


def test_is_clonable_by_sklearn() -> None:
    """Required for GridSearchCV: params must round-trip through get_params."""
    clf = MLPClassifier(alpha=0.01, hidden_layer_sizes=(8, 4), normalize=False)
    twin = clone(clf)
    assert twin.get_params() == clf.get_params()


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_grid_search_runs_and_selects_from_the_grid() -> None:
    X, y = synthetic_dataset()
    search = GridSearchCV(
        MLPClassifier(max_iter=100, random_state=0),
        {"hidden_layer_sizes": [(8, 4), (16, 8)], "alpha": [1e-3, 1e-2]},
        cv=3,
        n_jobs=1,
    )
    search.fit(X, y)
    assert search.best_params_["hidden_layer_sizes"] in [(8, 4), (16, 8)]
    assert search.best_score_ > 0.9
    assert len(search.cv_results_["mean_test_score"]) == 4


@pytest.mark.parametrize("name", sorted(PARAM_GRIDS))
def test_every_named_grid_uses_real_constructor_parameters(name: str) -> None:
    """A typo in PARAM_GRIDS should fail here, not 20 minutes into a search."""
    valid = set(MLPClassifier().get_params())
    assert set(PARAM_GRIDS[name]) <= valid, f"grid {name!r} has unknown parameters"


# --- inspection --------------------------------------------------------------


def test_input_weight_magnitude_has_one_entry_per_site() -> None:
    X, y = synthetic_dataset()
    clf = _fast().fit(X, y)
    assert clf.input_weight_magnitude.shape == (N_SITES,)
    assert np.all(clf.input_weight_magnitude >= 0.0)


def test_loss_curve_length_matches_n_iter() -> None:
    X, y = synthetic_dataset()
    clf = _fast().fit(X, y)
    assert clf.loss_curve.size == clf.n_iter


# --- real QW distributions ---------------------------------------------------


@pytest.mark.parametrize(
    "channel, window_deloc, window_loc",
    [
        ("discrete_coin", (0.0, 0.05), (0.45, THETA_0)),
        ("continuous_coin", (0.0, 0.05), (0.45, THETA_0)),
        ("random_translation", (0.0, 0.02), (0.45, 0.5)),
    ],
)
def test_separates_the_two_extreme_regimes(channel, window_deloc, window_loc) -> None:
    """Phase 1 acceptance: high accuracy on the far-delocalized vs far-localized problem.

    This proves the pipeline works, and nothing more -- the classes are
    one-peak vs two-peak and trivially separable (Paper A, Sec. III B 1).
    Small n and sample count keep the suite fast; the quotable numbers come
    from ``scripts/train_mlp.py``, not from here.
    """
    ds = make_dataset(30, channel, window_deloc, window_loc, n_samples=120,
                      theta_0=THETA_0, seed=0)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=0)
    clf.fit(ds.X, ds.y)
    assert clf.score(ds.X, ds.y) > 0.95
