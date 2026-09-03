"""SVM classifier tests (CLAUDE.md Sec. 7.1).

Validation order follows CLAUDE.md Sec. 1: synthetic / trivially separable data
first, then real QW distributions from the far-delocalized and far-localized
regimes.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generate import DELOCALIZED, LOCALIZED, make_dataset
from models.base import Classifier
from models.svm import SVMClassifier

N_SITES = 121
CENTRE = N_SITES // 2


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


# --- interface ---------------------------------------------------------------


def test_satisfies_the_classifier_protocol() -> None:
    assert isinstance(SVMClassifier(), Classifier)


def test_predict_proba_shape_and_normalisation() -> None:
    X, y = synthetic_dataset()
    clf = SVMClassifier(random_state=0)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def _fitted_with_prototypes(seed: int = 0):
    X, y = synthetic_dataset()
    clf = SVMClassifier(random_state=seed)
    clf.fit(X, y)
    return clf, X[y == DELOCALIZED].mean(axis=0), X[y == LOCALIZED].mean(axis=0)


def _blend(deloc: np.ndarray, loc: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Interpolate between the two class prototypes; w=0 delocalized, w=1 localized."""
    return np.array([(1 - t) * deloc + t * loc for t in np.atleast_1d(w)])


def _decision_crossing(clf, deloc: np.ndarray, loc: np.ndarray) -> float:
    """Blend weight at which the decision function changes sign."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if clf.decision_function(_blend(deloc, loc, mid))[0] < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_probabilities_are_graded_near_the_decision_boundary() -> None:
    """modified_huber must give graded probabilities; hinge would not.

    Phase 2's confusion point needs a continuum, so a model that only ever
    emitted 0/1 would be useless even at perfect accuracy. The graded band is
    narrow (see the saturation test below), so this samples around the
    crossing rather than across the whole blend.
    """
    clf, deloc, loc = _fitted_with_prototypes()
    centre = _decision_crossing(clf, deloc, loc)

    blends = _blend(deloc, loc, np.linspace(centre - 0.002, centre + 0.002, 41))
    p_deloc = clf.predict_proba(blends)[:, DELOCALIZED]

    interior = (p_deloc > 0.01) & (p_deloc < 0.99)
    assert interior.sum() >= 5, f"only {interior.sum()} graded points; output is hard"
    # and it must vary monotonically with the decision function, not jitter
    order = np.argsort(clf.decision_function(blends))
    assert np.all(np.diff(p_deloc[order]) <= 1e-12)


def test_probabilities_saturate_outside_the_unit_margin() -> None:
    """sklearn maps modified_huber output as P(localized) = clip(d+1, 0, 2)/2.

    So the probability is exactly 0 or 1 wherever |decision| >= 1, and the
    graded band is only two decision-units wide. On well-separated training
    data the decision function ranges over hundreds of units, which makes that
    band a very small fraction of any randomness sweep.

    This is pinned as a test because it bears directly on Phase 2: the SVM's
    "point of maximal confusion" is read from a narrow window, which is the
    suspected cause of the ML methods' systematically low exponents
    (CLAUDE.md Sec. 7.4). If a future change widens or softens this, the
    confusion-point extraction has to be revisited, not silently inherited.
    """
    clf, deloc, loc = _fitted_with_prototypes()
    blends = _blend(deloc, loc, np.linspace(0.0, 1.0, 201))
    d = clf.decision_function(blends)
    p_loc = clf.predict_proba(blends)[:, LOCALIZED]

    np.testing.assert_allclose(p_loc, np.clip(d + 1.0, 0.0, 2.0) / 2.0, atol=1e-12)
    assert np.all(p_loc[d <= -1.0] == 0.0)
    assert np.all(p_loc[d >= 1.0] == 1.0)


def test_probability_columns_track_the_labels() -> None:
    """Column 0 is P(delocalized), column 1 is P(localized). Never flipped."""
    X, y = synthetic_dataset()
    clf = SVMClassifier(random_state=0)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    assert proba[y == DELOCALIZED, DELOCALIZED].mean() > 0.5
    assert proba[y == LOCALIZED, LOCALIZED].mean() > 0.5
    np.testing.assert_array_equal(proba.argmax(axis=1), clf.predict(X))


def test_fit_rejects_a_single_class() -> None:
    X, y = synthetic_dataset()
    with pytest.raises(ValueError, match="expected both classes"):
        SVMClassifier().fit(X[y == DELOCALIZED], y[y == DELOCALIZED])


# --- normalisation flag ------------------------------------------------------


def test_normalization_is_toggleable_and_on_by_default() -> None:
    assert SVMClassifier().normalize is True
    assert SVMClassifier(normalize=False).normalize is False


def test_normalization_applies_to_inference_as_well_as_training() -> None:
    """A rescaled input must classify identically when normalize=True.

    P_max = 1 normalisation is scale-invariant, so multiplying a distribution
    by a constant cannot change its prediction. If it did, normalisation was
    being applied at fit time only.
    """
    X, y = synthetic_dataset()
    clf = SVMClassifier(random_state=0, normalize=True)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict_proba(X), clf.predict_proba(X * 7.5), atol=1e-12)


# --- reproducibility ---------------------------------------------------------


def test_same_random_state_gives_identical_models() -> None:
    X, y = synthetic_dataset()
    a, b = SVMClassifier(random_state=11), SVMClassifier(random_state=11)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


# --- accuracy ----------------------------------------------------------------


def test_separates_synthetic_one_peak_from_two_peak() -> None:
    X, y = synthetic_dataset(seed=1)
    clf = SVMClassifier(random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


@pytest.mark.parametrize(
    "channel,deloc,loc",
    [
        ("discrete_coin", (0.0, 0.05), (0.47, 0.52)),
        ("continuous_coin", (0.0, 0.05), (0.47, 0.52)),
        ("random_translation", (0.0, 0.02), (0.45, 0.50)),
    ],
)
def test_separates_real_qw_extremes(channel: str, deloc, loc) -> None:
    """Held-out accuracy on real distributions from the two extreme regimes.

    This is a pipeline smoke test, not a result: one-peak vs two-peak is
    trivially separable (CLAUDE.md Sec. 7.1). Windows here are deliberately
    far from the transition.
    """
    n, n_samples = 40, 120
    train = make_dataset(n, channel, deloc, loc, n_samples=n_samples, seed=0)
    test = make_dataset(n, channel, deloc, loc, n_samples=n_samples, seed=999)

    clf = SVMClassifier(random_state=0)
    clf.fit(train.X, train.y)
    assert clf.score(test.X, test.y) > 0.95


def test_weights_are_one_per_lattice_site() -> None:
    X, y = synthetic_dataset()
    clf = SVMClassifier(random_state=0)
    clf.fit(X, y)
    assert clf.weights.shape == (N_SITES,)
    assert np.isfinite(clf.intercept)
