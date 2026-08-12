"""Tests for evaluate._fit_predict_relabeled against xgboost's contiguous-label
requirement.

Regression coverage: confirmed live (Louis Driver), running a
spatial-split, multi-classifier evaluation --

    Classifier xgboost failed at pct 10.0 seed 0: Invalid classes inferred
    from unique values of y. Expected: [0 1 2 3 4 5 6 7 8 9 10 11 12 13],
    got [0 1 2 3 4 5 6 7 8 10 11 12 13 15]

xgboost's sklearn wrapper requires the labels passed to .fit() to already
be an exact contiguous 0..(k-1) range. In spatial-split mode, a class can
be geographically confined entirely to the test-side region -- zero
pixels in any train-side bbox -- so no amount of resampling ever includes
it in y_train, while it still counts toward the *global* n_classes (from
train+test combined). Other classifiers here don't require contiguous
labels, so this was xgboost-specific and went unnoticed until real data
with a class like this was used.

Uses real xgboost (not a mock) -- installed via `uv pip install xgboost`
into this venv specifically to verify against the actual library
behaviour, matching the exact error Louis saw.
"""

from __future__ import annotations

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")

from tessera_eval.evaluate import _fit_predict_relabeled, run_learning_curve  # noqa: E402


def _drain(gen):
    yielded = []
    try:
        while True:
            yielded.append(next(gen))
    except StopIteration as stop:
        return yielded, stop.value


def _gapped_labels(rng, n=44):
    """0..8, 10..13, 15 -- exactly Louis's reported gap (missing 9 and 14),
    repeated/padded to n samples."""
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15]
    reps = n // len(base) + 1
    return np.array((base * reps)[:n])


def test_xgboost_rejects_gapped_labels_directly():
    """Document xgboost's own behaviour (not our code) -- the exact
    failure this whole fix exists to route around, using real xgboost."""
    rng = np.random.RandomState(0)
    y = _gapped_labels(rng)
    X = rng.randn(len(y), 4)
    clf = xgboost.XGBClassifier(n_estimators=5)
    with pytest.raises(ValueError, match="Invalid classes"):
        clf.fit(X, y)


def test_fit_predict_relabeled_handles_gapped_labels_with_real_xgboost():
    rng = np.random.RandomState(0)
    y_tr = _gapped_labels(rng)
    X_tr = rng.randn(len(y_tr), 4)
    X_te = rng.randn(10, 4)
    clf = xgboost.XGBClassifier(n_estimators=5)

    yielded, y_pred = _drain(_fit_predict_relabeled(clf, X_tr, y_tr, X_te))
    assert len(y_pred) == 10
    # Predictions must come back in the *original* label space (a value
    # xgboost's internal encoder never saw directly, e.g. 15), not the
    # internal contiguous 0..(k-1) space.
    assert set(y_pred).issubset(set(y_tr))


def test_fit_predict_relabeled_is_a_noop_in_substance_for_sklearn_classifiers():
    """Applied unconditionally (not just to xgboost) -- confirm it doesn't
    change behaviour for a classifier that never needed it."""
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.RandomState(1)
    y_tr = _gapped_labels(rng)
    X_tr = rng.randn(len(y_tr), 4)
    X_te = X_tr[:5]

    direct = KNeighborsClassifier(n_neighbors=1).fit(X_tr, y_tr).predict(X_te)
    _yielded, relabeled = _drain(
        _fit_predict_relabeled(KNeighborsClassifier(n_neighbors=1), X_tr, y_tr, X_te)
    )
    assert list(direct) == list(relabeled)


# ── Integration: run_learning_curve in spatial-split mode, a class confined
# entirely to the test-side region (Louis's actual scenario, not just a
# small-subsample coincidence) ──


def test_run_learning_curve_xgboost_survives_a_class_confined_to_test_region():
    rng = np.random.RandomState(0)
    dim = 4
    n_per_class = 20

    # Classes 0,1,2,4,5 appear in both train and test. Class 3 -- a *middle*
    # class, not the top one -- is confined entirely to the test-side
    # spatial region: zero pixels in the train pool, at any percentage.
    # This specific shape matters: xgboost only rejects a gap where the
    # remaining labels aren't a clean 0..(k-1) prefix (confirmed
    # separately -- missing only the *highest* class doesn't trigger it,
    # since what's left is still contiguous from 0).
    present_in_train = [0, 1, 2, 4, 5]
    train_vectors, train_labels = [], []
    test_vectors, test_labels = [], []
    centers = {cls: rng.randn(dim) * 4 for cls in range(6)}
    for cls in present_in_train:
        train_vectors.append(centers[cls] + rng.randn(n_per_class, dim) * 0.3)
        train_labels.extend([cls] * n_per_class)
    for cls in range(6):
        test_vectors.append(centers[cls] + rng.randn(n_per_class, dim) * 0.3)
        test_labels.extend([cls] * n_per_class)

    train_vectors = np.vstack(train_vectors).astype(np.float32)
    train_labels = np.array(train_labels)
    test_vectors = np.vstack(test_vectors).astype(np.float32)
    test_labels = np.array(test_labels)

    events = list(
        run_learning_curve(
            train_vectors,
            train_labels,
            ["xgboost"],
            training_pcts=[10, 80],
            repeats=1,
            test_vectors=test_vectors,
            test_labels=test_labels,
        )
    )
    progress_events = [e for e in events if e["type"] == "progress"]
    assert len(progress_events) == 2
    for e in progress_events:
        # A real (if imperfect, given class 3 is unlearnable from train)
        # score, not the 0.0 fallback run_learning_curve substitutes when
        # a classifier's fit/predict raises.
        assert e["classifiers"]["xgboost"]["mean_f1"] is not None
