"""Tests for classifier/regressor failure reporting in run_learning_curve
and run_kfold_cv.

Background: a model that raises during .fit() (e.g. deep_mlp without torch
installed) was silently zero-filled -- logged server-side only, with no
signal on the wire at all. A flat 0.0 line sitting next to other models'
real scores is indistinguishable from "genuinely the worst model".
Confirmed live (Keshav): deep_mlp reported F1=0 for every fold in a local
run with no visible explanation in the UI, root cause a missing optional
dependency on that machine, not a real result.

Fixed with two changes, both covered here: (1) a {"type": "classifier_status",
"message": ...} event yielded the *first* time a given model fails (deduped
-- a hard dependency failure repeats identically at every pct/repeat/fold,
so only the first occurrence is reported, not one message per fold); (2)
a "failed": True flag added to that model's own metrics dict from then on,
so a consumer can tell "didn't run" apart from "scored zero" without
parsing log messages.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera_eval.evaluate import run_kfold_cv, run_learning_curve


def _classification_data(n_classes=3, per_class=40, dim=8, seed=0):
    rng = np.random.RandomState(seed)
    vectors, labels = [], []
    for cls in range(n_classes):
        center = rng.randn(dim) * 3
        vectors.append(center + rng.randn(per_class, dim) * 0.3)
        labels.extend([cls] * per_class)
    return np.vstack(vectors).astype(np.float32), np.array(labels)


@pytest.fixture
def always_failing_nn(monkeypatch):
    """Makes the 'nn' classifier raise on every .fit() call, without
    touching any other classifier -- mirrors test_fit_heartbeat.py's own
    technique of monkeypatching KNeighborsClassifier.fit directly."""
    from sklearn.neighbors import KNeighborsClassifier

    def broken_fit(self, X, y):
        raise RuntimeError("synthetic failure for testing")

    monkeypatch.setattr(KNeighborsClassifier, "fit", broken_fit)


def test_learning_curve_reports_classifier_status_once_not_per_repeat(
    always_failing_nn,
):
    vectors, labels = _classification_data()
    events = list(
        run_learning_curve(
            vectors, labels, ["nn"], training_pcts=[50, 80], repeats=3, task="classification"
        )
    )
    statuses = [e for e in events if e["type"] == "classifier_status"]
    failure_statuses = [e for e in statuses if "nn failed to train" in e["message"]]
    # 2 pcts x 3 repeats = 6 fit attempts, all failing identically -- only
    # the first should be reported, not all 6.
    assert len(failure_statuses) == 1
    assert "synthetic failure for testing" in failure_statuses[0]["message"]


def test_learning_curve_marks_failed_models_in_progress_events(always_failing_nn):
    vectors, labels = _classification_data()
    events = list(
        run_learning_curve(
            vectors, labels, ["nn"], training_pcts=[50, 80], repeats=1, task="classification"
        )
    )
    progress = [e for e in events if e["type"] == "progress"]
    assert len(progress) == 2
    for e in progress:
        assert e["classifiers"]["nn"]["failed"] is True
        # The numeric fallback is still present (0.0) so aggregation math
        # downstream doesn't need a special case.
        assert e["classifiers"]["nn"]["mean_f1"] == 0.0


def test_learning_curve_working_classifier_never_marked_failed():
    vectors, labels = _classification_data()
    events = list(
        run_learning_curve(
            vectors, labels, ["nn"], training_pcts=[80], repeats=1, task="classification"
        )
    )
    progress = [e for e in events if e["type"] == "progress"]
    assert "failed" not in progress[0]["classifiers"]["nn"]
    # classifier_status is also used for ordinary "training X..." progress
    # messages -- check specifically for a failure message, not the mere
    # presence of any classifier_status event.
    failure_statuses = [
        e
        for e in events
        if e["type"] == "classifier_status" and "failed to train" in e["message"]
    ]
    assert not failure_statuses


def test_learning_curve_one_failing_one_working_classifier_independent(
    always_failing_nn,
):
    vectors, labels = _classification_data()
    events = list(
        run_learning_curve(
            vectors, labels, ["nn", "rf"], training_pcts=[80], repeats=1, task="classification"
        )
    )
    progress = [e for e in events if e["type"] == "progress"]
    assert progress[0]["classifiers"]["nn"]["failed"] is True
    assert "failed" not in progress[0]["classifiers"]["rf"]
    assert progress[0]["classifiers"]["rf"]["mean_f1"] > 0.0


def test_kfold_reports_classifier_status_once_not_per_fold(always_failing_nn):
    vectors, labels = _classification_data()
    events = list(run_kfold_cv(vectors, labels, ["nn"], k=5, task="classification"))
    statuses = [e for e in events if e["type"] == "classifier_status"]
    failure_statuses = [e for e in statuses if "nn failed to train" in e["message"]]
    # 5 folds, all failing identically -- only the first should be reported.
    assert len(failure_statuses) == 1


def test_kfold_marks_failed_models_in_fold_result_and_aggregate(always_failing_nn):
    vectors, labels = _classification_data()
    events = list(run_kfold_cv(vectors, labels, ["nn"], k=3, task="classification"))
    fold_results = [e for e in events if e["type"] == "fold_result"]
    aggregate = [e for e in events if e["type"] == "aggregate"][0]
    assert len(fold_results) == 3
    for fr in fold_results:
        assert fr["models"]["nn"]["failed"] is True
    assert aggregate["models"]["nn"]["failed"] is True
    assert aggregate["models"]["nn"]["mean_f1"] == 0.0


def test_kfold_working_classifier_never_marked_failed():
    vectors, labels = _classification_data()
    events = list(run_kfold_cv(vectors, labels, ["nn"], k=3, task="classification"))
    aggregate = [e for e in events if e["type"] == "aggregate"][0]
    assert "failed" not in aggregate["models"]["nn"]
    assert not [e for e in events if e["type"] == "classifier_status"]


def test_kfold_regression_failure_also_reported(monkeypatch):
    from sklearn.neighbors import KNeighborsRegressor

    def broken_fit(self, X, y):
        raise RuntimeError("synthetic regressor failure")

    monkeypatch.setattr(KNeighborsRegressor, "fit", broken_fit)

    rng = np.random.RandomState(0)
    vectors = rng.randn(120, 6).astype(np.float32)
    labels = rng.randn(120).astype(np.float32)
    events = list(run_kfold_cv(vectors, labels, ["nn_reg"], k=3, task="regression"))
    aggregate = [e for e in events if e["type"] == "aggregate"][0]
    assert aggregate["models"]["nn_reg"]["failed"] is True
    statuses = [
        e
        for e in events
        if e["type"] == "classifier_status" and "nn_reg failed to train" in e["message"]
    ]
    assert len(statuses) == 1


def test_deep_mlp_missing_torch_reports_a_real_failure_message(monkeypatch):
    """The exact bug report this was built for: deep_mlp requested without
    torch installed must surface a clear message, not a silent zero.
    Monkeypatches deep_mlp._HAS_TORCH rather than relying on torch actually
    being absent from this environment -- deterministic either way, and
    exercises the real _require_torch() RuntimeError path make_classifier's
    deep_mlp branch (classify.py) hits."""
    import tessera_eval.deep_mlp as deep_mlp_mod

    monkeypatch.setattr(deep_mlp_mod, "_HAS_TORCH", False)

    vectors, labels = _classification_data()
    events = list(
        run_learning_curve(
            vectors, labels, ["deep_mlp"], training_pcts=[80], repeats=1, task="classification"
        )
    )
    statuses = [e for e in events if e["type"] == "classifier_status"]
    assert any("deep_mlp failed to train" in e["message"] for e in statuses)
    assert any("PyTorch" in e["message"] for e in statuses)
    progress = [e for e in events if e["type"] == "progress"]
    assert progress[0]["classifiers"]["deep_mlp"]["failed"] is True
