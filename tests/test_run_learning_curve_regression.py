"""Tests for run_learning_curve's regression support.

run_large_area (server.py) builds "_reg"-suffixed model names for regression
requests (nn_reg, rf_reg, ...) but run_learning_curve had no task parameter
at all and unconditionally called make_classifier -- confirmed live (Louis
Driver): every pixel classifier failed with "ValueError: Unknown classifier:
nn_reg" (etc.) the moment training actually started. A sibling function,
run_kfold_cv, already had correct task-based dispatch (make_classifier vs
make_regressor) and its own "aggregate" event shape that the frontend's
regression display is built against -- these tests confirm run_learning_curve
now has the same dispatch, and yields that same event shape so the
already-built frontend path actually gets data.
"""

from __future__ import annotations

import numpy as np

from tessera_eval.evaluate import run_learning_curve

RNG = np.random.RandomState(0)
DIM = 8
N = 400


def _classification_data():
    vectors = RNG.rand(N, DIM).astype(np.float32)
    # 3 well-separated clusters so classifiers actually have something to learn.
    centers = RNG.rand(3, DIM) * 5
    labels = RNG.randint(0, 3, size=N)
    vectors += centers[labels]
    return vectors.astype(np.float32), labels.astype(np.int32)


def _regression_data():
    vectors = RNG.rand(N, DIM).astype(np.float32)
    # Learnable linear target + a bit of noise, so r2 isn't meaninglessly ~0.
    weights = RNG.rand(DIM)
    labels = (vectors @ weights + RNG.normal(scale=0.05, size=N)).astype(np.float32)
    return vectors, labels


def _run(vectors, labels, names, task, training_pcts=(50, 80)):
    return list(
        run_learning_curve(
            vectors,
            labels,
            names,
            list(training_pcts),
            repeats=2,
            task=task,
        )
    )


def test_regression_does_not_crash_and_produces_r2_rmse_mae():
    vectors, labels = _regression_data()
    events = _run(vectors, labels, ["nn_reg", "rf_reg"], task="regression")

    progress = [e for e in events if e["type"] == "progress"]
    assert len(progress) == 2  # one per training_pct
    for e in progress:
        for name in ("nn_reg", "rf_reg"):
            m = e["classifiers"][name]
            assert set(m) == {"mean_r2", "std_r2", "mean_rmse", "std_rmse", "mean_mae", "std_mae"}
            # Learnable synthetic data -- a real regressor should do better than "no skill".
            assert m["mean_r2"] > 0.3


def test_regression_yields_aggregate_not_confusion_matrices():
    vectors, labels = _regression_data()
    events = _run(vectors, labels, ["nn_reg"], task="regression")

    assert not any(e["type"] == "confusion_matrices" for e in events)
    aggregates = [e for e in events if e["type"] == "aggregate"]
    assert len(aggregates) == 1
    assert set(aggregates[0]["models"]) == {"nn_reg"}
    assert set(aggregates[0]["models"]["nn_reg"]) == {
        "mean_r2",
        "std_r2",
        "mean_rmse",
        "std_rmse",
        "mean_mae",
        "std_mae",
    }


def test_regression_aggregate_matches_largest_pct_progress_event():
    """The aggregate event is documented as "the largest percentage's
    metrics" -- confirm it actually is, not some other aggregation."""
    vectors, labels = _regression_data()
    events = _run(vectors, labels, ["rf_reg"], task="regression", training_pcts=(30, 50, 80))

    progress = [e for e in events if e["type"] == "progress"]
    aggregate = next(e for e in events if e["type"] == "aggregate")
    largest_progress = max(progress, key=lambda e: e["pct"])
    assert aggregate["models"]["rf_reg"] == largest_progress["classifiers"]["rf_reg"]


def test_classification_still_works_unaffected_by_task_default():
    """task defaults to "classification" -- existing callers that never pass
    it (or pass it explicitly) must be completely unaffected."""
    vectors, labels = _classification_data()
    events = _run(vectors, labels, ["nn", "rf"], task="classification")

    progress = [e for e in events if e["type"] == "progress"]
    assert len(progress) == 2
    for e in progress:
        for name in ("nn", "rf"):
            m = e["classifiers"][name]
            assert set(m) == {"mean_f1", "std_f1", "mean_f1w", "std_f1w"}
            assert m["mean_f1"] > 0.5  # well-separated clusters, should be easy

    assert not any(e["type"] == "aggregate" for e in events)
    cm_events = [e for e in events if e["type"] == "confusion_matrices"]
    assert len(cm_events) == 1


def test_a_fit_time_failure_degrades_to_zero_not_crashing_the_stream(monkeypatch):
    """A regressor that exists but fails during fit/predict shouldn't kill
    the generator -- matches the classifier path's existing
    try/except-around-fit-and-append-0.0 behavior. (An *unknown name*
    raises immediately for both classification and regression alike --
    make_classifier/make_regressor are called before the try block in both
    branches, a pre-existing characteristic of this function, not something
    this fix changed.)"""
    import sklearn.neighbors

    def _broken_fit(self, X, y):
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(sklearn.neighbors.KNeighborsRegressor, "fit", _broken_fit)

    vectors, labels = _regression_data()
    events = _run(vectors, labels, ["nn_reg"], task="regression")

    progress = [e for e in events if e["type"] == "progress"]
    assert len(progress) == 2
    for e in progress:
        assert e["classifiers"]["nn_reg"]["mean_r2"] == 0.0
