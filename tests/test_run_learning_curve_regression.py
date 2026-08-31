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

import sys

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


def _spatial_regression_data(window, n=300):
    """Synthetic (spatial_vectors, spatial_labels) pair shaped like real
    neighbourhood-augmented features: (n, window*window*dim), target
    derived from the *centre* cell (a real spatial_mlp input would predict
    from the whole neighbourhood, but a centre-derived target is enough to
    confirm a real fit is happening, not a degenerate/constant one)."""
    dim = DIM
    vectors = RNG.rand(n, window * window * dim).astype(np.float32)
    weights = RNG.rand(dim)
    center = (window * window // 2) * dim
    center_features = vectors[:, center : center + dim]
    labels = (center_features @ weights + RNG.normal(scale=0.05, size=n)).astype(np.float32)
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
            assert {"mean_r2", "std_r2", "mean_rmse", "std_rmse", "mean_mae", "std_mae"} <= set(m)
            # Learnable synthetic data -- a real regressor should do better than "no skill".
            assert m["mean_r2"] > 0.3


def test_regression_yields_aggregate_not_confusion_matrices():
    vectors, labels = _regression_data()
    events = _run(vectors, labels, ["nn_reg"], task="regression")

    assert not any(e["type"] == "confusion_matrices" for e in events)
    aggregates = [e for e in events if e["type"] == "aggregate"]
    assert len(aggregates) == 1
    assert set(aggregates[0]["models"]) == {"nn_reg"}
    # "scatter" (predicted-vs-actual points) is present too -- see the
    # dedicated scatter tests below for its shape/content.
    assert {
        "mean_r2",
        "std_r2",
        "mean_rmse",
        "std_rmse",
        "mean_mae",
        "std_mae",
    } <= set(aggregates[0]["models"]["nn_reg"])


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
    # A regressor that never successfully fit has nothing to scatter --
    # confirm the key is just absent, not present-but-empty.
    aggregate = next(e for e in events if e["type"] == "aggregate")
    assert "scatter" not in aggregate["models"]["nn_reg"]


def test_spatial_mlp_regression_trains_and_produces_a_real_r2():
    """spatial_mlp/spatial_mlp_5x5 now support regression: make_regressor
    recognizes both names directly (no "_reg" suffix -- see its docstring),
    and _extract_tile_patches's spatial-label shift is now conditional on
    is_classification. Previously this crashed the whole evaluation stream
    outright (ValueError: Unknown regressor: spatial_mlp) -- confirmed live,
    Louis Driver."""
    vectors, labels = _regression_data()
    sp_vectors, sp_labels = _spatial_regression_data(window=3)

    events = list(
        run_learning_curve(
            vectors,
            labels,
            ["spatial_mlp"],
            [50, 80],
            repeats=2,
            spatial_vectors=sp_vectors,
            spatial_labels=sp_labels,
            task="regression",
        )
    )

    progress = [e for e in events if e["type"] == "progress"]
    assert len(progress) == 2
    for e in progress:
        m = e["classifiers"]["spatial_mlp"]
        assert {"mean_r2", "std_r2", "mean_rmse", "std_rmse", "mean_mae", "std_mae"} <= set(m)
        # Learnable synthetic data -- a real fit should beat "no skill".
        assert m["mean_r2"] > 0.3

    aggregate = next(e for e in events if e["type"] == "aggregate")
    assert set(aggregate["models"]) == {"spatial_mlp"}


def test_spatial_mlp_5x5_regression_trains_and_produces_a_real_r2():
    vectors, labels = _regression_data()
    sp_vectors, sp_labels = _spatial_regression_data(window=5)

    events = list(
        run_learning_curve(
            vectors,
            labels,
            ["spatial_mlp_5x5"],
            [50, 80],
            repeats=2,
            spatial_vectors_5x5=sp_vectors,
            spatial_labels=sp_labels,
            task="regression",
        )
    )

    progress = [e for e in events if e["type"] == "progress"]
    assert len(progress) == 2
    for e in progress:
        assert e["classifiers"]["spatial_mlp_5x5"]["mean_r2"] > 0.3


def test_aggregate_includes_scatter_points_matching_the_test_set():
    """The core feature: predicted-vs-actual points for a scatter plot,
    carried on the largest percentage's aggregate event."""
    vectors, labels = _regression_data()
    events = _run(vectors, labels, ["rf_reg"], task="regression", training_pcts=(50, 80))

    aggregate = next(e for e in events if e["type"] == "aggregate")
    scatter = aggregate["models"]["rf_reg"]["scatter"]
    assert set(scatter) == {"y_true", "y_pred"}
    assert len(scatter["y_true"]) == len(scatter["y_pred"])
    assert len(scatter["y_true"]) > 0
    # Real regression targets/predictions, not something degenerate.
    assert len(set(scatter["y_true"])) > 1
    # And a real fit -- predictions should correlate with actuals on this
    # learnable synthetic data, not just be noise.
    corr = np.corrcoef(scatter["y_true"], scatter["y_pred"])[0, 1]
    assert corr > 0.5


def test_scatter_is_only_on_the_largest_percentage_not_every_progress_event():
    vectors, labels = _regression_data()
    events = _run(vectors, labels, ["rf_reg"], task="regression", training_pcts=(30, 50, 80))

    progress = [e for e in events if e["type"] == "progress"]
    non_largest = [e for e in progress if e["pct"] != max(e["pct"] for e in progress)]
    for e in non_largest:
        assert "scatter" not in e["classifiers"]["rf_reg"]


def test_scatter_points_are_capped_for_a_large_test_set(monkeypatch):
    """A big evaluation shouldn't turn into an unbounded SSE payload just
    because someone wants to eyeball fit quality."""
    # `import tessera_eval.evaluate as ev` is *also* an attribute lookup on
    # the `tessera_eval` package under the hood (import a.b as c ==
    # import a.b; c = a.b), and tessera_eval/__init__.py re-exports a
    # function named `evaluate` that shadows the submodule on that
    # attribute -- sys.modules is the only reliable way to get the real
    # submodule to patch.
    ev = sys.modules["tessera_eval.evaluate"]

    monkeypatch.setattr(ev, "_MAX_SCATTER_POINTS", 10)

    rng = np.random.RandomState(1)
    n = 500
    vectors = rng.rand(n, DIM).astype(np.float32)
    weights = rng.rand(DIM)
    labels = (vectors @ weights).astype(np.float32)

    events = _run(vectors, labels, ["rf_reg"], task="regression", training_pcts=(80,))
    aggregate = next(e for e in events if e["type"] == "aggregate")
    scatter = aggregate["models"]["rf_reg"]["scatter"]
    assert len(scatter["y_true"]) == 10


# --- Out-of-range prediction flag (bug 8, Louis Driver) -------------------
#
# Regression pct_results carry oor_frac (fraction of largest-pct test
# predictions beyond the training targets' full span) and train_range
# [min, max], for the UI's "Outside range" column. Nothing about the
# scores or predictions is altered by this.


def test_regression_reports_oor_frac_and_train_range_on_largest_pct():
    vectors, labels = _regression_data()
    events = _run(vectors, labels, ["nn_reg", "rf_reg"], task="regression", training_pcts=(50, 80))

    lo = round(float(np.min(labels)), 4)
    hi = round(float(np.max(labels)), 4)

    progress = sorted((e for e in events if e["type"] == "progress"), key=lambda e: e["pct"])
    smallest, largest = progress[0], progress[-1]

    for name in ("nn_reg", "rf_reg"):
        # Present only at the largest percentage (the "final" result).
        assert "oor_frac" not in smallest["classifiers"][name]
        m = largest["classifiers"][name]
        assert m["train_range"] == [lo, hi]
        # RF / kNN average stored targets -- they cannot predict outside the
        # training span, so this is a deterministic 0.0, not just "small".
        assert m["oor_frac"] == 0.0

    aggregate = next(e for e in events if e["type"] == "aggregate")
    assert aggregate["models"]["rf_reg"]["oor_frac"] == 0.0
    assert aggregate["models"]["rf_reg"]["train_range"] == [lo, hi]


def test_oor_frac_catches_a_model_that_extrapolates(monkeypatch):
    """A regressor whose predictions all land above the training span must
    report oor_frac == 1.0 -- and its R2/RMSE/MAE are still computed from
    the raw (unclamped) predictions."""
    import sklearn.neighbors

    real_predict = sklearn.neighbors.KNeighborsRegressor.predict

    def _shifted_predict(self, X):
        return real_predict(self, X) + 1e6  # far above any real target

    monkeypatch.setattr(sklearn.neighbors.KNeighborsRegressor, "predict", _shifted_predict)

    vectors, labels = _regression_data()
    events = _run(vectors, labels, ["nn_reg"], task="regression", training_pcts=(80,))

    aggregate = next(e for e in events if e["type"] == "aggregate")
    m = aggregate["models"]["nn_reg"]
    assert m["oor_frac"] == 1.0
    # Scores are NOT clamped -- a wildly shifted prediction tanks R2.
    assert m["mean_r2"] < 0.0


def test_classification_has_no_oor_fields():
    vectors, labels = _classification_data()
    events = _run(vectors, labels, ["nn", "rf"], task="classification")
    for e in (x for x in events if x["type"] == "progress"):
        for name in ("nn", "rf"):
            assert "oor_frac" not in e["classifiers"][name]
            assert "train_range" not in e["classifiers"][name]
