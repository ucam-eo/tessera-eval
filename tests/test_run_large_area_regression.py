"""End-to-end test for /api/evaluation/run-large-area in regression mode.

Confirmed live (Louis Driver): every pixel regressor (nn_reg/rf_reg/
xgboost_reg/mlp_reg) crashed the SSE stream with "ValueError: Unknown
classifier: nn_reg" the moment training started -- run_learning_curve had no
task-aware dispatch to make_regressor at all. Fixed in run_learning_curve
(evaluate.py) and threaded through here (server.py's lc_kwargs). This test
drives the real endpoint (not just run_learning_curve in isolation, which
tests/test_run_learning_curve_regression.py already covers) with GeoTessera
faked out, the same pattern as test_train_test_year_split.py.
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.evaluate  # noqa: F401 -- registers tessera_eval.evaluate in sys.modules
import tessera_eval.server as srv

ev = sys.modules["tessera_eval.evaluate"]

EMBED_DIM = 128


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [object()]


class _FakeGeoTessera:
    """Unlike test_train_test_year_split.py's constant-per-year fake, this
    one returns embeddings correlated with the regression target (each
    point's vector is a noisy multiple of its target height) -- constant
    features would make r2 degenerate and wouldn't exercise the actual
    fit/predict path meaningfully."""

    def __init__(self, embeddings_dir=None):
        self.registry = _FakeRegistry()
        self._rng = np.random.RandomState(0)

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        if progress_callback:
            progress_callback(len(points), len(points), "done")
        n = len(points)
        # Target height is recovered from `points` by the caller via the
        # gdf join, so just return points-correlated noise here: enough
        # for a regressor to find *some* signal without needing the fake
        # to know the true labels.
        base = np.array([[p[0] + p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.1, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _make_gdf():
    """A continuous 'height' field over several small polygons -- enough
    points/classes-as-bins for run_large_area's sampling to produce a usable
    training set without needing a huge fixture."""
    heights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 4  # more rows -> more sample points
    geoms = [box(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05) for i in range(len(heights))]
    return gpd.GeoDataFrame({"height": heights}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _make_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    return srv.app.test_client()


def _run(client, **body):
    body.setdefault("field", "height")
    body.setdefault("task", "regression")
    body.setdefault("classifiers", ["nn", "rf"])  # mapped to nn_reg/rf_reg via _CLF_TO_REG
    body.setdefault("classifier_params", {"nn_reg": {"n_neighbors": 1}})
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 500)
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"run-large-area returned error event(s): {errors}"
    return events


def _training_progress_events(events):
    """'progress' events come in two shapes: tile-fetch progress
    ({"event": "progress", "pct": int, "message": ...}, no "classifiers"
    key -- see evaluation.js's ev.event === 'progress' && !ev.classifiers
    branch) and per-training-percentage classifier progress (has
    "classifiers"). Only the latter is relevant here."""
    return [e for e in events if e["event"] == "progress" and "classifiers" in e]


def test_pixel_regressors_run_without_the_unknown_classifier_crash(client):
    events = _run(client)

    progress = _training_progress_events(events)
    assert progress, "expected at least one training-progress event"
    for e in progress:
        for name in ("nn_reg", "rf_reg"):
            m = e["classifiers"][name]
            assert set(m) == {"mean_r2", "std_r2", "mean_rmse", "std_rmse", "mean_mae", "std_mae"}


def test_aggregate_event_reaches_the_client(client):
    """server.py used to have no case for run_learning_curve's "aggregate"
    event type at all -- silently dropped even if evaluate.py produced one.
    Confirm it's actually forwarded over the wire."""
    events = _run(client)

    aggregates = [e for e in events if e["event"] == "aggregate"]
    assert len(aggregates) == 1
    assert set(aggregates[0]["models"]) == {"nn_reg", "rf_reg"}

    cm_events = [e for e in events if e["event"] == "confusion_matrices"]
    assert not cm_events, "regression mode shouldn't produce a confusion matrix"


def test_regression_labels_are_real_field_values_not_label_encoder_ranks(client, monkeypatch):
    """The actual bug: labels used to always be LabelEncoder(field).transform(...)
    -- e.g. heights [1.5, 3.2, 3.3, 10.7, 100.5, 200.9] became ranks
    [0, 1, 2, 3, 4, 5], and *those* were what regressors actually trained
    against. Capture what run_learning_curve is called with and confirm
    every label is one of the real height values, not a small rank int."""
    captured = {}

    def _fake_run_learning_curve(vectors, labels, active_models, training_pcts, **lc_kwargs):
        captured["labels"] = labels
        return iter(())

    monkeypatch.setattr(ev, "run_learning_curve", _fake_run_learning_curve)

    real_heights = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    _run(client)

    labels = captured["labels"]
    assert len(labels) > 0
    assert set(np.unique(labels).tolist()) <= real_heights, (
        f"expected only real height values, got {sorted(set(labels.tolist()))}"
    )
    # The old (broken) behavior specifically produced small contiguous rank
    # ints -- [0..5] here, coincidentally overlapping 1..5 of the real
    # values, which is exactly why this needs an explicit set comparison
    # rather than "looks like small integers": 6.0 (a real height, but not
    # a valid LabelEncoder rank for 6 classes, which would top out at 5)
    # must be present, since it's the most-sampled row in the fixture.
    assert 6.0 in labels
