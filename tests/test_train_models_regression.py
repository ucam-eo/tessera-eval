"""End-to-end test for /api/evaluation/train-models (the "Download Models"
button) in regression mode.

Confirmed live (Louis Driver): after a successful kNN regression evaluation,
clicking "Download Models" failed with "Failed to train model 'nn_reg':
Unknown classifier: nn_reg". train_models() never dispatched to
make_regressor at all -- it unconditionally called make_classifier(name),
the same bug class already fixed in run_learning_curve/run_large_area
(test_run_large_area_regression.py) but missed here, since this endpoint
retrains a final model in a separate request after evaluation finishes, not
inline. Fixed by stashing `is_classification` in `_tile_cache` at the end of
run_large_area and dispatching on it in train_models().
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

sys.modules["tessera_eval.evaluate"]

EMBED_DIM = 128


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [object()]


class _FakeGeoTessera:
    def __init__(self, embeddings_dir=None):
        self.registry = _FakeRegistry()
        self._rng = np.random.RandomState(0)

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        if progress_callback:
            progress_callback(len(points), len(points), "done")
        n = len(points)
        base = np.array([[p[0] + p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.1, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _make_gdf():
    heights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 4
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


def _run_evaluation(client, **body):
    body.setdefault("field", "height")
    body.setdefault("task", "regression")
    body.setdefault("classifiers", ["nn"])  # mapped to nn_reg via _CLF_TO_REG
    body.setdefault("classifier_params", {"nn_reg": {"n_neighbors": 1}})
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 500)
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"run-large-area returned error event(s): {errors}"
    return events


def _train_models(client):
    resp = client.post("/api/evaluation/train-models", json={})
    assert resp.status_code == 200
    return [json.loads(line) for line in resp.text.strip().splitlines()]


def test_download_models_trains_a_pixel_regressor_without_unknown_classifier(client):
    _run_evaluation(client)
    events = _train_models(client)

    failures = [
        e
        for e in events
        if e.get("event") == "status" and "Failed to train" in e.get("message", "")
    ]
    assert not failures, f"train-models reported a training failure: {failures}"

    ready = [e["classifier"] for e in events if e.get("event") == "model_ready"]
    assert "nn_reg" in ready

    done = [e for e in events if e.get("event") == "done"]
    assert done and "nn_reg" in done[0]["models_available"]


def test_downloaded_regressor_model_file_loads_and_predicts(client):
    _run_evaluation(client)
    _train_models(client)

    resp = client.get("/api/evaluation/download-model/nn_reg")
    assert resp.status_code == 200

    import io

    import joblib

    bundle = joblib.load(io.BytesIO(resp.data))
    assert bundle["class_names"] == []  # regression: no class taxonomy
    preds = bundle["model"].predict(np.zeros((3, EMBED_DIM), dtype=np.float32))
    assert preds.shape == (3,)
