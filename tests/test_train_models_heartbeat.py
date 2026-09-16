"""Tests for server._fit_with_wire_heartbeat and its use in train_models()
("Download Models").

Background: every model in train_models() (and create_map()) used to train
with a single raw, blocking .fit()/train_unet_on_patches() call and no
heartbeat at all -- unlike run_learning_curve/run_kfold_cv, which already
got evaluate.py's _fit_with_heartbeat fix for exactly this failure mode (a
long enough fit leaves the SSE stream completely silent, and whatever's
carrying the connection reads that as dead and drops it). Confirmed live
(Moustafa Eweda): a U-Net "Download Models" run that completed
successfully server-side surfaced in the browser as "Training error:
network error" -- the download endpoint went silent for the whole run.

_fit_with_wire_heartbeat wraps evaluate.py's _fit_with_heartbeat, translated
into the JSON-string wire format train_models()/create_map()'s own stream()
generators use (see its own docstring). Isolated unit tests here mirror
test_fit_heartbeat.py's own coverage of the underlying helper; the
integration test below is the one that matches Moustafa's actual report.
"""

from __future__ import annotations

import json
import sys
import time

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.evaluate  # noqa: F401 -- registers the submodule
import tessera_eval.server as srv

EMBED_DIM = 128


def _drain(gen):
    yielded = []
    try:
        while True:
            yielded.append(next(gen))
    except StopIteration as stop:
        return yielded, stop.value


def test_fast_fit_yields_no_heartbeats_and_returns_result(monkeypatch):
    monkeypatch.setattr(sys.modules["tessera_eval.evaluate"], "_HEARTBEAT_INTERVAL_S", 1.0)
    yielded, result = _drain(srv._fit_with_wire_heartbeat(lambda: 42))
    assert yielded == []
    assert result == 42


def test_slow_fit_yields_wire_format_heartbeats_and_still_returns_result(monkeypatch):
    monkeypatch.setattr(sys.modules["tessera_eval.evaluate"], "_HEARTBEAT_INTERVAL_S", 0.05)

    def slow():
        time.sleep(0.22)
        return "done"

    yielded, result = _drain(srv._fit_with_wire_heartbeat(slow))
    assert result == "done"
    assert len(yielded) >= 2
    assert all(line == json.dumps({"event": "heartbeat"}) + "\n" for line in yielded)


def test_fit_exception_is_reraised_after_wire_heartbeats(monkeypatch):
    monkeypatch.setattr(sys.modules["tessera_eval.evaluate"], "_HEARTBEAT_INTERVAL_S", 0.05)

    def slow_then_raise():
        time.sleep(0.11)
        raise ValueError("boom")

    gen = srv._fit_with_wire_heartbeat(slow_then_raise)
    with pytest.raises(ValueError, match="boom"):
        while True:
            next(gen)


# ── Integration: train_models() actually emits heartbeats for a slow fit ──


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
        base = np.array([[p[0] - p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.05, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _classification_gdf():
    classes = ["oak", "beech", "pine"] * 12
    geoms = [box(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05) for i in range(len(classes))]
    return gpd.GeoDataFrame({"species": classes}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _classification_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    return srv.app.test_client()


def test_download_models_emits_heartbeats_for_a_slow_generic_fit(client, monkeypatch):
    """The exact bug class Moustafa hit, reproduced on the generic (non-
    spatial, non-U-Net) branch: a fit slow enough to cross the heartbeat
    interval must actually produce heartbeat lines in train_models()'s own
    response, not leave the stream silent for the whole duration."""
    from sklearn.ensemble import RandomForestClassifier

    monkeypatch.setattr(sys.modules["tessera_eval.evaluate"], "_HEARTBEAT_INTERVAL_S", 0.05)
    real_fit = RandomForestClassifier.fit

    def slow_fit(self, X, y):
        time.sleep(0.16)
        return real_fit(self, X, y)

    monkeypatch.setattr(RandomForestClassifier, "fit", slow_fit)

    body = {
        "field": "species",
        "task": "classification",
        "sampling": "equal",
        "max_training_samples": 400,
        "classifiers": ["rf"],
    }
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert not [e for e in events if e.get("event") == "error"], events

    resp = client.post("/api/evaluation/train-models", json={})
    assert resp.status_code == 200
    train_events = [json.loads(line) for line in resp.text.strip().splitlines()]

    failures = [
        e
        for e in train_events
        if e.get("event") == "status" and "Failed to train" in e.get("message", "")
    ]
    assert not failures, f"train-models reported a training failure: {failures}"
    heartbeats = [e for e in train_events if e.get("event") == "heartbeat"]
    ready = [e["classifier"] for e in train_events if e.get("event") == "model_ready"]
    assert len(heartbeats) >= 1
    assert "rf" in ready
