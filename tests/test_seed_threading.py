"""A single seed keys every random draw in an evaluation run.

Threaded from the CLI --seed / the web request's `seed` through
run_learning_curve / run_kfold_cv (resampling + fold splits) and into
make_classifier / make_regressor (estimator random_state). Same seed =>
identical results; different seed => different.
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.evaluate  # noqa: F401
import tessera_eval.server as srv
from tessera_eval.classify import make_classifier, make_regressor

sys.modules["tessera_eval.evaluate"]

EMBED_DIM = 128


def test_make_classifier_and_regressor_take_the_seed():
    assert make_classifier("rf", seed=7).random_state == 7
    assert make_classifier("xgboost", seed=7).random_state == 7 if _has_xgb() else True
    assert make_classifier("mlp", seed=7).random_state == 7
    assert make_regressor("rf_reg", seed=7).random_state == 7
    assert make_regressor("mlp_reg", seed=7).random_state == 7
    # default unchanged
    assert make_classifier("rf").random_state == 42


def _has_xgb():
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


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
        # Deterministic per-point noise so repeated runs see the same
        # embeddings -- only the run's own seed should move the result.
        rng = np.random.RandomState(abs(hash(tuple(round(x, 4) for x in points[0]))) % (2**31))
        noise = rng.normal(scale=0.1, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _gdf():
    classes = ["oak", "beech", "pine", "elm"] * 12
    geoms = [box(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05) for i in range(len(classes))]
    return gpd.GeoDataFrame({"species": classes}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    return srv.app.test_client()


def _run(client, **body):
    body.setdefault("field", "species")
    body.setdefault("task", "classification")
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 400)
    body.setdefault("classifiers", ["rf"])
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert not [e for e in events if e.get("event") == "error"], events
    return events


def _final_f1(events):
    progs = [e for e in events if e.get("event") == "progress" and "classifiers" in e]
    assert progs
    return progs[-1]["classifiers"]["rf"]["mean_f1"]


def test_start_event_carries_the_seed(client):
    events = _run(client, seed=123)
    start = next(e for e in events if e["event"] == "start")
    assert start["seed"] == 123


def test_same_seed_reproduces_the_learning_curve(client):
    a = _final_f1(_run(client, seed=7))
    # fresh cache so sampling re-runs too, not just the models
    srv._tile_cache.clear()
    srv._tile_cache.update({"key": None, "vectors": None})
    b = _final_f1(_run(client, seed=7))
    assert a == b


def test_different_seed_changes_the_learning_curve(client):
    a = _final_f1(_run(client, seed=7))
    srv._tile_cache.clear()
    srv._tile_cache.update({"key": None, "vectors": None})
    b = _final_f1(_run(client, seed=99))
    assert a != b


def test_kfold_seed_is_honoured(client):
    def folds(seed):
        srv._tile_cache.clear()
        srv._tile_cache.update({"key": None, "vectors": None})
        ev = _run(client, seed=seed, eval_mode="kfold", kfold_k=3)
        agg = next(e for e in ev if e["event"] == "aggregate")
        return agg["models"]["rf"]["mean_f1"]

    assert folds(7) == folds(7)
    assert folds(7) != folds(31)
