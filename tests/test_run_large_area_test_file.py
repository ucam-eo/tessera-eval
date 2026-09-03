"""run-large-area with a separate held-out test shapefile (Louis Driver).

Upload one shapefile as the training ground truth and a *different* one as
the test set; run_large_area samples the test file at test_year and uses it
as the fixed test set, ignoring any drawn train/test rectangles. Lets a
repeat survey be evaluated for between-year transfer with real surface
change in the data.
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
        base = np.array([[p[0] - p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.05, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _grid(vals, col, x0=0.0):
    geoms = [
        box(x0 + i * 0.1, i * 0.1, x0 + i * 0.1 + 0.05, i * 0.1 + 0.05) for i in range(len(vals))
    ]
    return gpd.GeoDataFrame({col: vals}, geometry=geoms, crs="EPSG:4326")


def _train_clf():
    return _grid(["oak", "beech", "pine"] * 12, "species")


def _test_clf(classes=("oak", "beech", "pine")):
    return _grid(list(classes) * 8, "species", x0=10.0)


def _train_reg():
    return _grid([float(h) for h in range(1, 66, 2)] * 2, "height")


def _test_reg():
    return _grid([float(h) for h in range(2, 60, 2)] * 2, "height", x0=10.0)


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    return srv.app.test_client()


def _run(client, train_gdf, test_gdf, monkeypatch, **body):
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: train_gdf)
    monkeypatch.setattr(srv, "_get_test_merged_gdf", lambda: test_gdf)
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 400)
    body.setdefault("classifiers", ["rf"])
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    return [json.loads(line) for line in resp.text.strip().splitlines()]


def test_separate_test_file_is_used_as_the_fixed_test_set(client, monkeypatch):
    events = _run(
        client,
        _train_clf(),
        _test_clf(),
        monkeypatch,
        field="species",
        task="classification",
        train_year=2024,
        test_year=2020,
    )
    assert not [e for e in events if e.get("event") == "error"], events
    start = next(e for e in events if e["event"] == "start")
    assert start.get("file_split") is True
    assert "year_split" not in start and "spatial_split" not in start
    assert start["train_count"] > 0 and start["test_count"] > 0
    # the run actually completes
    assert any(e.get("event") == "progress" and "classifiers" in e for e in events)
    assert any(e.get("event") == "done" for e in events)


def test_test_file_overrides_drawn_rectangles(client, monkeypatch):
    events = _run(
        client,
        _train_clf(),
        _test_clf(),
        monkeypatch,
        field="species",
        task="classification",
        train_bboxes=[[0, 0, 5, 5]],
        test_bboxes=[[6, 6, 9, 9]],
    )
    assert not [e for e in events if e.get("event") == "error"], events
    start = next(e for e in events if e["event"] == "start")
    assert start.get("file_split") is True
    assert "spatial_split" not in start


def test_unknown_class_in_test_file_is_an_error(client, monkeypatch):
    events = _run(
        client,
        _train_clf(),
        _test_clf(classes=("oak", "beech", "willow")),
        monkeypatch,
        field="species",
        task="classification",
    )
    errs = [e for e in events if e.get("event") == "error"]
    assert errs and "not present in the training data" in errs[0]["message"]
    assert "willow" in errs[0]["message"]


def test_missing_test_field_is_a_400(client, monkeypatch):
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _train_clf())
    monkeypatch.setattr(srv, "_get_test_merged_gdf", lambda: _test_clf())
    resp = client.post(
        "/api/evaluation/run-large-area",
        json={"field": "species", "test_field": "nope", "classifiers": ["rf"]},
    )
    assert resp.status_code == 400
    assert "not found in the test shapefile" in resp.get_json()["error"]


def test_regression_test_file(client, monkeypatch):
    events = _run(
        client,
        _train_reg(),
        _test_reg(),
        monkeypatch,
        field="height",
        task="regression",
        regressors=["rf_reg"],
        test_year=2019,
    )
    assert not [e for e in events if e.get("event") == "error"], events
    start = next(e for e in events if e["event"] == "start")
    assert start.get("file_split") is True
    assert start["test_count"] > 0


def test_kfold_ignores_the_test_file_with_a_note(client, monkeypatch):
    events = _run(
        client,
        _train_clf(),
        _test_clf(),
        monkeypatch,
        field="species",
        task="classification",
        eval_mode="kfold",
        kfold_k=3,
    )
    assert not [e for e in events if e.get("event") == "error"], events
    start = next(e for e in events if e["event"] == "start")
    assert "file_split" not in start
    msgs = " ".join(e.get("message", "") for e in events if e["event"] == "status")
    assert "Test file ignored" in msgs
    assert [e for e in events if e["event"] == "fold_result"]
