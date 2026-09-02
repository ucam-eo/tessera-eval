"""run-large-area with eval_mode="kfold" -- k-fold cross-validation from the
Validation panel (previously CLI-only via `tessera-eval kfold`).

The endpoint reuses all the tile-fetch / sampling / caching machinery and
just swaps run_learning_curve for run_kfold_cv at the evaluation step:
no learning curve, no train/test bboxes, pixel models only. Events:
fold_result (per fold), aggregate (mean +/- std across folds),
confusion_matrices (classification).
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.evaluate  # noqa: F401 -- registers the submodule
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
        # Separable by the sign of (x - y) so a classifier has real signal.
        base = np.array([[p[0] - p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.05, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _regression_gdf():
    heights = [float(h) for h in range(1, 66, 2)] * 2  # 33 distinct values -> regression
    geoms = [box(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05) for i in range(len(heights))]
    return gpd.GeoDataFrame({"height": heights}, geometry=geoms, crs="EPSG:4326")


def _classification_gdf():
    classes = ["oak", "beech", "pine"] * 12  # 36 polygons, 12 per class
    geoms = [box(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05) for i in range(len(classes))]
    return gpd.GeoDataFrame({"species": classes}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    return srv.app.test_client()


def _run(client, **body):
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 400)
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"run-large-area returned error event(s): {errors}"
    return events


def test_kfold_regression_folds_and_aggregate(client, monkeypatch):
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _regression_gdf())
    events = _run(
        client,
        field="height",
        task="regression",
        eval_mode="kfold",
        kfold_k=3,
        classifiers=["nn", "rf"],
    )

    start = next(e for e in events if e["event"] == "start")
    assert start["mode"] == "kfold"
    assert start["k"] == 3

    folds = [e for e in events if e["event"] == "fold_result"]
    assert [f["fold"] for f in folds] == [1, 2, 3]
    assert all(f["total_folds"] == 3 for f in folds)
    assert all(set(f["models"]) == {"nn_reg", "rf_reg"} for f in folds)

    agg = next(e for e in events if e["event"] == "aggregate")
    for name in ("nn_reg", "rf_reg"):
        m = agg["models"][name]
        assert {"mean_r2", "std_r2", "mean_rmse", "std_rmse", "mean_mae", "std_mae"} <= set(m)

    # No learning curve in k-fold mode.
    assert not [e for e in events if e["event"] == "progress" and "classifiers" in e]


def test_kfold_classification_has_confusion_matrix(client, monkeypatch):
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _classification_gdf())
    events = _run(
        client,
        field="species",
        task="classification",
        eval_mode="kfold",
        kfold_k=3,
        classifiers=["rf"],
    )

    start = next(e for e in events if e["event"] == "start")
    assert start["mode"] == "kfold"

    folds = [e for e in events if e["event"] == "fold_result"]
    assert len(folds) == 3
    assert all("mean_f1" in f["models"]["rf"] for f in folds)

    agg = next(e for e in events if e["event"] == "aggregate")
    assert {"mean_f1", "std_f1", "mean_f1w", "std_f1w"} <= set(agg["models"]["rf"])

    cm = next(e for e in events if e["event"] == "confusion_matrices")
    assert "rf" in cm["confusion_matrices"]


def test_kfold_skips_spatial_and_unet_models(client, monkeypatch):
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _classification_gdf())
    events = _run(
        client,
        field="species",
        task="classification",
        eval_mode="kfold",
        kfold_k=2,
        classifiers=["rf", "spatial_mlp", "unet"],
    )
    msgs = " ".join(e.get("message", "") for e in events if e["event"] == "status")
    assert "spatial_mlp skipped" in msgs
    assert "unet skipped" in msgs
    folds = [e for e in events if e["event"] == "fold_result"]
    assert all(set(f["models"]) == {"rf"} for f in folds)


def test_kfold_k_is_clamped(client, monkeypatch):
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _regression_gdf())
    events = _run(
        client,
        field="height",
        task="regression",
        eval_mode="kfold",
        kfold_k=999,
        classifiers=["rf"],
    )
    start = next(e for e in events if e["event"] == "start")
    assert start["k"] == 20
    assert len([e for e in events if e["event"] == "fold_result"]) == 20


def test_default_mode_is_still_learning_curve(client, monkeypatch):
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _regression_gdf())
    events = _run(client, field="height", task="regression", classifiers=["rf"])
    start = next(e for e in events if e["event"] == "start")
    assert start["mode"] == "learning_curve"
    assert not [e for e in events if e["event"] == "fold_result"]
    assert [e for e in events if e["event"] == "progress" and "classifiers" in e]
