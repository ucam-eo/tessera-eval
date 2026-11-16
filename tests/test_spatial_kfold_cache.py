"""Spatial k-fold must get the point coordinates of the sample it is run on.

Reported 2026-11-16 (Moustafa Eweda): after a run with a different seed,
k-fold with Spatial k-fold on said "point coordinates don't match the
training data". The disk-cache hit replaced the in-memory vectors but left
the previous sample's coordinates in place, and the disk cache did not store
coordinates at all, so spatial k-fold could not work from it either.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.server as srv

EMBED_DIM = 128


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [object()]


class _FakeGeoTessera:
    calls = 0

    def __init__(self, embeddings_dir=None, **kwargs):
        self.registry = _FakeRegistry()

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        type(self).calls += 1
        if progress_callback:
            progress_callback(len(points), len(points), "done")
        rng = np.random.RandomState(0)
        return rng.normal(size=(len(points), EMBED_DIM)).astype(np.float32)


def _make_gdf():
    # A 6 x 4 grid of fields, two classes, so a spatial block grid has room.
    names, geoms = [], []
    for i in range(6):
        for j in range(4):
            names.append("a" if (i + j) % 2 else "b")
            geoms.append(box(i * 0.1, j * 0.1, i * 0.1 + 0.05, j * 0.1 + 0.05))
    return gpd.GeoDataFrame({"habitat": names}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _make_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("tessera_eval.dataset.ZarrClient", _FakeGeoTessera)
    _FakeGeoTessera.calls = 0
    return srv.app.test_client()


def _run(client, seed=42, spatial_kfold=False, max_samples=600):
    resp = client.post(
        "/api/evaluation/run-large-area",
        json={
            "field": "habitat",
            "task": "classification",
            "classifiers": ["nn"],
            "classifier_params": {"nn": {"n_neighbors": 1}},
            "sampling": "equal",
            "max_training_samples": max_samples,
            "seed": seed,
            "eval_mode": "kfold",
            "kfold_k": 2,
            "group_by_field": True,
            "spatial_kfold": spatial_kfold,
        },
    )
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert not [e for e in events if e.get("event") == "error"], events
    start = next(e for e in events if e.get("event") == "start")
    ignored = [
        e["message"]
        for e in events
        if e.get("event") == "status" and "Spatial k-fold ignored" in e.get("message", "")
    ]
    return start, ignored


def _restart_server():
    srv._tile_cache.clear()
    srv._tile_cache.update({"key": None, "vectors": None})


def test_spatial_kfold_after_a_run_with_another_seed(client):
    """Moustafa's sequence: seed 42, then seed 43, then seed 42 with spatial k-fold."""
    _run(client, seed=42)
    _run(client, seed=43, max_samples=500)  # fresh sample, different size
    _run(client, seed=42)  # disk-cache hit
    start, ignored = _run(client, seed=42, spatial_kfold=True)
    assert not ignored, ignored
    assert start.get("spatial_kfold") is True


def test_spatial_kfold_on_a_disk_cache_hit_needs_no_re_extraction(client):
    _run(client, seed=42)
    assert _FakeGeoTessera.calls == 1
    _restart_server()
    start, ignored = _run(client, seed=42, spatial_kfold=True)
    assert not ignored, ignored
    assert start.get("spatial_kfold") is True
    assert _FakeGeoTessera.calls == 1, "coordinates should come from the disk cache"


def test_old_cache_without_coordinates_is_re_extracted_for_spatial_kfold(client, tmp_path):
    _run(client, seed=42)
    [path] = list(tmp_path.glob("result_*.npz"))
    d = dict(np.load(path, allow_pickle=True))
    d.pop("points", None)
    np.savez_compressed(path, **d)

    _restart_server()
    start, ignored = _run(client, seed=42, spatial_kfold=True)
    assert not ignored, ignored
    assert start.get("spatial_kfold") is True
    assert _FakeGeoTessera.calls == 2


def test_disk_cache_round_trips_points(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    gdf = _make_gdf()
    vectors = np.zeros((3, EMBED_DIM), dtype=np.float32)
    labels = np.array([0, 1, 1], dtype=np.int32)
    points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    srv._save_cached_result("habitat", 2024, gdf, vectors, labels, ["a", "b"], {}, points=points)
    loaded = srv._load_cached_result("habitat", 2024, gdf)
    np.testing.assert_array_equal(loaded[5], points)
