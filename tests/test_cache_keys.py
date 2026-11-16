"""The sample caches must not outlive the data they were drawn from.

Found 2026-11-16 while fixing the spatial k-fold cache bug: the in-memory
cache ignored which shapefile was loaded, the disk cache ignored label
values (it hashed only row count, columns and bounds), and neither recorded
the embeddings dataset version.
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


STATE = {}


def _gdf(names):
    geoms = [box(i * 0.1, 0, i * 0.1 + 0.05, 0.05) for i in range(len(names))]
    return gpd.GeoDataFrame({"habitat": names}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    STATE["gdf"] = _gdf(["a", "b"] * 6)
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: STATE["gdf"])
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("tessera_eval.dataset.ZarrClient", _FakeGeoTessera)
    _FakeGeoTessera.calls = 0
    return srv.app.test_client()


def _run(client):
    resp = client.post(
        "/api/evaluation/run-large-area",
        json={
            "field": "habitat",
            "task": "classification",
            "classifiers": ["nn"],
            "classifier_params": {"nn": {"n_neighbors": 1}},
            "sampling": "equal",
            "max_training_samples": 400,
            "seed": 42,
        },
    )
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert not [e for e in events if e.get("event") == "error"], events
    start = next(e for e in events if e.get("event") == "start")
    return sorted(c["name"] for c in start["classes"])


def _restart_server():
    srv._tile_cache.clear()
    srv._tile_cache.update({"key": None, "vectors": None})


def test_new_shapefile_without_restart_is_resampled(client):
    assert _run(client) == ["a", "b"]
    STATE["gdf"] = _gdf(["a", "b", "c"] * 6)
    assert _run(client) == ["a", "b", "c"], "reused the previous shapefile's sample"
    assert _FakeGeoTessera.calls == 2


def test_relabelled_shapefile_is_resampled_after_restart(client):
    _run(client)
    _restart_server()
    STATE["gdf"] = _gdf(["x", "y"] * 6)  # same polygons, new labels
    assert _run(client) == ["x", "y"], "reused old labels from the disk cache"


def test_unchanged_shapefile_still_uses_the_caches(client):
    _run(client)
    _run(client)
    _restart_server()
    STATE["gdf"] = _gdf(["a", "b"] * 6)  # a fresh but identical upload
    _run(client)
    assert _FakeGeoTessera.calls == 1


def test_cache_from_another_dataset_version_is_not_reused(client, monkeypatch):
    monkeypatch.setattr("tessera_eval.dataset.EMBEDDINGS_DATASET_VERSION", "v1.0")
    _run(client)
    _restart_server()
    monkeypatch.setattr("tessera_eval.dataset.EMBEDDINGS_DATASET_VERSION", "v1.1")
    _run(client)
    assert _FakeGeoTessera.calls == 2
