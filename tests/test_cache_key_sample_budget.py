"""Max pixel samples and seed must be part of the sample cache key (memory and disk).

Confirmed live (austria.zip, 2026-10-02): a first run with Max pixel samples
~2,000 (left over from a small manual-labels shapefile) generated 1,989
points, and every later run of the same field/year/sampling reused them --
even after the budget was set back to 200,000 -- because neither the
in-memory _tile_cache key nor the on-disk result-cache filename included the
budget. Under a spatial split that left ~150 training pixels ("0.0K" in the
Validation panel) and F1 near zero.
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
    names = ["a", "b"] * 6
    geoms = [box(i * 0.1, 0, i * 0.1 + 0.05, 0.05) for i in range(len(names))]
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


def _total_pixels(client, budget, seed=42):
    resp = client.post(
        "/api/evaluation/run-large-area",
        json={
            "field": "habitat",
            "task": "classification",
            "classifiers": ["nn"],
            "classifier_params": {"nn": {"n_neighbors": 1}},
            "sampling": "equal",
            "max_training_samples": budget,
            "seed": seed,
        },
    )
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert not [e for e in events if e.get("event") == "error"], events
    start = next(e for e in events if e.get("event") == "start")
    return sum(c["pixels"] for c in start["classes"])


def test_changing_budget_resamples_instead_of_reusing_the_cache(client):
    small = _total_pixels(client, 40)
    assert _FakeGeoTessera.calls == 1

    large = _total_pixels(client, 400)
    assert _FakeGeoTessera.calls == 2, "a different budget must not hit the cached sample"
    assert large > small


def test_same_budget_still_hits_the_in_memory_cache(client):
    _total_pixels(client, 400)
    _total_pixels(client, 400)
    assert _FakeGeoTessera.calls == 1


def test_disk_cache_entries_are_separate_per_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    gdf = _make_gdf()
    vectors = np.zeros((10, EMBED_DIM), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int32)
    srv._save_cached_result("habitat", 2024, gdf, vectors, labels, ["a"], {}, "equal", 2000)

    assert srv._load_cached_result("habitat", 2024, gdf, "equal", 2000) is not None
    assert srv._load_cached_result("habitat", 2024, gdf, "equal", 200_000) is None


def test_changing_seed_resamples_instead_of_reusing_the_cache(client):
    """seed drives the point sampler, so a new seed must draw a new sample --
    otherwise "try a different seed" silently reused the cached points."""
    _total_pixels(client, 400, seed=1)
    _total_pixels(client, 400, seed=2)
    assert _FakeGeoTessera.calls == 2, "a different seed must not hit the cached sample"
    _total_pixels(client, 400, seed=2)
    assert _FakeGeoTessera.calls == 2, "the same seed should still hit the cache"


def test_disk_cache_entries_are_separate_per_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    gdf = _make_gdf()
    vectors = np.zeros((10, EMBED_DIM), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int32)
    srv._save_cached_result("habitat", 2024, gdf, vectors, labels, ["a"], {}, "equal", 2000, seed=1)

    assert srv._load_cached_result("habitat", 2024, gdf, "equal", 2000, seed=1) is not None
    assert srv._load_cached_result("habitat", 2024, gdf, "equal", 2000, seed=2) is None
