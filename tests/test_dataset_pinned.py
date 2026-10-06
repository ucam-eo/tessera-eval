"""tessera-eval must ask for its embeddings dataset explicitly.

geotessera 0.11.0 changed what a bare GeoTessera() / GeoTesseraZarr() reads
(v1.1-cambridge NPY / v1.1-dclimate Zarr instead of v1.0). With only
geotessera>=0.10.1 required, fresh installs silently switched every
evaluation to different embeddings -- Austria 2022 went from 70 tiles to 0.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.server as srv
from tessera_eval.dataset import EMBEDDINGS_DATASET_VERSION, zarr_store_url


def test_pinned_to_v1_0():
    assert EMBEDDINGS_DATASET_VERSION == "v1.0"
    assert zarr_store_url().rstrip("/").endswith("/zarr/v1")


class _Reg:
    def load_blocks_for_region(self, bbox, year):
        return [object()]


class _RecordingGT:
    kwargs_seen = []

    def __init__(self, **kwargs):
        type(self).kwargs_seen.append(kwargs)
        self.registry = _Reg()

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        return np.random.RandomState(0).normal(size=(len(points), 128)).astype(np.float32)


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    gdf = gpd.GeoDataFrame(
        {"habitat": ["a", "b"] * 4},
        geometry=[box(i * 0.1, 0, i * 0.1 + 0.05, 0.05) for i in range(8)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: gdf)
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _RecordingGT)
    _RecordingGT.kwargs_seen = []
    return srv.app.test_client()


def test_run_large_area_requests_the_pinned_dataset(client):
    resp = client.post(
        "/api/evaluation/run-large-area",
        json={
            "field": "habitat",
            "task": "classification",
            "classifiers": ["nn"],
            "sampling": "equal",
            "max_training_samples": 200,
        },
    )
    events = [json.loads(line) for line in resp.text.strip().splitlines() if line.strip()]
    assert not [e for e in events if e.get("event") == "error"], events
    assert _RecordingGT.kwargs_seen
    assert all(k.get("dataset_version") == "v1.0" for k in _RecordingGT.kwargs_seen)
