"""The single embeddings reader: GeoTesseraZarr on v1.1-dclimate.

Every read (pixel sampling, tile/patch extraction, test-year and test-file
sampling, Create Map) goes through tessera_eval.dataset.ZarrClient. The
dataset is named explicitly so a geotessera upgrade can't silently change
it (0.11 made a bare GeoTessera() mean the sparse v1.1-cambridge run).
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.dataset as ds
import tessera_eval.server as srv


def test_dataset_is_v1_1_dclimate():
    assert (ds.EMBEDDINGS_DATASET_VERSION, ds.EMBEDDINGS_DATASET_VARIANT) == ("v1.1", "dclimate")


# --------------------------------------------------------------- tile grid


def test_tiles_cover_the_bbox_on_the_x_x5_grid():
    tiles = ds.tiles_for_bbox((16.3, 48.0, 16.5, 48.2), 2022)
    assert sorted(t[1] for t in tiles) == [16.35] * 2 + [16.45] * 2
    assert sorted({t[2] for t in tiles}) == [48.05, 48.15]
    assert all(t[0] == 2022 for t in tiles)


def test_a_point_sized_bbox_still_gets_its_tile():
    assert ds.tiles_for_bbox((16.32, 48.07, 16.32, 48.07), 2022) == [(2022, 16.35, 48.05)]


def test_no_tile_straddles_a_utm_zone_edge():
    for west, east in ((17.5, 18.5), (-3.2, -2.8), (5.9, 6.1)):
        for _y, lon, _lat in ds.tiles_for_bbox((west, 48.0, east, 48.1), 2022):
            lo, hi = lon - ds.TILE_DEG / 2, lon + ds.TILE_DEG / 2
            assert int((lo + 1e-9 + 180) // 6) == int((hi - 1e-9 + 180) // 6), lon


# ------------------------------------------------------- ZarrClient, faked


class _FakeStore:
    years = [2022, 2025]

    def __init__(self, fail_reads=0, **kwargs):
        self.kwargs = kwargs
        self.fail_reads = fail_reads
        self.sample_calls = []

    def sample_points(self, coords, year, progress=True):
        self.sample_calls.append(len(coords))
        out = np.ones((len(coords), 128), np.float32)
        out[0] = np.nan  # e.g. a point over water
        return out

    def read_region(self, bbox, year):
        if self.fail_reads:
            self.fail_reads -= 1
            raise OSError("HTTP 502")
        return np.ones((4, 4, 128), np.int8), "transform", "EPSG:32633"


@pytest.fixture
def client_with(monkeypatch):
    monkeypatch.setattr(ds, "zarr_store_url", lambda: "store://v1.1-dclimate")
    monkeypatch.setattr(ds, "_RETRY_BACKOFF_S", 0)

    def make(**store_kwargs):
        store = _FakeStore(**store_kwargs)
        monkeypatch.setattr("geotessera.store.GeoTesseraZarr", lambda **kw: store)
        return ds.ZarrClient(cache_dir="/tmp/x"), store

    return make


def test_points_are_sampled_in_batches_with_progress(client_with, monkeypatch):
    monkeypatch.setattr(ds, "_POINT_BATCH", 10)
    client, store = client_with()
    seen = []
    vecs = client.sample_embeddings_at_points(
        [(16.5, 48.2)] * 25, year=2022, progress_callback=lambda c, t, s: seen.append((c, t))
    )
    assert vecs.shape == (25, 128) and vecs.dtype == np.float32
    assert store.sample_calls == [10, 10, 5]
    assert seen == [(10, 25), (20, 25), (25, 25)]
    assert np.isnan(vecs[0]).all()  # no-data rows pass through as NaN


def test_a_transient_read_error_is_retried(client_with):
    client, _ = client_with(fail_reads=1)
    ((_y, _lon, _lat, emb, crs, _t),) = list(client.fetch_embeddings([(2022, 16.55, 48.25)]))
    assert emb is not None and emb.dtype == np.float32 and crs == "EPSG:32633"


def test_a_tile_that_keeps_failing_yields_none_in_order(client_with):
    client, _ = client_with(fail_reads=ds._READ_ATTEMPTS)
    out = list(client.fetch_embeddings([(2022, 16.55, 48.25), (2022, 16.65, 48.25)]))
    assert [o[1] for o in out] == [16.55, 16.65]  # still one result per tile, in order
    assert out[0][3] is None and out[1][3] is not None


def test_registry_lists_nothing_for_a_year_the_store_lacks(client_with):
    client, _ = client_with()
    assert client.registry.load_blocks_for_region((16.5, 48.2, 16.6, 48.3), 2019) == []
    assert client.registry.load_blocks_for_region((16.5, 48.2, 16.6, 48.3), 2022)


# ------------------------------------------------- run-large-area wiring


class _RecordingClient:
    created = []

    def __init__(self, **kwargs):
        type(self).created.append(kwargs)
        self.registry = self

    def load_blocks_for_region(self, bbox, year):
        return [object()]

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        return np.random.RandomState(0).normal(size=(len(points), 128)).astype(np.float32)


def test_run_large_area_reads_through_the_zarr_client(tmp_path, monkeypatch):
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
    monkeypatch.setattr(ds, "ZarrClient", _RecordingClient)
    _RecordingClient.created = []

    resp = srv.app.test_client().post(
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
    assert len(_RecordingClient.created) == 1
    assert str(_RecordingClient.created[0]["cache_dir"]).endswith("zarr")
