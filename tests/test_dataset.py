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
    """Tiles on a plain lon/lat grid (EPSG:4326, 0.001 deg pixels) whose
    pixel values encode their own (row, col), so a test can check that each
    point picked up the pixel it falls in."""

    years = [2022, 2025]

    def __init__(self, fail_reads=0, fail_tiles=(), **kwargs):
        self.kwargs = kwargs
        self.fail_reads = fail_reads
        self.fail_tiles = set(fail_tiles)
        self.reads = []

    def _region(self, bbox):
        from affine import Affine

        west, south, east, north = bbox
        key = (round((west + east) / 2, 2), round((south + north) / 2, 2))
        self.reads.append(key)
        if self.fail_reads:
            self.fail_reads -= 1
            raise OSError("HTTP 502")
        if key in self.fail_tiles:
            raise OSError("HTTP 502")
        h = max(1, round((north - south) / 0.001))
        w = max(1, round((east - west) / 0.001))
        rows, cols = np.mgrid[0:h, 0:w]
        q = np.zeros((h, w, 128), np.int8)
        q[..., 0], q[..., 1] = rows, cols
        return q, Affine(0.001, 0, west, 0, -0.001, north)

    def read_region_quantized(self, bbox, year):
        q, transform = self._region(bbox)
        return q, np.ones(q.shape[:2], np.float32), transform, "EPSG:4326"

    def read_region(self, bbox, year):
        q, transform = self._region(bbox)
        return q.astype(np.float32), transform, "EPSG:4326"


class _FakeRegistry:
    """Embedded blocks: only the western half of the test area has data in
    2022; nothing in 2025."""

    def tiles(self, bbox=None, year=None):
        import pandas as pd

        rows = [
            dict(
                year=2022,
                embedded=True,
                bbox_west=16.30,
                bbox_south=48.20,
                bbox_east=16.40,
                bbox_north=48.30,
            ),
            dict(
                year=2022,
                embedded=False,
                bbox_west=16.40,
                bbox_south=48.20,
                bbox_east=16.50,
                bbox_north=48.30,
            ),
        ]
        df = pd.DataFrame(rows)
        return df if year is None else df[df.year == year]


@pytest.fixture
def client_with(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "zarr_store_url", lambda: "store://v1.1-dclimate")
    monkeypatch.setattr(ds, "_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(ds, "_open_tile_registry", lambda cache_dir: None)

    def make(registry=None, **store_kwargs):
        store = _FakeStore(**store_kwargs)
        monkeypatch.setattr("geotessera.store.GeoTesseraZarr", lambda **kw: store)
        if registry is not None:
            monkeypatch.setattr(ds, "_open_tile_registry", lambda cache_dir: registry)
        return ds.ZarrClient(cache_dir=tmp_path), store

    return make


def test_points_are_read_one_tile_at_a_time(client_with):
    client, store = client_with()
    # Three points in tile (16.35, 48.25), one in tile (16.45, 48.25).
    pts = [(16.3005, 48.2995), (16.3105, 48.2895), (16.3995, 48.2005), (16.4505, 48.2995)]
    seen = []
    vecs = client.sample_embeddings_at_points(
        pts, year=2022, progress_callback=lambda c, t, s: seen.append((c, t))
    )
    assert vecs.shape == (4, 128) and vecs.dtype == np.float32
    assert sorted(store.reads) == [(16.35, 48.25), (16.45, 48.25)]  # one read per tile
    assert seen == [(1, 2), (2, 2)]
    # (row, col) of each point within its tile: 0.001-degree pixels from the NW corner.
    assert vecs[:, :2].tolist() == [[0, 0], [10, 10], [99, 99], [0, 50]]


def test_points_in_an_unreadable_tile_come_back_nan(client_with):
    client, _ = client_with(fail_tiles={(16.45, 48.25)})
    vecs = client.sample_embeddings_at_points([(16.3005, 48.2995), (16.4505, 48.2995)], year=2022)
    assert np.isfinite(vecs[0]).all()
    assert np.isnan(vecs[1]).all()


def test_tiles_are_cached_on_disk(client_with):
    client, store = client_with()
    first = client.read_tile(2022, 16.35, 48.25)
    second = client.read_tile(2022, 16.35, 48.25)
    assert store.reads == [(16.35, 48.25)]  # second read came from disk
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1] and tuple(first[2]) == tuple(second[2])
    # A new client on the same cache dir (a later run) doesn't fetch it again.
    again, store2 = client_with()
    again.read_tile(2022, 16.35, 48.25)
    assert store2.reads == []


def test_cache_is_bounded(client_with):
    client, _ = client_with()
    client._tile_cache_max = 1  # bytes: only the newest tile may stay
    client.read_tile(2022, 16.35, 48.25)
    client.read_tile(2022, 16.45, 48.25)
    cached = list(client._tile_dir.rglob("*.npz"))
    assert len(cached) <= 1


def test_registry_skips_tiles_without_embeddings(client_with):
    client, _ = client_with(registry=_FakeRegistry())
    tiles = client.registry.load_blocks_for_region((16.30, 48.20, 16.50, 48.30), 2022)
    assert [(t[1], t[2]) for t in tiles] == [(16.35, 48.25)]  # 16.45 isn't embedded
    assert client.registry.load_blocks_for_region((16.30, 48.20, 16.50, 48.30), 2025) == []


def test_year_coverage_needs_no_embedding_reads(client_with):
    client, store = client_with(registry=_FakeRegistry())
    cov = ds.year_coverage(client, (16.30, 48.20, 16.50, 48.30), years=[2022, 2025])
    assert cov == {"2022": 1, "2025": 0}
    assert store.reads == []


def test_map_reads_only_the_map_area(client_with):
    client, store = client_with()
    clip = (16.32, 48.22, 16.34, 48.24)  # a small corner of tile (16.35, 48.25)
    ((_y, _lon, _lat, emb, crs, _t),) = list(
        client.fetch_embeddings([(2022, 16.35, 48.25)], clip_bbox=clip)
    )
    assert emb.shape[:2] == (20, 20)  # not the whole 100 x 100 tile


def test_a_transient_read_error_is_retried(client_with):
    client, _ = client_with(fail_reads=1)
    ((_y, _lon, _lat, emb, crs, _t),) = list(client.fetch_embeddings([(2022, 16.55, 48.25)]))
    assert emb is not None and emb.dtype == np.float32 and crs == "EPSG:4326"


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


def test_a_map_covering_most_of_a_tile_reads_it_through_the_cache(client_with):
    client, store = client_with()
    clip = (16.30, 48.20, 16.39, 48.29)  # 81% of tile (16.35, 48.25)
    list(client.fetch_embeddings([(2022, 16.35, 48.25)], clip_bbox=clip))
    list(client.fetch_embeddings([(2022, 16.35, 48.25)], clip_bbox=clip))  # re-run the map
    assert store.reads == [(16.35, 48.25)]  # whole tile once, then from disk
