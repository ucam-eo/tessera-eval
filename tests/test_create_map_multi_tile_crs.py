"""create_map must predict on native embedding grids and still merge maps
that span more than one UTM zone.

Embeddings are produced on each tile's native UTM grid, and the geotessera
guidance is to classify on that grid and reproject only the result.  The
NPY fallback used to reproject the embeddings themselves to EPSG:4326
before predicting (resampling every 128-dimensional vector), and a
since-removed chunked path handed rasterio.merge blocks in whichever UTM
zone each chunk fell in, which raised "CRS mismatch with source" for map
areas crossing a zone boundary.  Maps must now predict on the grid the embeddings arrive
on, with only the per-block *prediction* rasters reprojected onto a common
CRS before merging.

The tiles here straddle the 18 degrees east meridian: zone 33 (EPSG:32633)
to the west, zone 34 (EPSG:32634) to the east.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

import tessera_eval.server as srv

EMBED_DIM = 8
TILE_SIZE = 16
RES_M = 10.0

WEST_TILE = (17.95, 48.25, "EPSG:32633")
EAST_TILE = (18.05, 48.25, "EPSG:32634")
MAP_BBOX = [48.2, 17.9, 48.3, 18.1]  # [south, west, north, east]


def _native_tile(tlon, tlat, crs):
    """A TILE_SIZE x TILE_SIZE embedding block on a native UTM grid whose
    top-left corner sits at (tlon, tlat)."""
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = to_utm.transform(tlon, tlat)
    transform = Affine(RES_M, 0, x0, 0, -RES_M, y0)
    rng = np.random.RandomState(int(tlon * 100))
    emb = rng.uniform(0, 50, size=(TILE_SIZE, TILE_SIZE, EMBED_DIM)).astype(np.float32)
    return emb, transform, crs


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [(year, WEST_TILE[0], WEST_TILE[1]), (year, EAST_TILE[0], EAST_TILE[1])]


class _FakeGeoTessera:
    """Serves each tile on its own native UTM grid, like the real tile
    store.  fetch_mosaic_for_region reprojects embeddings before analysis,
    so create_map must never call it."""

    def __init__(self, embeddings_dir=None, **kwargs):
        self.registry = _FakeRegistry()

    def fetch_embeddings(self, tiles, clip_bbox=None):
        def gen():
            for _yr, tlon, tlat in tiles:
                crs = WEST_TILE[2] if tlon < 18.0 else EAST_TILE[2]
                emb, transform, crs = _native_tile(tlon, tlat, crs)
                yield (None, None, None, emb, crs, transform)

        return gen()

    def fetch_mosaic_for_region(self, *args, **kwargs):
        raise AssertionError(
            "create_map must not use fetch_mosaic_for_region: it reprojects "
            "embeddings before prediction"
        )


def _client(monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr("tessera_eval.dataset.ZarrClient", _FakeGeoTessera)

    rng = np.random.RandomState(0)
    n = 100
    vectors = rng.uniform(0, 50, size=(n, EMBED_DIM)).astype(np.float32)
    labels = rng.randint(0, 2, size=n).astype(np.int32)
    monkeypatch.setattr(
        srv,
        "_tile_cache",
        {
            "key": ("habitat", 2024, 2024, "equal"),
            "vectors": vectors,
            "labels": labels,
            "class_names": ["grass", "water"],
            "_model_params": {},
        },
    )
    monkeypatch.setattr(srv, "_generated_maps", {})
    return srv.app.test_client()


@pytest.fixture
def npy_client(monkeypatch):
    return _client(monkeypatch)


def _run(client):
    body = {"classifier": "rf", "map_bboxes": [MAP_BBOX]}
    resp = client.post("/api/evaluation/create-map", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"create-map returned error event(s): {errors}"
    return events


def test_npy_path_predicts_on_native_grids_and_merges_across_zones(npy_client):
    events = _run(npy_client)
    ready = next(e for e in events if e["event"] == "map_ready")
    assert ready["width"] > 0 and ready["height"] > 0
    assert ready["crs"].startswith("EPSG:326"), (
        "map_ready must report the output CRS now that maps are written on "
        "native UTM grids rather than always EPSG:4326"
    )


def test_merge_prediction_rasters_handles_mixed_crs():
    a = np.full((4, 4), 1, dtype=np.uint8)
    b = np.full((4, 4), 2, dtype=np.uint8)
    to_33 = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    to_34 = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
    ax, ay = to_33.transform(17.95, 48.25)
    bx, by = to_34.transform(18.05, 48.25)

    merged, transform, crs = srv._merge_prediction_rasters(
        [
            (a, Affine(10, 0, ax, 0, -10, ay), "EPSG:32633"),
            (b, Affine(10, 0, bx, 0, -10, by), "EPSG:32634"),
        ],
        out_dtype="uint8",
        out_nodata=0,
    )

    assert set(np.unique(merged)) >= {1, 2}
    assert str(crs) in ("EPSG:32633", "EPSG:32634")


def test_registry_failure_yields_an_error_event_not_a_dead_stream(monkeypatch):
    class _BrokenRegistry:
        def load_blocks_for_region(self, bbox, year):
            raise RuntimeError("registry unavailable")

    class _BrokenGeoTessera(_FakeGeoTessera):
        def __init__(self, embeddings_dir=None, **kwargs):
            self.registry = _BrokenRegistry()

    client = _client(monkeypatch)
    monkeypatch.setattr("tessera_eval.dataset.ZarrClient", _BrokenGeoTessera)
    monkeypatch.setattr(srv, "_geotessera_instance", None)

    resp = client.post(
        "/api/evaluation/create-map",
        json={"classifier": "rf", "map_bboxes": [MAP_BBOX]},
    )
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert any(e.get("event") == "error" for e in events), (
        "a registry failure must surface as an error event, not truncate the stream"
    )
