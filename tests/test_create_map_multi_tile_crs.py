"""create_map's NPY fallback must merge multi-tile/multi-CRS chunks correctly.

Confirmed live (Louis Driver), traceback:
    rasterio.errors.RasterioError: CRS mismatch with source: ...
raised from rasterio.merge.merge() inside create_map()'s chunk-merging step,
for a large map area. Root cause: the previous NPY fallback called
gt.registry.load_blocks_for_region() + gt.fetch_embeddings() and took only
the *first* tile via next(tile_gen) -- for a chunk_bbox overlapping multiple
embedding tiles, this silently dropped the rest, and different chunks ended
up carrying whatever native UTM CRS their (arbitrarily-chosen) first tile
happened to be in. rasterio.merge.merge() requires every source dataset to
share one CRS. This was previously masked for large areas because zarr's
read_region already reprojected everything to a shared EPSG:4326 grid before
this code path was hit at all -- it only surfaced once zarr was disabled
(2026-08-19, the UTM-boundary-bug fix) and the NPY fallback became the only
path.

Fixed by calling gt.fetch_mosaic_for_region(chunk_bbox, target_crs=
"EPSG:4326") instead -- the library's own purpose-built method for exactly
this ("dense raster operations like classification"), which merges every
overlapping tile *and* reprojects to a common CRS internally.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from affine import Affine

import tessera_eval.server as srv

EMBED_DIM = 8


class _FakeRegistry:
    """Deliberately NOT exercised by create_map()'s NPY fallback anymore --
    present only so nothing else on the fake object breaks if some other
    code path still reaches for it."""

    def load_blocks_for_region(self, bbox, year):
        raise AssertionError(
            "create_map()'s NPY fallback must call fetch_mosaic_for_region, "
            "not registry.load_blocks_for_region directly"
        )


class _FakeGeoTessera:
    """Each chunk_bbox gets embeddings in a *different* native UTM-like CRS
    from fetch_mosaic_for_region's caller's perspective -- but since
    fetch_mosaic_for_region is responsible for reprojecting to target_crs
    before returning, this fake always honours target_crs in its return
    value, exactly like the real method's contract. If create_map ever
    reverts to using per-tile CRS without reprojecting, this fake alone
    won't catch it (it's not simulating the merge-internals bug) -- the
    real assertion is in test_map_with_multiple_chunks_merges_without_crs_
    mismatch, which drives >1 chunk through the *real* rasterio merge.
    """

    def __init__(self, embeddings_dir=None):
        self.registry = _FakeRegistry()
        self._call_count = 0

    def fetch_mosaic_for_region(
        self, bbox, year=2024, target_crs="EPSG:4326", auto_download=True, progress_callback=None
    ):
        self._call_count += 1
        rng = np.random.RandomState(self._call_count)  # different data per chunk
        emb = rng.uniform(0, 50, size=(16, 16, EMBED_DIM)).astype(np.float32)
        west, south, _east, _north = bbox
        transform = Affine(0.001, 0, west, 0, -0.001, south + 0.016)
        return emb, transform, target_crs


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "get_zarr", lambda: None)  # force the NPY fallback path
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)

    rng = np.random.RandomState(0)
    n = 100
    vectors = rng.rand(n, EMBED_DIM).astype(np.float32)
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


def test_map_with_multiple_chunks_merges_without_crs_mismatch(client):
    """A map area larger than one 0.1deg chunk -- >1 chunk gets merged by
    the real rasterio.merge.merge(), the exact call that raised
    'CRS mismatch with source' live."""
    body = {
        "classifier": "rf",
        # 0.25 x 0.15 deg -- multiple 0.1deg chunks in both directions.
        "map_bboxes": [[48.0, 16.0, 48.15, 16.25]],
    }
    resp = client.post("/api/evaluation/create-map", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]

    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"create-map returned error event(s): {errors}"

    chunk_events = [e for e in events if e.get("event") == "map_progress"]
    assert len(chunk_events) > 1, "test bbox must actually span multiple chunks"

    ready = next(e for e in events if e["event"] == "map_ready")
    assert ready["width"] > 0 and ready["height"] > 0
