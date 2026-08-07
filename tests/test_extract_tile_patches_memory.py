"""_extract_tile_patches must not leak whole source tiles into unet_patches.

Regression for a real OOM: emb_patch = tile_emb[r0:r1, c0:c1] is a numpy
*view* (basic slicing), so appending it to unet_patches (which lives for the
whole evaluation run) kept the entire multi-hundred-MB source tile alive for
every tile that contributed even one all-finite patch -- confirmed via dmesg
("Out of memory: Killed process ... (tee-compute)") on a real evaluation run.
"""

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from affine import Affine
from shapely.geometry import box as shp_box
from sklearn.preprocessing import LabelEncoder

from tessera_eval.server import _extract_tile_patches


class _FakeRegistry:
    def __init__(self, tiles):
        self._tiles = tiles

    def load_blocks_for_region(self, bbox, year):  # noqa: ARG002
        return list(self._tiles)


class _FakeGeoTessera:
    """Minimal gt stand-in: one synthetic, all-finite tile via the NPY path."""

    def __init__(self, tile_emb, crs, transform, tile_key):
        self.registry = _FakeRegistry([tile_key])
        self._tile_emb = tile_emb
        self._crs = crs
        self._transform = transform

    def fetch_embeddings(self, tiles_to_fetch):  # noqa: ARG002
        # (year, lon, lat, embedding, crs, transform) per tile -- matches
        # server.py's `_, _, _, tile_emb, crs, transform = next(tiles_gen)`.
        for _ in tiles_to_fetch:
            yield (None, None, None, self._tile_emb, self._crs, self._transform)


def test_extract_tile_patches_copies_out_of_the_source_tile(monkeypatch):
    """Every patch in unet_patches must be a standalone array (base is None),
    not a view still anchored to the full per-tile embedding array -- even for
    an all-finite patch (the no-NaN branch is exactly what used to leak)."""
    monkeypatch.setattr("tessera_eval.server.get_zarr", lambda: None)  # force NPY path

    tile_size_px, dim = 300, 8
    tlon, tlat, year = 10.05, 45.05, 2024
    # All-finite tile: the no-NaN branch, which never copied before this fix.
    tile_emb = np.random.default_rng(0).random((tile_size_px, tile_size_px, dim)).astype(
        np.float32
    )
    crs = "EPSG:4326"
    res = 0.1 / tile_size_px  # tile spans 0.1 deg, matching real tile granularity
    transform = Affine(res, 0, tlon - 0.05, 0, -res, tlat + 0.05)

    gt = _FakeGeoTessera(tile_emb, crs, transform, (year, tlon, tlat))

    poly = shp_box(tlon - 0.05, tlat - 0.05, tlon + 0.05, tlat + 0.05)
    gdf = gpd.GeoDataFrame({"cls": ["a"]}, geometry=[poly], crs="EPSG:4326")
    le = LabelEncoder().fit(["a"])

    unet_patches, _s3, _s5, _pv, _sl3, _sl5 = _extract_tile_patches(
        gt,
        gdf,
        "cls",
        year,
        le,
        n_classes=1,
        patch_size=32,
        max_patches=3,
        needs_spatial_3x3=False,
        needs_spatial_5x5=False,
        sample_points_lonlat=None,
        logger=None,
        progress_cb=None,
        cancel_flag=None,
    )

    assert len(unet_patches) > 0  # sanity: the fixture actually produced patches
    for emb_patch, label_patch in unet_patches:
        # .base is None for an array that owns its own memory; a numpy view
        # (e.g. tile_emb[r0:r1, c0:c1] without .copy()) has .base pointing at
        # (an ancestor of) the ~300x300x8 source tile instead.
        assert emb_patch.base is None, (
            "emb_patch is a view into the source tile, not an independent copy -- "
            "the source tile's full memory stays alive as long as unet_patches does"
        )
        assert label_patch.base is None
