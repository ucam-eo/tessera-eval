"""Unit test for _extract_tile_patches's regression branch (server.py).

The riskiest untested integration point in the U-Net regression change: the
inline tile-rasterization/patch-building loop in _extract_tile_patches
(distinct from, and not going through, unet.py's standalone
extract_labelled_patches_regression -- this is server.py's own patch
pipeline, the one run_large_area actually uses). Confirms it produces
float32/NaN-masked patches carrying real field values when
is_classification=False, not int32 LabelEncoder ranks.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

import tessera_eval.server as srv

EMBED_DIM = 8
TILE_SIZE = 64


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [(year, 16.63, 48.32)]  # one tile, matching the transform below


class _FakeGeoTessera:
    def __init__(self, tile_emb, transform, crs="EPSG:4326"):
        self.registry = _FakeRegistry()
        self._tile_emb = tile_emb
        self._transform = transform
        self._crs = crs

    def fetch_embeddings(self, tiles):
        def gen():
            for _yr, _lon, _lat in tiles:
                yield (None, None, None, self._tile_emb, self._crs, self._transform)

        return gen()


@pytest.fixture
def fake_tile(monkeypatch):
    rng = np.random.RandomState(0)
    tile_emb = rng.rand(TILE_SIZE, TILE_SIZE, EMBED_DIM).astype(np.float32)
    # Simple degrees-per-pixel transform, no real UTM reprojection needed --
    # rasterize/gather_spatial_features only care about the pixel<->coord
    # mapping, not what CRS it nominally represents.
    transform = Affine(0.001, 0, 16.6, 0, -0.001, 48.35)
    monkeypatch.setattr(srv, "get_zarr", lambda: None)  # force the NPY fallback path
    return tile_emb, transform


def _make_gdf():
    # Covers most of the 64x64 tile (16.6-16.664 lon, 48.286-48.35 lat).
    return gpd.GeoDataFrame(
        {"height": [3.7]},
        geometry=[box(16.6, 48.29, 16.66, 48.35)],
        crs="EPSG:4326",
    )


def test_regression_patches_carry_real_values_not_label_encoder_ranks(fake_tile):
    tile_emb, transform = fake_tile
    gt = _FakeGeoTessera(tile_emb, transform)
    gdf = _make_gdf()

    unet_patches, spatial_3x3, spatial_5x5, point_vectors, _sl3, _sl5 = srv._extract_tile_patches(
        gt,
        gdf,
        "height",
        2024,
        le=None,
        n_classes=0,
        patch_size=32,
        max_patches=5,
        is_classification=False,
    )

    assert len(unet_patches) > 0
    emb_patch, target_patch = unet_patches[0]
    assert emb_patch.dtype == np.float32
    assert target_patch.dtype == np.float32
    # The real field value (3.7), not a LabelEncoder rank (which would be 0
    # for a single-class field) or an int-cast class ID.
    valid = ~np.isnan(target_patch)
    assert valid.any()
    assert np.allclose(target_patch[valid], 3.7)


def test_regression_patches_do_not_crash_with_spatial_features_requested(fake_tile):
    """The le=None safety fix (v1.3.3) -- confirms the whole call succeeds
    when spatial_mlp features are requested alongside regression (now a
    genuinely supported combination, not just a "shouldn't crash" case)."""
    tile_emb, transform = fake_tile
    gt = _FakeGeoTessera(tile_emb, transform)
    gdf = _make_gdf()

    result = srv._extract_tile_patches(
        gt,
        gdf,
        "height",
        2024,
        le=None,
        n_classes=0,
        patch_size=32,
        max_patches=5,
        needs_spatial_3x3=True,
        is_classification=False,
    )
    assert result is not None


def test_regression_spatial_labels_are_not_shifted_by_one(fake_tile):
    """The actual bug in the spatial-mlp-regression data pipeline (not just
    the ValueError crash it led to elsewhere): the 1-based-to-0-based '-1'
    shift applied to spatial labels is meaningful for classification (class
    IDs) but corrupts real continuous regression values -- every height (or
    whatever field) would silently come out 1.0 too low. _make_gdf's single
    polygon has height=3.7 everywhere it covers, so any shift is directly
    visible."""
    tile_emb, transform = fake_tile
    gt = _FakeGeoTessera(tile_emb, transform)
    gdf = _make_gdf()

    _unet_patches, spatial_3x3, spatial_5x5, _pv, spatial_labels_3x3, spatial_labels_5x5 = (
        srv._extract_tile_patches(
            gt,
            gdf,
            "height",
            2024,
            le=None,
            n_classes=0,
            patch_size=32,
            max_patches=5,
            needs_spatial_3x3=True,
            needs_spatial_5x5=True,
            is_classification=False,
        )
    )

    assert spatial_3x3 is not None and len(spatial_labels_3x3) > 0
    assert np.allclose(spatial_labels_3x3, 3.7), (
        f"expected the real field value 3.7, got {np.unique(spatial_labels_3x3)} "
        "-- looks shifted by the classification-only '-1'"
    )
    assert spatial_5x5 is not None and len(spatial_labels_5x5) > 0
    assert np.allclose(spatial_labels_5x5, 3.7)
