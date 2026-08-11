"""Unit tests for load_embeddings_for_raster's bbox clipping.

Uses a synthetic, hand-built raster and a fake GeoTessera-like object
exposing only the two methods load_embeddings_for_raster actually calls —
no network access and no real Tessera data, per the contributing guide.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tessera_eval.data import load_embeddings_for_raster


@pytest.fixture
def synthetic_raster(tmp_path):
    """A 10x10 raster, 1 degree per pixel, EPSG:4326, where every pixel's
    value is row*10 + col — unique per pixel, so we can check exactly
    which pixels made it through by their value."""
    transform = from_origin(west=0, north=10, xsize=1, ysize=1)
    values = np.fromfunction(lambda r, c: r * 10 + c, (10, 10), dtype=int).astype(np.int32)

    path = tmp_path / "synthetic.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="int32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(values, 1)

    return path, transform


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [("dummy_tile",)]  # only its length is used by the caller


class _FakeGeoTessera:
    """Stands in for a real GeoTessera instance, yielding one fake tile
    whose grid exactly matches the synthetic raster's full extent."""

    def __init__(self, transform, dim=4):
        self.registry = _FakeRegistry()
        self._transform = transform
        self._dim = dim

    def fetch_embeddings(self, tiles):
        rng = np.random.RandomState(0)
        tile_emb = rng.randn(10, 10, self._dim).astype(np.float32)
        yield (2020, 0, 0, tile_emb, "EPSG:4326", self._transform)


class TestLoadEmbeddingsForRasterBboxClipping:
    def test_only_pixels_inside_bbox_are_returned(self, synthetic_raster):
        raster_path, transform = synthetic_raster
        gt = _FakeGeoTessera(transform)

        # Pixel centers at col+0.5, 9.5-row. This bbox (west=3, south=4,
        # east=6, north=7) covers exactly rows 3-5, cols 3-5 by center —
        # 9 known pixels, out of the tile's full 100.
        vectors, labels, class_names, stats, task = load_embeddings_for_raster(
            str(raster_path),
            bbox=(3, 4, 6, 7),
            year=2020,
            gt_instance=gt,
            task="classification",
        )

        expected_values = {33, 34, 35, 43, 44, 45, 53, 54, 55}
        returned_values = {int(float(c)) for c in class_names}

        assert returned_values == expected_values
        assert vectors.shape == (9, 4)
        assert stats["total_pixels"] == 9
        # The bug being guarded against: without bbox clipping, this would
        # return close to the full 10x10=100-pixel tile instead of 9.
        assert stats["total_pixels"] < 100

    def test_bbox_covering_whole_tile_returns_everything(self, synthetic_raster):
        raster_path, transform = synthetic_raster
        gt = _FakeGeoTessera(transform)

        vectors, labels, class_names, stats, task = load_embeddings_for_raster(
            str(raster_path),
            bbox=(0, 0, 10, 10),
            year=2020,
            gt_instance=gt,
            task="classification",
        )

        assert stats["total_pixels"] == 100
