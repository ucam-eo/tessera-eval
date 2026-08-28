"""Sentinel nodata values must be removed before resampling, not after.

Reference rasters such as MS-NFI use large sentinel values (32766/32767)
for missing data. align_raster_to_grid used to reproject first and strip
sentinels afterwards by exact equality -- but bilinear resampling blends a
sentinel with its real neighbours, producing large in-between values that
no longer equal the sentinel and so survive as apparently valid targets.
"""

import numpy as np
import rasterio
from affine import Affine

from tessera_eval.rasterize import align_raster_to_grid

SENTINEL = 32767.0


def _write_raster(path, data, transform, crs="EPSG:4326"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_bilinear_does_not_blend_sentinels_into_valid_pixels(tmp_path):
    data = np.full((8, 8), 10.0, dtype=np.float32)
    data[4, 4] = SENTINEL
    src_transform = Affine(0.001, 0, 0.0, 0, -0.001, 1.0)
    path = tmp_path / "ref.tif"
    _write_raster(path, data, src_transform)

    # Same resolution, shifted by half a pixel, so every destination pixel
    # interpolates between four source pixels.
    dst_transform = Affine(0.001, 0, 0.0005, 0, -0.001, 0.9995)
    aligned = align_raster_to_grid(
        str(path),
        dst_transform,
        "EPSG:4326",
        width=7,
        height=7,
        resampling="bilinear",
        nodata_values=[SENTINEL],
    )

    valid = aligned[~np.isnan(aligned)]
    assert valid.size > 0
    assert valid.max() < 100, (
        f"sentinel value leaked into interpolated output: max valid value {valid.max():.1f}"
    )


def test_nearest_with_sentinels_still_masks_them(tmp_path):
    data = np.full((8, 8), 10.0, dtype=np.float32)
    data[4, 4] = SENTINEL
    src_transform = Affine(0.001, 0, 0.0, 0, -0.001, 1.0)
    path = tmp_path / "ref.tif"
    _write_raster(path, data, src_transform)

    aligned = align_raster_to_grid(
        str(path),
        src_transform,
        "EPSG:4326",
        width=8,
        height=8,
        resampling="nearest",
        nodata_values=[SENTINEL],
    )

    assert np.isnan(aligned[4, 4])
    valid = aligned[~np.isnan(aligned)]
    assert np.allclose(valid, 10.0)


def test_destination_outside_raster_returns_all_nan(tmp_path):
    data = np.full((8, 8), 10.0, dtype=np.float32)
    src_transform = Affine(0.001, 0, 0.0, 0, -0.001, 1.0)
    path = tmp_path / "ref.tif"
    _write_raster(path, data, src_transform)

    far_away = Affine(0.001, 0, 5.0, 0, -0.001, 6.0)
    aligned = align_raster_to_grid(
        str(path),
        far_away,
        "EPSG:4326",
        width=4,
        height=4,
        resampling="bilinear",
        nodata_values=[SENTINEL],
    )

    assert np.isnan(aligned).all()


def test_destination_barely_overlapping_the_padding_returns_all_nan(tmp_path):
    """A destination grid just outside the raster, but within the window's
    two-pixel interpolation margin, used to round to a zero-width read and
    crash inside reproject instead of returning all-NaN."""
    data = np.full((8, 8), 10.0, dtype=np.float32)
    src_transform = Affine(0.001, 0, 0.0, 0, -0.001, 1.0)
    path = tmp_path / "ref.tif"
    _write_raster(path, data, src_transform)

    just_west = Affine(0.001, 0, -0.0057, 0, -0.001, 1.0)
    aligned = align_raster_to_grid(
        str(path),
        just_west,
        "EPSG:4326",
        width=4,
        height=4,
        resampling="bilinear",
        nodata_values=[SENTINEL],
    )

    assert np.isnan(aligned).all()
