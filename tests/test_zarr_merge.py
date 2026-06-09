"""read_region_chunked: cross-UTM-zone merge into a shared EPSG:4326 grid.

Uses a fake GeoTesseraZarr whose read_region returns each chunk in its own
centre-zone UTM CRS, filled with a per-side tag value, so we can assert the
merge places both zones correctly (the old code dropped non-first-zone chunks).
"""

import numpy as np
import pytest
from affine import Affine

rasterio_warp = pytest.importorskip("rasterio.warp")
transform_bounds = rasterio_warp.transform_bounds

from tessera_eval.zarr_utils import _utm_zone, read_region_chunked


def _utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


class FakeZarr:
    """Mimics GeoTesseraZarr.read_region: a chunk in its centre's UTM zone."""

    def __init__(self, bands=4, res_m=10.0, value_fn=None):
        self.bands = bands
        self.res_m = res_m
        self.value_fn = value_fn or (lambda lon, lat: 1.0)

    def read_region(self, bounds, year):
        lon0, lat0, lon1, lat1 = bounds
        clon, clat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
        crs = _utm_epsg(clon, clat)
        left, bottom, right, top = transform_bounds("EPSG:4326", crs, lon0, lat0, lon1, lat1)
        w = max(1, round((right - left) / self.res_m))
        h = max(1, round((top - bottom) / self.res_m))
        tfm = Affine(self.res_m, 0, left, 0, -self.res_m, top)
        emb = np.full((h, w, self.bands), self.value_fn(clon, clat), dtype=np.float32)
        return emb, tfm, crs


def _covered(mosaic):
    return ~np.isnan(mosaic).all(axis=2)


def test_utm_zone_boundary():
    assert _utm_zone(-0.03) == 30  # zone 30N: lon [-6, 0)
    assert _utm_zone(0.07) == 31  # zone 31N: lon [0, 6)
    assert _utm_zone(-0.03) != _utm_zone(0.07)


def test_cross_zone_merge_covers_both_sides():
    # Straddles 0degE -> chunk west of 0 is UTM 30N, east is 31N (different CRSs).
    bounds = (-0.08, 52.0, 0.12, 52.1)
    gtz = FakeZarr(value_fn=lambda lon, lat: 10.0 if lon < 0 else 20.0)
    mosaic, transform, crs = read_region_chunked(gtz, bounds, 2024)

    assert crs == "EPSG:4326"
    h, w, b = mosaic.shape
    assert h > 0 and w > 0 and b == 4
    cov = _covered(mosaic)
    assert cov.mean() > 0.9, f"coverage only {cov.mean():.2f} — chunks dropped/mis-placed?"

    # Column -> centre longitude via the dst transform.
    px_lon = transform.a
    west = transform.c
    lons = west + (np.arange(w) + 0.5) * px_lon
    west_cols = np.where(lons < 0)[0]
    east_cols = np.where(lons > 0)[0]
    assert west_cols.size and east_cols.size
    # Both zones present (old code dropped the second zone -> east all-NaN).
    assert cov[:, west_cols].any(), "no data west of 0degE"
    assert cov[:, east_cols].any(), "no data east of 0degE (second UTM zone dropped?)"

    # Placement: west tagged 10, east tagged 20 (nearest-neighbour keeps constants).
    wv = mosaic[:, west_cols, 0]
    ev = mosaic[:, east_cols, 0]
    assert np.nanmedian(wv) == pytest.approx(10.0)
    assert np.nanmedian(ev) == pytest.approx(20.0)


def test_single_zone_large_full_coverage():
    # 0.25deg span (> CHUNK_THRESHOLD) entirely in UTM 32N (lon 6..12).
    bounds = (10.0, 50.0, 10.25, 50.25)
    gtz = FakeZarr(value_fn=lambda lon, lat: 5.0)
    mosaic, transform, crs = read_region_chunked(gtz, bounds, 2024)
    assert crs == "EPSG:4326"
    assert _covered(mosaic).mean() > 0.9
    assert np.nanmedian(mosaic) == pytest.approx(5.0)


def test_small_single_zone_fast_path():
    # <= 0.2deg, single zone -> single read + reproject (no merge).
    bounds = (10.0, 50.0, 10.1, 50.1)
    gtz = FakeZarr(value_fn=lambda lon, lat: 7.0)
    mosaic, transform, crs = read_region_chunked(gtz, bounds, 2024)
    assert crs == "EPSG:4326"
    assert _covered(mosaic).any()
    assert np.nanmedian(mosaic) == pytest.approx(7.0)


def test_no_data_returns_none():
    class Empty:
        def read_region(self, bounds, year):
            return None, None, None

    assert read_region_chunked(Empty(), (10.0, 50.0, 10.3, 50.3), 2024) == (None, None, None)
