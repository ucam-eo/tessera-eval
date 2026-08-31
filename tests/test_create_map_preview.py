"""_render_map_preview: the in-browser map overlay for create_map (feature 5).

The GeoTIFF is still the real output; this is a small lat/lon PNG + legend
the viewer drops on the map as an L.imageOverlay so a map can be eyeballed
without opening QGIS. It must be best-effort -- any failure returns None
and never breaks map generation.
"""

from __future__ import annotations

import base64
import json
import struct

import numpy as np
from affine import Affine

from tessera_eval.server import _render_map_preview


def _utm33_transform():
    # 10 m pixels, top-left near Vienna in EPSG:32633 (UTM 33N).
    return Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5_350_000.0)


def _png_size(data_url):
    """(width, height) straight out of the PNG IHDR -- no imaging library."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", raw[16:24])


def test_classification_preview_png_bounds_and_legend():
    arr = np.zeros((40, 60), dtype=np.uint8)
    arr[:20] = 1
    arr[20:] = 3  # non-contiguous class ids are fine
    arr[0, 0] = 0  # a nodata pixel

    p = _render_map_preview(
        arr, _utm33_transform(), "EPSG:32633", True, ["Water", "Scrub", "Forest"], None
    )
    assert p is not None
    assert p["is_classification"] is True
    assert p["png"].startswith("data:image/png;base64,")

    (south, west), (north, east) = p["bounds"]
    assert south < north and west < east
    assert 40 < south < 55 and 10 < west < 20  # roughly UTM 33N / central Europe

    w, h = _png_size(p["png"])
    assert w <= 1024 and h <= 1024
    assert abs(w / h - (east - west) / (north - south)) < 0.05  # aspect matches the bbox

    assert [item["value"] for item in p["legend"]] == [1, 3]  # 0 (nodata) excluded
    assert p["legend"][0]["label"] == "Water"
    assert p["legend"][1]["label"] == "Forest"
    assert all(item["color"].startswith("#") for item in p["legend"])


def test_regression_preview_carries_the_value_range_and_ramp():
    arr = np.linspace(0.0, 40.0, 40 * 60, dtype=np.float32).reshape(40, 60)
    arr[0, :5] = np.nan  # nodata band

    p = _render_map_preview(arr, _utm33_transform(), "EPSG:32633", False, [], (0.0, 40.0))
    assert p is not None
    assert p["is_classification"] is False
    assert p["legend"]["min"] == 0.0
    assert p["legend"]["max"] == 40.0
    assert isinstance(p["legend"]["ramp"], list) and len(p["legend"]["ramp"]) >= 2
    _png_size(p["png"])  # decodes as a valid PNG


def test_regression_preview_falls_back_to_data_range_without_a_clamp():
    arr = np.full((20, 20), 7.5, dtype=np.float32)
    p = _render_map_preview(arr, _utm33_transform(), "EPSG:32633", False, [], None)
    assert p is not None
    assert p["legend"]["min"] == 7.5 and p["legend"]["max"] == 7.5  # (hi-lo)=0 handled


def test_preview_is_json_serialisable():
    arr = np.ones((10, 12), dtype=np.uint8)
    p = _render_map_preview(arr, _utm33_transform(), "EPSG:32633", True, ["A"], None)
    json.dumps(p)  # must not raise -- it rides in the map_ready SSE line


def test_preview_returns_none_on_bad_input_instead_of_raising():
    # 1-D array -> `h, w = arr.shape` unpack fails -> swallowed -> None
    assert (
        _render_map_preview(np.zeros(5), _utm33_transform(), "EPSG:32633", True, [], None) is None
    )


def test_preview_downscales_a_large_raster_to_the_cap():
    arr = (np.arange(3000 * 2000, dtype=np.uint8) % 4 + 1).reshape(3000, 2000)
    p = _render_map_preview(arr, _utm33_transform(), "EPSG:32633", True, ["a", "b", "c", "d"], None)
    w, h = _png_size(p["png"])
    assert max(w, h) == 1024
