"""load_embeddings_for_shapefile_vq: region/chunk loader over a mosaic-fetching client.

Uses a fake client (no network) whose fetch_mosaic_for_region returns a constant
EPSG:4326 mosaic for the requested bbox — so we can check chunking, the
touched-chunk filter, label extraction, and the no-coverage path.
"""

import numpy as np
import pytest
from affine import Affine

gpd = pytest.importorskip("geopandas")
from shapely.geometry import box as shp_box

from tessera_eval import load_embeddings_for_shapefile_vq


class FakeClient:
    def __init__(self, res_deg=0.001, bands=8, value=2.0, fail=False):
        self.res_deg = res_deg
        self.bands = bands
        self.value = value
        self.fail = fail
        self.calls = []

    def fetch_mosaic_for_region(self, bbox, year=2024, target_crs="EPSG:4326"):
        self.calls.append(tuple(bbox))
        if self.fail:
            raise RuntimeError("no VQ coverage")
        lon0, lat0, lon1, lat1 = bbox
        w = max(1, round((lon1 - lon0) / self.res_deg))
        h = max(1, round((lat1 - lat0) / self.res_deg))
        tfm = Affine(self.res_deg, 0, lon0, 0, -self.res_deg, lat1)
        mosaic = np.full((h, w, self.bands), self.value, dtype=np.float32)
        return mosaic, tfm, "EPSG:4326"


def _gdf(geom_box, cls="a"):
    return gpd.GeoDataFrame({"cls": [cls]}, geometry=[shp_box(*geom_box)], crs="EPSG:4326")


def test_vq_loader_basic():
    gdf = _gdf((0.01, 50.01, 0.03, 50.03))  # ~2 km square -> a single chunk
    client = FakeClient(value=2.0, bands=8)
    vectors, labels, class_names, stats = load_embeddings_for_shapefile_vq(gdf, "cls", 2024, client)

    assert vectors.shape[1] == 8
    assert vectors.shape[0] > 0
    assert set(np.unique(labels).tolist()) == {0}
    assert class_names == ["a"]
    assert np.allclose(vectors, 2.0)
    assert stats["total_pixels"] == vectors.shape[0]
    assert stats["chunks_with_data"] >= 1
    assert client.calls


def test_vq_loader_skips_untouched_chunks():
    # Two tiny polygons far apart: the empty chunks between them must not be fetched.
    g = gpd.GeoDataFrame(
        {"cls": ["a", "b"]},
        geometry=[shp_box(0.0, 50.0, 0.005, 50.005), shp_box(0.30, 50.30, 0.305, 50.305)],
        crs="EPSG:4326",
    )
    client = FakeClient()
    _, _, class_names, stats = load_embeddings_for_shapefile_vq(g, "cls", 2024, client, max_km=10.0)

    assert stats["chunk_count"] > len(client.calls)  # not every chunk fetched
    assert len(client.calls) <= 4  # only the two corners (+ maybe edge rounding)
    assert set(class_names) == {"a", "b"}
    assert stats["chunks_with_data"] >= 1


def test_vq_loader_reprojects_non_4326_input():
    gdf = _gdf((0.01, 50.01, 0.03, 50.03)).to_crs("EPSG:3857")
    client = FakeClient()
    vectors, _, _, _ = load_embeddings_for_shapefile_vq(gdf, "cls", 2024, client)
    assert vectors.shape[0] > 0
    # fetched bboxes are lon/lat degrees, not Web-Mercator metres
    assert all(abs(c[0]) < 360 for c in client.calls)


def test_vq_loader_no_coverage_raises():
    gdf = _gdf((0.0, 50.0, 0.01, 50.01))
    client = FakeClient(fail=True)
    with pytest.raises(ValueError):
        load_embeddings_for_shapefile_vq(gdf, "cls", 2024, client)
