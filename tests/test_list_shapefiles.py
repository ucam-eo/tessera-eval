"""GET /api/evaluation/list-shapefiles -- what's currently in the merged
ground-truth set. Uploads accumulate (multi-shapefile merge), and the
viewer shows this on entering Validation so an earlier upload still in the
set is visible rather than a surprise (Keshav, 2026-09-02)."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

import tessera_eval.server as srv


@pytest.fixture
def client():
    srv.app.config["TESTING"] = True
    return srv.app.test_client()


def _gdf(n):
    return gpd.GeoDataFrame(
        {"v": list(range(n))},
        geometry=[Point(i, i) for i in range(n)],
        crs="EPSG:4326",
    )


def test_lists_each_uploaded_file_with_its_feature_count(client, monkeypatch):
    monkeypatch.setattr(
        srv, "_uploaded_shapefiles", [("austria.zip", _gdf(3)), ("snowdon.zip", _gdf(1))]
    )
    resp = client.get("/api/evaluation/list-shapefiles")
    assert resp.status_code == 200
    assert resp.get_json()["files"] == [
        {"name": "austria.zip", "features": 3},
        {"name": "snowdon.zip", "features": 1},
    ]


def test_empty_when_nothing_uploaded(client, monkeypatch):
    monkeypatch.setattr(srv, "_uploaded_shapefiles", [])
    resp = client.get("/api/evaluation/list-shapefiles")
    assert resp.status_code == 200
    assert resp.get_json() == {"files": []}
