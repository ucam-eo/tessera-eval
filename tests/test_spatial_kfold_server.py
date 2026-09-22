"""Integration tests for the spatial_kfold request flag wired through
run-large-area -- StratifiedGroupKFold over geographic blocks instead of
plain StratifiedKFold, so no two neighbouring points (or points from the
same field) are guaranteed to land on opposite sides of a k-fold fold.

See server.py's spatial_kfold comment and CHANGELOG for the motivation:
ordinary k-fold shuffles points with no notion of location at all, unlike
the learning curve's Spatial Train/Test Split. _spatial_block_groups (see
test_spatial_block_groups.py) supplies the grouping; this file tests that
it's actually wired into the request correctly.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.evaluate  # noqa: F401 -- registers the submodule
import tessera_eval.server as srv

EMBED_DIM = 128


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [object()]


class _FakeGeoTessera:
    def __init__(self, embeddings_dir=None):
        self.registry = _FakeRegistry()
        self._rng = np.random.RandomState(0)

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        if progress_callback:
            progress_callback(len(points), len(points), "done")
        n = len(points)
        base = np.array([[p[0] - p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.05, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _spread_out_gdf(n=36, ncols=6):
    """Polygons spread across a real 2D grid (not a 1D diagonal -- lon and
    lat must vary independently, or a quantile grid over strongly
    correlated axes collapses to effectively one dimension, giving far
    fewer distinct spatial blocks than intended; see
    test_spatial_block_groups.py's own tests for that behaviour in
    isolation) so quantile-binned sample points actually land in multiple
    distinct spatial blocks."""
    classes = ["oak", "beech", "pine"] * (n // 3)
    geoms = [
        box(
            (i % ncols) * 0.15,
            (i // ncols) * 0.15,
            (i % ncols) * 0.15 + 0.05,
            (i // ncols) * 0.15 + 0.05,
        )
        for i in range(len(classes))
    ]
    return gpd.GeoDataFrame({"species": classes}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _spread_out_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    return srv.app.test_client()


def _run(client, **body):
    body.setdefault("field", "species")
    body.setdefault("task", "classification")
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 400)
    body.setdefault("classifiers", ["rf"])
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"run-large-area returned error event(s): {errors}"
    return events


def test_spatial_kfold_activates_in_kfold_mode(client):
    events = _run(client, eval_mode="kfold", kfold_k=3, spatial_kfold=True)
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and starts[0].get("spatial_kfold") is True
    agg = [e for e in events if e.get("event") == "aggregate"]
    assert agg and "rf" in agg[0]["models"]


def test_spatial_kfold_ignored_outside_kfold_mode(client):
    events = _run(client, eval_mode="learning_curve", spatial_kfold=True)
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and "spatial_kfold" not in starts[0]
    statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
    assert any("Spatial k-fold ignored" in m and "only applies to k-fold" in m for m in statuses)


def test_spatial_kfold_takes_precedence_over_group_by_field(client):
    events = _run(
        client, eval_mode="kfold", kfold_k=3, spatial_kfold=True, group_by_field=True
    )
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and starts[0].get("spatial_kfold") is True
    assert "group_by_field" not in starts[0]
    statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
    assert any("Group by field ignored" in m and "Spatial k-fold is active" in m for m in statuses)


def test_group_by_field_still_works_alone_in_kfold_mode(client):
    """Unaffected by the spatial_kfold precedence change -- group_by_field
    on its own (no spatial_kfold) must behave exactly as it did before."""
    events = _run(client, eval_mode="kfold", kfold_k=3, group_by_field=True)
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and starts[0].get("group_by_field") is True


def test_neither_flag_set_is_a_no_op(client):
    events = _run(client, eval_mode="kfold", kfold_k=3)
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and "spatial_kfold" not in starts[0] and "group_by_field" not in starts[0]
    agg = [e for e in events if e.get("event") == "aggregate"]
    assert agg and "rf" in agg[0]["models"]
