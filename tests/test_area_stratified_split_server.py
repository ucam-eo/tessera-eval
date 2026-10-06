"""Integration tests for the area_stratified_split request flag wired
through run-large-area -- reproducing Frank Feng's (TESSERA paper co-
author) Austrian-crop split: "a field-level, area-stratified split (30% of
each class's area -> train; rest 1/7 val, 6/7 test)". See
_area_stratified_field_split (tests/test_area_stratified_field_split.py)
for the underlying per-field allocation this wires up as a fixed test set,
the same way spatial/year/file splits already do.
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
    def __init__(self, embeddings_dir=None, **kwargs):
        self.registry = _FakeRegistry()
        self._rng = np.random.RandomState(0)

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        if progress_callback:
            progress_callback(len(points), len(points), "done")
        n = len(points)
        base = np.array([[p[0] - p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.05, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _many_fields_gdf(n=60, ncols=10):
    """Many small, similarly-sized polygons across a real 2D grid, 3
    classes -- enough fields per class for a 30/10/60 area split to have
    real train and test pools (a handful of fields per class, like
    test_spatial_kfold_server.py's fixture, can't support this: with 4
    fields in a class, a 30% quota rounds up to swallow more than one
    field)."""
    classes = ["oak", "beech", "pine"] * (n // 3)
    geoms = [
        box(
            (i % ncols) * 0.1,
            (i // ncols) * 0.1,
            (i % ncols) * 0.1 + 0.04,
            (i // ncols) * 0.1 + 0.04,
        )
        for i in range(len(classes))
    ]
    return gpd.GeoDataFrame({"species": classes}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _many_fields_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("tessera_eval.dataset.ZarrClient", _FakeGeoTessera)
    return srv.app.test_client()


def _run(client, **body):
    body.setdefault("field", "species")
    body.setdefault("task", "classification")
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 600)
    body.setdefault("classifiers", ["rf"])
    body.setdefault("training_pcts", [50, 100])
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"run-large-area returned error event(s): {errors}"
    return events


def test_area_stratified_split_activates_and_reports_a_fixed_test_set(client):
    events = _run(client, area_stratified_split=True)
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and starts[0].get("area_stratified_split") is True
    assert starts[0]["train_count"] > 0
    assert starts[0]["test_count"] > 0
    # Roughly 30/60 train/test (10% val dropped) -- loose bound, this is a
    # tiny synthetic fixture, not a precision check (see
    # test_area_stratified_field_split.py for the tight version).
    ratio = starts[0]["train_count"] / starts[0]["test_count"]
    assert 0.2 < ratio < 1.0
    progress = [e for e in events if e.get("event") == "progress"]
    assert progress and "rf" in progress[-1]["classifiers"]


def test_area_stratified_split_off_by_default(client):
    events = _run(client)
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and "area_stratified_split" not in starts[0]


def test_area_stratified_split_ignored_in_kfold_mode(client):
    events = _run(client, eval_mode="kfold", kfold_k=3, area_stratified_split=True)
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and "area_stratified_split" not in starts[0]
    statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
    assert any(
        "Area-stratified split ignored" in m and "only applies to the learning curve" in m
        for m in statuses
    )
    agg = [e for e in events if e.get("event") == "aggregate"]
    assert agg and "rf" in agg[0]["models"]


def test_area_stratified_split_takes_precedence_over_group_by_field(client):
    events = _run(client, area_stratified_split=True, group_by_field=True)
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and starts[0].get("area_stratified_split") is True
    assert "group_by_field" not in starts[0]
    statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
    assert any(
        "Group by field ignored" in m and "already fixing the test set" in m for m in statuses
    )


def test_area_stratified_split_ignored_for_regression(client, monkeypatch):
    def _continuous_gdf():
        gdf = _many_fields_gdf()
        rng = np.random.RandomState(0)
        gdf["yield_t_ha"] = rng.uniform(1, 10, size=len(gdf))
        return gdf

    monkeypatch.setattr(srv, "_get_merged_gdf", _continuous_gdf)
    events = _run(
        client,
        field="yield_t_ha",
        task="regression",
        classifiers=["rf_reg"],
        area_stratified_split=True,
    )
    starts = [e for e in events if e.get("event") == "start"]
    assert starts and "area_stratified_split" not in starts[0]
    statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
    assert any("Area-stratified split ignored" in m and "regression target" in m for m in statuses)
