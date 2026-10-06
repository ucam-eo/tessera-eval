"""User-settable limits (no silent caps) and point ground truth.

- Spatial MLP training points and spatial features per patch are now
  settings (Validation panel), not hard-coded 50,000 / 5,000.
- Every learning-curve step runs the full number of repeats (it used to
  drop to 3 then 2 at larger steps, and U-Net ran only one).
- With a separate test set the learning curve reaches 100% of the training
  pool; with a random split it stops at 80% (the rest is the test set).
- Point ground truth works: each point is one labelled pixel. It used to
  fail in every sampling mode ("No sample points generated", "float
  division by zero", "cannot convert float NaN to integer").
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import MultiPoint, Point, box

import tessera_eval.evaluate  # noqa: F401 -- registers the module in sys.modules
import tessera_eval.server as srv
from tessera_eval.evaluate import run_learning_curve

# The package re-exports a function named `evaluate`, so the attribute lookup
# `tessera_eval.evaluate` gives that function, not the module.
ev = sys.modules["tessera_eval.evaluate"]

# ---------------------------------------------------------------- helpers


def test_optional_limit():
    assert srv._optional_limit({}, "k", 50_000) == 50_000  # older clients
    for blank in (None, "", 0, "0"):
        assert srv._optional_limit({"k": blank}, "k", 50_000) is None  # no limit
    assert srv._optional_limit({"k": "1234"}, "k", 50_000) == 1234


def test_points_count_as_one_pixel_of_area():
    gdf = gpd.GeoDataFrame(
        geometry=[
            Point(0.1, 52.1),
            MultiPoint([(0.2, 52.2), (0.3, 52.3)]),
            box(0, 52, 0.001, 52.001),
        ],
        crs="EPSG:4326",
    )
    area = srv._label_area_m2(gdf)
    assert area.iloc[0] == 100.0
    assert area.iloc[1] == 200.0
    assert area.iloc[2] > 1000  # a real polygon keeps its real area


def test_sampler_takes_points_as_they_are():
    gdf = gpd.GeoDataFrame(geometry=[Point(i, 0) for i in range(5)], crs="EPSG:4326")
    coords, idx = srv._sample_points_within_budget(gdf, 100, np.random.RandomState(0))
    assert sorted(coords[:, 0].tolist()) == [0, 1, 2, 3, 4]
    assert sorted(idx.tolist()) == [0, 1, 2, 3, 4]


def test_sampler_subsamples_points_over_budget_and_mixes_with_polygons():
    pts = gpd.GeoDataFrame(geometry=[Point(i, 0) for i in range(50)], crs="EPSG:4326")
    coords, _ = srv._sample_points_within_budget(pts, 10, np.random.RandomState(0))
    assert len(coords) == 10

    mixed = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0), box(5, 5, 6, 6)], crs="EPSG:4326")
    coords, idx = srv._sample_points_within_budget(mixed, 20, np.random.RandomState(0))
    assert {0, 1} <= set(idx.tolist())
    assert (idx == 2).sum() > 0  # the polygon gets the remaining budget


# ------------------------------------------------- endpoint, point data


class _Reg:
    def load_blocks_for_region(self, bbox, year):
        return [object()]


class _PointGT:
    def __init__(self, embeddings_dir=None, **kwargs):
        self.registry = _Reg()

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        return np.random.RandomState(0).normal(size=(len(points), 128)).astype(np.float32)


def _point_gdf():
    rng = np.random.RandomState(1)
    pts = [Point(0.3 + rng.rand() * 0.1, 52.5 + rng.rand() * 0.1) for _ in range(300)]
    return gpd.GeoDataFrame(
        {"habitat": rng.choice(["a", "b", "c"], 300)}, geometry=pts, crs="EPSG:4326"
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("tessera_eval.dataset.ZarrClient", _PointGT)
    return srv.app.test_client()


def _events(client, **body):
    body.setdefault("field", "habitat")
    body.setdefault("task", "classification")
    body.setdefault("classifiers", ["rf"])
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    return [json.loads(line) for line in resp.text.strip().splitlines() if line.strip()]


@pytest.mark.parametrize("sampling", ["equal", "sqrt", "proportional"])
def test_point_ground_truth_runs_in_every_sampling_mode(client, monkeypatch, sampling):
    monkeypatch.setattr(srv, "_get_merged_gdf", _point_gdf)
    events = _events(client, sampling=sampling, max_training_samples=2000)
    assert not [e for e in events if e.get("event") == "error"], events
    start = next(e for e in events if e.get("event") == "start")
    # Budget exceeds the data: every point is used, each exactly once.
    assert sum(c["pixels"] for c in start["classes"]) == 300
    assert any(e.get("event") == "done" for e in events)


def test_learning_curve_reaches_100pct_with_a_separate_test_set(client, monkeypatch):
    monkeypatch.setattr(srv, "_get_merged_gdf", _point_gdf)
    # [south, west, north, east]: left half trains, right half tests.
    events = _events(
        client,
        sampling="equal",
        max_training_samples=2000,
        train_bboxes=[[52.4, 0.2, 52.7, 0.35]],
        test_bboxes=[[52.4, 0.35, 52.7, 0.5]],
    )
    start = next(e for e in events if e.get("event") == "start")
    assert start["training_pcts"][-1] == 100
    pcts = [e["pct"] for e in events if e.get("event") == "progress" and "classifiers" in e]
    assert pcts[-1] == 100


def test_random_split_stops_at_80pct(client, monkeypatch):
    monkeypatch.setattr(srv, "_get_merged_gdf", _point_gdf)
    events = _events(client, sampling="equal", max_training_samples=2000)
    start = next(e for e in events if e.get("event") == "start")
    assert start["training_pcts"][-1] == 80


# ------------------------------------------------ learning-curve engine


def _pixel_data(n=300):
    rng = np.random.RandomState(0)
    labels = rng.randint(0, 3, size=n)
    vectors = (rng.normal(size=(n, 8)) + labels[:, None]).astype(np.float32)
    return vectors, labels


def test_every_step_gets_every_repeat():
    vectors, labels = _pixel_data()
    events = list(run_learning_curve(vectors, labels, ["nn"], [10, 50, 80], repeats=5))
    msgs = [e["message"] for e in events if e.get("type") == "classifier_status"]
    for pct in (10, 50, 80):
        assert any(f"Pct {pct}%" in m and "/5)" in m for m in msgs), (pct, msgs)


def test_random_split_drops_steps_above_80_but_a_test_set_keeps_100():
    vectors, labels = _pixel_data()
    events = list(run_learning_curve(vectors, labels, ["nn"], [50, 80, 100], repeats=1))
    assert [e["pct"] for e in events if e["type"] == "progress"] == [50, 80]

    tv, tl = _pixel_data(60)
    events = list(
        run_learning_curve(
            vectors, labels, ["nn"], [50, 80, 100], repeats=1, test_vectors=tv, test_labels=tl
        )
    )
    assert [e["pct"] for e in events if e["type"] == "progress"] == [50, 80, 100]


def test_spatial_training_cap_is_passed_through(monkeypatch):
    seen = []
    real = ev.augment_spatial

    def spy(X, y, window, dim, cap=ev.DEFAULT_AUGMENT_CAP, seed=42):
        seen.append(cap)
        return real(X, y, window, dim, cap=cap, seed=seed)

    monkeypatch.setattr(ev, "augment_spatial", spy)
    vectors, labels = _pixel_data(120)
    sp = np.random.RandomState(1).normal(size=(120, 9 * 8)).astype(np.float32)
    for cap in (None, 7):
        seen.clear()
        list(
            run_learning_curve(
                vectors,
                labels,
                ["spatial_mlp"],
                [50],
                repeats=1,
                classifier_params={"spatial_mlp": {"max_iter": 5}},
                spatial_vectors=sp,
                spatial_labels=labels,
                spatial_augment_cap=cap,
            )
        )
        assert seen and set(seen) == {cap}


# ------------------------------------------------------ patch extraction

TILE = 64


class _TileReg:
    def load_blocks_for_region(self, bbox, year):
        return [(year, 16.6 + i * 0.2, 48.3) for i in range(2)]


class _TileGT:
    def __init__(self):
        self.registry = _TileReg()
        self._emb = np.random.RandomState(0).rand(TILE, TILE, 8).astype(np.float32)

    def fetch_embeddings(self, tiles):
        def gen():
            for _yr, lon, _lat in tiles:
                yield (
                    None,
                    None,
                    None,
                    self._emb,
                    "EPSG:4326",
                    Affine(0.001, 0, lon, 0, -0.001, 48.35),
                )

        return gen()


def _extract(gdf, **kw):
    out = srv._extract_tile_patches(
        _TileGT(),
        gdf,
        "height",
        2024,
        le=None,
        n_classes=0,
        patch_size=16,
        max_patches=20,
        is_classification=False,
        **kw,
    )
    return out[0], out[1]


def _polygon_gdf():
    geoms = [box(16.6 + i * 0.2, 48.29, 16.6 + i * 0.2 + 0.06, 48.35) for i in range(2)]
    return gpd.GeoDataFrame({"height": [3.7, 4.2]}, geometry=geoms, crs="EPSG:4326")


def test_spatial_features_per_patch_is_settable():
    patches, sp_limited = _extract(_polygon_gdf(), needs_spatial_3x3=True, max_spatial_px=5)
    _, sp_all = _extract(_polygon_gdf(), needs_spatial_3x3=True, max_spatial_px=None)
    assert len(sp_limited) <= 5 * len(patches)
    assert len(sp_all) > len(sp_limited)


def test_sparse_points_still_yield_patches():
    pts = [Point(16.6 + i * 0.2 + 0.03, 48.32) for i in range(2)]
    gdf = gpd.GeoDataFrame({"height": [1.0, 2.0]}, geometry=pts, crs="EPSG:4326")
    assert len(_extract(gdf)[0]) == 0  # polygon default: >=10 labelled px per patch
    assert len(_extract(gdf, min_labelled_px=1)[0]) > 0
