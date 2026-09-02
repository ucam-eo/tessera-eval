"""create_map()/train_models() must not silently fall back to classification.

Confirmed live (Louis Driver): a regression height map came out quantised
onto the training label values (1, 3, ... 65) with a discrete class palette,
even though the evaluation run had correctly reported R2. Root cause: the
map/final-model endpoints read the task type only from
_tile_cache["_is_classification"], which run_large_area wrote *once*, at the
very end of its response stream -- after every model had trained. An
evaluation cut short before that final event (client disconnect / tab
throttle mid-run, or a cancel) never wrote the key, so create_map() hit its
`cache.get("_is_classification", True)` default, fit a classifier on the
continuous heights, and wrote uint8 predictions snapped onto the label set.

Fixes under test:
  * run_large_area commits _is_classification to the tile cache *with* the
    vectors it applies to (both cache-population sites), so it's set before
    any training and survives an interrupted stream.
  * create_map/train_models resolve the task via _resolve_task(): an
    explicit request override first, then the cached flag, then a
    data-derived fallback (regression runs leave class_names empty) --
    never a blind default to classification.
  * create_map accepts a "task" body param (the frontend passes the task
    its evaluation ran as) and reports the task it used on map_ready.
"""

from __future__ import annotations

import io
import json
import sys

import numpy as np
import pytest
import rasterio
from affine import Affine

import tessera_eval.evaluate  # noqa: F401 -- registers the submodule in sys.modules
import tessera_eval.server as srv

# tessera_eval.evaluate the attribute is a function; the submodule lives in
# sys.modules (name collision -- see test_start_event_task_on_cache_hit.py).
ev_mod = sys.modules["tessera_eval.evaluate"]

EMBED_DIM = 8


# --- _resolve_task unit tests ------------------------------------------------


def test_resolve_task_explicit_override_beats_stale_cache():
    # A stale _is_classification=True (regression run that never finished
    # writing the real value) must lose to an explicit request override.
    assert srv._resolve_task({"_is_classification": True}, "regression") is False
    assert srv._resolve_task({"_is_classification": False}, "classification") is True


def test_resolve_task_ignores_non_committal_override():
    # None / "auto" are not overrides -- fall through to the cache.
    assert srv._resolve_task({"_is_classification": False}, None) is False
    assert srv._resolve_task({"_is_classification": False}, "auto") is False


def test_resolve_task_uses_cached_flag_when_present():
    assert srv._resolve_task({"_is_classification": True}, None) is True
    assert srv._resolve_task({"_is_classification": False}, None) is False


def test_resolve_task_falls_back_to_class_names_not_classification():
    # Key absent (old cache entry, or an interrupted run) -> infer from the
    # data instead of defaulting to classification.
    assert srv._resolve_task({"class_names": []}, None) is False
    assert srv._resolve_task({"class_names": ["oak", "beech"]}, None) is True
    assert srv._resolve_task({}, None) is False


def test_resolve_task_cached_flag_wins_over_class_names_fallback():
    assert srv._resolve_task({"_is_classification": False, "class_names": ["a"]}, None) is False


# --- run_large_area commits the task before training ------------------------


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
        base = np.array([[p[0] + p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.1, size=(n, 128)).astype(np.float32)
        return base + noise


def _make_gdf():
    import geopandas as gpd
    from shapely.geometry import box

    heights = [float(h) for h in range(1, 66, 2)] * 2  # 33 distinct values > 20
    geoms = [box(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05) for i in range(len(heights))]
    return gpd.GeoDataFrame({"height": heights}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def eval_client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _make_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    return srv.app.test_client()


def test_task_committed_to_cache_even_when_training_blows_up(eval_client, monkeypatch):
    """The interrupted-stream scenario: training raises partway, the stream
    ends on an error event -- but the task type is already in the cache
    because it's written with the vectors, not at the end."""

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated mid-run failure")
        yield  # pragma: no cover -- makes this a generator

    # server.py imports this lazily from tessera_eval.evaluate inside the
    # stream generator, so patch it at the source module.
    monkeypatch.setattr(ev_mod, "run_learning_curve", _boom)

    resp = eval_client.post(
        "/api/evaluation/run-large-area",
        json={
            "field": "height",
            "task": "regression",
            "sampling": "equal",
            "max_training_samples": 200,
            "classifiers": ["nn"],
        },
    )
    # The failure surfaces mid-stream (run_large_area doesn't wrap the
    # learning-curve loop), which is the real-world interrupted-run shape.
    with pytest.raises(RuntimeError, match="simulated mid-run failure"):
        resp.get_data()

    assert srv._tile_cache.get("vectors") is not None, "vectors should be cached before training"
    assert srv._tile_cache.get("_is_classification") is False, (
        "task type must be committed to the cache with the vectors, so an "
        "interrupted eval can't leave create_map guessing"
    )


# --- create_map without / against the cached flag -------------------------


class _FakeMapRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [(year, 16.62, 48.22)]


class _FakeMapGeoTessera:
    def __init__(self, embeddings_dir=None):
        self.registry = _FakeMapRegistry()

    def fetch_embeddings(self, tiles):
        def gen():
            for yr, _lon, _lat in tiles:
                rng = np.random.RandomState(1)
                emb = rng.uniform(0, 50, size=(16, 16, EMBED_DIM)).astype(np.float32)
                transform = Affine(0.001, 0, 16.6, 0, -0.001, 48.25)
                yield (None, None, None, emb, "EPSG:4326", transform)

        return gen()


def _map_cache(extra):
    rng = np.random.RandomState(0)
    vectors = rng.uniform(0, 50, size=(200, EMBED_DIM)).astype(np.float32)
    # Continuous heights floored to odd integers -- the real shape of
    # Louis's field. > 20 distinct values, so genuinely a regression.
    labels = (rng.uniform(0, 32, size=200).astype(int) * 2 + 1).astype(np.float32)
    base = {
        "key": ("height", 2025, 2025, "equal"),
        "vectors": vectors,
        "labels": labels,
        "class_names": [],
        "_model_params": {},
    }
    base.update(extra)
    return base


@pytest.fixture
def map_client(monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_zarr", lambda: None)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr("geotessera.GeoTessera", _FakeMapGeoTessera)
    monkeypatch.setattr(srv, "_generated_maps", {})
    return srv.app.test_client()


def _create_map(client, **body):
    body.setdefault("classifier", "rf")
    body.setdefault("map_bboxes", [[48.2, 16.6, 48.25, 16.65]])
    resp = client.post("/api/evaluation/create-map", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert not [e for e in events if e.get("event") == "error"], events
    return events


def _map_array(client, ready):
    resp = client.get(ready["download_url"])
    assert resp.status_code == 200
    with rasterio.open(io.BytesIO(resp.data)) as ds:
        return ds.dtypes[0], ds.read(1)


def test_missing_cached_flag_infers_regression_from_empty_class_names(map_client, monkeypatch):
    monkeypatch.setattr(srv, "_tile_cache", _map_cache({}))  # no _is_classification key
    events = _create_map(map_client)
    ready = next(e for e in events if e["event"] == "map_ready")
    assert ready["task"] == "regression"
    dtype, arr = _map_array(map_client, ready)
    assert dtype == "float32", "regression map must be float32, not a uint8 class raster"
    valid = arr[~np.isnan(arr)]
    assert len(np.unique(valid)) > 5, "predictions collapsed onto a class-like value set"


def test_stale_classification_flag_overridden_by_task_body_param(map_client, monkeypatch):
    # The exact bug: an interrupted earlier classification run left
    # _is_classification=True in the cache; the frontend now passes the task
    # its evaluation actually ran as.
    monkeypatch.setattr(srv, "_tile_cache", _map_cache({"_is_classification": True}))
    events = _create_map(map_client, task="regression")
    ready = next(e for e in events if e["event"] == "map_ready")
    assert ready["task"] == "regression"
    dtype, _ = _map_array(map_client, ready)
    assert dtype == "float32"


def test_cached_regression_flag_is_honoured_without_a_body_param(map_client, monkeypatch):
    monkeypatch.setattr(srv, "_tile_cache", _map_cache({"_is_classification": False}))
    events = _create_map(map_client)
    ready = next(e for e in events if e["event"] == "map_ready")
    assert ready["task"] == "regression"
    dtype, _ = _map_array(map_client, ready)
    assert dtype == "float32"
