"""'start' event must carry task, and must do so even on a cache-hit run.

Confirmed live (Louis Driver): after the first evaluation in a session, R²
stopped showing in the GUI (despite being logged server-side/CLI) for every
following evaluation, and the learning curve failed to build. Root cause:
the frontend's only source for "is this run classification or regression"
was the 'field_start' event -- but that event is gated by the tile cache
key changing (`_tile_cache["key"] != cache_key`), so it's only emitted on a
cache miss. Re-running with the same field/year/sampling (e.g. just
changing which classifiers are checked) hits the in-memory cache and skips
'field_start' entirely, leaving the frontend's task-tracking state stuck at
whatever the *previous* run left it at.

'start' is unconditional regardless of cache state -- these tests confirm
it now carries "task" directly, on both a cache-miss run and a
same-session cache-hit run.
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.evaluate  # noqa: F401 -- registers tessera_eval.evaluate in sys.modules
import tessera_eval.server as srv

sys.modules["tessera_eval.evaluate"]

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
        base = np.array([[p[0] + p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.1, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _make_gdf():
    heights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 4
    geoms = [box(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05) for i in range(len(heights))]
    return gpd.GeoDataFrame({"height": heights}, geometry=geoms, crs="EPSG:4326")


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _make_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    return srv.app.test_client()


def _run(client, **body):
    body.setdefault("field", "height")
    body.setdefault("task", "regression")
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 300)
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"run-large-area returned error event(s): {errors}"
    return events


def test_start_event_carries_task_on_first_cache_miss_run(client):
    events = _run(client, classifiers=["nn"])
    field_starts = [e for e in events if e["event"] == "field_start"]
    assert field_starts, "the session's first run should be a cache miss (field_start fires)"

    start = next(e for e in events if e["event"] == "start")
    assert start["task"] == "regression"


def test_start_event_carries_task_on_same_session_cache_hit_run(client):
    """The actual bug: a second run with the same field/year/sampling (only
    the classifier selection differs) hits the in-memory tile cache."""
    _run(client, classifiers=["nn"])
    events = _run(client, classifiers=["rf"])

    field_starts = [e for e in events if e["event"] == "field_start"]
    assert not field_starts, (
        "expected a cache hit on the second run (no field_start) -- if this fails, "
        "the test setup no longer reproduces the scenario this bug needs"
    )

    start = next(e for e in events if e["event"] == "start")
    assert start["task"] == "regression", (
        "'start' must carry task itself -- the frontend can no longer rely on "
        "field_start alone, since it doesn't fire on this cache-hit run"
    )
