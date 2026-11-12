"""group_by_field must survive the on-disk sample cache.

Reported 2026-11-12 (Louis Driver / Moustafa Eweda): learning curves run
one model at a time with Group by field on. The first run read the labels
fresh and was grouped; every later run hit the disk result cache, which did
not store each pixel's source polygon, so group_by_field was silently
dropped (a brief status line only) and the run used an ungrouped, optimistic
random split.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import tessera_eval.server as srv
from tests.test_cache_key_sample_budget import EMBED_DIM, _FakeGeoTessera, _make_gdf


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _make_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("tessera_eval.dataset.ZarrClient", _FakeGeoTessera)
    _FakeGeoTessera.calls = 0
    return srv.app.test_client()


def _run(client, group_by_field):
    resp = client.post(
        "/api/evaluation/run-large-area",
        json={
            "field": "habitat",
            "task": "classification",
            "classifiers": ["nn"],
            "classifier_params": {"nn": {"n_neighbors": 1}},
            "sampling": "equal",
            "max_training_samples": 400,
            "seed": 42,
            "group_by_field": group_by_field,
        },
    )
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert not [e for e in events if e.get("event") == "error"], events
    start = next(e for e in events if e.get("event") == "start")
    ignored = [
        e["message"]
        for e in events
        if e.get("event") == "status" and "Group by field ignored" in e.get("message", "")
    ]
    return start, ignored


def _restart_server():
    """Drop the in-memory cache, as a tee-compute restart does; the disk cache stays."""
    srv._tile_cache.clear()
    srv._tile_cache.update({"key": None, "vectors": None})


def test_grouping_applies_on_a_disk_cache_hit(client):
    start, ignored = _run(client, group_by_field=True)
    assert start.get("group_by_field") is True and not ignored
    assert _FakeGeoTessera.calls == 1

    _restart_server()
    start, ignored = _run(client, group_by_field=True)
    assert _FakeGeoTessera.calls == 1, "should reuse the disk cache, not re-extract"
    assert start.get("group_by_field") is True, "grouping silently dropped on a cache hit"
    assert not ignored, ignored


def test_old_cache_without_polygon_ids_is_re_extracted_for_grouping(client, tmp_path):
    """A cache written before polygon IDs were stored can't be grouped: re-extract."""
    _run(client, group_by_field=False)
    assert _FakeGeoTessera.calls == 1
    # Rewrite the cache file the way pre-1.18.1 versions did (no groups).
    [path] = list(tmp_path.glob("result_*.npz"))
    d = dict(np.load(path, allow_pickle=True))
    d.pop("groups", None)
    np.savez_compressed(path, **d)

    _restart_server()
    start, ignored = _run(client, group_by_field=True)
    assert _FakeGeoTessera.calls == 2, "old cache must not be reused for a grouped run"
    assert start.get("group_by_field") is True and not ignored

    # ...and the rewritten cache now carries polygon IDs.
    _restart_server()
    start, ignored = _run(client, group_by_field=True)
    assert _FakeGeoTessera.calls == 2
    assert start.get("group_by_field") is True and not ignored


def test_old_cache_still_used_when_grouping_is_off(client, tmp_path):
    _run(client, group_by_field=False)
    [path] = list(tmp_path.glob("result_*.npz"))
    d = dict(np.load(path, allow_pickle=True))
    d.pop("groups", None)
    np.savez_compressed(path, **d)

    _restart_server()
    _run(client, group_by_field=False)
    assert _FakeGeoTessera.calls == 1


def test_in_memory_cache_without_groups_is_reloaded_for_grouping(client):
    _run(client, group_by_field=False)
    srv._tile_cache["groups"] = None  # e.g. filled from an old disk cache
    start, ignored = _run(client, group_by_field=True)
    # Reloaded past the in-memory entry; the disk cache (written by this
    # version) has the polygon IDs, so no re-download is needed.
    assert _FakeGeoTessera.calls == 1
    assert start.get("group_by_field") is True and not ignored


def test_disk_cache_round_trips_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    gdf = _make_gdf()
    vectors = np.zeros((4, EMBED_DIM), dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    groups = np.array([3, 3, 7, 8])
    srv._save_cached_result("habitat", 2024, gdf, vectors, labels, ["a", "b"], {}, groups=groups)
    *_, loaded = srv._load_cached_result("habitat", 2024, gdf)
    np.testing.assert_array_equal(loaded, groups)

    srv._save_cached_result("habitat", 2024, gdf, vectors, labels, ["a", "b"], {})
    *_, loaded = srv._load_cached_result("habitat", 2024, gdf)
    assert loaded is None
