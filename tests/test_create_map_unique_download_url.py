"""create_map's download URL must be unique per call.

map_name used to be just f"map_{bbox_idx + 1}" -- identical across every
create_map() call for the same bbox slot, so /api/evaluation/download-map/
map_1's URL never changed between generations. Confirmed harmless for the
normal frontend flow (it downloads immediately after each run's own "done"
event), but a real risk regardless: any cache keying purely on URL (browser,
proxy) has no reason to know a *different* file now lives behind it. Two
separate create_map() calls for the same bbox must now get distinct
download_url values, and the response must say not to cache it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from affine import Affine

import tessera_eval.server as srv

EMBED_DIM = 8


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [(year, 16.62, 48.22)]


class _FakeGeoTessera:
    def __init__(self, embeddings_dir=None):
        self.registry = _FakeRegistry()

    def fetch_embeddings(self, tiles):
        def gen():
            for yr, _lon, _lat in tiles:
                emb = np.full((16, 16, EMBED_DIM), float(yr), dtype=np.float32)
                transform = Affine(0.001, 0, 16.6, 0, -0.001, 48.25)
                yield (None, None, None, emb, "EPSG:4326", transform)

        return gen()

    def fetch_mosaic_for_region(self, *args, **kwargs):
        raise AssertionError(
            "create_map must not use fetch_mosaic_for_region: it reprojects "
            "embeddings before prediction"
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_zarr", lambda: None)  # force the NPY fallback path
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)

    rng = np.random.RandomState(0)
    n = 50
    vectors = rng.rand(n, EMBED_DIM).astype(np.float32)
    labels = rng.randint(0, 2, size=n).astype(np.int32)
    monkeypatch.setattr(
        srv,
        "_tile_cache",
        {
            "key": ("habitat", 2024, 2024, "equal"),
            "vectors": vectors,
            "labels": labels,
            "class_names": ["grass", "water"],
            "_model_params": {},
        },
    )
    monkeypatch.setattr(srv, "_generated_maps", {})
    return srv.app.test_client()


def _run(client):
    body = {"classifier": "rf", "map_bboxes": [[48.2, 16.6, 48.25, 16.65]]}
    resp = client.post("/api/evaluation/create-map", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"create-map returned error event(s): {errors}"
    return next(e for e in events if e["event"] == "map_ready")


def test_two_calls_for_the_same_bbox_get_different_download_urls(client):
    ready_a = _run(client)
    ready_b = _run(client)
    assert ready_a["download_url"] != ready_b["download_url"], (
        "repeat create_map() calls for the same bbox slot must not reuse a URL "
        "-- a stale intermediate cache could serve the wrong file"
    )


def test_download_response_says_not_to_cache(client):
    ready = _run(client)
    resp = client.get(ready["download_url"])
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == "no-store"
