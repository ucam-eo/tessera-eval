"""Tests for create_map's map_year (server.py).

Confirmed real need (Julia Jones / Deri, via Keshav, 2026-08-18): train a
model on one year (already supported, Validation pane), then use that
*already-trained* model as pure inference against a *different* year's
embeddings to produce a map -- e.g. train on 2025, map 2018, to look for
change over time. Distinct from the train/test-year Validation feature
(v1.3.0): that scores a classifier against held-out ground truth from a
different year at the *same labelled points* (evaluation); this is
unlabelled whole-area inference with no scoring at all. create_map()
already fetched embeddings tile-by-tile for the map bboxes and already had
a cached, trained model -- it just always used the *training* year for
that fetch. map_year, when given, overrides the fetch year independently
of the year the cached model was trained on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from affine import Affine

import tessera_eval.server as srv

EMBED_DIM = 8


class _FakeRegistry:
    def __init__(self, seen_years):
        self._seen_years = seen_years

    def load_blocks_for_region(self, bbox, year):
        self._seen_years.append(year)
        return [(year, 16.62, 48.22)]


class _FakeGeoTessera:
    """fetch_embeddings' embeddings encode the requested year (via the tile
    tuple's year, echoed straight through) so the test can assert exactly
    which year's data reached the prediction step, without needing a real
    embeddings store."""

    def __init__(self, embeddings_dir=None):
        self.seen_years = []
        self.registry = _FakeRegistry(self.seen_years)

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
    n = 200
    vectors = rng.rand(n, EMBED_DIM).astype(np.float32)
    labels = rng.randint(0, 2, size=n).astype(np.int32)
    monkeypatch.setattr(
        srv,
        "_tile_cache",
        {
            "key": ("habitat", 2025, 2025, "equal"),  # trained on 2025
            "vectors": vectors,
            "labels": labels,
            "class_names": ["grass", "water"],
            "_model_params": {},
        },
    )
    monkeypatch.setattr(srv, "_generated_maps", {})
    return srv.app.test_client()


def _run(client, **body):
    body.setdefault("classifier", "rf")
    body.setdefault("map_bboxes", [[48.2, 16.6, 48.25, 16.65]])  # single 0.05deg chunk
    resp = client.post("/api/evaluation/create-map", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"create-map returned error event(s): {errors}"
    return events


def test_map_year_defaults_to_training_year_when_omitted(client, monkeypatch):
    fake = _FakeGeoTessera()
    monkeypatch.setattr("geotessera.GeoTessera", lambda embeddings_dir=None: fake)

    events = _run(client)

    assert fake.seen_years == [2025]
    ready = next(e for e in events if e["event"] == "map_ready")
    assert ready["train_year"] == 2025
    assert ready["map_year"] == 2025


def test_map_year_override_fetches_the_requested_year_not_training_year(client, monkeypatch):
    fake = _FakeGeoTessera()
    monkeypatch.setattr("geotessera.GeoTessera", lambda embeddings_dir=None: fake)

    events = _run(client, map_year=2018)

    assert fake.seen_years == [2018], "must fetch map_year's embeddings, not the training year"
    ready = next(e for e in events if e["event"] == "map_ready")
    assert ready["train_year"] == 2025  # the model's actual training year, unchanged
    assert ready["map_year"] == 2018

    status_messages = " ".join(e.get("message", "") for e in events if e["event"] == "status")
    assert "2025" in status_messages and "2018" in status_messages


def test_map_ready_carries_an_in_browser_preview(client, monkeypatch):
    """map_ready includes a lat/lon PNG + legend for the viewer to overlay
    (feature 5). The GeoTIFF download is unaffected."""
    fake = _FakeGeoTessera()
    monkeypatch.setattr("geotessera.GeoTessera", lambda embeddings_dir=None: fake)

    events = _run(client)
    ready = next(e for e in events if e["event"] == "map_ready")

    preview = ready["preview"]
    assert preview is not None
    assert preview["png"].startswith("data:image/png;base64,")
    assert preview["is_classification"] is True
    (south, west), (north, east) = preview["bounds"]
    assert south < north and west < east
    # class_names from the tile cache flow into the legend labels
    labels = {item["label"] for item in preview["legend"]}
    assert labels <= {"grass", "water"} and labels
    # the .tif is still downloadable
    assert ready["download_url"].endswith(ready["name"])
