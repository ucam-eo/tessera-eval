"""Tests for create_map() in regression mode.

Confirmed live (Louis Driver): create_map() was never adapted for
regression at all -- it unconditionally called make_classifier(classifier_
name, ...) and wrote predictions as uint8 with nodata=0. XGBoost's
classifier validates class labels strictly against a contiguous integer
range and crashed outright ("Invalid classes inferred from unique values of
y") the moment continuous height values were passed as y. k-NN/RF/MLP don't
validate that, so they didn't crash -- they silently trained as an
enormous multi-class classifier over what looked like ~40 arbitrary
"classes" (each distinct height value), then truncated real predictions to
uint8 and collided height=0 with the nodata sentinel. "Mapping is working"
for those was the dangerous outcome, not the safe one.

Fixed: create_map() now reads is_classification from the tile cache (set by
run_large_area, same mechanism as the Download Models fix in v1.5.1),
dispatches make_classifier/make_regressor via a UI-name -> "_reg"-suffixed
lookup (_CLF_TO_REG, hoisted to module level), and writes predictions/
GeoTIFF as float32 with NaN nodata for regression instead of uint8/0.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import rasterio
from affine import Affine

import tessera_eval.server as srv

try:
    import xgboost  # noqa: F401

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

_needs_xgboost = pytest.mark.skipif(not _HAS_XGBOOST, reason="xgboost not installed")

EMBED_DIM = 8


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [(year, 16.62, 48.22)]


class _FakeGeoTessera:
    """Embeddings vary per-pixel (not constant) so a regressor has something
    real to predict from, and so predictions vary too -- a constant map
    wouldn't distinguish "real values" from "degenerate single value"."""

    def __init__(self, embeddings_dir=None):
        self.registry = _FakeRegistry()

    def fetch_embeddings(self, tiles):
        def gen():
            for yr, _lon, _lat in tiles:
                rng = np.random.RandomState(1)
                emb = rng.uniform(0, 50, size=(16, 16, EMBED_DIM)).astype(np.float32)
                transform = Affine(0.001, 0, 16.6, 0, -0.001, 48.25)
                yield (None, None, None, emb, "EPSG:4326", transform)

        return gen()

    def fetch_mosaic_for_region(
        self, bbox, year=2024, target_crs="EPSG:4326", auto_download=True, progress_callback=None
    ):
        # create_map()'s NPY fallback calls this directly now (not
        # fetch_embeddings/registry.load_blocks_for_region) -- it's the one
        # that handles multi-tile merge + reprojection to a common CRS.
        rng = np.random.RandomState(1)
        emb = rng.uniform(0, 50, size=(16, 16, EMBED_DIM)).astype(np.float32)
        transform = Affine(0.001, 0, 16.6, 0, -0.001, 48.25)
        return emb, transform, target_crs


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "get_zarr", lambda: None)  # force the NPY fallback path
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)

    rng = np.random.RandomState(0)
    n = 200
    vectors = rng.uniform(0, 50, size=(n, EMBED_DIM)).astype(np.float32)
    # Real continuous heights, deliberately non-contiguous when sorted/cast
    # to int (e.g. a gap) -- this is exactly the shape that broke XGBoost's
    # classifier label validation.
    labels = rng.uniform(0.0, 42.0, size=n).astype(np.float32)
    monkeypatch.setattr(
        srv,
        "_tile_cache",
        {
            "key": ("height", 2025, 2025, "equal"),
            "vectors": vectors,
            "labels": labels,
            "class_names": [],
            "_model_params": {},
            "_is_classification": False,
        },
    )
    monkeypatch.setattr(srv, "_generated_maps", {})
    return srv.app.test_client()


def _run(client, **body):
    body.setdefault("classifier", "xgboost")
    body.setdefault("map_bboxes", [[48.2, 16.6, 48.25, 16.65]])
    resp = client.post("/api/evaluation/create-map", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"create-map returned error event(s): {errors}"
    return events


@pytest.mark.parametrize(
    "classifier",
    ["nn", "rf", pytest.param("xgboost", marks=_needs_xgboost), "mlp"],
)
def test_pixel_regressors_create_a_map_without_crashing(client, classifier):
    events = _run(client, classifier=classifier)
    ready = [e for e in events if e["event"] == "map_ready"]
    assert ready, f"{classifier}: no map_ready event"


@_needs_xgboost
def test_xgboost_no_longer_raises_invalid_classes(client):
    """The exact reported crash: XGBClassifier's fit-time label validation
    rejected continuous, non-contiguous height values."""
    events = _run(client, classifier="xgboost")
    messages = " ".join(e.get("message", "") for e in events)
    assert "Invalid classes" not in messages
    assert "Unknown classifier" not in messages


def test_map_geotiff_is_float32_with_nan_nodata_not_uint8_with_zero(client):
    events = _run(client, classifier="rf")
    ready = next(e for e in events if e["event"] == "map_ready")

    resp = client.get(ready["download_url"])
    assert resp.status_code == 200

    with rasterio.open(io.BytesIO(resp.data)) as ds:
        assert ds.dtypes[0] == "float32"
        assert ds.nodata is not None and np.isnan(ds.nodata)
        arr = ds.read(1)

    # Real predicted heights, not truncated/rank-like small integers, and
    # not the old classification convention (uint8, 1-based class IDs).
    valid = arr[~np.isnan(arr)]
    assert valid.size > 0
    assert valid.max() > 5, "predictions look truncated to tiny class-ID-like values"


def test_predictions_are_not_forced_into_a_class_taxonomy(client):
    """Even without crashing (nn/rf/mlp), the old code path trained a
    classifier over continuous values as if they were class IDs. Confirm
    the model actually trained is a regressor by checking predictions span
    a real continuous range rather than landing on a small fixed set of
    class-like integers."""
    events = _run(client, classifier="rf")
    ready = next(e for e in events if e["event"] == "map_ready")
    resp = client.get(ready["download_url"])
    with rasterio.open(io.BytesIO(resp.data)) as ds:
        arr = ds.read(1)
    valid = arr[~np.isnan(arr)]
    n_unique = len(np.unique(valid))
    assert n_unique > 5, "predictions collapsed to a handful of class-like values"
