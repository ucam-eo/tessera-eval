"""train_models() ("Download Models") for Spatial MLP.

Background: _tile_cache["spatial_3x3"]/["spatial_5x5"] are always None --
that key's same-request cache-hit meaning is unrelated (see _tile_cache's
own comment) and is never actually populated, by design, within a single
run_large_area request. train_models() used to read that same always-None
key, so its spatial_mlp/spatial_mlp_5x5 branches never ran: any Spatial MLP
"download" silently fell through to the generic branch and trained a
plain, non-windowed MLPClassifier on the raw pixel vectors under the
spatial name, with no error -- a downloaded model that didn't match what
the evaluation scored.

Fixed by stashing the real spatial_3x3/spatial_5x5 *and their own
spatial_labels* under new, dedicated _spatial_3x3/_spatial_5x5/
_spatial_labels_3x3/_spatial_labels_5x5 keys at the end of a successful
run_large_area (mirroring the pre-existing _unet_patches stash), and
reading those in train_models(). The point count of spatial_3x3 differs
from vectors/labels (patch-derived vs pixel-sampled), so it must be
paired with spatial_labels_3x3, not the pixel labels array -- these tests
use deliberately different lengths for the two so a reintroduced
labels/spatial_labels mix-up fails loudly (a sklearn length-mismatch
error) instead of silently training on misaligned data.
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.evaluate  # noqa: F401 -- registers the submodule
import tessera_eval.server as srv

sys.modules["tessera_eval.evaluate"]

EMBED_DIM = 128
N_SPATIAL = 60  # deliberately != the pixel sample count


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [object()]


class _FakeGeoTessera:
    def __init__(self, embeddings_dir=None):
        self.registry = _FakeRegistry()
        self._rng = np.random.RandomState(0)

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        # Used for the pixel-only fetch path (spatial not needed -- e.g. a
        # fixed test region, which skips spatial feature extraction).
        if progress_callback:
            progress_callback(len(points), len(points), "done")
        n = len(points)
        base = np.array([[p[0] - p[1]] for p in points], dtype=np.float32)
        noise = self._rng.normal(scale=0.05, size=(n, EMBED_DIM)).astype(np.float32)
        return base + noise


def _classification_gdf():
    classes = ["oak", "beech", "pine"] * 12  # 36 polygons -> more than N_SPATIAL pixel samples
    geoms = [box(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05) for i in range(len(classes))]
    return gpd.GeoDataFrame({"species": classes}, geometry=geoms, crs="EPSG:4326")


def _fake_extract(gt, gdf, field_name, year, le, n_classes, **kw):
    """Stand-in for _extract_tile_patches: point vectors sized to the real
    sample_points_lonlat (so run_large_area's own bookkeeping stays happy),
    plus a spatial_3x3 set of a *different* size (N_SPATIAL) with its own
    labels -- the realistic case, since spatial points are patch-derived.
    """
    pts = kw.get("sample_points_lonlat") or []
    n = len(pts)
    rng = np.random.RandomState(0)
    vectors = np.array([[p[0] - p[1]] for p in pts], dtype=np.float32) + rng.normal(
        scale=0.05, size=(n, EMBED_DIM)
    ).astype(np.float32)

    s3 = s5 = lbls = None
    if kw.get("needs_spatial_3x3"):
        lbls = np.array([0, 1, 2] * (N_SPATIAL // 3))
        s3 = (rng.normal(size=(N_SPATIAL, 9 * EMBED_DIM)) + lbls[:, None] * 3.0).astype(
            np.float32
        )
    if kw.get("needs_spatial_5x5"):
        lbls = np.array([0, 1, 2] * (N_SPATIAL // 3))
        s5 = (rng.normal(size=(N_SPATIAL, 25 * EMBED_DIM)) + lbls[:, None] * 3.0).astype(
            np.float32
        )
    return ([], s3, s5, vectors, lbls, lbls)


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _classification_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    monkeypatch.setattr(srv, "_extract_tile_patches", _fake_extract)
    return srv.app.test_client()


def _run_evaluation(client, **body):
    body.setdefault("field", "species")
    body.setdefault("task", "classification")
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 400)
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"run-large-area returned error event(s): {errors}"
    return events


def _train_models(client):
    resp = client.post("/api/evaluation/train-models", json={})
    assert resp.status_code == 200
    return [json.loads(line) for line in resp.text.strip().splitlines()]


def test_download_models_trains_a_real_spatial_mlp(client):
    _run_evaluation(client, classifiers=["spatial_mlp"])
    events = _train_models(client)

    failures = [
        e for e in events if e.get("event") == "status" and "Failed to train" in e.get("message", "")
    ]
    assert not failures, f"train-models reported a training failure: {failures}"

    ready = [e["classifier"] for e in events if e.get("event") == "model_ready"]
    assert "spatial_mlp" in ready

    import io

    import joblib

    resp = client.get("/api/evaluation/download-model/spatial_mlp")
    assert resp.status_code == 200
    bundle = joblib.load(io.BytesIO(resp.data))
    # 9x EMBED_DIM in, not EMBED_DIM -- the model trained on the windowed
    # spatial features, not the raw (un-windowed) pixel vectors. This is
    # exactly the assertion the original bug would fail: the silently
    # wrong model that used to get saved here had n_features_in_ == 128.
    assert bundle["model"].n_features_in_ == 9 * EMBED_DIM


def test_download_models_trains_a_real_spatial_mlp_5x5(client):
    _run_evaluation(client, classifiers=["spatial_mlp_5x5"])
    events = _train_models(client)

    failures = [
        e for e in events if e.get("event") == "status" and "Failed to train" in e.get("message", "")
    ]
    assert not failures, f"train-models reported a training failure: {failures}"

    ready = [e["classifier"] for e in events if e.get("event") == "model_ready"]
    assert "spatial_mlp_5x5" in ready

    import io

    import joblib

    resp = client.get("/api/evaluation/download-model/spatial_mlp_5x5")
    bundle = joblib.load(io.BytesIO(resp.data))
    assert bundle["model"].n_features_in_ == 25 * EMBED_DIM


def test_spatial_mlp_never_reaches_train_models_for_a_fixed_test_set(client):
    """A spatial/year/file split skips spatial feature extraction entirely
    (has_fixed_test_set), so run_large_area's own active_models filter
    already excludes spatial_mlp before train_models() ever sees it --
    confirming that path stays clean (no attempted, failed, or bogus
    spatial_mlp download) after this change."""
    _run_evaluation(
        client,
        classifiers=["spatial_mlp", "rf"],
        train_bboxes=[[0, 0, 1.8, 1.8]],
        test_bboxes=[[1.8, 1.8, 3.7, 3.7]],
    )
    events = _train_models(client)

    ready = [e["classifier"] for e in events if e.get("event") == "model_ready"]
    assert "spatial_mlp" not in ready
    assert "rf" in ready  # the pixel model still trains fine

    failures = [
        e for e in events if e.get("event") == "status" and "Failed to train" in e.get("message", "")
    ]
    assert not failures


def test_spatial_mlp_skips_cleanly_if_cached_spatial_data_is_ever_missing(client):
    """Defence in depth: even if active_models legitimately includes
    spatial_mlp (a successful spatial evaluation) but the cached spatial
    data is somehow absent by the time train_models() runs, it must skip
    with a clear message -- never fall through to the generic branch and
    silently train a plain MLP under the spatial name. Constructed
    directly (rather than via a real run) because a normal run always
    persists active_models and the spatial cache together, in the same
    request -- this simulates the cache having fallen out of sync."""
    _run_evaluation(client, classifiers=["spatial_mlp"])
    # Simulate the spatial stash going missing while active_models still
    # lists spatial_mlp (the state train_models() must defend against).
    srv._tile_cache["_spatial_3x3"] = None
    srv._tile_cache["_spatial_labels_3x3"] = None

    events = _train_models(client)

    ready = [e["classifier"] for e in events if e.get("event") == "model_ready"]
    assert "spatial_mlp" not in ready

    skips = [
        e
        for e in events
        if e.get("event") == "status" and "spatial_mlp skipped" in e.get("message", "")
    ]
    assert skips, f"expected a clean skip status for spatial_mlp, got: {events}"

    failures = [
        e for e in events if e.get("event") == "status" and "Failed to train" in e.get("message", "")
    ]
    assert not failures, f"must not attempt (and fail) to train spatial_mlp here: {failures}"
