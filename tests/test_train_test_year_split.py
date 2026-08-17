"""Tests for run_large_area's train/test-year split (server.py).

When test_year != train_year, the test role's points get their embeddings
re-fetched at test_year and fed into run_learning_curve's pre-existing
test_vectors/test_labels fixed-test-set mechanism -- see the "Train/test-year
split" block in run_large_area's stream() for the full rationale. These
tests exercise the actual /api/evaluation/run-large-area endpoint end to
end, with GeoTessera and run_learning_curve replaced by fakes:

- _FakeGeoTessera.sample_embeddings_at_points returns an array filled with
  the requested `year` (as a float), for every point -- so "did the test
  set come from test_year, not train_year" is a trivial equality check
  rather than something that needs a real embeddings store.
- run_learning_curve is replaced with a stub that just records the
  positional vectors/labels and the test_vectors/test_labels kwargs it was
  called with (an empty generator otherwise) -- these tests are about the
  wiring into that call, not about the ML pipeline itself, which has its
  own coverage elsewhere.
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

# `import tessera_eval.evaluate as ev` would silently bind `ev` to the
# `evaluate` *function* re-exported by tessera_eval/__init__.py's
# `from tessera_eval.evaluate import (..., evaluate, ...)`, not the
# `tessera_eval.evaluate` submodule -- `import a.b as x` is defined to mean
# `import a.b; x = a.b` (attribute access on `a`), and that attribute gets
# overwritten by the package's own `from .evaluate import evaluate` re-export.
# sys.modules keys by the real dotted path regardless, so use that instead.
import tessera_eval.evaluate  # noqa: F401 -- ensures it's registered in sys.modules
import tessera_eval.server as srv

ev = sys.modules["tessera_eval.evaluate"]

EMBED_DIM = 128


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        return [object()]  # only len() is used, by the tile-count stat


class _FakeGeoTessera:
    def __init__(self, embeddings_dir=None):
        self.registry = _FakeRegistry()

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        if progress_callback:
            progress_callback(len(points), len(points), "done")
        return np.full((len(points), EMBED_DIM), float(year), dtype=np.float32)


def _make_gdf():
    """Two classes in two disjoint regions, so bbox-based train/test splits
    can cleanly separate them: 'grass' in [0,0]-[1,1], 'water' in
    [10,10]-[11,11] (lon, lat)."""
    return gpd.GeoDataFrame(
        {"habitat": ["grass", "water"]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    srv.app.config["TESTING"] = True
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _make_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("geotessera.GeoTessera", _FakeGeoTessera)
    return srv.app.test_client()


@pytest.fixture
def captured_lc_call(monkeypatch):
    """Stub out run_learning_curve; capture how it was called."""
    captured = {}

    def _fake_run_learning_curve(vectors, labels, active_models, training_pcts, **lc_kwargs):
        captured["vectors"] = vectors
        captured["labels"] = labels
        captured["lc_kwargs"] = lc_kwargs
        return iter(())  # no progress events needed

    monkeypatch.setattr(ev, "run_learning_curve", _fake_run_learning_curve)
    return captured


def _run(client, **body):
    body.setdefault("field", "habitat")
    body.setdefault("classifiers", ["nn"])
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 20)
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"run-large-area returned error event(s): {errors}"
    return events


def test_same_year_is_unaffected(client, captured_lc_call):
    """train_year == test_year (the default): no test_vectors/test_labels at
    all, exactly today's behavior -- the year-split logic must be a no-op."""
    _run(client, train_year=2024, test_year=2024)
    assert "test_vectors" not in captured_lc_call["lc_kwargs"]
    assert "test_labels" not in captured_lc_call["lc_kwargs"]
    assert np.all(captured_lc_call["vectors"] == 2024.0)


def test_different_years_no_bboxes_uses_test_year_as_fixed_test_set(client, captured_lc_call):
    """Case 3: no bboxes drawn -- every point is both a training example (at
    train_year) and a test example (re-embedded at test_year)."""
    events = _run(client, train_year=2022, test_year=2023)

    lc_kwargs = captured_lc_call["lc_kwargs"]
    assert np.all(captured_lc_call["vectors"] == 2022.0), (
        "train pool must be train_year's embeddings"
    )
    assert np.all(lc_kwargs["test_vectors"] == 2023.0), "test set must be test_year's embeddings"
    # Same points serve both roles, so the label arrays must be the same length.
    assert len(lc_kwargs["test_labels"]) == len(captured_lc_call["labels"])

    start = next(e for e in events if e["event"] == "start")
    assert start["train_year"] == 2022
    assert start["test_year"] == 2023
    assert start.get("year_split") is True

    done = next(e for e in events if e["event"] == "done")
    assert done["train_year"] == 2022
    assert done["test_year"] == 2023


def test_different_years_with_bboxes_only_test_region_uses_test_year(client, captured_lc_call):
    """Case 4: bboxes drawn -- train region stays at train_year, test region
    is re-embedded at test_year instead of train_year."""
    _run(
        client,
        train_year=2022,
        test_year=2023,
        train_bboxes=[[0, 0, 1, 1]],  # south, west, north, east -- covers 'grass'
        test_bboxes=[[10, 10, 11, 11]],  # covers 'water'
    )

    lc_kwargs = captured_lc_call["lc_kwargs"]
    assert np.all(captured_lc_call["vectors"] == 2022.0), "train region must stay at train_year"
    assert np.all(lc_kwargs["test_vectors"] == 2023.0), "test region must use test_year"
    assert len(lc_kwargs["test_vectors"]) > 0
    assert len(captured_lc_call["vectors"]) > 0


def test_same_year_with_bboxes_is_unaffected(client, captured_lc_call):
    """train_year == test_year with bboxes drawn: today's existing spatial
    split, using train_year for both regions (year-split logic is a no-op)."""
    _run(
        client,
        train_year=2024,
        test_year=2024,
        train_bboxes=[[0, 0, 1, 1]],
        test_bboxes=[[10, 10, 11, 11]],
    )
    lc_kwargs = captured_lc_call["lc_kwargs"]
    assert np.all(captured_lc_call["vectors"] == 2024.0)
    assert np.all(lc_kwargs["test_vectors"] == 2024.0)
