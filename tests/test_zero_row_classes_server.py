"""Integration test: run-large-area surfaces an explicit status message
when a classifier's confusion matrix has a class with zero true test
examples (a "zero row") -- instead of a silent, misleading-looking zero
row that reads identically to "the model failed on this class". See
_zero_row_classes (tests/test_zero_row_classes.py) for the underlying
per-class check this wires up.

Uses a monkeypatched run_learning_curve/run_kfold_cv so the test exercises
server.py's own forwarding logic directly, rather than trying to
reconstruct the exact sampling conditions (a small/geographically
clustered class a capped, non-stratified patch sample misses) that
produce a real zero row -- see test_zero_row_classes_server.py's sibling,
test_learning_curve_spatial_confusion_matrix_sizing.py, for that scenario
from the matrix-*sizing* side.
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import tessera_eval.evaluate  # noqa: F401 -- registers the submodule in sys.modules
import tessera_eval.server as srv

# tessera_eval/__init__.py does `from tessera_eval.evaluate import (..., evaluate,
# ...)`, which rebinds the *package's* `evaluate` attribute to that function --
# so `import tessera_eval.evaluate as X` (an attribute lookup on the package)
# would bind X to the function, not the submodule. Go through sys.modules
# instead, which always holds the real submodule object.
evaluate_mod = sys.modules["tessera_eval.evaluate"]

EMBED_DIM = 8


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
        return self._rng.normal(size=(n, EMBED_DIM)).astype(np.float32)


def _two_class_gdf(n=20, ncols=5):
    # Alphabetical class names so LabelEncoder's fitted order is
    # unambiguous: "alpha" -> index 0, "zeta" -> index 1.
    classes = ["alpha", "zeta"] * (n // 2)
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
    monkeypatch.setattr(srv, "_get_merged_gdf", lambda: _two_class_gdf())
    monkeypatch.setattr(srv, "_tile_disk_cache_dir", tmp_path)
    monkeypatch.setattr(srv, "_geotessera_instance", None)
    monkeypatch.setattr(srv, "_tile_cache", {"key": None, "vectors": None})
    monkeypatch.setattr("tessera_eval.dataset.ZarrClient", _FakeGeoTessera)
    return srv.app.test_client()


def _run(client, **body):
    body.setdefault("field", "species")
    body.setdefault("task", "classification")
    body.setdefault("sampling", "equal")
    body.setdefault("max_training_samples", 200)
    body.setdefault("classifiers", ["rf"])
    resp = client.post("/api/evaluation/run-large-area", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().splitlines()]
    errors = [e for e in events if e.get("event") == "error"]
    assert not errors, f"run-large-area returned error event(s): {errors}"
    return events


def _fake_run_learning_curve_with_zeta_zero_row(*args, **kwargs):
    yield {
        "type": "progress",
        "pct": 100,
        "classifiers": {"rf": {"mean_f1": 0.9, "std_f1": 0.0, "mean_f1w": 0.9, "std_f1w": 0.0}},
        "pixel_train_count": 10,
        "unet_train_count": 0,
        "total_pixels": 30,
        "total_unet_pixels": 0,
    }
    yield {
        "type": "confusion_matrices",
        "confusion_matrices": {
            "rf": [
                [5, 1],  # "alpha": real test examples
                [0, 0],  # "zeta": zero -- no true test examples this run
            ]
        },
    }


def _fake_run_kfold_cv_with_zeta_zero_row(*args, **kwargs):
    yield {
        "type": "fold_result",
        "fold": 1,
        "models": {"rf": {"f1": 0.9, "f1w": 0.9}},
    }
    yield {
        "type": "aggregate",
        "models": {"rf": {"mean_f1": 0.9, "std_f1": 0.0, "mean_f1w": 0.9, "std_f1w": 0.0}},
    }
    yield {
        "type": "confusion_matrices",
        "confusion_matrices": {
            "rf": [
                [5, 1],
                [0, 0],
            ]
        },
    }


def test_zero_row_surfaces_a_status_message_naming_the_class_and_classifier(client, monkeypatch):
    monkeypatch.setattr(
        evaluate_mod, "run_learning_curve", _fake_run_learning_curve_with_zeta_zero_row
    )
    events = _run(client)
    statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
    assert any("rf" in m and "zeta" in m and "no test samples" in m for m in statuses), (
        f"expected a zero-row status message, got: {statuses}"
    )
    # The raw confusion matrix event itself is still forwarded unchanged --
    # this is an addition, not a replacement.
    cms = [e for e in events if e.get("event") == "confusion_matrices"]
    assert cms and cms[0]["confusion_matrices"]["rf"] == [[5, 1], [0, 0]]


def test_no_zero_row_means_no_extra_status_message(client, monkeypatch):
    def _fake_no_zero_row(*args, **kwargs):
        yield {
            "type": "progress",
            "pct": 100,
            "classifiers": {"rf": {"mean_f1": 0.9, "std_f1": 0.0, "mean_f1w": 0.9, "std_f1w": 0.0}},
            "pixel_train_count": 10,
            "unet_train_count": 0,
            "total_pixels": 30,
            "total_unet_pixels": 0,
        }
        yield {
            "type": "confusion_matrices",
            "confusion_matrices": {"rf": [[5, 1], [2, 4]]},
        }

    monkeypatch.setattr(evaluate_mod, "run_learning_curve", _fake_no_zero_row)
    events = _run(client)
    statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
    assert not any("no test samples" in m for m in statuses)


def test_zero_row_surfaces_in_kfold_mode_too(client, monkeypatch):
    monkeypatch.setattr(evaluate_mod, "run_kfold_cv", _fake_run_kfold_cv_with_zeta_zero_row)
    events = _run(client, eval_mode="kfold", kfold_k=2)
    statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
    assert any("rf" in m and "zeta" in m and "no test samples" in m for m in statuses), (
        f"expected a zero-row status message, got: {statuses}"
    )
