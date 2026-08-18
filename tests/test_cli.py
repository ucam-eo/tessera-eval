"""Unit tests for the tessera-eval CLI (tessera_eval/cli.py).

Uses synthetic, in-memory fixtures only — no network access and no real
Tessera data, per the contributing guide. The --data fresh-download path
(load_embeddings_for_shapefile/load_embeddings_for_raster + GeoTessera) is
intentionally not covered for successful runs here, since it requires both;
validation-only behaviour (bad bbox, missing --year, etc.) is covered since
it never reaches the network.
"""

import sys
import types

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from tessera_eval.cli import _detect_source_kind, app

runner = CliRunner()

# ── Synthetic data fixtures ──


@pytest.fixture
def classification_npz(tmp_path, monkeypatch):
    """A cached vectors.npz for a 3-class classification problem, in an
    isolated working directory so it doesn't touch a real vectors.npz."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.RandomState(42)
    n_per_class = 30
    dim = 10
    vectors, labels = [], []
    for cls in range(3):
        center = rng.randn(dim) * 2
        vectors.append(center + rng.randn(n_per_class, dim) * 0.5)
        labels.extend([cls] * n_per_class)
    vectors = np.vstack(vectors).astype(np.float32)
    labels = np.array(labels)
    class_names = np.array(["a", "b", "c"], dtype=object)

    path = tmp_path / "vectors.npz"
    np.savez(
        path,
        vectors=vectors,
        labels=labels,
        class_names=class_names,
        source_data="synthetic.geojson",
        source_field="class",
        source_year=2024,
        source_kind="shapefile",
        task="classification",
    )
    return path


@pytest.fixture
def regression_npz(tmp_path, monkeypatch):
    """A cached vectors.npz for a regression problem, in an isolated
    working directory so it doesn't touch a real vectors.npz."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.RandomState(42)
    n, dim = 90, 10
    vectors = rng.randn(n, dim).astype(np.float32)
    weights = rng.randn(dim)
    labels = (vectors @ weights + rng.randn(n) * 0.5).astype(np.float32)
    class_names = np.array([], dtype=object)

    path = tmp_path / "vectors.npz"
    np.savez(
        path,
        vectors=vectors,
        labels=labels,
        class_names=class_names,
        source_data="synthetic.geojson",
        source_field="volume",
        source_year=2024,
        source_kind="shapefile",
        task="regression",
    )
    return path


@pytest.fixture
def fake_geotessera(monkeypatch):
    """Stub the geotessera package so raster/spatial-holdout code paths can
    run without the optional dependency installed or any network access."""
    fake_module = types.ModuleType("geotessera")

    class FakeGeoTessera:
        def __init__(self, *args, **kwargs):
            pass

    fake_module.GeoTessera = FakeGeoTessera
    monkeypatch.setitem(sys.modules, "geotessera", fake_module)
    return fake_module


@pytest.fixture
def raster_npz(tmp_path, monkeypatch):
    """A cached vectors.npz simulating `load`'s raster output."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.RandomState(0)
    vectors = rng.randn(60, 10).astype(np.float32)
    labels = rng.randint(0, 3, size=60)
    class_names = np.array(["1.0", "2.0", "3.0"], dtype=object)

    path = tmp_path / "vectors.npz"
    np.savez(
        path,
        vectors=vectors,
        labels=labels,
        class_names=class_names,
        source_data="fake.tif",
        source_field="",
        source_year=2020,
        source_kind="raster",
        source_bbox="27.0,67.0,27.1,67.1",
        source_band=1,
        source_nodata="",
        task="classification",
    )
    return path


def _fake_load_embeddings_for_raster(task_to_return, class_names=("1.0", "2.0", "3.0")):
    """Build a fake load_embeddings_for_raster returning small synthetic data,
    so learning_curve's branching can be tested without network access."""

    def _fake(
        raster_path, bbox, year, gt_instance, band=1, task=None, nodata_values=None, callback=None
    ):
        rng = np.random.RandomState(1)
        n = 40
        vectors = rng.randn(n, 10).astype(np.float32)
        if task_to_return == "regression":
            labels = rng.rand(n).astype(np.float32) * 100
            names = []
        else:
            labels = rng.randint(0, len(class_names), size=n)
            names = list(class_names)
        stats = {
            "tile_count": 1,
            "tiles_with_data": 1,
            "total_pixels": n,
            "n_classes": len(names) if names else None,
        }
        return vectors, labels, names, stats, task_to_return

    return _fake


# ── TestVersion ──


class TestVersion:
    def test_version_runs_and_prints_something(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.stdout.strip() != ""


# ── TestDetectSourceKind ──


class TestDetectSourceKind:
    def test_tif_is_raster(self):
        assert _detect_source_kind("foo.tif") == "raster"
        assert _detect_source_kind("foo.tiff") == "raster"

    def test_geojson_is_shapefile(self):
        assert _detect_source_kind("foo.geojson") == "shapefile"
        assert _detect_source_kind("foo.shp") == "shapefile"
        assert _detect_source_kind("foo.gpkg") == "shapefile"

    def test_unknown_extension_raises(self):
        with pytest.raises(typer.BadParameter):
            _detect_source_kind("foo.csv")


# ── TestLoadValidation ──


class TestLoadValidation:
    def test_invalid_bbox_format(self):
        result = runner.invoke(
            app, ["load", "--data", "fake.tif", "--year", "2020", "--bbox", "not,a,box"]
        )
        assert result.exit_code != 0
        assert "must be four comma-separated numbers" in result.output

    def test_bbox_out_of_range_for_degrees(self):
        # Values that look like metres (e.g. EPSG:3067), not lon/lat degrees.
        result = runner.invoke(
            app,
            [
                "load",
                "--data",
                "fake.tif",
                "--year",
                "2020",
                "--bbox",
                "500000,7517000,506000,7523000",
            ],
        )
        assert result.exit_code != 0
        assert "out of range for EPSG:4326" in result.output

    def test_bbox_required_for_raster(self):
        result = runner.invoke(app, ["load", "--data", "fake.tif", "--year", "2020"])
        assert result.exit_code != 0
        assert "--bbox is required" in result.output

    def test_year_required(self):
        result = runner.invoke(app, ["load", "--data", "fake.tif", "--bbox", "27.0,67.0,27.1,67.1"])
        assert result.exit_code != 0
        assert "--year is required" in result.output

    def test_field_required_for_shapefile(self):
        result = runner.invoke(app, ["load", "--data", "fake.geojson", "--year", "2020"])
        assert result.exit_code != 0
        assert "--field is required" in result.output


# ── TestKfoldClassification ──


class TestKfoldClassification:
    def test_runs_and_prints_macro_f1(self, classification_npz):
        result = runner.invoke(
            app, ["kfold", "--vectors", str(classification_npz), "--models", "nn"]
        )
        assert result.exit_code == 0
        assert "macro-F1" in result.stdout

    def test_confusion_summary_always_printed(self, classification_npz):
        result = runner.invoke(
            app, ["kfold", "--vectors", str(classification_npz), "--models", "nn"]
        )
        assert result.exit_code == 0
        assert "Confusion summary" in result.stdout

    def test_confusion_matrix_flag_adds_full_matrix(self, classification_npz):
        result = runner.invoke(
            app,
            [
                "kfold",
                "--vectors",
                str(classification_npz),
                "--models",
                "nn",
                "--confusion-matrix",
            ],
        )
        assert result.exit_code == 0
        assert "Confusion summary" in result.stdout
        assert "Confusion matrix" in result.stdout

    def test_confusion_matrix_absent_by_default(self, classification_npz):
        result = runner.invoke(
            app, ["kfold", "--vectors", str(classification_npz), "--models", "nn"]
        )
        assert "Confusion matrix" not in result.stdout

    def test_confusion_never_reports_self_confusion(self, classification_npz):
        result = runner.invoke(
            app, ["kfold", "--vectors", str(classification_npz), "--models", "nn"]
        )
        assert result.exit_code == 0
        for line in result.output.splitlines():
            if "most confused with" in line:
                class_name = line.split(":")[0].strip()
                confused_with = line.split("with ")[1].rstrip(")")
                assert confused_with != class_name, f"Self-confusion reported: {line}"


# ── TestKfoldRegression ──


class TestKfoldRegression:
    def test_runs_and_prints_r2(self, regression_npz):
        result = runner.invoke(
            app, ["kfold", "--vectors", str(regression_npz), "--models", "rf_reg"]
        )
        assert result.exit_code == 0
        assert "R²" in result.stdout

    def test_no_confusion_summary_for_regression(self, regression_npz):
        result = runner.invoke(
            app, ["kfold", "--vectors", str(regression_npz), "--models", "rf_reg"]
        )
        assert "Confusion summary" not in result.stdout


# ── TestLearningCurve ──


class TestLearningCurve:
    def test_runs_and_prints_macro_f1(self, classification_npz):
        result = runner.invoke(
            app,
            [
                "learning-curve",
                "--vectors",
                str(classification_npz),
                "--models",
                "rf",
                "--training-pcts",
                "50,100",
                "--repeats",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert "macro-F1" in result.stdout

    def test_regression_task_is_rejected(self, regression_npz):
        # run_learning_curve has no task parameter, so regression must be
        # rejected explicitly rather than silently producing wrong results.
        result = runner.invoke(
            app, ["learning-curve", "--vectors", str(regression_npz), "--models", "rf_reg"]
        )
        assert result.exit_code != 0
        assert "does not currently support regression" in result.output


# ── TestMissingCache ──


class TestMissingCache:
    def test_kfold_without_vectors_or_data_fails_clearly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["kfold"])
        assert result.exit_code != 0
        assert "No cached vectors found" in result.output

    def test_learning_curve_spatial_holdout_without_cache_fails_clearly(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["learning-curve", "--spatial-holdout"])
        assert result.exit_code != 0
        assert "No cached vectors found" in result.output


# ── TestLearningCurveRaster ──


class TestLearningCurveRaster:
    def test_spatial_holdout_raster_requires_bbox(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "learning-curve",
                "--data",
                "fake.tif",
                "--year",
                "2020",
                "--spatial-holdout",
                "--models",
                "rf",
            ],
        )
        assert result.exit_code != 0
        assert "requires --bbox" in result.output

    def test_spatial_holdout_raster_invalid_bbox(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "learning-curve",
                "--data",
                "fake.tif",
                "--year",
                "2020",
                "--spatial-holdout",
                "--bbox",
                "bad",
                "--models",
                "rf",
            ],
        )
        assert result.exit_code != 0
        assert "must be four comma-separated numbers" in result.output

    def test_spatial_holdout_raster_regression_rejected(
        self, raster_npz, fake_geotessera, monkeypatch
    ):
        monkeypatch.setattr(
            "tessera_eval.cli.load_embeddings_for_raster",
            _fake_load_embeddings_for_raster("regression"),
        )
        result = runner.invoke(app, ["learning-curve", "--spatial-holdout", "--models", "rf"])
        assert result.exit_code != 0
        assert "does not currently support regression" in result.output

    def test_spatial_holdout_raster_runs(self, raster_npz, fake_geotessera, monkeypatch):
        monkeypatch.setattr(
            "tessera_eval.cli.load_embeddings_for_raster",
            _fake_load_embeddings_for_raster("classification"),
        )
        result = runner.invoke(
            app,
            [
                "learning-curve",
                "--spatial-holdout",
                "--models",
                "rf",
                "--training-pcts",
                "50,100",
                "--repeats",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert "macro-F1" in result.output


# ── TestTestYear ──


class TestTestYear:
    def test_test_year_requires_test_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "learning-curve",
                "--data",
                "fake.geojson",
                "--field",
                "habitat",
                "--year",
                "2019",
                "--test-year",
                "2023",
                "--models",
                "rf",
            ],
        )
        assert result.exit_code != 0
        assert "requires --test-data" in result.output
