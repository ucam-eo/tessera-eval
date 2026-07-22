"""Unit tests for the tessera-eval CLI (tessera_eval/cli.py).

Uses synthetic, in-memory fixtures only — no network access and no real
Tessera data, per the contributing guide. The --data/--field fresh-download
path (load_embeddings_for_shapefile + GeoTessera) is intentionally not
covered here, since it requires both.
"""

import numpy as np
import pytest
from typer.testing import CliRunner

from tessera_eval.cli import app

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
        task="regression",
    )
    return path


# ── TestVersion ──


class TestVersion:
    def test_version_runs_and_prints_something(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.stdout.strip() != ""


# ── TestKfoldClassification ──


class TestKfoldClassification:
    def test_runs_and_prints_macro_f1(self, classification_npz):
        result = runner.invoke(
            app, ["kfold", "--vectors", str(classification_npz), "--models", "nn"]
        )
        assert result.exit_code == 0
        assert "macro-F1" in result.stdout

    def test_confusion_summary_printed_by_default(self, classification_npz):
        result = runner.invoke(
            app, ["kfold", "--vectors", str(classification_npz), "--models", "nn"]
        )
        assert "Confusion summary" in result.stdout

    def test_no_confusion_flag_suppresses_summary(self, classification_npz):
        result = runner.invoke(
            app,
            [
                "kfold",
                "--vectors",
                str(classification_npz),
                "--models",
                "nn",
                "--no-confusion",
            ],
        )
        assert result.exit_code == 0
        assert "Confusion summary" not in result.stdout


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
