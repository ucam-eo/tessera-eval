"""Tests for DeepMLPClassifier (BatchNorm+Dropout+AdamW PyTorch MLP,
classify.py's "deep_mlp") and its wiring through make_classifier.

torch is an optional extra ([torch], not installed in CI) -- these tests are
skipped entirely when it isn't available, matching unet.py's own _HAS_TORCH
pattern. See deep_mlp.py's module docstring for the diagnosis this model
exists to address.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera_eval.deep_mlp import _HAS_TORCH

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _separable_classes(n=400, dim=16, k=4, seed=0):
    """Synthetic per-class Gaussian blobs -- a real, learnable signal, not
    noise, so a passing accuracy assertion means the model actually learned
    something."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, dim) * 3
    y = rng.randint(0, k, n)
    X = (rng.randn(n, dim) + centers[y]).astype(np.float32)
    return X, y


def test_fit_predict_learns_a_real_signal():
    from tessera_eval.deep_mlp import DeepMLPClassifier

    X, y = _separable_classes()
    clf = DeepMLPClassifier(hidden=(32, 16), epochs=20, batch_size=64, seed=42)
    clf.fit(X, y)
    pred = clf.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.8, f"expected the model to learn real signal, got acc={acc:.3f}"


def test_predict_proba_rows_sum_to_one_and_match_predict_argmax():
    from tessera_eval.deep_mlp import DeepMLPClassifier

    X, y = _separable_classes()
    clf = DeepMLPClassifier(hidden=(16,), epochs=10, batch_size=64, seed=1)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(clf.classes_[proba.argmax(axis=1)], clf.predict(X))


def test_classes_need_not_be_a_dense_zero_indexed_range():
    """Unlike the run_learning_curve caller convention (_fit_predict_relabeled
    always passes a dense 0..k-1 range), DeepMLPClassifier itself makes no
    such assumption -- classes_ is whatever np.unique(y) finds, and
    predict() reports labels in that original space."""
    from tessera_eval.deep_mlp import DeepMLPClassifier

    rng = np.random.RandomState(2)
    X = rng.randn(120, 8).astype(np.float32)
    y = rng.choice([5, 10, 20], size=120)
    clf = DeepMLPClassifier(hidden=(8,), epochs=5, batch_size=32, seed=2)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert set(np.unique(pred)) <= {5, 10, 20}
    assert list(clf.classes_) == [5, 10, 20]


def test_tiny_training_set_does_not_crash_batchnorm():
    """Fewer samples than max(20, 2*n_classes) skips the internal
    validation split entirely (see fit()'s docstring) -- and even the
    training loop itself must tolerate a final batch of size 1 without
    BatchNorm1d raising."""
    from tessera_eval.deep_mlp import DeepMLPClassifier

    rng = np.random.RandomState(3)
    X = rng.randn(5, 4).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0])
    clf = DeepMLPClassifier(hidden=(4,), epochs=3, batch_size=2, seed=3)
    clf.fit(X, y)  # must not raise
    pred = clf.predict(X)
    assert pred.shape == (5,)


def test_fit_raises_on_no_samples():
    from tessera_eval.deep_mlp import DeepMLPClassifier

    clf = DeepMLPClassifier(hidden=(4,), epochs=1)
    with pytest.raises(ValueError, match="No training samples"):
        clf.fit(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64))


def test_predict_before_fit_raises():
    from tessera_eval.deep_mlp import DeepMLPClassifier

    clf = DeepMLPClassifier(hidden=(4,), epochs=1)
    with pytest.raises(RuntimeError, match="fit"):
        clf.predict(np.zeros((3, 4), dtype=np.float32))


def test_seeded_runs_are_reproducible():
    X, y = _separable_classes(seed=5)
    from tessera_eval.deep_mlp import DeepMLPClassifier

    clf_a = DeepMLPClassifier(hidden=(16,), epochs=10, batch_size=64, seed=99)
    clf_a.fit(X, y)
    clf_b = DeepMLPClassifier(hidden=(16,), epochs=10, batch_size=64, seed=99)
    clf_b.fit(X, y)
    assert np.array_equal(clf_a.predict(X), clf_b.predict(X))


def test_make_classifier_builds_deep_mlp_with_hidden_layers_param():
    from tessera_eval.classify import make_classifier

    clf = make_classifier(
        "deep_mlp", {"hidden_layers": "64,32", "epochs": 3, "batch_size": 16}, seed=11
    )
    assert clf.hidden == (64, 32)
    assert clf.epochs == 3
    assert clf.seed == 11


def test_deep_mlp_listed_in_available_classifiers_when_torch_present():
    from tessera_eval.classify import available_classifiers

    assert "deep_mlp" in available_classifiers()


def test_make_classifier_deep_mlp_end_to_end_through_the_factory():
    """The exact call shape run_learning_curve/run_kfold_cv use: build via
    make_classifier, fit, predict -- no direct DeepMLPClassifier import."""
    from tessera_eval.classify import make_classifier

    X, y = _separable_classes(n=200, k=3, seed=6)
    clf = make_classifier(
        "deep_mlp", {"hidden_layers": "16,8", "epochs": 15, "batch_size": 32}, seed=6
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    assert (pred == y).mean() > 0.75
