"""U-Net learning-curve patch limits: uncapped by default, configurable.

The learning curve used to silently train U-Net on at most 20 patches and
test it on at most 10 -- ten 256x256 test patches can't contain most classes
of a 38-class habitat map, so most confusion-matrix rows were empty by
construction (Moustafa Eweda, 2026-10-05). Now it trains on pct% of the
patches and tests on all the rest, unless lc_max_train_patches /
lc_max_test_patches are set (shown in the Validation panel).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera_eval.unet as unet_mod
from tessera_eval.evaluate import _unet_lc_split_sizes, run_learning_curve


def test_split_is_uncapped_by_default():
    assert _unet_lc_split_sizes(147, 80) == (117, 30)
    assert _unet_lc_split_sizes(147, 50) == (73, 74)


def test_split_keeps_at_least_one_patch_each_side():
    assert _unet_lc_split_sizes(10, 1) == (1, 9)
    assert _unet_lc_split_sizes(10, 100) == (9, 1)


def test_split_honours_configured_limits():
    params = {"lc_max_train_patches": 20, "lc_max_test_patches": 10}
    assert _unet_lc_split_sizes(147, 80, params) == (20, 10)
    # A limit larger than what's available changes nothing.
    assert _unet_lc_split_sizes(147, 80, {"lc_max_test_patches": 500}) == (117, 30)


@pytest.fixture
def fake_unet(monkeypatch):
    """Record what U-Net is trained/tested on without needing PyTorch."""
    seen = {"train": [], "test": 0, "params": []}

    def fake_train(patches, n_classes, params, seed=42):
        seen["train"].append(len(patches))
        seen["params"].append(dict(params))
        return object()

    def fake_predict(model, emb_patch, patch_size=None):
        seen["test"] += 1
        return np.ones(emb_patch.shape[:2], dtype=np.int32)

    monkeypatch.setattr(unet_mod, "_HAS_TORCH", True)
    monkeypatch.setattr(unet_mod, "train_unet_on_patches", fake_train, raising=False)
    monkeypatch.setattr(unet_mod, "predict_unet_tile", fake_predict, raising=False)
    return seen


def _run(params):
    rng = np.random.RandomState(0)
    patches = [
        (
            rng.normal(size=(8, 8, 4)).astype(np.float32),
            rng.randint(1, 3, size=(8, 8)).astype(np.int32),
        )
        for _ in range(40)
    ]
    vectors = rng.normal(size=(200, 4)).astype(np.float32)
    labels = rng.randint(0, 2, size=200)
    events = list(
        run_learning_curve(
            vectors,
            labels,
            ["unet"],
            [50],
            repeats=1,
            classifier_params={"unet": params},
            unet_patches=patches,
        )
    )
    return [e["message"] for e in events if e.get("type") == "classifier_status"]


def test_learning_curve_uses_all_remaining_patches_by_default(fake_unet):
    messages = _run({"epochs": 1})
    assert fake_unet["train"] == [20]
    assert fake_unet["test"] == 20
    assert any("training unet on 20 patches, testing on 20" in m for m in messages)


def test_learning_curve_honours_limits_and_strips_them_from_training_params(fake_unet):
    _run({"epochs": 1, "lc_max_train_patches": 5, "lc_max_test_patches": 3})
    assert fake_unet["train"] == [5]
    assert fake_unet["test"] == 3
    assert "lc_max_train_patches" not in fake_unet["params"][0]
    assert "lc_max_test_patches" not in fake_unet["params"][0]
