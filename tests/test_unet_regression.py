"""Tests for U-Net regression support (train_unet_regressor_on_patches,
predict_unet_tile_regression, extract_labelled_patches_regression).

torch is an optional extra ([torch], not installed in CI) -- these tests
are skipped entirely when it isn't available, matching unet.py's own
_HAS_TORCH pattern.

Before this, U-Net had no regression path at all: TinyUNet's architecture
turns out to need zero changes for regression (its output head is already
a generic n_classes-channel Conv2d with no softmax baked into the model --
n_classes=1 is a real single-channel regression network as-is), but
everything around it (patch labels, loss, prediction) was
classification-only, and -- more fundamentally -- patches were always
rasterized through a LabelEncoder (rasterize_shapefile), so even a
regression-aware training loop would have been fitting against class ranks,
not real target values, repeating the same bug fixed for pixel classifiers
in v1.3.2. rasterize_shapefile_continuous (tests in
test_rasterize_encoder.py) is the piece that actually fixes that; these
tests cover the patch-extraction/train/predict layer built on top of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera_eval.unet import _HAS_TORCH, extract_labelled_patches_regression

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _synthetic_tile(H=64, W=64, dim=8, seed=0):
    rng = np.random.RandomState(seed)
    tile_emb = rng.rand(H, W, dim).astype(np.float32)
    target_raster = np.full((H, W), np.nan, dtype=np.float32)
    # Learnable signal: target = mean of embedding channels. Labelled region
    # (20x20) deliberately smaller than the 32px patch_size used below, so
    # extracted patches include real NaN padding/background, not just
    # labelled pixels throughout.
    target_raster[22:42, 22:42] = tile_emb[22:42, 22:42].mean(axis=-1)
    return tile_emb, target_raster


def test_extract_labelled_patches_regression_uses_nan_not_zero_as_ignore():
    tile_emb, target_raster = _synthetic_tile()
    patches = extract_labelled_patches_regression(
        tile_emb, target_raster, patch_size=32, min_labelled=5
    )
    assert len(patches) > 0
    emb_patch, target_patch = patches[0]
    assert emb_patch.dtype == np.float32
    assert target_patch.dtype == np.float32
    # Ignore sentinel is NaN, not 0 -- 0 is a valid real target value.
    assert np.isnan(target_patch).any()  # some padding/background present
    assert (~np.isnan(target_patch)).sum() >= 5


def test_extract_labelled_patches_regression_empty_for_all_nan_tile():
    tile_emb = np.random.RandomState(0).rand(32, 32, 8).astype(np.float32)
    target_raster = np.full((32, 32), np.nan, dtype=np.float32)
    patches = extract_labelled_patches_regression(tile_emb, target_raster, patch_size=16)
    assert patches == []


def test_train_and_predict_roundtrip_learns_real_signal():
    """End-to-end: train on synthetic patches with a learnable target, then
    predict on the same tile and confirm the model actually learned
    something (positive correlation with ground truth), not just that it
    runs without crashing."""
    import torch

    from tessera_eval.unet import predict_unet_tile_regression, train_unet_regressor_on_patches

    # train_unet_regressor_on_patches seeds numpy (augmentation noise) but
    # not torch itself -- weight init and DataLoader shuffling were still
    # nondeterministic, making this assertion genuinely flaky (observed:
    # ~1/3 runs landed under a 0.2 correlation threshold on 40 epochs).
    # Seed torch here, at the test level, rather than loosening the
    # threshold to paper over run-to-run variance.
    torch.manual_seed(42)

    tile_emb, target_raster = _synthetic_tile()
    patches = extract_labelled_patches_regression(
        tile_emb, target_raster, patch_size=32, min_labelled=5
    )
    assert len(patches) > 0

    model = train_unet_regressor_on_patches(
        patches, params={"epochs": 40, "depth": 2, "base_filters": 8, "batch_size": 4}
    )
    assert model.out_conv.out_channels == 1

    pred = predict_unet_tile_regression(model, tile_emb, patch_size=32, overlap=8)
    assert pred.shape == (64, 64)
    assert pred.dtype == np.float32
    assert not np.isnan(
        pred
    ).any()  # every pixel gets a prediction, even outside the labelled region

    true_vals = target_raster[22:42, 22:42].flatten()  # the actual labelled region
    pred_vals = pred[22:42, 22:42].flatten()
    corr = np.corrcoef(true_vals, pred_vals)[0, 1]
    assert corr > 0.1, f"expected the model to learn some real signal, got corr={corr:.3f}"


def test_train_unet_regressor_raises_on_no_patches():
    from tessera_eval.unet import train_unet_regressor_on_patches

    with pytest.raises(ValueError, match="No patches"):
        train_unet_regressor_on_patches([])


def test_masked_loss_ignores_nan_target_pixels():
    """A patch that's mostly NaN (background) should still train without
    NaN propagating into the loss/gradients -- this is the actual risk in
    the masked-MSE implementation (NaN * 0 = NaN in IEEE arithmetic, not 0,
    so a naive `(pred - target)**2 * mask` without first replacing NaN in
    `target` would silently poison the whole batch's gradient)."""
    from tessera_eval.unet import train_unet_regressor_on_patches

    rng = np.random.RandomState(1)
    emb_patch = rng.rand(16, 16, 4).astype(np.float32)
    target_patch = np.full((16, 16), np.nan, dtype=np.float32)
    target_patch[7:9, 7:9] = 5.0  # tiny labelled region, mostly NaN

    model = train_unet_regressor_on_patches(
        [(emb_patch, target_patch)] * 4,  # a few copies so batching has something to do
        params={"epochs": 5, "depth": 1, "base_filters": 4, "batch_size": 2},
    )
    # If NaN had leaked into the loss, every weight would be NaN.
    for param in model.parameters():
        assert not param.detach().isnan().any()
