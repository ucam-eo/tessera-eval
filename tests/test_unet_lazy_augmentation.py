"""Tests for _AugmentedPatches, the lazy replacement for U-Net's eager
16x augmentation stack.

torch is an optional extra ([torch], not installed in CI) -- these tests
are skipped entirely when it isn't available, matching unet.py's own
_HAS_TORCH pattern.

Background: train_unet_on_patches / train_unet_regressor_on_patches used to
build all 16 augmentation variants (4 rotations x {as-is, +noise} x
{as-is, h-flipped}) of every patch as one eager np.stack before training
started. For a 128-dim, 256x256 patch that's 32 MiB per variant, so
len(patches) * 16 * 32 MiB up front -- confirmed live as a real OOM
("Unable to allocate 74.5 GiB for an array with shape (2384, 128, 256,
256)", Moustafa Eweda, hit via train_models()'s uncapped patch set, which
unlike the learning curve's per-pct 20-patch cap has no size limit at all).

_AugmentedPatches (a torch Dataset) generates the same 16 variants lazily
in __getitem__ instead, so peak memory is O(batch_size) rather than
O(len(patches) * 16). These tests check: (1) it reproduces the same 16
variants, in the same order, as the old eager algorithm (label arrays and
non-noisy geometric variants must match byte-for-byte -- only the noise on
the two noisy variants per rotation is allowed to differ, since it's now
independently seeded per sample instead of drawn from one sequential
stream); (2) it doesn't eagerly materialize the augmented set; (3) an
end-to-end smoke test that train_unet_on_patches (classification path)
still trains and predicts after the refactor -- the only path this
repo previously had no smoke test for at all (the regression path is
covered by test_unet_regression.py's roundtrip test).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera_eval.unet import _HAS_TORCH, extract_labelled_patches

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _synthetic_patches(n=3, dim=6, size=8, seed=0):
    rng = np.random.RandomState(seed)
    patches = []
    for _ in range(n):
        emb = rng.randn(size, size, dim).astype(np.float32)
        lbl = rng.randint(0, 4, size=(size, size)).astype(np.int64)
        patches.append((emb, lbl))
    return patches


def _old_eager_reference(patches, seed):
    """Exact replica of the pre-fix eager augmentation loop -- the oracle
    _AugmentedPatches must match for anything that isn't randomised noise."""
    rng_aug = np.random.RandomState(seed)
    emb_list, lbl_list = [], []
    for emb_patch, lbl_patch in patches:
        emb = emb_patch.transpose(2, 0, 1)
        lbl = lbl_patch.astype(np.int64)
        for k in range(4):
            emb_r = np.rot90(emb, k, axes=(1, 2)).copy()
            lbl_r = np.rot90(lbl, k, axes=(0, 1)).copy()
            emb_list.append(emb_r)
            lbl_list.append(lbl_r)
            noise = rng_aug.normal(0, 0.02, emb_r.shape).astype(np.float32)
            emb_list.append(emb_r + noise)
            lbl_list.append(lbl_r)
            emb_f = emb_r[:, :, ::-1].copy()
            lbl_f = lbl_r[:, ::-1].copy()
            emb_list.append(emb_f)
            lbl_list.append(lbl_f)
            noise = rng_aug.normal(0, 0.02, emb_f.shape).astype(np.float32)
            emb_list.append(emb_f + noise)
            lbl_list.append(lbl_f)
    return emb_list, lbl_list


class TestAugmentedPatchesDataset:
    def test_length_is_16x_input_patches(self):
        from tessera_eval.unet import _AugmentedPatches

        patches = _synthetic_patches(n=5)
        ds = _AugmentedPatches(patches, seed=42, label_dtype=np.int64)
        assert len(ds) == 5 * 16

    def test_does_not_eagerly_materialize_augmented_copies(self):
        """The whole point of the fix: constructing the Dataset must not
        build the 16x-larger augmented set -- it should just hold the
        original patches until something is actually indexed."""
        from tessera_eval.unet import _AugmentedPatches

        patches = _synthetic_patches(n=5)
        ds = _AugmentedPatches(patches, seed=42, label_dtype=np.int64)
        assert len(ds.patches) == 5
        assert ds.patches is patches

    def test_matches_old_eager_algorithm_index_for_index(self):
        """Labels never carry noise, so they must match the old eager path
        exactly at every index. Non-noisy geometric variants (rotation and
        rotation+flip, without the added-noise ones) must also match
        byte-for-byte -- same rotations, same flip, same order."""
        from tessera_eval.unet import _AugmentedPatches

        patches = _synthetic_patches(n=3)
        seed = 42
        old_emb, old_lbl = _old_eager_reference(patches, seed)
        ds = _AugmentedPatches(patches, seed, label_dtype=np.int64)

        assert len(ds) == len(old_emb)
        for i in range(len(ds)):
            new_emb, new_lbl = ds[i]
            new_emb, new_lbl = np.asarray(new_emb), np.asarray(new_lbl)
            assert np.array_equal(new_lbl, old_lbl[i]), f"label mismatch at {i}"
            is_noisy = (i % 4) % 2 == 1
            if not is_noisy:
                assert np.array_equal(new_emb, old_emb[i]), f"geometry mismatch at {i}"
            else:
                assert new_emb.shape == old_emb[i].shape

    def test_noise_only_touches_embeddings_never_labels(self):
        from tessera_eval.unet import _AugmentedPatches

        patches = _synthetic_patches(n=2)
        ds = _AugmentedPatches(patches, seed=1, label_dtype=np.int64)
        # every 4-block of variants for a given rotation has 2 identical
        # label arrays before the flip and 2 identical label arrays after
        for base in range(0, len(ds), 4):
            _, lbl_a = ds[base]
            _, lbl_b = ds[base + 1]
            assert np.array_equal(np.asarray(lbl_a), np.asarray(lbl_b))
            _, lbl_c = ds[base + 2]
            _, lbl_d = ds[base + 3]
            assert np.array_equal(np.asarray(lbl_c), np.asarray(lbl_d))

    def test_same_seed_is_fully_reproducible(self):
        from tessera_eval.unet import _AugmentedPatches

        patches = _synthetic_patches(n=3)
        a = _AugmentedPatches(patches, seed=7, label_dtype=np.int64)
        b = _AugmentedPatches(patches, seed=7, label_dtype=np.int64)
        for i in range(len(a)):
            ea, la = a[i]
            eb, lb = b[i]
            assert np.array_equal(np.asarray(ea), np.asarray(eb))
            assert np.array_equal(np.asarray(la), np.asarray(lb))

    def test_different_seed_changes_the_noise(self):
        from tessera_eval.unet import _AugmentedPatches

        patches = _synthetic_patches(n=3)
        a = _AugmentedPatches(patches, seed=7, label_dtype=np.int64)
        b = _AugmentedPatches(patches, seed=8, label_dtype=np.int64)
        noisy_idx = [i for i in range(len(a)) if (i % 4) % 2 == 1]
        assert any(
            not np.array_equal(np.asarray(a[i][0]), np.asarray(b[i][0])) for i in noisy_idx
        )

    def test_regression_targets_preserve_nan_through_rotation_and_flip(self):
        from tessera_eval.unet import _AugmentedPatches

        rng = np.random.RandomState(3)
        size, dim = 8, 4
        emb = rng.randn(size, size, dim).astype(np.float32)
        tgt = rng.randn(size, size).astype(np.float32)
        tgt[rng.random((size, size)) < 0.3] = np.nan
        n_nan = int(np.isnan(tgt).sum())

        ds = _AugmentedPatches([(emb, tgt)], seed=5, label_dtype=np.float32)
        for i in range(len(ds)):
            _, t = ds[i]
            assert int(np.isnan(np.asarray(t)).sum()) == n_nan


def _synthetic_labelled_tile(H=64, W=64, dim=6, seed=0):
    """A classification tile with a real, learnable 2-class signal (sign of
    the mean embedding), matching test_unet_regression.py's fixture style."""
    rng = np.random.RandomState(seed)
    tile_emb = rng.rand(H, W, dim).astype(np.float32) - 0.5
    class_raster = np.zeros((H, W), dtype=np.int32)
    region = tile_emb[16:48, 16:48].mean(axis=-1)
    class_raster[16:48, 16:48] = np.where(region > 0, 1, 2)
    return tile_emb, class_raster


class TestTrainUnetOnPatchesStillWorks:
    """End-to-end smoke test for the classification training path -- the
    only path this repo had no training smoke test for at all before this
    change. Catches wiring bugs the Dataset-level tests above can't, e.g.
    the in_channels lookup that used to come from the eager X tensor."""

    def test_train_and_predict_roundtrip_after_the_refactor(self):
        import torch

        from tessera_eval.unet import predict_unet_tile, train_unet_on_patches

        torch.manual_seed(42)

        tile_emb, class_raster = _synthetic_labelled_tile()
        patches = extract_labelled_patches(tile_emb, class_raster, patch_size=32, min_labelled=5)
        assert len(patches) > 0

        model = train_unet_on_patches(
            patches,
            n_classes=2,
            params={"epochs": 30, "depth": 2, "base_filters": 8, "batch_size": 4},
        )
        # n_classes + 1 outputs: index 0 is the ignore/background class
        assert model.out_conv.out_channels == 3

        pred = predict_unet_tile(model, tile_emb, patch_size=32, overlap=8)
        assert pred.shape == (64, 64)

        true_vals = class_raster[16:48, 16:48].flatten()
        pred_vals = pred[16:48, 16:48].flatten()
        from sklearn.metrics import f1_score

        macro_f1 = f1_score(true_vals, pred_vals, average="macro", zero_division=0)
        assert macro_f1 > 0.3, f"expected the model to learn some real signal, got F1={macro_f1:.3f}"
