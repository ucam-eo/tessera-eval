"""Tests for augment_spatial's size cap.

Background: augment_spatial's 4x flip expansion is materialized eagerly
(scikit-learn's .fit() needs the whole array at once, unlike U-Net's
PyTorch DataLoader -- see unet.py's _AugmentedPatches and
test_unet_lazy_augmentation.py for that version of this same problem).
train_models() ("Download Models") used to pass its full, uncapped cached
spatial point set straight through, which is a real OOM risk at scale --
the same shape of bug as the U-Net one, confirmed live for U-Net
("Unable to allocate 74.5 GiB...", Moustafa Eweda). augment_spatial now
subsamples down to `cap` rows (default DEFAULT_AUGMENT_CAP) before
augmenting, so every call site is protected without needing its own
capping logic, whether or not it already does its own (semantically
different) subsampling on top.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera_eval.classify import DEFAULT_AUGMENT_CAP, augment_spatial


def _points(n, window=3, dim=4, seed=0, classification=True):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, window * window * dim).astype(np.float32)
    y = rng.randint(0, 3, size=n) if classification else rng.randn(n).astype(np.float32)
    return X, y


class TestAugmentSpatialCap:
    def test_below_cap_is_unaffected(self):
        X, y = _points(20, window=3, dim=4)
        X_aug, y_aug = augment_spatial(X, y, window=3, dim=4, cap=10_000)
        assert X_aug.shape == (20 * 4, 9 * 4)
        assert y_aug.shape == (20 * 4,)

    def test_above_cap_is_subsampled_before_augmenting(self):
        X, y = _points(500, window=3, dim=4)
        X_aug, y_aug = augment_spatial(X, y, window=3, dim=4, cap=50)
        assert X_aug.shape == (50 * 4, 9 * 4)
        assert y_aug.shape == (50 * 4,)

    def test_default_cap_is_50_000(self):
        assert DEFAULT_AUGMENT_CAP == 50_000

    def test_default_cap_applies_when_not_overridden(self):
        X, y = _points(DEFAULT_AUGMENT_CAP + 500, window=3, dim=2)
        X_aug, y_aug = augment_spatial(X, y, window=3, dim=2)
        assert X_aug.shape[0] == DEFAULT_AUGMENT_CAP * 4
        assert y_aug.shape[0] == DEFAULT_AUGMENT_CAP * 4

    def test_cap_none_disables_it(self):
        X, y = _points(200, window=3, dim=4)
        X_aug, y_aug = augment_spatial(X, y, window=3, dim=4, cap=None)
        assert X_aug.shape[0] == 200 * 4

    def test_subsampling_is_deterministic_for_a_given_seed(self):
        X, y = _points(500, window=3, dim=4)
        a_X, a_y = augment_spatial(X, y, window=3, dim=4, cap=50, seed=7)
        b_X, b_y = augment_spatial(X, y, window=3, dim=4, cap=50, seed=7)
        assert np.array_equal(a_X, b_X)
        assert np.array_equal(a_y, b_y)

    def test_different_seed_picks_a_different_subsample(self):
        X, y = _points(500, window=3, dim=4)
        a_X, _ = augment_spatial(X, y, window=3, dim=4, cap=50, seed=1)
        b_X, _ = augment_spatial(X, y, window=3, dim=4, cap=50, seed=2)
        assert not np.array_equal(a_X, b_X)

    def test_works_with_float_regression_targets(self):
        X, y = _points(500, window=5, dim=3, classification=False)
        X_aug, y_aug = augment_spatial(X, y, window=5, dim=3, cap=40)
        assert X_aug.shape == (40 * 4, 25 * 3)
        assert y_aug.dtype == np.float32
        assert y_aug.shape == (40 * 4,)

    @pytest.mark.parametrize("cap", [1, 2, 10])
    def test_never_exceeds_cap_times_four_rows(self, cap):
        X, y = _points(1000, window=3, dim=2)
        X_aug, y_aug = augment_spatial(X, y, window=3, dim=2, cap=cap)
        assert X_aug.shape[0] == cap * 4
        assert y_aug.shape[0] == cap * 4
