"""_predict_raster's regression clamp (bug 8, Louis Driver).

MLP/XGBoost regressors extrapolate past the training targets' span
(negative canopy heights, biomass above anything observed) on embeddings
unlike their training set. create_map now passes the observed [min, max]
as clip_range so the dense raster stays physical; classification and the
unclamped default must be untouched.
"""

from __future__ import annotations

import numpy as np

from tessera_eval.server import _predict_raster


class _ConstReg:
    """Predicts a fixed value regardless of input -- lets a test force
    predictions to land outside any given clip_range."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(X.shape[0], self.value, dtype=np.float64)


class _RampReg:
    """Predicts the row index -- spans a wide, known range."""

    def predict(self, X):
        return np.arange(X.shape[0], dtype=np.float64)


def _emb(h=4, w=5, c=3, nan_at=None):
    e = np.ones((h, w, c), dtype=np.float32)
    if nan_at is not None:
        e[nan_at] = np.nan
    return e


def test_clip_range_clamps_regression_predictions_into_the_band():
    out = _predict_raster(
        _ConstReg(1000.0), _emb(), is_classification=False, clip_range=(0.0, 42.0)
    )
    valid = out[~np.isnan(out)]
    assert valid.size == 20
    assert np.all(valid == 42.0)  # every over-range prediction pinned to the ceiling


def test_clip_range_clamps_below_the_floor_too():
    out = _predict_raster(_ConstReg(-5.0), _emb(), is_classification=False, clip_range=(0.0, 42.0))
    valid = out[~np.isnan(out)]
    assert np.all(valid == 0.0)


def test_ramp_predictions_are_clipped_at_both_ends():
    # 20 pixels -> predictions 0..19; clip to [5, 12].
    out = _predict_raster(
        _RampReg(), _emb(4, 5, 3), is_classification=False, clip_range=(5.0, 12.0)
    )
    valid = np.sort(out[~np.isnan(out)])
    assert valid.min() == 5.0
    assert valid.max() == 12.0
    # interior values pass through unchanged
    assert set(valid.tolist()) == {5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}


def test_no_clip_range_leaves_out_of_range_predictions_untouched():
    out = _predict_raster(_ConstReg(1000.0), _emb(), is_classification=False, clip_range=None)
    valid = out[~np.isnan(out)]
    assert np.all(valid == 1000.0)


def test_nan_pixels_stay_nan_and_are_not_clamped():
    out = _predict_raster(
        _ConstReg(1000.0), _emb(nan_at=(1, 2)), is_classification=False, clip_range=(0.0, 42.0)
    )
    assert np.isnan(out[1, 2])
    assert np.all(out[~np.isnan(out)] == 42.0)


def test_clip_range_is_ignored_for_classification():
    class _Cls:
        def predict(self, X):
            return np.full(X.shape[0], 7, dtype=np.int64)

    # clip_range must not touch the classification branch (1-based class IDs).
    out = _predict_raster(_Cls(), _emb(), is_classification=True, clip_range=(0.0, 1.0))
    assert np.all(out == 8)  # 7 + 1 (1-based), unclamped
