"""Unit tests for server._spatial_block_groups -- the quantile-grid geographic
grouping used by the "spatial k-fold" option (see that flag's own docstring
in server.py, and CHANGELOG for the full motivation: k-fold's existing split
has no notion of location at all, so two points from neighbouring fields --
or the same field -- can land on opposite sides of a fold)."""

from __future__ import annotations

import numpy as np

from tessera_eval.server import _spatial_block_groups


def test_returns_correct_length_int_array():
    rng = np.random.RandomState(0)
    pts = rng.uniform(size=(200, 2))
    groups = _spatial_block_groups(pts, n_blocks=6)
    assert groups.shape == (200,)
    assert np.issubdtype(groups.dtype, np.integer)


def test_roughly_n_blocks_distinct_groups_for_well_spread_points():
    rng = np.random.RandomState(1)
    pts = rng.uniform(low=[0, 0], high=[10, 10], size=(500, 2))
    groups = _spatial_block_groups(pts, n_blocks=5)
    n_distinct = len(np.unique(groups))
    # The grid rounds up to a near-square shape (see docstring) -- close to
    # but not necessarily exactly n_blocks.
    assert 4 <= n_distinct <= 9


def test_deterministic_same_input_same_output():
    rng = np.random.RandomState(2)
    pts = rng.uniform(size=(100, 2))
    a = _spatial_block_groups(pts, n_blocks=4)
    b = _spatial_block_groups(pts, n_blocks=4)
    assert np.array_equal(a, b)


def test_all_points_at_same_location_collapse_to_one_group():
    pts = np.tile([5.0, 5.0], (50, 1))
    groups = _spatial_block_groups(pts, n_blocks=8)
    assert len(np.unique(groups)) == 1


def test_points_along_a_single_line_of_latitude():
    """lat is constant, lon varies -- must not crash when one axis has zero
    spread (the collapse-to-a-single-bin path for that axis)."""
    rng = np.random.RandomState(3)
    lons = rng.uniform(0, 10, size=60)
    pts = np.column_stack([lons, np.full(60, 3.0)])
    groups = _spatial_block_groups(pts, n_blocks=4)
    assert groups.shape == (60,)
    assert len(np.unique(groups)) >= 2  # lon still provides real separation


def test_tiny_point_count_does_not_crash():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    groups = _spatial_block_groups(pts, n_blocks=10)
    assert groups.shape == (3,)
    assert len(np.unique(groups)) <= 3


def test_group_sizes_are_reasonably_balanced_for_uniform_data():
    rng = np.random.RandomState(4)
    pts = rng.uniform(low=[0, 0], high=[10, 10], size=(1000, 2))
    groups = _spatial_block_groups(pts, n_blocks=5)
    counts = np.bincount(groups)
    counts = counts[counts > 0]
    # A quantile grid should keep block sizes within a few x of each other
    # for uniformly spread data -- not, say, one giant block and several
    # near-empty ones.
    assert counts.max() / counts.min() < 3.0
