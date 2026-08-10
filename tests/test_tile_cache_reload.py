"""Tests for tessera_eval.server._cached_tiles_need_reload.

Regression coverage: run_large_area's in-memory-cache-hit branch used to
force a reload by setting its local `vectors = None`, without also
invalidating _tile_cache["key"] -- so the later "actually reload from
GeoTessera" block (guarded by `_tile_cache["key"] != cache_key`) never
triggered either, since the key still matched. vectors stayed None all the
way through and crashed downstream at `len(vectors)` with
TypeError: object of type 'NoneType' has no len(), confirmed live on a
spatial_mlp request that hit a cache entry from a prior non-spatial run.
"""

from __future__ import annotations

from tessera_eval.server import _cached_tiles_need_reload


def test_no_reload_needed_when_nothing_extra_is_required():
    assert not _cached_tiles_need_reload(
        None,
        None,
        None,
        needs_spatial_3x3=False,
        needs_spatial_5x5=False,
        has_spatial_bboxes=False,
    )


def test_no_reload_needed_when_required_data_is_already_cached():
    assert not _cached_tiles_need_reload(
        "cached_3x3",
        "cached_5x5",
        "cached_points",
        needs_spatial_3x3=True,
        needs_spatial_5x5=True,
        has_spatial_bboxes=True,
    )


def test_reload_needed_when_spatial_3x3_required_but_not_cached():
    """The exact scenario from the live crash: a plain (non-spatial) run
    populated the cache, then a spatial_mlp run hits that same cache key."""
    assert _cached_tiles_need_reload(
        None,
        None,
        None,
        needs_spatial_3x3=True,
        needs_spatial_5x5=False,
        has_spatial_bboxes=False,
    )


def test_reload_needed_when_spatial_5x5_required_but_not_cached():
    assert _cached_tiles_need_reload(
        None,
        None,
        None,
        needs_spatial_3x3=False,
        needs_spatial_5x5=True,
        has_spatial_bboxes=False,
    )


def test_reload_not_needed_for_5x5_when_only_3x3_is_missing():
    """Each spatial resolution is checked independently -- a cache missing
    3x3 features doesn't force a reload for a request that only needs 5x5
    (already-cached) features."""
    assert not _cached_tiles_need_reload(
        None,
        "cached_5x5",
        None,
        needs_spatial_3x3=False,
        needs_spatial_5x5=True,
        has_spatial_bboxes=False,
    )


def test_reload_needed_when_spatial_bboxes_need_sample_points():
    assert _cached_tiles_need_reload(
        "cached_3x3",
        "cached_5x5",
        None,
        needs_spatial_3x3=False,
        needs_spatial_5x5=False,
        has_spatial_bboxes=True,
    )
