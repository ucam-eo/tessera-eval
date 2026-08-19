"""Tests for tessera_eval.server._cached_tiles_need_reload.

Regression coverage: run_large_area's in-memory-cache-hit branch used to
force a reload by setting its local `vectors = None`, without also
invalidating _tile_cache["key"] -- so the later "actually reload from
GeoTessera" block (guarded by `_tile_cache["key"] != cache_key`) never
triggered either, since the key still matched. vectors stayed None all the
way through and crashed downstream at `len(vectors)` with
TypeError: object of type 'NoneType' has no len(), confirmed live on a
spatial_mlp request that hit a cache entry from a prior non-spatial run.

Also covers the same bug for U-Net (confirmed live, Louis Driver): running
any plain pixel regressor first (field/year/sampling unchanged, no U-Net
needed) cached unet_patches=[]. Selecting U-Net next, with the same cache
key, hit this in-memory cache -- and since the function didn't check
needs_unet at all, it reported "no reload needed", silently reusing the
empty patch list. U-Net then got filtered out of active_models entirely,
producing a 0-classifier run that finished suspiciously fast with no error.
"""

from __future__ import annotations

from tessera_eval.server import _cached_tiles_need_reload


def test_no_reload_needed_when_nothing_extra_is_required():
    assert not _cached_tiles_need_reload(
        None,
        None,
        None,
        None,
        needs_spatial_3x3=False,
        needs_spatial_5x5=False,
        needs_unet=False,
        has_spatial_bboxes=False,
    )


def test_no_reload_needed_when_required_data_is_already_cached():
    assert not _cached_tiles_need_reload(
        "cached_3x3",
        "cached_5x5",
        [("emb", "lbl")],
        "cached_points",
        needs_spatial_3x3=True,
        needs_spatial_5x5=True,
        needs_unet=True,
        has_spatial_bboxes=True,
    )


def test_reload_needed_when_spatial_3x3_required_but_not_cached():
    """The exact scenario from the live crash: a plain (non-spatial) run
    populated the cache, then a spatial_mlp run hits that same cache key."""
    assert _cached_tiles_need_reload(
        None,
        None,
        None,
        None,
        needs_spatial_3x3=True,
        needs_spatial_5x5=False,
        needs_unet=False,
        has_spatial_bboxes=False,
    )


def test_reload_needed_when_spatial_5x5_required_but_not_cached():
    assert _cached_tiles_need_reload(
        None,
        None,
        None,
        None,
        needs_spatial_3x3=False,
        needs_spatial_5x5=True,
        needs_unet=False,
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
        None,
        needs_spatial_3x3=False,
        needs_spatial_5x5=True,
        needs_unet=False,
        has_spatial_bboxes=False,
    )


def test_reload_needed_when_spatial_bboxes_need_sample_points():
    assert _cached_tiles_need_reload(
        "cached_3x3",
        "cached_5x5",
        [("emb", "lbl")],
        None,
        needs_spatial_3x3=False,
        needs_spatial_5x5=False,
        needs_unet=True,
        has_spatial_bboxes=True,
    )


def test_reload_needed_when_unet_required_but_cache_has_no_patches():
    """The exact live bug: a prior plain-regressor run cached
    unet_patches=[] (never needed patches). Selecting U-Net next, same
    cache key, must force a reload rather than silently training on zero
    patches."""
    assert _cached_tiles_need_reload(
        None,
        None,
        [],
        "cached_points",
        needs_spatial_3x3=False,
        needs_spatial_5x5=False,
        needs_unet=True,
        has_spatial_bboxes=False,
    )


def test_no_reload_needed_when_unet_patches_already_cached():
    assert not _cached_tiles_need_reload(
        None,
        None,
        [("emb", "lbl")],
        "cached_points",
        needs_spatial_3x3=False,
        needs_spatial_5x5=False,
        needs_unet=True,
        has_spatial_bboxes=False,
    )


def test_reload_not_needed_for_unet_when_unet_not_requested():
    """An empty cached patch list is fine when this request doesn't need
    U-Net at all -- only relevant when needs_unet is True."""
    assert not _cached_tiles_need_reload(
        None,
        None,
        [],
        "cached_points",
        needs_spatial_3x3=False,
        needs_spatial_5x5=False,
        needs_unet=False,
        has_spatial_bboxes=False,
    )
