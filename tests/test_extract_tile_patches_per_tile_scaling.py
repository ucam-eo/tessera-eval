"""Regression test: patches_per_tile must scale with the max_patches
budget (floored at 5 for geographic diversity even on a small budget),
not sit at a fixed 5 regardless of the budget.

Background: a 2026-04-03 commit ("Shuffle tiles and cap at 5 patches per
tile for geographic diversity", cfe8578) fixed a real bug -- tiles were
fetched in spatial order, clustering patches geographically -- by
shuffling tile read order. In the same diff it accidentally flattened
`patches_per_tile = max(max_patches // total_tiles, 5)` down to a fixed
`patches_per_tile = 5`, silently breaking "increase Max patches" as a way
to get more spatial-model patch coverage. At the ~70-100 tile counts and
max_patches=500 default typical of a real run, the two formulas land
within about 1 of each other (500 // 73 = 6 vs. the floor of 5) -- close
enough that this went unnoticed for months. Confirmed live (Moustafa
Eweda, 2026-09-29): raising Max patches 500 -> 600 -> 700 changed total
patches extracted by almost nothing (147 -> 144 -> 145 across three real
runs, all effectively pinned at total_tiles * 5).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

import tessera_eval.server as srv

EMBED_DIM = 8
TILE_SIZE = 64
N_TILES = 4


class _FakeRegistry:
    def load_blocks_for_region(self, bbox, year):
        # N_TILES distinct locations -- distinct enough to be treated as
        # separate tiles, identical content is fine for a pure
        # patch-count-scaling test (no geographic realism needed here).
        return [(year, 16.6 + i * 0.2, 48.3) for i in range(N_TILES)]


class _FakeGeoTessera:
    """Yields a *distinct* transform per tile, anchored at that tile's own
    lon -- each fake tile's geographic footprint must actually move to
    match _make_gdf's per-tile polygons, or every "tile" ends up
    geo-referenced to the same patch of ground and only one polygon ever
    intersects (confirmed the hard way: an earlier version of this fixture
    reused one shared transform for all N_TILES, silently making every
    test collapse to "only tile 0 ever contributes any patches")."""

    def __init__(self, tile_emb, crs="EPSG:4326"):
        self.registry = _FakeRegistry()
        self._tile_emb = tile_emb
        self._crs = crs

    def fetch_embeddings(self, tiles, clip_bbox=None):
        def gen():
            for _yr, lon, _lat in tiles:
                transform = Affine(0.001, 0, lon, 0, -0.001, 48.35)
                yield (None, None, None, self._tile_emb, self._crs, transform)

        return gen()


@pytest.fixture
def fake_tiles(monkeypatch):
    rng = np.random.RandomState(0)
    tile_emb = rng.rand(TILE_SIZE, TILE_SIZE, EMBED_DIM).astype(np.float32)
    return tile_emb


def _make_gdf():
    # A big polygon covering nearly the whole tile at every one of
    # _FakeRegistry's N_TILES locations, so every tile has abundant
    # valid patch centers (far more than any patches_per_tile value this
    # test exercises) -- the constraint under test is purely the
    # per-tile cap, not running out of labelled area.
    geoms = [box(16.6 + i * 0.2, 48.29, 16.6 + i * 0.2 + 0.06, 48.35) for i in range(N_TILES)]
    return gpd.GeoDataFrame({"height": [3.7] * N_TILES}, geometry=geoms, crs="EPSG:4326")


def _total_patches(fake_tiles, max_patches):
    tile_emb = fake_tiles
    gt = _FakeGeoTessera(tile_emb)
    gdf = _make_gdf()

    unet_patches, *_ = srv._extract_tile_patches(
        gt,
        gdf,
        "height",
        2024,
        le=None,
        n_classes=0,
        patch_size=16,
        max_patches=max_patches,
        is_classification=False,
    )
    return len(unet_patches)


def test_small_budget_still_gets_the_five_per_tile_floor(fake_tiles):
    # max_patches // N_TILES < 5 here (20 // 4 = 5, right at the floor) --
    # use something clearly below to exercise the floor unambiguously.
    total = _total_patches(fake_tiles, max_patches=8)
    # 8 budget, floor of 5/tile -- capped by the overall max_patches, not
    # starved by an under-sized floor.
    assert total == 8


def test_larger_budget_draws_more_than_the_old_fixed_five_per_tile(fake_tiles):
    """The actual regression test: at a budget well above
    total_tiles * 5 (4 tiles * 5 = 20), more patches must actually be
    drawn -- the pre-fix code would have returned exactly 20 here
    regardless of how much higher max_patches went."""
    total = _total_patches(fake_tiles, max_patches=80)
    # patches_per_tile = max(5, 80 // 4) = 20, so up to 4*20 = 80 --
    # bounded only by max_patches itself here since the fixture's
    # labelled area comfortably exceeds 20 valid centers per tile.
    assert total == 80


def test_patch_count_scales_with_max_patches_not_flat(fake_tiles):
    low = _total_patches(fake_tiles, max_patches=40)
    high = _total_patches(fake_tiles, max_patches=120)
    assert high > low, (
        f"expected more patches at a higher max_patches budget, got {low} then {high} "
        "-- patches_per_tile may have regressed back to a fixed 5/tile"
    )
