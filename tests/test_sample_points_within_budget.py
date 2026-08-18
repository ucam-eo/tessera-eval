"""Tests for _sample_points_within_budget (server.py).

geopandas' sample_points(size=N) generates N points PER ROW, not N total.
run_large_area's point-generation used to draw from every row with a floor
of >=1 point/row regardless of how that compared to the requested budget --
confirmed live (Louis Driver): a 420,000-row shapefile with a 200,000-point
default budget generated ~420,000 points, ignoring the budget entirely
whenever there were more rows than the budget allowed. This is the smaller,
residual piece of that bug: the ~25x multiplicative part (many LabelEncoder
"classes" each independently hitting this floor) was already fixed in
v1.3.2 by not treating regression as N-way classification in the first
place; this fixes the same floor's remaining ~2x-scale overrun, for both
classification and regression.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from tessera_eval.server import _sample_points_within_budget


def _make_gdf(n_rows):
    """n_rows tiny, disjoint 1x1-degree squares side by side."""
    geoms = [box(i, 0, i + 1, 1) for i in range(n_rows)]
    return gpd.GeoDataFrame({"id": range(n_rows)}, geometry=geoms, crs="EPSG:4326")


def test_respects_budget_when_rows_exceed_it():
    """The actual bug: 1000 rows, budget of 100 -- must return ~100 points,
    not ~1000."""
    gdf = _make_gdf(1000)
    rng = np.random.RandomState(0)
    coords, row_idx = _sample_points_within_budget(gdf, budget=100, rng=rng)
    assert len(coords) == 100
    assert len(row_idx) == 100
    # Drawn from a genuine subset of rows, not literally every row.
    assert len(set(row_idx.tolist())) == 100


def test_subsampled_rows_are_not_just_the_first_n():
    """Row order can correlate with geography (e.g. a raster-to-polygon
    conversion scanning left-to-right) -- confirm the chosen rows aren't
    trivially rows[:budget]."""
    gdf = _make_gdf(1000)
    rng = np.random.RandomState(0)
    _coords, row_idx = _sample_points_within_budget(gdf, budget=50, rng=rng)
    assert sorted(row_idx.tolist()) != list(range(50))


def test_every_row_gets_at_least_one_point_when_rows_fit_in_budget():
    """When rows <= budget, behavior matches the original floor logic:
    every row contributes at least one point."""
    gdf = _make_gdf(10)
    rng = np.random.RandomState(0)
    coords, row_idx = _sample_points_within_budget(gdf, budget=100, rng=rng)
    assert len(coords) > 0
    assert set(row_idx.tolist()) == set(range(10))
    # budget=100 over 10 rows -> 10 points/row -> 100 points total.
    assert len(coords) == 100


def test_empty_gdf_returns_empty_arrays():
    gdf = _make_gdf(0)
    rng = np.random.RandomState(0)
    coords, row_idx = _sample_points_within_budget(gdf, budget=100, rng=rng)
    assert len(coords) == 0
    assert len(row_idx) == 0


def test_zero_budget_returns_empty_arrays():
    gdf = _make_gdf(10)
    rng = np.random.RandomState(0)
    coords, row_idx = _sample_points_within_budget(gdf, budget=0, rng=rng)
    assert len(coords) == 0
    assert len(row_idx) == 0
