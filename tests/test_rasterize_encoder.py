"""Test rasterize_shapefile with a pre-fitted LabelEncoder (Fix 1)."""

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box
from sklearn.preprocessing import LabelEncoder

from tessera_eval.rasterize import rasterize_shapefile, rasterize_shapefile_continuous


@pytest.fixture
def sample_gdf():
    """GDF with three habitat types covering a small grid."""
    return gpd.GeoDataFrame(
        {
            "geometry": [
                box(0, 0, 5, 5),
                box(5, 0, 10, 5),
            ],
            "habitat": ["grassland", "woodland"],
        },
        crs="EPSG:4326",
    )


def test_prefitted_encoder_matches_ordering(sample_gdf):
    """A pre-fitted encoder whose classes are a superset still produces
    correct 1-based IDs that match the encoder's ordering."""
    le = LabelEncoder()
    # Superset: includes 'wetland' which is NOT in the GDF
    le.fit(["grassland", "wetland", "woodland"])

    transform = Affine(1, 0, 0, 0, -1, 10)  # 1 px = 1 unit

    raster = rasterize_shapefile(
        sample_gdf,
        "habitat",
        transform,
        width=10,
        height=10,
        label_encoder=le,
    )

    # Encoder ordering: grassland=0, wetland=1, woodland=2
    # 1-based: grassland=1, wetland=2, woodland=3
    grassland_id = le.transform(["grassland"])[0] + 1  # 1
    woodland_id = le.transform(["woodland"])[0] + 1  # 3

    # Left half (x 0-5) should be grassland
    assert raster[5, 2] == grassland_id
    # Right half (x 5-10) should be woodland
    assert raster[5, 7] == woodland_id
    # No wetland pixels should exist
    wetland_id = le.transform(["wetland"])[0] + 1  # 2
    assert wetland_id not in raster


def test_without_encoder_fits_locally(sample_gdf):
    """Without a pre-fitted encoder the function fits its own (backward-compat)."""
    transform = Affine(1, 0, 0, 0, -1, 10)
    raster = rasterize_shapefile(sample_gdf, "habitat", transform, width=10, height=10)
    # Should have exactly two class IDs plus 0 (nodata at edges if any)
    unique = set(np.unique(raster)) - {0}
    assert unique == {1, 2}


def test_rasterize_continuous_burns_real_values_not_class_ids():
    """Regression counterpart: the raster should carry the field's actual
    numeric values, not LabelEncoder ranks -- this is the piece that was
    missing for U-Net regression (patches would otherwise train against
    ranks, repeating the same bug fixed in run_large_area's point sampling
    for pixel classifiers)."""
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [box(0, 0, 5, 5), box(5, 0, 10, 5)],
            "height": [1.5, 3.2],
        },
        crs="EPSG:4326",
    )
    transform = Affine(1, 0, 0, 0, -1, 10)
    raster = rasterize_shapefile_continuous(gdf, "height", transform, width=10, height=10)

    assert raster.dtype == np.float32
    assert np.isclose(raster[5, 2], 1.5)  # left half
    assert np.isclose(raster[5, 7], 3.2)  # right half


def test_rasterize_continuous_fill_is_nan_not_zero():
    """NaN as the nodata sentinel (not 0) is the whole point -- 0 is a
    perfectly valid real target value (e.g. tree height at a bare patch),
    and would silently collide with "no data" if reused as the fill."""
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 5, 5)], "height": [0.0]},
        crs="EPSG:4326",
    )
    transform = Affine(1, 0, 0, 0, -1, 10)
    raster = rasterize_shapefile_continuous(gdf, "height", transform, width=10, height=10)

    # Covered pixel: real value 0.0, distinguishable from nodata.
    assert raster[5, 2] == 0.0
    assert not np.isnan(raster[5, 2])
    # Uncovered pixel: nodata.
    assert np.isnan(raster[5, 8])
