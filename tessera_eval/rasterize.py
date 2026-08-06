"""Rasterize shapefile polygons onto a pixel grid."""

import numpy as np
import rasterio.features
from sklearn.preprocessing import LabelEncoder


def rasterize_shapefile(gdf, field, transform, width, height, label_encoder=None):
    """Rasterize a shapefile field onto a pixel grid.

    Each polygon in the GeoDataFrame is burned into a raster using the
    specified attribute field as the class label. Class IDs are 1-based
    (0 = nodata).

    Args:
        gdf: GeoDataFrame with geometry and attribute columns
        field: Name of the attribute column to use as class labels
        transform: Affine transform mapping pixel coords to geographic coords
        width: Raster width in pixels
        height: Raster height in pixels
        label_encoder: Optional pre-fitted LabelEncoder. When provided,
            uses transform() instead of fit_transform(), ensuring consistent
            class IDs across tiles.

    Returns:
        int32 array, shape (height, width) — 0=nodata, 1..N=class IDs
    """
    valid = gdf.dropna(subset=[field])
    if label_encoder is not None:
        class_ids = label_encoder.transform(valid[field]) + 1  # 1-based (0 = nodata)
    else:
        le = LabelEncoder()
        class_ids = le.fit_transform(valid[field]) + 1  # 1-based (0 = nodata)

    shapes = list(zip(valid.geometry, class_ids))

    class_raster = rasterio.features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=True,
    )

    return class_raster


def align_raster_to_grid(
    raster_path, transform, crs, width, height, band=1, resampling="nearest", nodata_values=None
):
    """Reproject one band of a raster onto a target pixel grid.

    The raster-to-raster counterpart to rasterize_shapefile: instead of
    burning polygons onto a grid, this resamples an existing raster (e.g.
    a forest-inventory GeoTIFF) onto the exact grid described by transform,
    crs, width, and height.

    Args:
        raster_path: path to a GeoTIFF (or any rasterio-readable raster)
        transform: Affine transform of the destination grid
        crs: CRS of the destination grid
        width: destination grid width in pixels
        height: destination grid height in pixels
        band: 1-based band index to read (default 1)
        resampling: 'nearest' (for categorical data) or 'bilinear' (for
            continuous data)
        nodata_values: optional list of extra sentinel values to treat as
            missing, beyond the raster's own declared nodata

    Returns:
        float64 array, shape (height, width) — NaN where missing.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    resampling_enum = Resampling.nearest if resampling == "nearest" else Resampling.bilinear
    nodata_values = nodata_values or []

    aligned = np.full((height, width), np.nan, dtype=np.float64)
    with rasterio.open(raster_path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=resampling_enum,
        )
    for v in nodata_values:
        aligned[aligned == v] = np.nan

    return aligned
