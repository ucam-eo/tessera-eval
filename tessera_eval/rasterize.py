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


def rasterize_shapefile_continuous(gdf, field, transform, width, height):
    """Rasterize a shapefile field's real values onto a pixel grid (regression).

    The regression counterpart to rasterize_shapefile: burns the field's
    actual numeric values directly, not a LabelEncoder-assigned class ID.
    Deliberately a separate function rather than a mode flag on
    rasterize_shapefile -- that one's int32/1-based/0-as-nodata contract is
    baked into every caller (U-Net's patch code checks `> 0` for "labelled"),
    and mixing a float/NaN-as-nodata contract into the same function via a
    branch would risk breaking the classification path for a regression-only
    need.

    Args:
        gdf: GeoDataFrame with geometry and a numeric attribute column
        field: Name of the attribute column to use as the regression target
        transform: Affine transform mapping pixel coords to geographic coords
        width: Raster width in pixels
        height: Raster height in pixels

    Returns:
        float32 array, shape (height, width) -- NaN = nodata, everywhere
        else the real field value of whichever polygon covers that pixel.
    """
    valid = gdf.dropna(subset=[field])
    values = valid[field].astype(np.float64).to_numpy()

    shapes = list(zip(valid.geometry, values))

    target_raster = rasterio.features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype=np.float32,
        all_touched=True,
    )

    return target_raster


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
        if nodata_values:
            # Sentinels must become nodata before resampling: bilinear would
            # otherwise blend them into neighbouring pixels, producing large
            # in-between values that no longer match the sentinel exactly.
            source, src_transform = _read_window_with_sentinels_masked(
                src, band, transform, crs, width, height, nodata_values
            )
            if source is None:
                return aligned
            reproject(
                source=source,
                destination=aligned,
                src_transform=src_transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=resampling_enum,
            )
        else:
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

    return aligned


def _read_window_with_sentinels_masked(src, band, transform, crs, width, height, nodata_values):
    """Read the part of *src* covering the destination grid, with the
    raster's declared nodata and the extra sentinel values converted to NaN.

    Reading only the covering window (plus a small interpolation margin)
    keeps memory bounded by the destination grid, not the source raster.
    Returns (data, window_transform), or (None, None) when the destination
    grid lies entirely outside the raster.
    """
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    dst_bounds = array_bounds(height, width, transform)
    src_bounds = transform_bounds(crs, src.crs, *dst_bounds)
    window = from_bounds(*src_bounds, transform=src.transform)
    window = Window(window.col_off - 2, window.row_off - 2, window.width + 4, window.height + 4)
    try:
        window = window.intersection(Window(0, 0, src.width, src.height))
    except rasterio.errors.WindowError:
        return None, None
    window = window.round_offsets().round_lengths()
    if window.width < 1 or window.height < 1:
        return None, None

    data = src.read(band, window=window, masked=True).astype(np.float64).filled(np.nan)
    for v in nodata_values:
        data[data == v] = np.nan
    return data, src.window_transform(window)
