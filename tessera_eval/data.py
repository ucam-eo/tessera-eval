"""Load and dequantize Tessera embeddings from various formats."""

import gzip
import io
import json
import logging
import math
from pathlib import Path

import numpy as np
from rasterio.transform import array_bounds as _array_bounds
from shapely.geometry import box as _box

logger = logging.getLogger(__name__)

_M_PER_DEG_LAT = 111_320.0  # metres per degree of latitude (WGS84, approx)


def dequantize_uint8(quantized, dim_min, dim_max):
    """Convert uint8 embeddings to float32 vectors using per-dimension min/max.

    This is TEE's quantization format: uint8 values in [0, 255] mapped to
    [dim_min, dim_max] per dimension.

    Args:
        quantized: uint8 array, shape (N, 128) or (H, W, 128)
        dim_min: float32 array, shape (128,)
        dim_max: float32 array, shape (128,)

    Returns:
        float32 array, same shape as quantized
    """
    dim_min = np.asarray(dim_min, dtype=np.float32)
    dim_max = np.asarray(dim_max, dtype=np.float32)
    dim_scale = dim_max - dim_min
    dim_scale[dim_scale == 0] = 1
    return quantized.astype(np.float32) / 255.0 * dim_scale + dim_min


def dequantize_int8(quantized, scales):
    """Convert int8 embeddings to float32 vectors using per-pixel scales.

    This is GeoTessera's quantization format: int8 values multiplied by
    float32 scale factors.

    Args:
        quantized: int8 array, shape (H, W, 128)
        scales: float32 array, shape (H, W) or (H, W, 128)

    Returns:
        float32 array, shape (H, W, 128)
    """
    if scales.ndim == 2 and quantized.ndim == 3:
        scales = scales[..., np.newaxis]
    return quantized.astype(np.float32) * scales


def load_tee_vectors(vector_dir):
    """Load dequantized float32 vectors from TEE's vector directory format.

    Reads: all_embeddings_uint8.npy.gz, quantization.json, pixel_coords.npy.gz,
    metadata.json from the given directory.

    Args:
        vector_dir: Path to directory containing TEE vector files
            (e.g., '/data/vectors/cambridge/2024')

    Returns:
        Tuple of (vectors, coords, metadata):
        - vectors: float32 array, shape (N, 128)
        - coords: int32 array, shape (N, 2) — pixel (x, y) coordinates
        - metadata: dict with geotransform, mosaic dimensions, etc.

    Raises:
        FileNotFoundError: If required files are missing
    """
    vector_dir = Path(vector_dir)

    emb_path = vector_dir / "all_embeddings_uint8.npy.gz"
    quant_path = vector_dir / "quantization.json"
    coords_path = vector_dir / "pixel_coords.npy.gz"
    meta_path = vector_dir / "metadata.json"

    for p in [emb_path, quant_path, coords_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing vector file: {p}")

    with open(quant_path) as f:
        quant = json.load(f)
    dim_min = np.array(quant["dim_min"], dtype=np.float32)
    dim_max = np.array(quant["dim_max"], dtype=np.float32)

    with gzip.open(emb_path, "rb") as f:
        quantized = np.load(io.BytesIO(f.read()))

    vectors = dequantize_uint8(quantized, dim_min, dim_max)

    with gzip.open(coords_path, "rb") as f:
        coords = np.load(io.BytesIO(f.read()))

    with open(meta_path) as f:
        metadata = json.load(f)

    return vectors, coords, metadata


def load_geotessera_tile(embedding_path, scales_path):
    """Load a single GeoTessera tile and dequantize.

    Args:
        embedding_path: Path to .npy file (int8, shape H×W×128)
        scales_path: Path to _scales.npy file (float32)

    Returns:
        float32 array, shape (H, W, 128)
    """
    quantized = np.load(embedding_path)
    scales = np.load(scales_path)
    return dequantize_int8(quantized, scales)


def load_embeddings_for_shapefile(gdf, field, year, gt_instance, callback=None):
    """Load embeddings tile-by-tile for all pixels overlapping a shapefile.

    Memory-bounded: processes one GeoTessera tile at a time, only accumulates
    labelled pixels. Suitable for large-area (county/country) shapefiles.

    Args:
        gdf: GeoDataFrame with geometry and the target field (EPSG:4326)
        field: Name of the attribute column to use as labels
        year: Year of embeddings to load
        gt_instance: GeoTessera instance (with registry and embeddings_dir)
        callback: Optional function(current_tile, total_tiles) for progress

    Returns:
        Tuple of (vectors, labels, class_names, stats) where:
        - vectors: float32 array, shape (N, 128)
        - labels: int array, shape (N,) — 0-indexed class labels
        - class_names: list of str — class name for each label index
        - stats: dict with tile_count, total_pixels, etc.

    Raises:
        ValueError: If no labelled pixels found
    """
    from sklearn.preprocessing import LabelEncoder

    from tessera_eval.rasterize import rasterize_shapefile

    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

    tiles = gt_instance.registry.load_blocks_for_region(bbox, year)
    total_tiles = len(tiles)
    if total_tiles == 0:
        raise ValueError(f"No GeoTessera tiles found for bbox {bbox}, year {year}")

    # Fit label encoder on the full shapefile
    le = LabelEncoder()
    le.fit(gdf[field].dropna().unique())
    class_names = le.classes_.tolist()

    all_vectors = []
    all_labels = []
    tiles_with_data = 0

    for tile_idx, (yr, tile_lon, tile_lat, tile_emb, tile_crs, tile_transform) in enumerate(
        gt_instance.fetch_embeddings(tiles)
    ):
        if callback:
            callback(tile_idx + 1, total_tiles)

        h, w, dim = tile_emb.shape

        # Reproject GDF to tile CRS, then filter to tile bbox
        tile_bounds = _array_bounds(h, w, tile_transform)
        gdf_proj = gdf.to_crs(tile_crs) if gdf.crs != tile_crs else gdf
        tile_gdf = gdf_proj[gdf_proj.intersects(_box(*tile_bounds))]
        if tile_gdf.empty:
            continue

        # Rasterize shapefile onto this tile's grid
        class_raster = rasterize_shapefile(tile_gdf, field, tile_transform, w, h, label_encoder=le)

        # Extract labelled pixels
        labelled_mask = class_raster > 0
        n_labelled = int(labelled_mask.sum())
        if n_labelled == 0:
            continue

        tiles_with_data += 1
        # class_raster is 1-based (from rasterize_shapefile), convert to 0-based
        tile_labels = class_raster[labelled_mask] - 1
        tile_vectors = tile_emb[labelled_mask]  # (n_labelled, 128)

        all_vectors.append(tile_vectors)
        all_labels.append(tile_labels)

    if not all_vectors:
        raise ValueError("No labelled pixels found across any tiles")

    vectors = np.concatenate(all_vectors, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)

    stats = {
        "tile_count": total_tiles,
        "tiles_with_data": tiles_with_data,
        "total_pixels": len(labels),
        "n_classes": len(class_names),
    }

    return vectors, labels, class_names, stats


def load_embeddings_for_shapefile_vq(
    gdf,
    field,
    year,
    client,
    *,
    max_km=10.0,
    target_crs="EPSG:4326",
    callback=None,
):
    """Load labelled embeddings via a VQ bolt-on (or any mosaic-fetching client).

    Like :func:`load_embeddings_for_shapefile`, but instead of iterating raw
    GeoTessera tiles it pulls **reconstructed** embeddings region-by-region from
    ``client.fetch_mosaic_for_region(bbox, year, target_crs)``. Use it to evaluate
    downstream accuracy on VQ-reconstructed embeddings (the cost of compression),
    versus the raw-tile loader (the reference).

    The shapefile's bounding box is split into ``<= max_km`` chunks because the VQ
    bolt-on caps the bbox per request; chunks the polygons don't touch are skipped
    without a fetch, and chunks the bolt-on has no coverage for are skipped with a
    warning. Class IDs are consistent across chunks (one shared ``LabelEncoder``).

    Args:
        gdf: GeoDataFrame with geometry + the ``field`` column (any CRS).
        field: attribute column used as labels.
        year: embedding year.
        client: any object exposing
            ``fetch_mosaic_for_region(bbox, year, target_crs) -> (mosaic, transform, crs)``
            where ``mosaic`` is ``(H, W, 128)`` float32 in ``target_crs``. Both
            ``tessera_vq.VQTessera`` (the VQ bolt-on) and ``geotessera.GeoTessera``
            satisfy this; pass a ``VQTessera`` for the VQ path. (Not imported here —
            construct it yourself, so tessera-eval keeps no tessera-vq dependency.)
        max_km: max chunk side in km (default 10, matching the bolt-on's default cap;
            a 10% safety margin is applied).
        target_crs: CRS for the fetched mosaics (default EPSG:4326).
        callback: optional ``function(current_chunk, total_chunks)`` for progress.

    Returns:
        ``(vectors, labels, class_names, stats)`` — same contract as
        :func:`load_embeddings_for_shapefile`. ``stats`` has ``chunk_count``,
        ``chunks_with_data``, ``total_pixels``, ``n_classes``.

    Raises:
        ValueError: if no labelled pixels are recovered from any chunk.
    """
    from sklearn.preprocessing import LabelEncoder

    from tessera_eval.rasterize import rasterize_shapefile

    gdf4326 = gdf if _is_4326(gdf.crs) else gdf.to_crs(target_crs)
    west, south, east, north = (float(v) for v in gdf4326.total_bounds)

    # Chunk side in degrees, kept under max_km on both axes (lon shrinks with lat).
    midlat = (south + north) / 2.0
    span_m = max_km * 1000.0 * 0.9  # 10% margin under the server cap
    dlat = span_m / _M_PER_DEG_LAT
    dlon = span_m / (_M_PER_DEG_LAT * max(math.cos(math.radians(midlat)), 1e-6))

    chunks = []
    lat = south
    while lat < north:
        lon = west
        while lon < east:
            chunks.append((lon, lat, min(lon + dlon, east), min(lat + dlat, north)))
            lon += dlon
        lat += dlat

    le = LabelEncoder()
    le.fit(gdf4326[field].dropna().unique())
    class_names = le.classes_.tolist()
    sindex = gdf4326.sindex

    all_vectors, all_labels = [], []
    chunks_with_data = 0
    for i, cb in enumerate(chunks):
        if callback:
            callback(i + 1, len(chunks))
        # Skip chunks no polygon touches — avoids a wasted bolt-on round-trip.
        candidates = list(sindex.intersection(cb))
        if not candidates:
            continue
        sub = gdf4326.iloc[candidates]
        sub = sub[sub.intersects(_box(*cb))]
        if sub.empty:
            continue
        try:
            mosaic, transform, _crs = client.fetch_mosaic_for_region(
                cb, year=year, target_crs=target_crs
            )
        except Exception as exc:  # no VQ coverage / server error for this chunk
            logger.warning("VQ chunk %s skipped: %s", cb, exc)
            continue
        if mosaic is None or mosaic.size == 0:
            continue

        h, w = mosaic.shape[:2]
        class_raster = rasterize_shapefile(sub, field, transform, w, h, label_encoder=le)
        labelled = class_raster > 0
        if not labelled.any():
            continue
        labels = class_raster[labelled] - 1  # 1-based -> 0-based
        vectors = mosaic[labelled]
        finite = np.isfinite(vectors).all(axis=1)  # drop nodata pixels
        if not finite.any():
            continue
        all_vectors.append(vectors[finite].astype(np.float32))
        all_labels.append(labels[finite].astype(np.int32))
        chunks_with_data += 1

    if not all_vectors:
        raise ValueError("No labelled VQ-reconstructed pixels found across any chunk")

    vectors = np.concatenate(all_vectors, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    stats = {
        "chunk_count": len(chunks),
        "chunks_with_data": chunks_with_data,
        "total_pixels": len(labels),
        "n_classes": len(class_names),
    }
    return vectors, labels, class_names, stats


def _is_4326(crs):
    """True if a GeoDataFrame CRS is (effectively) EPSG:4326 / WGS84."""
    if crs is None:
        return True  # assume already lon/lat
    try:
        return int(crs.to_epsg() or 0) == 4326
    except Exception:
        return "4326" in str(crs)
