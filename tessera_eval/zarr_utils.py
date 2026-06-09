"""Shared zarr utilities for GeoTessera tile access.

Provides cached zarr instance, coverage probing, and chunked region reading.
Used by both the evaluation server (server.py) and viewport processing
(process_viewport.py).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton zarr instance ──

_zarr_instance = None  # None = not tried, False = tried and failed


def get_zarr():
    """Return a cached GeoTesseraZarr instance, or None if unavailable.

    Only attempts the import once; caches the result (including failure).
    """
    global _zarr_instance
    if _zarr_instance is None:
        try:
            from geotessera.store import GeoTesseraZarr

            _zarr_instance = GeoTesseraZarr()
            logger.info("GeoTesseraZarr available: %s", _zarr_instance.url)
        except Exception:
            _zarr_instance = False
            logger.info("GeoTesseraZarr not available")
    return _zarr_instance if _zarr_instance is not False else None


def probe_zarr_coverage(gtz, bounds, year):
    """Probe zarr store for coverage at the centre of bounds.

    Returns True if zarr has non-NaN data for (year, centre-of-bounds).
    """
    try:
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        probe = gtz.sample_at(cx, cy, year)
        return not np.isnan(probe).all()
    except Exception:
        return False


# ── Chunked region reading ──

CHUNK_THRESHOLD = 0.2  # degrees — regions larger than this get split
CHUNK_SIZE = 0.1  # degrees per chunk


def _reproject_to_4326(mosaic, transform, src_crs):
    """Reproject a (H, W, B) mosaic to EPSG:4326.

    geotessera's zarr ``read_region`` returns data in the native UTM zone CRS
    (e.g. EPSG:32631, metre coordinates), but the viewport pipeline — the crop
    math in process_viewport, pyramid georeferencing, the 4326 vectors
    metadata, and the Leaflet frontend — all assume lon/lat degrees. Reproject
    here so the zarr fast path matches the NPY path (which already requests
    target_crs='EPSG:4326').

    Nearest-neighbour resampling preserves exact embedding vectors (bilinear
    would blend the 128-d embeddings and corrupt similarity search). NaN
    nodata is carried through. No-op if already EPSG:4326.

    Returns (mosaic_4326, transform_4326, 'EPSG:4326').
    """
    dst_crs = "EPSG:4326"
    if str(src_crs).upper().replace(" ", "") in ("EPSG:4326", "WGS84"):
        return mosaic, transform, dst_crs

    from rasterio.warp import Resampling, calculate_default_transform, reproject

    h, w, bands = mosaic.shape
    left, top = transform.c, transform.f
    right = left + transform.a * w
    bottom = top + transform.e * h

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h, left=left, bottom=bottom, right=right, top=top
    )

    src = np.ascontiguousarray(np.transpose(mosaic, (2, 0, 1)))  # (B, H, W)
    dst = np.full((bands, dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=src,
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    logger.info(
        "Reprojected zarr mosaic %s (%dx%d) -> EPSG:4326 (%dx%d)", src_crs, w, h, dst_w, dst_h
    )
    return np.transpose(dst, (1, 2, 0)), dst_transform, dst_crs


def _utm_zone(lon):
    """UTM zone number for a longitude (geotessera routes each chunk by its centre)."""
    return int((lon + 180.0) // 6.0) + 1


def _reproject_chunk_into(mosaic, dst_transform, emb, src_transform, src_crs, chunk_bounds):
    """Reproject one native-CRS chunk into the shared EPSG:4326 ``mosaic`` in place.

    Only the chunk's geographic sub-window is written, and existing non-NaN values
    are kept where the reprojected chunk is NaN — so overlapping seams don't punch
    holes. Nearest-neighbour only (never blend 128-d embeddings). Temp memory is
    bounded to one chunk window.
    """
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    h_full, w_full, bands = mosaic.shape
    px_lon, px_lat = dst_transform.a, -dst_transform.e
    west0, north0 = dst_transform.c, dst_transform.f
    lon0, lat0, lon1, lat1 = chunk_bounds

    # Pixel window of this chunk in the dst grid (+1 px pad to avoid seam gaps).
    c0 = max(0, int(np.floor((lon0 - west0) / px_lon)) - 1)
    c1 = min(w_full, int(np.ceil((lon1 - west0) / px_lon)) + 1)
    r0 = max(0, int(np.floor((north0 - lat1) / px_lat)) - 1)
    r1 = min(h_full, int(np.ceil((north0 - lat0) / px_lat)) + 1)
    if c1 <= c0 or r1 <= r0:
        return

    win_transform = Affine(px_lon, 0.0, west0 + c0 * px_lon, 0.0, -px_lat, north0 - r0 * px_lat)
    src = np.ascontiguousarray(np.transpose(emb, (2, 0, 1)))  # (B, h, w)
    tmp = np.full((bands, r1 - r0, c1 - c0), np.nan, dtype=np.float32)
    reproject(
        source=src,
        destination=tmp,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=win_transform,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    tmp = np.transpose(tmp, (1, 2, 0))  # (rows, cols, B)
    covered = ~np.isnan(tmp).all(axis=2)
    mosaic[r0:r1, c0:c1][covered] = tmp[covered]


def read_region_chunked(gtz, bounds, year):
    """Read a region via zarr, chunking if larger than CHUNK_THRESHOLD.

    Args:
        gtz: GeoTesseraZarr instance
        bounds: (west, south, east, north) in EPSG:4326
        year: int

    Returns:
        (mosaic, transform, crs) where mosaic is (H, W, 128) float32 and the
        transform/crs are reprojected to EPSG:4326 (geotessera returns native
        UTM; downstream assumes lon/lat). Returns (None, None, None) if no
        data available.
    """
    west, south, east, north = bounds
    lon_span = east - west
    lat_span = north - south
    single_zone = _utm_zone(west) == _utm_zone(east)

    # Small, single-zone region — one native read + reproject (fast path).
    if lon_span <= CHUNK_THRESHOLD and lat_span <= CHUNK_THRESHOLD and single_zone:
        mosaic, transform, crs = gtz.read_region(bounds, year)
        return _reproject_to_4326(mosaic, transform, crs)

    # Otherwise — large, and/or spanning >1 UTM zone. geotessera's read_region
    # routes a whole bbox to a single centre-zone grid and clips the rest, and a
    # naive metre-offset merge cannot place chunks from different zones. So read
    # 0.1deg chunks (each in its own native zone) and reproject each into ONE
    # shared EPSG:4326 grid defined up front.
    from affine import Affine
    from rasterio.warp import calculate_default_transform

    chunk_lons, lon = [], west
    while lon < east:
        chunk_lons.append((lon, min(lon + CHUNK_SIZE, east)))
        lon += CHUNK_SIZE
    chunk_lats, lat = [], south
    while lat < north:
        chunk_lats.append((lat, min(lat + CHUNK_SIZE, north)))
        lat += CHUNK_SIZE
    logger.info(
        "Reading %d zarr chunks (%d x %d) -> shared EPSG:4326 grid",
        len(chunk_lons) * len(chunk_lats),
        len(chunk_lons),
        len(chunk_lats),
    )

    # Read every chunk (each in its native zone); keep its requested 4326 bounds.
    read = []  # (emb, tfm, crs, (lon0, lat0, lon1, lat1))
    for lat0, lat1 in chunk_lats:
        for lon0, lon1 in chunk_lons:
            cb = (lon0, lat0, lon1, lat1)
            try:
                emb, tfm, crs = gtz.read_region(cb, year)
            except Exception as e:
                logger.warning("Zarr chunk %s failed: %s", cb, e)
                continue
            if emb is None or emb.size == 0:
                continue
            read.append((emb, tfm, crs, cb))

    if not read:
        return None, None, None

    # One resolution for the whole target grid (don't let each chunk choose its
    # own — that would create sub-pixel seams). Derive it from the first chunk's
    # native -> 4326 reprojection.
    emb0, tfm0, crs0, _ = read[0]
    h0, w0, bands = emb0.shape
    dt0, _, _ = calculate_default_transform(
        crs0,
        "EPSG:4326",
        w0,
        h0,
        left=tfm0.c,
        bottom=tfm0.f + tfm0.e * h0,
        right=tfm0.c + tfm0.a * w0,
        top=tfm0.f,
    )
    px_lon, px_lat = dt0.a, -dt0.e

    width = max(1, int(np.ceil(lon_span / px_lon)))
    height = max(1, int(np.ceil(lat_span / px_lat)))
    dst_transform = Affine(px_lon, 0.0, west, 0.0, -px_lat, north)
    mosaic = np.full((height, width, bands), np.nan, dtype=np.float32)

    for emb, tfm, crs, cb in read:
        _reproject_chunk_into(mosaic, dst_transform, emb, tfm, crs, cb)

    return mosaic, dst_transform, "EPSG:4326"
