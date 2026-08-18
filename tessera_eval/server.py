"""Lightweight compute server for tessera-eval.

Handles ML evaluation locally and proxies everything else (UI, tiles,
label sharing) to a hosted TEE server. This lets users run compute on
their own machine while using the hosted server for data.

Usage:
    tee-compute --hosted https://tee.cl.cam.ac.uk
    tee-compute --hosted https://tee.cl.cam.ac.uk --port 8001
"""

import argparse
import json
import logging
import tempfile
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import requests
from flask import Flask, Response, jsonify, request, send_file

logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.after_request
def _add_cors(response):
    """Allow browsers on any origin to talk to tee-compute directly."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# Pixel classifier UI name -> its regression counterpart (make_regressor's
# naming convention). Used wherever a request carries a plain classifier name
# ("rf") that needs to become the right model for the cached task
# (run_large_area, create_map) -- make_regressor only recognizes the "_reg"
# suffixed names.
_CLF_TO_REG = {"nn": "nn_reg", "rf": "rf_reg", "mlp": "mlp_reg", "xgboost": "xgboost_reg"}

# ── State (single-user, one process) ──

_uploaded_shapefiles = []  # list of (filename, gdf) tuples
_merged_gdf = None
_trained_models = {}  # classifier name → temp file path
_generated_maps = {}  # map name → temp file path (GeoTIFF)
_finish_classifiers = set()
_tile_cache = {
    "key": None,
    "vectors": None,
    "labels": None,
    "class_names": None,
    "stats": None,
    "spatial_3x3": None,
    "spatial_5x5": None,
}
_hosted_url = None
_tile_disk_cache_dir = None  # set in main()
_geotessera_instance = None  # cached to avoid 10-30s registry init per run
_cancel_flag = None  # threading.Event, set when user cancels
# Shared across every proxy() call so the TCP+TLS connection to _hosted_url is
# kept alive and reused (requests' connection-pooling adapter), instead of a
# fresh handshake per proxied request -- see proxy()'s docstring for why this
# matters. Sharing one Session across threads is standard/safe for this: no
# concurrent mutation of cookies/state beyond urllib3's own thread-safe pools.
_proxy_session = requests.Session()

FLUSH_PAD = 18 * 1024  # pad NDJSON lines to force Waitress flush


def _get_cache_dir():
    """Return the cache directory, creating it if needed."""
    global _tile_disk_cache_dir
    if _tile_disk_cache_dir is None:
        _tile_disk_cache_dir = Path.home() / ".cache" / "tessera-eval"
    _tile_disk_cache_dir.mkdir(parents=True, exist_ok=True)
    return _tile_disk_cache_dir


def _result_cache_path(field, year, gdf_hash, sampling="equal"):
    """Return the disk path for cached evaluation results (vectors + labels)."""
    return _get_cache_dir() / f"result_{field}_{year}_{sampling}_{gdf_hash}.npz"


def _gdf_hash(gdf):
    """Quick hash of a GeoDataFrame for cache keying."""
    import hashlib

    h = hashlib.md5()
    h.update(str(len(gdf)).encode())
    h.update(str(sorted(gdf.columns.tolist())).encode())
    bounds = gdf.total_bounds
    h.update(f"{bounds[0]:.4f},{bounds[1]:.4f},{bounds[2]:.4f},{bounds[3]:.4f}".encode())
    return h.hexdigest()[:12]


def _load_cached_result(field, year, gdf, sampling="equal"):
    """Load cached evaluation result. Returns (vectors, labels, class_names, stats) or None."""
    path = _result_cache_path(field, year, _gdf_hash(gdf), sampling)
    if path.exists():
        try:
            data = np.load(path, allow_pickle=True)
            return (
                data["vectors"],
                data["labels"],
                data["class_names"].tolist(),
                dict(data["stats"].item()),
            )
        except Exception:
            path.unlink(missing_ok=True)
    return None


from tessera_zarr_utils import get_zarr, probe_zarr_coverage


def _sample_points_within_budget(rows_gdf, budget, rng):
    """Sample up to `budget` points total from rows_gdf's polygons.

    geopandas' sample_points(size=N) generates N points PER ROW, not N
    total -- so a caller wanting "at least 1 point per row" (so no row is
    left with zero representation) combined with "no more than `budget`
    points total" has a real conflict whenever there are more rows than
    budget: drawing 1 point from every row produces len(rows_gdf) points,
    ignoring the budget outright. Confirmed live (Louis Driver): a
    420,000-row shapefile with a 200,000-point default budget generated
    ~420,000 points -- every row got its floor-guaranteed 1 point,
    regardless of budget.

    When there are more rows than budget, this subsamples *which* rows to
    draw from first (a random subset, not just the first `budget` rows --
    a shapefile's row order often correlates with something geographic,
    e.g. a raster-to-polygon conversion scanning left-to-right/top-to-bottom)
    rather than drawing 1 point from every row and blowing past the budget.

    Returns (coords, row_index): row_index maps each point back to its
    source row's index in rows_gdf (0 points -> two empty arrays).
    """
    import warnings

    n_rows = len(rows_gdf)
    empty_coords = np.empty((0, 2))
    empty_idx = np.empty((0,), dtype=rows_gdf.index.dtype if n_rows else np.int64)
    if n_rows == 0 or budget <= 0:
        return empty_coords, empty_idx

    if n_rows > budget:
        chosen = rng.choice(rows_gdf.index.to_numpy(), size=budget, replace=False)
        source = rows_gdf.loc[chosen]
        pts_per_row = 1
    else:
        source = rows_gdf
        pts_per_row = max(1, budget // n_rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pts = source.sample_points(size=pts_per_row)
    pts_exploded = pts[~pts.is_empty].explode(index_parts=False)
    if len(pts_exploded) == 0:
        return empty_coords, empty_idx
    coords = np.array([(p.x, p.y) for p in pts_exploded])
    row_index = pts_exploded.index.to_numpy()
    return coords, row_index


def _extract_tile_patches(
    gt,
    gdf,
    field_name,
    year,
    le,
    n_classes,
    patch_size=256,
    max_patches=500,
    needs_spatial_3x3=False,
    needs_spatial_5x5=False,
    sample_points_lonlat=None,
    logger=None,
    progress_cb=None,
    cancel_flag=None,
    is_classification=True,
):
    """Extract pixel-aligned 2D patches and optionally point samples from tiles.

    Uses zarr read_region() when available (~0.2s/patch vs ~15s/tile via NPY).
    Falls back to gt.fetch_embeddings() for NPY tile downloads.

    is_classification=False (regression): patches carry real continuous
    target values (via rasterize_shapefile_continuous, NaN = unlabelled)
    instead of LabelEncoder class IDs (via rasterize_shapefile, 0 =
    unlabelled) -- le/n_classes are unused in that case (spatial_mlp/
    spatial_mlp_5x5 regression isn't supported yet regardless -- no
    regressor variant exists for either -- so needs_spatial_3x3/5x5 should
    never actually be True alongside is_classification=False in practice,
    but this function doesn't crash if it happens).

    Returns (unet_patches, spatial_3x3, spatial_5x5, point_vectors) where
    point_vectors is a (N, 128) array if sample_points_lonlat was given, else
    None. unet_patches' label_patch is int32 for classification, float32
    (NaN = ignore) for regression.
    """
    import rasterio.transform
    from rasterio.transform import array_bounds
    from shapely.geometry import box as _box

    from tessera_eval.classify import gather_spatial_features_2d
    from tessera_eval.rasterize import rasterize_shapefile, rasterize_shapefile_continuous

    rng = np.random.RandomState(42)

    # Find tiles overlapping the shapefile
    bounds = gdf.total_bounds

    # Try zarr — but verify coverage with a single-pixel probe first,
    # since the zarr store only has 2025 for some regions.
    gtz = get_zarr()
    use_zarr = gtz is not None and probe_zarr_coverage(gtz, bounds, year)
    if logger:
        logger.info(
            "Using %s for tile reads", "zarr (fast)" if use_zarr else "NPY tiles with local cache"
        )
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
    tiles_to_fetch = gt.registry.load_blocks_for_region(bbox, year)
    tiles_to_fetch = list(tiles_to_fetch)
    rng.shuffle(tiles_to_fetch)

    # Pre-group sample points by tile for efficient extraction
    point_vectors = None
    points_by_tile = {}
    if sample_points_lonlat is not None and len(sample_points_lonlat) > 0:
        point_vectors = np.full((len(sample_points_lonlat), 128), np.nan, dtype=np.float32)
        # Vectorized tile grouping
        pts_arr = np.array(sample_points_lonlat)  # (N, 2)
        tile_lons = np.round((pts_arr[:, 0] - 0.05) / 0.1) * 0.1 + 0.05
        tile_lats = np.round((pts_arr[:, 1] - 0.05) / 0.1) * 0.1 + 0.05
        tile_lons = np.round(tile_lons, 2)
        tile_lats = np.round(tile_lats, 2)
        for pt_idx in range(len(pts_arr)):
            key = (tile_lons[pt_idx], tile_lats[pt_idx])
            if key not in points_by_tile:
                points_by_tile[key] = []
            points_by_tile[key].append(pt_idx)

    if logger:
        pts_info = (
            f", {len(sample_points_lonlat)} sample points"
            if sample_points_lonlat is not None
            else ""
        )
        logger.info("Reading %d tiles (shuffled)%s...", len(tiles_to_fetch), pts_info)

    unet_patches = []
    all_spatial_3x3 = [] if needs_spatial_3x3 else None
    all_spatial_5x5 = [] if needs_spatial_5x5 else None
    all_spatial_labels_3x3 = [] if needs_spatial_3x3 else None
    all_spatial_labels_5x5 = [] if needs_spatial_5x5 else None

    patches_per_tile = 5
    total_tiles = len(tiles_to_fetch)

    # For NPY fallback, create the tile generator (lazy, one tile at a time)
    # Note: fetch_embeddings downloads a landmask per tile for CRS/transform,
    # even when embedding files are cached. This is a GeoTessera issue —
    # landmask CRS should be cached per UTM zone.
    tiles_gen = gt.fetch_embeddings(tiles_to_fetch) if not use_zarr else None

    for t_idx, (yr_t, tlon, tlat) in enumerate(tiles_to_fetch):
        if cancel_flag and cancel_flag.is_set():
            if logger:
                logger.info("Tile extraction cancelled")
            break
        if progress_cb:
            progress_cb(t_idx, total_tiles)

        try:
            if use_zarr:
                tile_bbox = (tlon - 0.05, tlat - 0.05, tlon + 0.05, tlat + 0.05)
                tile_emb, transform, crs = gtz.read_region(tile_bbox, year)
            else:
                _, _, _, tile_emb, crs, transform = next(tiles_gen)
                tile_emb = tile_emb.astype(np.float32)
        except Exception as e:
            if logger:
                logger.warning("Failed to load tile (%.2f, %.2f): %s", tlon, tlat, e)
            continue

        h, w = tile_emb.shape[:2]

        # Extract point samples from this tile (vectorized)
        tile_key = (round(tlon, 2), round(tlat, 2))
        n_extracted = 0
        if tile_key in points_by_tile:
            from pyproj import Transformer

            pt_indices = np.array(points_by_tile[tile_key])
            lons = np.array([sample_points_lonlat[i][0] for i in pt_indices])
            lats = np.array([sample_points_lonlat[i][1] for i in pt_indices])
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)
            rows, cols = rasterio.transform.rowcol(transform, xs, ys)
            rows = np.asarray(rows)
            cols = np.asarray(cols)
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            valid_pt_idx = pt_indices[valid]
            valid_rows = rows[valid]
            valid_cols = cols[valid]
            point_vectors[valid_pt_idx] = tile_emb[valid_rows, valid_cols]
            n_extracted = int(valid.sum())
        if logger and t_idx < 3:  # debug first 3 tiles
            n_pts = len(points_by_tile.get(tile_key, []))
            logger.info(
                "  tile_key=%s, %d points matched, %d extracted, tile shape=%s, crs=%s",
                tile_key,
                n_pts,
                n_extracted,
                tile_emb.shape,
                crs,
            )

        patches_full = len(unet_patches) >= max_patches
        if patches_full:
            continue  # still iterate for point samples, skip patch extraction

        if h < patch_size or w < patch_size:
            if logger:
                logger.info(
                    "  Skipping tile (%d×%d) — smaller than patch size %d", h, w, patch_size
                )
            continue

        if logger:
            logger.info(
                "Tile %d/%d (%.2f, %.2f): %s, extracting patches...",
                t_idx + 1,
                total_tiles,
                tlon,
                tlat,
                tile_emb.shape[:2],
            )

        # Filter GDF to tile area BEFORE reprojecting (avoids reprojecting 42K features)
        tile_bbox_lonlat = _box(
            tlon - 0.06, tlat - 0.06, tlon + 0.06, tlat + 0.06
        )  # slight padding
        tile_gdf = gdf[gdf.intersects(tile_bbox_lonlat)]
        if tile_gdf.empty:
            continue
        tile_gdf = tile_gdf.to_crs(crs)
        tile_bounds = array_bounds(h, w, transform)
        tile_gdf = tile_gdf[tile_gdf.intersects(_box(*tile_bounds))]
        if tile_gdf.empty:
            continue

        # Rasterize labels for the full tile
        if is_classification:
            tile_labels = rasterize_shapefile(
                tile_gdf, field_name, transform, h, w, label_encoder=le
            )
            labelled_mask_tile = tile_labels > 0
        else:
            tile_labels = rasterize_shapefile_continuous(tile_gdf, field_name, transform, h, w)
            labelled_mask_tile = ~np.isnan(tile_labels)

        # Find rows/cols where labels exist, with enough margin for a patch
        labelled_rows, labelled_cols = np.where(labelled_mask_tile)
        if len(labelled_rows) == 0:
            continue

        margin = patch_size // 2
        valid = (
            (labelled_rows >= margin)
            & (labelled_rows < h - margin)
            & (labelled_cols >= margin)
            & (labelled_cols < w - margin)
        )
        valid_rows = labelled_rows[valid]
        valid_cols = labelled_cols[valid]
        if len(valid_rows) == 0:
            continue

        # Pick random centers
        n_pick = min(patches_per_tile, len(valid_rows), max_patches - len(unet_patches))
        idx = rng.choice(len(valid_rows), size=n_pick, replace=False)

        for i in idx:
            r, c = valid_rows[i], valid_cols[i]
            r0, r1 = r - margin, r + margin
            c0, c1 = c - margin, c + margin

            emb_patch = tile_emb[r0:r1, c0:c1]
            label_patch = tile_labels[r0:r1, c0:c1]

            if emb_patch.shape != (patch_size, patch_size, tile_emb.shape[2]):
                continue
            if label_patch.shape != (patch_size, patch_size):
                continue
            n_patch_labelled = (
                (label_patch > 0).sum() if is_classification else (~np.isnan(label_patch)).sum()
            )
            if n_patch_labelled < 10:
                continue

            # Basic slicing above returns a *view* into tile_emb -- copy() is not
            # optional here, even though nothing after this point looks like it
            # mutates emb_patch on the no-NaN path. unet_patches (below) is kept
            # for the rest of the evaluation run; without this copy, every patch
            # keeps its *entire* source tile (H*W*128*4 bytes -- several hundred
            # MB, not the ~patch_size*patch_size*128*4 bytes ~32MB the patch
            # itself needs) alive in memory for as long as unet_patches lives.
            # With patches drawn from many different tiles across a large
            # shapefile, that's the difference between tens of MB and tens of
            # GB retained -- confirmed as the proximate cause of an OOM kill on
            # a real evaluation run (dmesg: "Out of memory: Killed process
            # ... (tee-compute) ... anon-rss:2407060kB"). The NaN branch below
            # already copied (needed there to avoid mutating the shared
            # buffer), which accidentally made the leak conditional on which
            # patches happened to contain NaN pixels -- easy to miss in review.
            emb_patch = emb_patch.copy()

            # Replace NaN with 0
            nan_mask = np.isnan(emb_patch)
            if nan_mask.any():
                emb_patch[nan_mask] = 0.0

            # int32 for classification (class IDs); float32 as-is for
            # regression (real values, NaN = ignore -- casting to int32
            # would destroy both the precision and the NaN sentinel).
            unet_patches.append(
                (emb_patch, label_patch.astype(np.int32) if is_classification else label_patch)
            )

            # Subsample labelled pixels for spatial features to cap memory
            # (~300MB per full 256×256 patch at 3×3, ~800MB at 5×5)
            labelled_mask = label_patch > 0 if is_classification else ~np.isnan(label_patch)
            MAX_SPATIAL_PX = 5000  # per patch — 100 patches × 5K = 500K total
            n_labelled = labelled_mask.sum()
            if n_labelled > MAX_SPATIAL_PX and (needs_spatial_3x3 or needs_spatial_5x5):
                # Randomly zero out excess pixels in the mask
                rows, cols = np.where(labelled_mask)
                keep = rng.choice(len(rows), size=MAX_SPATIAL_PX, replace=False)
                labelled_mask = np.zeros_like(labelled_mask)
                labelled_mask[rows[keep], cols[keep]] = True

            # spatial_mlp/spatial_mlp_5x5 regression isn't supported (no
            # regressor variant exists), so these branches stay
            # classification-shaped (the "- 1" 1-based→0-based shift would
            # be meaningless for regression) -- needs_spatial_3x3/5x5 should
            # never actually be True here when is_classification is False.
            if needs_spatial_3x3:
                sf = gather_spatial_features_2d(emb_patch, radius=1, mask=labelled_mask)
                all_spatial_3x3.append(sf)
                all_spatial_labels_3x3.append(label_patch[labelled_mask] - 1)  # 1-based → 0-based
            if needs_spatial_5x5:
                sf = gather_spatial_features_2d(emb_patch, radius=2, mask=labelled_mask)
                all_spatial_5x5.append(sf)
                all_spatial_labels_5x5.append(label_patch[labelled_mask] - 1)

        if logger:
            logger.info("  %d patches so far (%d from this tile)", len(unet_patches), n_pick)

    spatial_3x3 = (
        np.concatenate(all_spatial_3x3, axis=0).astype(np.float32) if all_spatial_3x3 else None
    )
    spatial_5x5 = (
        np.concatenate(all_spatial_5x5, axis=0).astype(np.float32) if all_spatial_5x5 else None
    )
    spatial_labels_3x3 = (
        np.concatenate(all_spatial_labels_3x3).astype(np.int32) if all_spatial_labels_3x3 else None
    )
    spatial_labels_5x5 = (
        np.concatenate(all_spatial_labels_5x5).astype(np.int32) if all_spatial_labels_5x5 else None
    )

    if logger:
        s3 = f", spatial_3x3={spatial_3x3.shape}" if spatial_3x3 is not None else ""
        s5 = f", spatial_5x5={spatial_5x5.shape}" if spatial_5x5 is not None else ""
        logger.info("Tile patches: %d total%s%s", len(unet_patches), s3, s5)

    return (
        unet_patches,
        spatial_3x3,
        spatial_5x5,
        point_vectors,
        spatial_labels_3x3,
        spatial_labels_5x5,
    )


def _save_cached_result(field, year, gdf, vectors, labels, class_names, stats, sampling="equal"):
    """Save evaluation result to disk cache."""
    try:
        path = _result_cache_path(field, year, _gdf_hash(gdf), sampling)
        np.savez_compressed(
            path,
            vectors=vectors,
            labels=labels,
            class_names=np.array(class_names),
            stats=np.array(stats),
        )
    except Exception as e:
        logger.debug("Failed to save result cache: %s", e)


def _cached_tiles_need_reload(
    cached_spatial_3x3,
    cached_spatial_5x5,
    cached_sample_points,
    *,
    needs_spatial_3x3,
    needs_spatial_5x5,
    has_spatial_bboxes,
):
    """Does an in-memory _tile_cache hit (same key, vectors present) still
    need a fresh reload, because this specific request needs data the cached
    entry doesn't have?

    True when the cache was populated by an earlier request that didn't need
    spatial_mlp/spatial_mlp_5x5 features, or didn't need a spatial train/test
    split (so never cached sample-point coordinates), and *this* request
    does. Whoever calls this with True must also invalidate the cache's own
    key (not just its own local `vectors` variable) -- confirmed live as a
    real bug otherwise: run_large_area's cache-hit branch used to set
    `vectors = None` here without touching `_tile_cache["key"]`, so the
    later "reload from GeoTessera" block (guarded by
    `_tile_cache["key"] != cache_key`) never triggered either, since the key
    still matched. vectors stayed None all the way through and crashed
    downstream at `len(vectors)` (TypeError: object of type 'NoneType' has
    no len()) on a spatial_mlp request that hit a cache entry from a prior
    non-spatial run.
    """
    if (needs_spatial_3x3 and cached_spatial_3x3 is None) or (
        needs_spatial_5x5 and cached_spatial_5x5 is None
    ):
        return True
    return bool(has_spatial_bboxes and cached_sample_points is None)


def _padded(gen):
    """Pad each NDJSON line to exceed Waitress send_bytes buffer."""
    for chunk in gen:
        if len(chunk) < FLUSH_PAD:
            yield chunk + " " * (FLUSH_PAD - len(chunk))
        else:
            yield chunk


def _get_merged_gdf():
    """Return merged GeoDataFrame from all uploaded shapefiles."""
    global _merged_gdf
    if _merged_gdf is not None:
        return _merged_gdf
    if not _uploaded_shapefiles:
        return None
    import pandas as pd

    _merged_gdf = gpd.GeoDataFrame(
        pd.concat([g for _, g in _uploaded_shapefiles], ignore_index=True)
    )
    return _merged_gdf


# ── Local evaluation endpoints ──


@app.route("/api/evaluation/upload-shapefile", methods=["POST"])
def upload_shapefile():
    """Accept a .zip containing .shp/.dbf/.shx/.prj, append to shapefile list."""
    global _merged_gdf
    uploaded = request.files.get("file")
    if not uploaded:
        return jsonify({"error": "No file uploaded"}), 400

    if not uploaded.filename.endswith(".zip"):
        return jsonify({"error": "File must be a .zip"}), 400

    tmp_dir = tempfile.mkdtemp(prefix="tee_eval_")
    zip_path = Path(tmp_dir) / uploaded.filename
    uploaded.save(str(zip_path))

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid zip file"}), 400

    shp_files = list(Path(tmp_dir).rglob("*.shp"))
    if not shp_files:
        return jsonify({"error": "No .shp file found in zip"}), 400

    try:
        import pandas as pd

        gdfs = [gpd.read_file(shp) for shp in shp_files]
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True)) if len(gdfs) > 1 else gdfs[0]
    except Exception as e:
        return jsonify({"error": f"Failed to read shapefile: {e}"}), 400

    if len(gdf) == 0:
        return jsonify({"error": "Shapefile is empty (0 features)"}), 400

    if "geometry" not in gdf.columns or gdf.geometry.is_empty.all():
        return jsonify({"error": "Shapefile has no geometry"}), 400

    # Reproject to EPSG:4326
    if gdf.crs is None:
        logger.warning("Shapefile has no CRS — assuming EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    _uploaded_shapefiles.append((uploaded.filename, gdf))
    _merged_gdf = None  # invalidate merged GDF cache
    # Note: _tile_cache is NOT invalidated here — tiles don't depend on shapefile.
    # The cache key is (field, year) which naturally misses if field changes.
    logger.info(
        "Uploaded '%s': %d features, %d fields",
        uploaded.filename,
        len(gdf),
        len([c for c in gdf.columns if c != "geometry"]),
    )

    merged = _get_merged_gdf()

    # Build field info with non-null counts
    fields = []
    for col in merged.columns:
        if col == "geometry":
            continue
        total = len(merged)
        non_null = int(merged[col].notna().sum())
        unique_count = merged[col].nunique()
        samples = merged[col].dropna().head(10).tolist()
        samples = [s if isinstance(s, (str, int, float)) else str(s) for s in samples]
        # Per-class polygon counts (from full GDF, not truncated GeoJSON)
        class_counts = merged[col].dropna().value_counts().to_dict()
        class_counts = {str(k): int(v) for k, v in class_counts.items()}
        fields.append(
            {
                "name": col,
                "unique_count": int(unique_count),
                "non_null": non_null,
                "total": total,
                "samples": samples,
                "class_counts": class_counts,
            }
        )

    # Build GeoJSON for map overlay
    MAX_OVERLAY = 10_000
    if len(merged) > MAX_OVERLAY:
        geojson = json.loads(merged.iloc[:MAX_OVERLAY].to_json())
        geojson["truncated"] = len(merged)
    else:
        geojson = json.loads(merged.to_json())

    # Estimate total labelled pixels from polygon areas at 10m resolution
    try:
        area_crs = merged.estimate_utm_crs()
        total_area_m2 = merged.to_crs(area_crs).geometry.area.sum()
        estimated_labelled_pixels = int(total_area_m2 / 100)  # 10m × 10m per pixel
    except Exception:
        estimated_labelled_pixels = 0

    return jsonify(
        {
            "fields": fields,
            "geojson": geojson,
            "files": [f for f, _ in _uploaded_shapefiles],
            "estimated_labelled_pixels": estimated_labelled_pixels,
        }
    )


@app.route("/api/evaluation/clear-shapefiles", methods=["POST"])
def clear_shapefiles():
    """Clear all uploaded shapefiles."""
    global _merged_gdf
    _uploaded_shapefiles.clear()
    _merged_gdf = None
    return jsonify({"ok": True})


@app.route("/api/evaluation/cancel", methods=["POST"])
def cancel_evaluation():
    """Cancel the running evaluation."""
    global _cancel_flag
    if _cancel_flag is not None:
        _cancel_flag.set()
        logger.info("Evaluation cancelled by user")
        return jsonify({"ok": True, "message": "Cancellation requested"})
    return jsonify({"ok": False, "message": "No evaluation running"})


@app.route("/api/evaluation/finish-classifier", methods=["POST"])
def finish_classifier():
    """Mark a classifier as finished for early stop."""
    try:
        body = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    name = body.get("classifier")
    if not name:
        return jsonify({"error": "classifier is required"}), 400
    _finish_classifiers.add(name)
    logger.info("Classifier '%s' marked for early finish", name)
    return jsonify({"ok": True})


@app.route("/api/evaluation/run-large-area", methods=["POST"])
def run_large_area():
    """Run evaluation: GeoTessera tile loading + learning curve.

    Supports all classifiers including spatial MLP and U-Net (per-tile).
    """
    try:
        body = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    field_name = body.get("field")
    # "year" is accepted as an alias for train_year so any stray old
    # client/config keeps working. test_year defaults to train_year, i.e.
    # zero behavior change for anyone who doesn't set it.
    train_year = body.get("train_year", body.get("year", 2024))
    test_year = body.get("test_year", train_year)
    classifiers = body.get("classifiers", ["nn", "rf"])
    classifier_params = body.get("classifier_params", {})
    max_train = body.get("max_training_samples")
    if max_train is not None:
        max_train = int(max_train)
    sampling = body.get("sampling", "sqrt")  # equal, proportional, sqrt
    max_patches = int(body.get("max_patches", 500))
    train_bboxes = body.get("train_bboxes", [])
    test_bboxes = body.get("test_bboxes", [])

    if not field_name:
        return jsonify({"error": "field is required"}), 400

    gdf = _get_merged_gdf()
    if gdf is None:
        return jsonify({"error": "No shapefile uploaded. Upload first."}), 400

    if field_name not in gdf.columns:
        return jsonify({"error": f"Field '{field_name}' not found in shapefile"}), 400

    # Auto-detect task type
    from tessera_eval.evaluate import detect_field_type

    task = body.get("task")
    if task is None or task == "auto":
        task = detect_field_type(gdf, field_name)

    is_classification = task == "classification"
    # Expand hyperparameter variants: if classifier_params[name] is a list,
    # each element becomes a separate variant (e.g., "mlp_v1", "mlp_v2").
    import re as _re

    def _expand_variants(names, params):
        """Expand classifier list for multi-variant hyperparameter sweeps.

        If params[name] is a list, each element becomes a named variant
        (name_v1, name_v2, ...). Single-object params are unchanged.
        """
        expanded_names = []
        expanded_params = {}
        for name in names:
            p = params.get(name, {})
            if isinstance(p, list):
                for i, variant_p in enumerate(p):
                    variant_name = f"{name}_v{i + 1}"
                    expanded_names.append(variant_name)
                    expanded_params[variant_name] = variant_p
            else:
                expanded_names.append(name)
                expanded_params[name] = p
        return expanded_names, expanded_params

    if is_classification:
        model_names, model_params = _expand_variants(classifiers, classifier_params)
    else:
        regressors = body.get("regressors", [])
        regressor_params = body.get("regressor_params", {})
        if regressors:
            model_names, model_params = _expand_variants(regressors, regressor_params)
        else:
            reg_names = [_CLF_TO_REG.get(c, c) for c in classifiers]
            reg_params = {_CLF_TO_REG.get(c, c): v for c, v in classifier_params.items()}
            model_names, model_params = _expand_variants(reg_names, reg_params)

    def _base_name(name):
        return _re.sub(r"_v\d+$", "", name)

    # Determine which spatial features are needed (check base names)
    needs_spatial_3x3 = any(_base_name(n) == "spatial_mlp" for n in model_names)
    needs_spatial_5x5 = any(_base_name(n) == "spatial_mlp_5x5" for n in model_names)
    needs_unet = any(_base_name(n) == "unet" for n in model_names)

    def stream():
        import threading

        global _cancel_flag
        _cancel_flag = threading.Event()

        from geotessera import GeoTessera
        from sklearn.preprocessing import LabelEncoder

        from tessera_eval.evaluate import run_learning_curve

        _finish_classifiers.clear()

        def _cancelled():
            return _cancel_flag is not None and _cancel_flag.is_set()

        # Clean up old models
        for old_path in _trained_models.values():
            try:
                Path(old_path).unlink(missing_ok=True)
            except OSError:
                pass
        _trained_models.clear()

        t0 = time.time()

        # Check in-memory cache first, then disk cache
        cache_key = (field_name, train_year, test_year, sampling)
        vectors = labels = class_names = stats = None
        spatial_3x3 = spatial_5x5 = unet_patches = None
        spatial_labels_3x3 = spatial_labels_5x5 = None
        all_sample_points = None  # (lon, lat) coordinates of all sample points
        all_valid_mask = None  # boolean mask: True for points with valid embeddings

        # Also true when train/test years differ: that path needs the same
        # sample-point coordinates (to re-fetch the test role at test_year)
        # that spatial bbox splitting needs, so it must force the same
        # reload-if-missing / skip-disk-cache-shortcut behavior below.
        has_spatial_bboxes = bool(train_bboxes or test_bboxes) or test_year != train_year
        if _tile_cache["key"] == cache_key and _tile_cache["vectors"] is not None:
            vectors = _tile_cache["vectors"]
            labels = _tile_cache["labels"]
            class_names = _tile_cache["class_names"]
            stats = _tile_cache["stats"]
            spatial_3x3 = _tile_cache.get("spatial_3x3")
            spatial_5x5 = _tile_cache.get("spatial_5x5")
            unet_patches = _tile_cache.get("unet_patches", [])
            all_sample_points = _tile_cache.get("sample_points")
            all_valid_mask = _tile_cache.get("valid_mask")
            logger.info(
                "In-memory cache hit for %s/%s (%d pixels)", field_name, train_year, len(labels)
            )

            # See _cached_tiles_need_reload's docstring: setting vectors =
            # None alone (without also invalidating _tile_cache["key"]) used
            # to leave vectors None all the way through and crash downstream
            # at `len(vectors)` -- confirmed live on a spatial_mlp request
            # that hit a cache entry from a prior non-spatial run.
            if _cached_tiles_need_reload(
                spatial_3x3,
                spatial_5x5,
                all_sample_points,
                needs_spatial_3x3=needs_spatial_3x3,
                needs_spatial_5x5=needs_spatial_5x5,
                has_spatial_bboxes=has_spatial_bboxes,
            ):
                if (needs_spatial_3x3 and spatial_3x3 is None) or (
                    needs_spatial_5x5 and spatial_5x5 is None
                ):
                    logger.info("Spatial features needed but not cached — reloading tiles")
                else:
                    logger.info("Spatial split needs point coordinates — reloading tiles")
                vectors = None  # force reload
                _tile_cache["key"] = None

        if vectors is None:
            # Check disk result cache (much smaller than raw tiles)
            cached_result = _load_cached_result(field_name, train_year, gdf, sampling)
            if (
                cached_result
                and not needs_spatial_3x3
                and not needs_spatial_5x5
                and not needs_unet
                and not has_spatial_bboxes
            ):
                vectors, labels, class_names, stats = cached_result
                logger.info(
                    "Disk result cache hit for %s/%s (%d pixels)",
                    field_name,
                    train_year,
                    len(labels),
                )

        if vectors is not None:
            yield (
                json.dumps(
                    {
                        "event": "download_progress",
                        "tile": stats.get("tile_count", 0),
                        "total": stats.get("tile_count", 0),
                        "cached": True,
                    }
                )
                + "\n"
            )
            # Update in-memory cache so we skip the GeoTessera fetch below
            _tile_cache.update(
                {
                    "key": cache_key,
                    "vectors": vectors,
                    "labels": labels,
                    "class_names": class_names,
                    "stats": stats,
                    "spatial_3x3": None,
                    "spatial_5x5": None,
                    "unet_patches": [],
                }
            )

        if _tile_cache["key"] != cache_key:
            # Emit early so the browser knows we're working
            yield (
                json.dumps(
                    {
                        "event": "field_start",
                        "field": field_name,
                        "type": task,
                        "status": "Loading GeoTessera tile index...",
                    }
                )
                + "\n"
            )

            # Reuse cached GeoTessera instance (avoids 10-30s registry init per run)
            global _geotessera_instance
            logger.info("Initializing GeoTessera...")
            yield json.dumps({"event": "status", "message": "Initializing GeoTessera..."}) + "\n"
            if _geotessera_instance is None:
                tile_cache_dir = _get_cache_dir() / "tiles"
                tile_cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    _geotessera_instance = GeoTessera(embeddings_dir=str(tile_cache_dir))
                except Exception as e:
                    # Unguarded before this fix: a network failure here (e.g. no route to
                    # the Tessera embeddings store) raised out of the generator and killed
                    # the SSE stream mid-response, surfacing to the browser as an opaque
                    # "failed to fetch" instead of a readable error.
                    logger.warning("GeoTessera initialization failed: %s", e)
                    yield (
                        json.dumps(
                            {
                                "event": "error",
                                "message": (
                                    "Could not initialize GeoTessera -- check that this "
                                    f"machine has network access to the Tessera embeddings "
                                    f"store: {e}"
                                ),
                            }
                        )
                        + "\n"
                    )
                    return
            gt = _geotessera_instance

            try:
                MAX_SAMPLE_PIXELS = max_train if max_train else 200_000

                # LabelEncoder/class_names/n_classes are classification-only:
                # regression targets are continuous, so there's no fixed
                # class vocabulary to fit. Confirmed live (Louis Driver):
                # this used to run unconditionally, silently LabelEncoding a
                # continuous field (e.g. tree height) into arbitrary
                # rank-order integers (0, 1, 2, ...) that then became the
                # actual training targets all the way through -- regressors
                # were fitting real-looking-but-meaningless R2/RMSE/MAE
                # against those ranks, not real height values. It also drove
                # the per-class point-budget/floor logic below, which is
                # exactly why a 420,000-row, 25-unique-height shapefile
                # generated ~25x more sample points than requested (each of
                # the 25 LabelEncoder "classes" independently hit the "at
                # least 1 point per polygon" floor).
                if is_classification:
                    le = LabelEncoder()
                    le.fit(gdf[field_name].dropna().unique())
                    class_names = le.classes_.tolist()
                    n_classes = len(class_names)
                else:
                    # le=None is a real, already-supported case in
                    # _extract_tile_patches -> rasterize_shapefile (falls
                    # back to its own per-tile LabelEncoder) -- needed here
                    # since _extract_tile_patches is called unconditionally
                    # below whenever spatial features are requested
                    # (spatial_mlp/unet), regression included, and `le`
                    # would otherwise be undefined for that call. n_classes
                    # is accepted but unused inside _extract_tile_patches;
                    # 0 is fine.
                    le = None
                    class_names = []
                    n_classes = 0

                # Generate random sample points within shapefile polygons
                if is_classification:
                    logger.info("Generating sample points across %d classes...", n_classes)
                    yield (
                        json.dumps(
                            {
                                "event": "status",
                                "message": f"Generating sample points across {n_classes} classes...",
                            }
                        )
                        + "\n"
                    )
                else:
                    logger.info("Generating sample points...")
                    yield (
                        json.dumps({"event": "status", "message": "Generating sample points..."})
                        + "\n"
                    )

                valid_gdf = gdf.dropna(subset=[field_name]).copy()

                sample_points = []
                sample_labels = []

                if is_classification:
                    label_ids = le.transform(valid_gdf[field_name])
                    valid_gdf["_label_id"] = label_ids

                    # Sampling strategy: equal, proportional, or sqrt-proportional
                    MIN_PER_CLASS = 50
                    if sampling in ("proportional", "sqrt"):
                        import math

                        area_crs = valid_gdf.estimate_utm_crs()
                        projected = valid_gdf.to_crs(area_crs)
                        projected["_area"] = projected.geometry.area
                        valid_gdf["_area"] = projected["_area"].values
                        class_areas = valid_gdf.groupby("_label_id")["_area"].sum()
                        if sampling == "sqrt":
                            weights = {c: math.sqrt(a) for c, a in class_areas.items()}
                        else:
                            weights = dict(class_areas)
                        total_weight = sum(weights.values())
                        raw_alloc = {
                            c: max(MIN_PER_CLASS, int(MAX_SAMPLE_PIXELS * w / total_weight))
                            for c, w in weights.items()
                        }
                        # Scale down if total exceeds budget
                        alloc_total = sum(raw_alloc.values())
                        if alloc_total > MAX_SAMPLE_PIXELS:
                            scale = MAX_SAMPLE_PIXELS / alloc_total
                            raw_alloc = {
                                c: max(MIN_PER_CLASS, int(n * scale)) for c, n in raw_alloc.items()
                            }
                    else:
                        # Equal per class
                        equal_n = MAX_SAMPLE_PIXELS // n_classes
                        raw_alloc = {c: equal_n for c in range(n_classes)}

                    sampling_rng = np.random.RandomState(42)
                    for cls_idx in range(n_classes):
                        cls_gdf = valid_gdf[valid_gdf["_label_id"] == cls_idx]
                        if cls_gdf.empty:
                            continue
                        per_class = raw_alloc.get(cls_idx, MIN_PER_CLASS)
                        try:
                            coords, _row_idx = _sample_points_within_budget(
                                cls_gdf, per_class, sampling_rng
                            )
                            if len(coords) > 0:
                                sample_points.extend(coords.tolist())
                                sample_labels.extend([cls_idx] * len(coords))
                        except Exception as e:
                            logger.warning("sample_points failed for class %d: %s", cls_idx, e)
                else:
                    # Regression: no classes, so no per-class weighting/floor
                    # either -- the "sampling" strategy param (equal/
                    # proportional/sqrt) only makes sense as a *class*
                    # weighting choice, so it's not applicable here; ignored
                    # for regression rather than repurposed into something
                    # that doesn't map cleanly. One combined budget across
                    # every labelled row instead.
                    sampling_rng = np.random.RandomState(42)
                    try:
                        coords, row_idx = _sample_points_within_budget(
                            valid_gdf, MAX_SAMPLE_PIXELS, sampling_rng
                        )
                        if len(coords) > 0:
                            # Look up each point's *real* field value via its
                            # source row, not a LabelEncoder rank.
                            values = valid_gdf.loc[row_idx, field_name].to_numpy(dtype=np.float64)
                            sample_points.extend(coords.tolist())
                            sample_labels.extend(values.tolist())
                    except Exception as e:
                        logger.warning("sample_points failed for regression: %s", e)

                n_points = len(sample_points)
                if n_points == 0:
                    yield (
                        json.dumps(
                            {
                                "event": "error",
                                "message": "No sample points generated from shapefile polygons",
                            }
                        )
                        + "\n"
                    )
                    return

                logger.info("Generated %d sample points across %d classes", n_points, n_classes)
                yield (
                    json.dumps(
                        {
                            "event": "status",
                            "message": f"Generated {n_points:,} sample points across {n_classes} classes",
                        }
                    )
                    + "\n"
                )

                if _cancelled():
                    yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
                    return

                import queue
                import threading

                progress_q = queue.Queue()
                spatial_3x3 = None
                spatial_5x5 = None
                spatial_labels_3x3 = None
                spatial_labels_5x5 = None
                unet_patches = []

                if needs_spatial_3x3 or needs_spatial_5x5 or needs_unet:
                    # Single tile pass: fetch tiles once, extract both point samples AND patches
                    logger.info("Loading embeddings for %d points + patches...", n_points)
                    yield (
                        json.dumps(
                            {
                                "event": "status",
                                "message": f"Loading embeddings for {n_points:,} points + patches...",
                            }
                        )
                        + "\n"
                    )

                    def _tile_progress(current, total):
                        progress_q.put(("tile", current, total))

                    tile_result = [None, None]

                    def _fetch_all():
                        try:
                            tile_result[0] = _extract_tile_patches(
                                gt,
                                gdf,
                                field_name,
                                train_year,
                                le,
                                n_classes,
                                max_patches=max_patches,
                                needs_spatial_3x3=needs_spatial_3x3,
                                needs_spatial_5x5=needs_spatial_5x5,
                                sample_points_lonlat=sample_points,
                                logger=logger,
                                progress_cb=_tile_progress,
                                cancel_flag=_cancel_flag,
                                is_classification=is_classification,
                            )
                        except Exception as e:
                            tile_result[1] = e
                        finally:
                            progress_q.put(None)

                    t = threading.Thread(target=_fetch_all, daemon=True)
                    t.start()

                    while True:
                        if _cancelled():
                            logger.info("Cancelled during tile fetch")
                            yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
                            return
                        try:
                            item = progress_q.get(timeout=5)
                        except queue.Empty:
                            yield json.dumps({"event": "heartbeat"}) + "\n"
                            continue
                        if item is None:
                            break
                        if item[0] == "tile":
                            _, cur, tot = item
                            pct = int(100 * cur / tot) if tot else 0
                            msg = f"Loading tile {cur}/{tot} ({pct}%)"
                            logger.info(msg)
                            yield (
                                json.dumps({"event": "progress", "pct": pct, "message": msg}) + "\n"
                            )

                    t.join()
                    if tile_result[1] is not None:
                        yield (
                            json.dumps(
                                {
                                    "event": "error",
                                    "message": f"Tile fetch failed: {tile_result[1]}",
                                }
                            )
                            + "\n"
                        )
                        return

                    (
                        unet_patches,
                        spatial_3x3,
                        spatial_5x5,
                        vectors,
                        spatial_labels_3x3,
                        spatial_labels_5x5,
                    ) = tile_result[0]
                    if vectors is not None:
                        n_valid = (~np.isnan(vectors).any(axis=1)).sum()
                        logger.info("Point vectors from tiles: %d/%d valid", n_valid, len(vectors))
                    else:
                        logger.warning("No point vectors returned from tile extraction")
                else:
                    # Pixel-only: use sample_embeddings_at_points (faster, no tile loading)
                    logger.info("Fetching embeddings for %d points...", n_points)
                    yield (
                        json.dumps(
                            {
                                "event": "status",
                                "message": f"Fetching embeddings for {n_points:,} points...",
                            }
                        )
                        + "\n"
                    )

                    result_holder = [None, None]

                    def _fetch():
                        try:

                            def _cb(current, total, status):
                                progress_q.put(("tile", current, total))

                            vecs = gt.sample_embeddings_at_points(
                                sample_points, year=train_year, progress_callback=_cb
                            )
                            result_holder[0] = vecs
                        except Exception as e:
                            result_holder[1] = e
                        finally:
                            progress_q.put(None)

                    t = threading.Thread(target=_fetch, daemon=True)
                    t.start()

                    last_reported = -1
                    while True:
                        if _cancelled():
                            logger.info("Cancelled during pixel fetch")
                            yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
                            return
                        try:
                            item = progress_q.get(timeout=5)
                        except queue.Empty:
                            yield json.dumps({"event": "heartbeat"}) + "\n"
                            continue
                        if item is None:
                            break
                        if item[0] == "tile":
                            _, current, total = item
                            if current == last_reported:
                                continue
                            last_reported = current
                            pct = int(100 * current / total) if total else 0
                            msg = f"Fetching embeddings: {current}/{total} tiles ({pct}%)"
                            logger.info(msg)
                            yield (
                                json.dumps({"event": "progress", "pct": pct, "message": msg}) + "\n"
                            )

                    t.join()
                    if result_holder[1] is not None:
                        yield (
                            json.dumps(
                                {
                                    "event": "error",
                                    "message": f"GeoTessera sampling failed: {result_holder[1]}",
                                }
                            )
                            + "\n"
                        )
                        return
                    vectors = result_holder[0]

                logger.info("Processing embeddings...")
                yield json.dumps({"event": "status", "message": "Processing embeddings..."}) + "\n"

                labels = np.array(
                    sample_labels, dtype=np.int32 if is_classification else np.float32
                )

                # Remove NaN rows (points outside tile coverage)
                valid_mask = ~np.isnan(vectors).any(axis=1)
                if valid_mask.sum() < len(vectors):
                    n_removed = len(vectors) - valid_mask.sum()
                    n_remaining = valid_mask.sum()
                    logger.info(
                        "Removed %d points outside coverage (%d remaining)", n_removed, n_remaining
                    )
                    yield (
                        json.dumps(
                            {
                                "event": "status",
                                "message": f"Removed {n_removed:,} points outside coverage ({n_remaining:,} remaining)",
                            }
                        )
                        + "\n"
                    )
                    vectors = vectors[valid_mask].astype(np.float32)
                    labels = labels[valid_mask]
                else:
                    vectors = vectors.astype(np.float32)

                if len(vectors) == 0:
                    yield (
                        json.dumps(
                            {
                                "event": "error",
                                "message": "No valid embeddings found at sample points",
                            }
                        )
                        + "\n"
                    )
                    return

                # Count tiles used
                bounds = gdf.total_bounds
                bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
                tiles = gt.registry.load_blocks_for_region(bbox, train_year)
                total_tiles = len(tiles)

                stats = {
                    "tile_count": total_tiles,
                    "tiles_with_data": total_tiles,
                    "total_pixels": len(labels),
                    "n_classes": n_classes,
                }

                # Cache in memory and on disk
                _tile_cache.update(
                    {
                        "key": cache_key,
                        "vectors": vectors,
                        "labels": labels,
                        "class_names": class_names,
                        "stats": stats,
                        "spatial_3x3": None,
                        "spatial_5x5": None,
                        "unet_patches": [],
                        "sample_points": sample_points,
                        "valid_mask": valid_mask,
                    }
                )
                _save_cached_result(
                    field_name, train_year, gdf, vectors, labels, class_names, stats, sampling
                )

                all_sample_points = sample_points
                all_valid_mask = valid_mask

                logger.info(
                    "Point sampling complete: %d pixels, %.1fMB", len(labels), vectors.nbytes / 1e6
                )

            except Exception as e:
                yield json.dumps({"event": "error", "message": str(e)}) + "\n"
                return
        else:
            spatial_3x3 = _tile_cache.get("spatial_3x3")
            spatial_5x5 = _tile_cache.get("spatial_5x5")
            unet_patches = _tile_cache.get("unet_patches", [])

        total_labelled = len(vectors)

        # ── Spatial split: partition points into train/test pools ──
        spatial_train_vectors = spatial_test_vectors = None
        spatial_train_labels = spatial_test_labels = None
        has_spatial_split = bool(train_bboxes or test_bboxes)

        if has_spatial_split:

            def _point_in_bboxes(lon, lat, bboxes):
                """Check if (lon, lat) falls inside any bbox [south, west, north, east]."""
                for south, west, north, east in bboxes:
                    if west <= lon <= east and south <= lat <= north:
                        return True
                return False

            if all_sample_points is None:
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "message": "Spatial split requires sample point coordinates (not available from cache)",
                        }
                    )
                    + "\n"
                )
                return

            # Apply valid_mask to match the filtered vectors/labels
            sp = np.array(all_sample_points)
            if all_valid_mask is not None and all_valid_mask.sum() < len(sp):
                sp = sp[all_valid_mask]

            train_mask = np.zeros(len(sp), dtype=bool)
            test_mask_sp = np.zeros(len(sp), dtype=bool)
            for i in range(len(sp)):
                lon, lat = sp[i]
                if train_bboxes and _point_in_bboxes(lon, lat, train_bboxes):
                    train_mask[i] = True
                elif test_bboxes and _point_in_bboxes(lon, lat, test_bboxes):
                    test_mask_sp[i] = True
                # Points in neither are discarded

            n_train = train_mask.sum()
            n_test = test_mask_sp.sum()
            n_discard = len(sp) - n_train - n_test
            logger.info(
                "Spatial split: %d train, %d test, %d discarded", n_train, n_test, n_discard
            )
            yield (
                json.dumps(
                    {
                        "event": "status",
                        "message": f"Spatial split: {int(n_train):,} train, {int(n_test):,} test, {int(n_discard):,} discarded",
                    }
                )
                + "\n"
            )

            if n_train == 0:
                yield (
                    json.dumps(
                        {"event": "error", "message": "No sample points in train bounding boxes"}
                    )
                    + "\n"
                )
                return
            if n_test == 0:
                yield (
                    json.dumps(
                        {"event": "error", "message": "No sample points in test bounding boxes"}
                    )
                    + "\n"
                )
                return
            if n_test < 100:
                logger.warning(
                    "Very small test set (%d pixels) — results may be unreliable", n_test
                )
                yield (
                    json.dumps(
                        {
                            "event": "status",
                            "message": f"Warning: only {int(n_test)} test pixels — results may be unreliable. Consider enlarging test area.",
                        }
                    )
                    + "\n"
                )

            spatial_train_vectors = vectors[train_mask]
            spatial_train_labels = labels[train_mask]
            spatial_test_vectors = vectors[test_mask_sp]
            spatial_test_labels = labels[test_mask_sp]

            # For the learning curve, vectors/labels = train pool
            vectors = spatial_train_vectors
            labels = spatial_train_labels
            total_labelled = len(vectors)

        # ── Train/test-year split ──
        # When test_year != train_year, the test role's points get their
        # embeddings re-fetched at test_year and fed into run_learning_curve's
        # pre-existing test_vectors/test_labels fixed-test-set mechanism --
        # the same mechanism tessera-eval's `learning-curve --test-year` CLI
        # command (ucam-eo/tessera-eval#2) validated for this exact use case,
        # just fed from server.py instead of the CLI. Two cases, mirroring
        # what the CLI supports:
        #   - no bboxes drawn: every point serves as both a training example
        #     (at train_year) and a test example (re-embedded at test_year)
        #     -- "does a classifier trained on year A still work on year B
        #     at the same locations".
        #   - bboxes drawn: keep today's spatial train/test-region split, but
        #     embed the test region at test_year instead of train_year.
        year_split_test_vectors = year_split_test_labels = None
        if test_year != train_year:
            if all_sample_points is None or all_valid_mask is None:
                # Shouldn't happen -- has_spatial_bboxes forces a fresh fetch
                # (which always populates these) whenever years differ. Kept
                # as a defensive check rather than trusting that invariant.
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "message": (
                                "Different train/test years requires sample point "
                                "coordinates, unexpectedly unavailable -- please retry."
                            ),
                        }
                    )
                    + "\n"
                )
                return

            if has_spatial_split:
                test_role_points = sp[test_mask_sp]
                test_role_labels_pool = spatial_test_labels
            else:
                sp_all = np.array(all_sample_points)
                if all_valid_mask.sum() < len(sp_all):
                    sp_all = sp_all[all_valid_mask]
                test_role_points = sp_all
                test_role_labels_pool = labels

            # Lazy-init GeoTessera, same pattern as the primary fetch above --
            # needed even on a cache hit for train_year, since that path never
            # touches GeoTessera at all. (No `global` redeclaration here: it's
            # already declared global earlier in this same function -- doing
            # it twice with a use in between is itself a SyntaxError.)
            if _geotessera_instance is None:
                tile_cache_dir = _get_cache_dir() / "tiles"
                tile_cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    _geotessera_instance = GeoTessera(embeddings_dir=str(tile_cache_dir))
                except Exception as e:
                    yield (
                        json.dumps(
                            {
                                "event": "error",
                                "message": f"Could not initialize GeoTessera for test-year fetch: {e}",
                            }
                        )
                        + "\n"
                    )
                    return
            gt = _geotessera_instance

            yield (
                json.dumps(
                    {
                        "event": "status",
                        "message": (
                            f"Fetching test-year ({test_year}) embeddings for "
                            f"{len(test_role_points):,} points..."
                        ),
                    }
                )
                + "\n"
            )

            ty_holder = [None, None]

            def _fetch_test_year():
                try:
                    ty_holder[0] = gt.sample_embeddings_at_points(
                        test_role_points.tolist(), year=test_year
                    )
                except Exception as e:
                    ty_holder[1] = e

            ty_thread = threading.Thread(target=_fetch_test_year, daemon=True)
            ty_thread.start()
            while ty_thread.is_alive():
                if _cancelled():
                    yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
                    return
                ty_thread.join(timeout=5)
                if ty_thread.is_alive():
                    yield json.dumps({"event": "heartbeat"}) + "\n"

            if ty_holder[1] is not None:
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "message": f"Test-year ({test_year}) sampling failed: {ty_holder[1]}",
                        }
                    )
                    + "\n"
                )
                return

            ty_vectors = ty_holder[0]
            ty_valid = ~np.isnan(ty_vectors).any(axis=1)
            if ty_valid.sum() < len(ty_vectors):
                n_removed = len(ty_vectors) - ty_valid.sum()
                logger.info(
                    "Test year %d: %d/%d points outside tile coverage, dropped",
                    test_year,
                    n_removed,
                    len(ty_vectors),
                )
                yield (
                    json.dumps(
                        {
                            "event": "status",
                            "message": (
                                f"Test year {test_year}: {n_removed:,} points outside "
                                f"coverage, dropped"
                            ),
                        }
                    )
                    + "\n"
                )
            if ty_valid.sum() == 0:
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "message": f"No valid embeddings found at test year {test_year}",
                        }
                    )
                    + "\n"
                )
                return

            year_split_test_vectors = ty_vectors[ty_valid].astype(np.float32)
            year_split_test_labels = test_role_labels_pool[ty_valid]

        # Training percentages (% of labelled area)
        training_pcts = [1, 3, 5, 10, 20, 30, 50, 80]
        if max_train:
            max_pct = min(80, int(100 * max_train / total_labelled))
            training_pcts = [p for p in training_pcts if p <= max_pct]
            if not training_pcts:
                training_pcts = [max_pct]

        # Class info (classification only -- regression labels are
        # continuous floats, not a small fixed vocabulary to enumerate, and
        # class_names[lbl] below would TypeError on a float lbl regardless;
        # not used downstream for regression anyway, see "classes": ...
        # if is_classification else [] at the start event).
        class_info = []
        if is_classification:
            unique_labels, counts = np.unique(labels, return_counts=True)
            for lbl, cnt in zip(unique_labels, counts):
                name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
                class_info.append({"name": str(name), "pixels": int(cnt)})

        # Filter classifiers that the user hasn't installed deps for
        active_models = []
        for name in model_names:
            bn = _base_name(name)
            if bn == "unet":
                try:
                    from tessera_eval.unet import _HAS_TORCH

                    if not _HAS_TORCH:
                        logger.warning("Skipping U-Net: PyTorch not installed")
                        yield (
                            json.dumps(
                                {
                                    "event": "status",
                                    "message": f"{name} skipped — PyTorch not installed",
                                }
                            )
                            + "\n"
                        )
                        continue
                except ImportError:
                    continue
                if not unet_patches:
                    yield (
                        json.dumps(
                            {
                                "event": "status",
                                "message": f"{name} skipped — no labelled patches found",
                            }
                        )
                        + "\n"
                    )
                    continue
            if bn in ("spatial_mlp", "spatial_mlp_5x5"):
                if (bn == "spatial_mlp" and spatial_3x3 is None) or (
                    bn == "spatial_mlp_5x5" and spatial_5x5 is None
                ):
                    yield (
                        json.dumps(
                            {"event": "status", "message": f"{name} skipped — no spatial features"}
                        )
                        + "\n"
                    )
                    continue
            active_models.append(name)

        start_event = {
            "event": "start",
            "classifiers": active_models,
            "classes": class_info if is_classification else [],
            "total_labelled_pixels": total_labelled,
            "confusion_matrix_labels": class_names if is_classification else [],
            "training_pcts": training_pcts,
            "stats": stats,
            "train_year": train_year,
            "test_year": test_year,
        }
        if has_spatial_split:
            start_event["spatial_split"] = True
            start_event["train_count"] = int(len(spatial_train_labels))
            start_event["test_count"] = int(len(spatial_test_labels))
        if year_split_test_vectors is not None:
            start_event["year_split"] = True
            start_event["train_count"] = int(len(labels))
            start_event["test_count"] = int(len(year_split_test_labels))
        yield json.dumps(start_event) + "\n"

        # Run learning curve (all classifiers including U-Net)
        lc_kwargs = dict(
            repeats=5,
            classifier_params=model_params,
            spatial_vectors=spatial_3x3,
            spatial_vectors_5x5=spatial_5x5,
            spatial_labels=spatial_labels_3x3
            if spatial_labels_3x3 is not None
            else spatial_labels_5x5,
            finish_classifiers=_finish_classifiers,
            unet_patches=unet_patches,
            task=task,
        )
        if has_spatial_split:
            lc_kwargs["test_vectors"] = spatial_test_vectors
            lc_kwargs["test_labels"] = spatial_test_labels
        if year_split_test_vectors is not None:
            # Overrides the has_spatial_split assignment above when both
            # apply (bboxes drawn AND years differ) -- the test role's
            # vectors must come from test_year, not train_year.
            lc_kwargs["test_vectors"] = year_split_test_vectors
            lc_kwargs["test_labels"] = year_split_test_labels

        for event in run_learning_curve(
            vectors,
            labels,
            active_models,
            training_pcts,
            **lc_kwargs,
        ):
            if _cancelled():
                logger.info("Evaluation cancelled during learning curve")
                yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
                return
            if event["type"] == "progress":
                yield (
                    json.dumps(
                        {
                            "event": "progress",
                            "pct": event["pct"],
                            "classifiers": event["classifiers"],
                            "pixel_train_count": event.get("pixel_train_count", 0),
                            "unet_train_count": event.get("unet_train_count", 0),
                            "total_pixels": event.get("total_pixels", 0),
                            "total_unet_pixels": event.get("total_unet_pixels", 0),
                        }
                    )
                    + "\n"
                )
            elif event["type"] == "classifier_status":
                yield (
                    json.dumps(
                        {
                            "event": "status",
                            "message": event["message"],
                        }
                    )
                    + "\n"
                )
            elif event["type"] == "confusion_matrices":
                yield (
                    json.dumps(
                        {
                            "event": "confusion_matrices",
                            "confusion_matrices": event["confusion_matrices"],
                        }
                    )
                    + "\n"
                )
            elif event["type"] == "aggregate":
                # Regression mode -- run_learning_curve's analog of
                # confusion_matrices above (largest-percentage summary). The
                # frontend's regression display already expects this exact
                # {"event": "aggregate", "models": {...}} shape (built
                # against run_kfold_cv's event, which run_large_area doesn't
                # use, but the wire format matches).
                yield (
                    json.dumps(
                        {
                            "event": "aggregate",
                            "models": event["models"],
                        }
                    )
                    + "\n"
                )
            elif event["type"] == "heartbeat":
                # From _fit_with_heartbeat, during a single slow classifier fit
                # (spatial_mlp, U-Net) -- same wire event the tile-fetch phase
                # above already sends and the frontend already ignores-by-design
                # (keep-alive only). Bonus: since this loop's _cancelled() check
                # runs on every yielded event, a long fit is now also actually
                # cancellable every ~5s instead of only between whole classifiers.
                yield json.dumps({"event": "heartbeat"}) + "\n"

        # Store active_models for deferred training
        _tile_cache["_active_models"] = active_models
        _tile_cache["_model_params"] = model_params
        _tile_cache["_unet_patches"] = unet_patches
        # train_models() (Download Models) runs in a later, separate request
        # with no body of its own -- it needs to know whether to dispatch to
        # make_classifier or make_regressor, so stash it here.
        _tile_cache["_is_classification"] = is_classification

        _cancel_flag = None  # reset cancellation flag
        elapsed = time.time() - t0
        yield (
            json.dumps(
                {
                    "event": "done",
                    "elapsed_seconds": round(elapsed, 1),
                    "field": field_name,
                    "train_year": train_year,
                    "test_year": test_year,
                    "models_available": list(_trained_models.keys()),
                }
            )
            + "\n"
        )

    return Response(
        _padded(stream()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/evaluation/train-models", methods=["POST"])
def train_models():
    """Train final models on full data for download. Deferred from evaluation."""
    cache = _tile_cache
    if cache.get("vectors") is None:
        return jsonify({"error": "No evaluation data. Run evaluation first."}), 400

    vectors = cache["vectors"]
    labels = cache["labels"]
    class_names = cache.get("class_names", [])
    active_models = cache.get("_active_models", [])
    model_params = cache.get("_model_params", {})
    unet_patches = cache.get("_unet_patches", [])
    spatial_3x3 = cache.get("spatial_3x3")
    spatial_5x5 = cache.get("spatial_5x5")
    # Older cache entries (from before this key existed) default to
    # classification -- the only task this endpoint supported at the time.
    is_classification = cache.get("_is_classification", True)

    if not active_models:
        return jsonify({"error": "No classifiers configured."}), 400

    # For regression, `labels` are continuous values (e.g. tree heights), not
    # class indices -- np.unique(labels) would produce one "class" per
    # distinct float and class_names is empty anyway (see run_large_area).
    valid_class_names = (
        [
            class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
            for lbl in sorted(np.unique(labels))
        ]
        if is_classification
        else []
    )

    def stream():
        from tessera_eval.classify import make_classifier, make_regressor

        # Clean up old models
        for old_path in _trained_models.values():
            try:
                Path(old_path).unlink(missing_ok=True)
            except OSError:
                pass
        _trained_models.clear()

        yield (
            json.dumps({"event": "status", "message": "Training final models for download..."})
            + "\n"
        )

        for name in active_models:
            logger.info("Training %s...", name)
            yield json.dumps({"event": "status", "message": f"Training {name}..."}) + "\n"
            try:
                import re as _re

                _bn = _re.sub(r"_v\d+$", "", name)
                if _bn == "unet":
                    import torch as _torch

                    if is_classification:
                        from tessera_eval.unet import _HAS_TORCH, train_unet_on_patches
                    else:
                        from tessera_eval.unet import (
                            _HAS_TORCH,
                        )
                        from tessera_eval.unet import (
                            train_unet_regressor_on_patches as train_unet_on_patches,
                        )

                    if _HAS_TORCH and unet_patches:
                        _unet_progress = []

                        def _unet_cb(epoch, total, loss):
                            _unet_progress.append((epoch, total, loss))

                        if is_classification:
                            n_cls = len(np.unique(labels))
                            model = train_unet_on_patches(
                                unet_patches,
                                n_cls,
                                model_params.get(name, {}),
                                progress_callback=_unet_cb,
                            )
                        else:
                            # Regression U-Net is single-channel -- no n_cls arg.
                            model = train_unet_on_patches(
                                unet_patches,
                                model_params.get(name, {}),
                                progress_callback=_unet_cb,
                            )
                        for ep, tot, loss in _unet_progress:
                            yield (
                                json.dumps(
                                    {
                                        "event": "status",
                                        "message": f"U-Net epoch {ep}/{tot} loss={loss:.4f}",
                                    }
                                )
                                + "\n"
                            )
                        tmp = tempfile.NamedTemporaryFile(
                            suffix=".pt", prefix=f"{name}_model_", delete=False
                        )
                        _torch.save(
                            {"model_state": model.state_dict(), "class_names": valid_class_names},
                            tmp.name,
                        )
                        _trained_models[name] = tmp.name
                    else:
                        yield (
                            json.dumps(
                                {
                                    "event": "status",
                                    "message": f"{name} skipped — no patches or PyTorch",
                                }
                            )
                            + "\n"
                        )
                        continue
                elif _bn in ("spatial_mlp", "spatial_mlp_5x5") and not is_classification:
                    # No regressor variant exists for spatial-context models yet.
                    yield (
                        json.dumps(
                            {
                                "event": "status",
                                "message": f"{name} skipped — spatial MLP regression isn't supported yet",
                            }
                        )
                        + "\n"
                    )
                    continue
                elif _bn == "spatial_mlp" and spatial_3x3 is not None:
                    from tessera_eval.classify import augment_spatial

                    X_aug, y_aug = augment_spatial(
                        spatial_3x3, labels, window=3, dim=vectors.shape[1]
                    )
                    clf = make_classifier(name, model_params.get(name, {}))
                    clf.fit(X_aug, y_aug)
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".joblib", prefix=f"{name}_model_", delete=False
                    )
                    joblib.dump({"model": clf, "class_names": valid_class_names}, tmp.name)
                    _trained_models[name] = tmp.name
                elif _bn == "spatial_mlp_5x5" and spatial_5x5 is not None:
                    from tessera_eval.classify import augment_spatial

                    X_aug, y_aug = augment_spatial(
                        spatial_5x5, labels, window=5, dim=vectors.shape[1]
                    )
                    clf = make_classifier(name, model_params.get(name, {}))
                    clf.fit(X_aug, y_aug)
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".joblib", prefix=f"{name}_model_", delete=False
                    )
                    joblib.dump({"model": clf, "class_names": valid_class_names}, tmp.name)
                    _trained_models[name] = tmp.name
                else:
                    clf = (
                        make_classifier(name, model_params.get(name, {}))
                        if is_classification
                        else make_regressor(name, model_params.get(name, {}))
                    )
                    clf.fit(vectors, labels)
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".joblib", prefix=f"{name}_model_", delete=False
                    )
                    joblib.dump({"model": clf, "class_names": valid_class_names}, tmp.name)
                    _trained_models[name] = tmp.name
                logger.info("Trained model '%s' → %s", name, tmp.name)
                yield json.dumps({"event": "model_ready", "classifier": name}) + "\n"
            except Exception as e:
                logger.warning("Failed to train model '%s': %s", name, e)
                yield (
                    json.dumps({"event": "status", "message": f"Failed to train {name}: {e}"})
                    + "\n"
                )

        yield json.dumps({"event": "done", "models_available": list(_trained_models.keys())}) + "\n"

    return Response(
        _padded(stream()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/evaluation/download-model/<name>", methods=["GET"])
def download_model(name):
    """Serve a trained model file."""
    path = _trained_models.get(name)
    if not path or not Path(path).exists():
        return jsonify({"error": f"No trained model for '{name}'"}), 404
    import re as _re

    _bn = _re.sub(r"_v\d+$", "", name)
    ext = ".pt" if _bn == "unet" else ".joblib"
    return send_file(path, as_attachment=True, download_name=f"{name}_model{ext}")


@app.route("/api/evaluation/create-map", methods=["POST"])
def create_map():
    """Train classifier on all cached data, predict every pixel in map bboxes, produce GeoTIFF.

    Only pixel-based classifiers are supported (k-NN, RF, XGBoost, MLP).
    Spatial MLP and U-Net require neighbourhood features at every pixel
    which is prohibitively expensive for dense prediction.

    Processes map bboxes in 0.1 deg geographic chunks to cap memory (~50MB/chunk).
    """
    try:
        body = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    classifier_name = body.get("classifier", "rf")
    map_bboxes = body.get("map_bboxes", [])
    # Optional: predict using a *different* year's embeddings than the model
    # was trained on -- e.g. train on 2025 (validated in the Validation
    # pane), then map 2018 to look for change over time. Pure inference: no
    # ground truth needed for map_year, nothing gets scored, the model is
    # just applied as-is. Defaults to the training year (today's existing
    # behavior) when omitted.
    map_year_override = body.get("map_year")

    if not map_bboxes:
        return jsonify(
            {"error": "No map bounding boxes provided. Draw green map areas first."}
        ), 400

    cache = _tile_cache
    if cache.get("vectors") is None:
        return jsonify({"error": "No evaluation data cached. Run evaluation first."}), 400

    vectors = cache["vectors"]
    labels = cache["labels"]
    class_names = cache.get("class_names", [])
    model_params = cache.get("_model_params", {})
    # Older cache entries (from before this key existed) default to
    # classification -- the only task this endpoint supported at the time.
    is_classification = cache.get("_is_classification", True)
    # classifier_name is a plain UI name ("rf", "xgboost", ...) either way --
    # the frontend doesn't know about the "_reg" suffix convention. For
    # regression this was previously fed straight into make_classifier(),
    # which either crashed (XGBoost validates class labels strictly) or
    # silently "succeeded" by treating continuous height values as an
    # arbitrary set of classes (k-NN/RF/MLP don't validate that) -- a wrong
    # map, not a safe one. model_key is what actually selects/looks up the
    # model; classifier_name (unchanged) still names the UI selection in
    # status messages and GeoTIFF tags.
    model_key = (
        _CLF_TO_REG.get(classifier_name, classifier_name)
        if not is_classification
        else classifier_name
    )

    import re as _re

    base_name = _re.sub(r"_v\d+$", "", classifier_name)
    unsupported = ("spatial_mlp", "spatial_mlp_5x5", "unet")
    if base_name in unsupported:
        return jsonify(
            {
                "error": f"'{classifier_name}' is not supported for map generation. "
                f"Spatial MLP and U-Net require neighbourhood features at every pixel, "
                f"which is too expensive for dense prediction. Use k-NN, RF, XGBoost, or MLP."
            }
        ), 400

    def stream():
        import threading

        global _cancel_flag
        _cancel_flag = threading.Event()

        def _cancelled():
            return _cancel_flag is not None and _cancel_flag.is_set()

        t0 = time.time()

        yield (
            json.dumps(
                {
                    "event": "status",
                    "message": f"Training {classifier_name} on all {len(vectors):,} labels...",
                }
            )
            + "\n"
        )

        try:
            from tessera_eval.classify import make_classifier, make_regressor

            clf = (
                make_classifier(model_key, model_params.get(model_key, {}))
                if is_classification
                else make_regressor(model_key, model_params.get(model_key, {}))
            )
            clf.fit(vectors, labels)
        except Exception as e:
            yield (
                json.dumps({"event": "error", "message": f"Failed to train classifier: {e}"}) + "\n"
            )
            return

        yield (
            json.dumps(
                {"event": "status", "message": "Classifier trained. Predicting map areas..."}
            )
            + "\n"
        )

        if _cancelled():
            yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
            return

        # Prepare GeoTessera and zarr
        from geotessera import GeoTessera

        global _geotessera_instance

        if _geotessera_instance is None:
            tile_cache_dir = _get_cache_dir() / "tiles"
            tile_cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                _geotessera_instance = GeoTessera(embeddings_dir=str(tile_cache_dir))
            except Exception as e:
                logger.warning("GeoTessera initialization failed: %s", e)
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "message": (
                                "Could not initialize GeoTessera -- check that this "
                                f"machine has network access to the Tessera embeddings "
                                f"store: {e}"
                            ),
                        }
                    )
                    + "\n"
                )
                return
        gt = _geotessera_instance

        # cache["key"] is (field_name, train_year, test_year, sampling) --
        # index 1 regardless of test_year. map_year defaults to this (today's
        # existing behavior: map the same year the model was trained on) but
        # can be overridden to run the trained model as pure inference
        # against a different year's embeddings entirely -- see map_year's
        # comment above for why.
        train_year = cache["key"][1] if cache.get("key") else 2024
        map_year = map_year_override or train_year

        if map_year != train_year:
            yield (
                json.dumps(
                    {
                        "event": "status",
                        "message": (
                            f"Trained on {train_year} -- predicting {map_year} "
                            f"embeddings (inference only, not re-evaluated)"
                        ),
                    }
                )
                + "\n"
            )

        # Clean up old map files
        for old_path in _generated_maps.values():
            try:
                Path(old_path).unlink(missing_ok=True)
            except OSError:
                pass
        _generated_maps.clear()

        for bbox_idx, bbox in enumerate(map_bboxes):
            if _cancelled():
                yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
                return

            # bbox = [south, west, north, east]
            south, west, north, east = bbox
            yield (
                json.dumps(
                    {
                        "event": "status",
                        "message": f"Map area {bbox_idx + 1}/{len(map_bboxes)}: ({south:.3f}, {west:.3f}) to ({north:.3f}, {east:.3f})",
                    }
                )
                + "\n"
            )

            # Split bbox into 0.1 deg chunks to manage memory
            CHUNK_SIZE = 0.1
            chunk_lons = []
            lon = west
            while lon < east:
                chunk_lons.append((lon, min(lon + CHUNK_SIZE, east)))
                lon += CHUNK_SIZE
            chunk_lats = []
            lat = south
            while lat < north:
                chunk_lats.append((lat, min(lat + CHUNK_SIZE, north)))
                lat += CHUNK_SIZE

            total_chunks = len(chunk_lons) * len(chunk_lats)
            yield (
                json.dumps(
                    {
                        "event": "status",
                        "message": f"Map area {bbox_idx + 1}: {total_chunks} chunks ({len(chunk_lons)} x {len(chunk_lats)})",
                    }
                )
                + "\n"
            )

            # Probe zarr coverage
            gtz = get_zarr()
            use_zarr = gtz is not None and probe_zarr_coverage(
                gtz, (west, south, east, north), map_year
            )

            yield (
                json.dumps(
                    {
                        "event": "status",
                        "message": f"Using {'zarr (fast)' if use_zarr else 'NPY tiles'} for predictions",
                    }
                )
                + "\n"
            )

            # We'll collect chunk arrays and merge at the end.
            # To build the final GeoTIFF, we need to know the CRS and resolution.
            # We get this from the first chunk that returns data.
            chunk_results = []  # list of (predicted_2d, transform, crs, chunk_bbox)
            chunk_counter = 0

            for lon_start, lon_end in chunk_lons:
                for lat_start, lat_end in chunk_lats:
                    if _cancelled():
                        yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
                        return

                    chunk_counter += 1
                    yield (
                        json.dumps(
                            {
                                "event": "map_progress",
                                "bbox_idx": bbox_idx,
                                "chunk": chunk_counter,
                                "total_chunks": total_chunks,
                                "message": f"Predicting chunk {chunk_counter}/{total_chunks}",
                            }
                        )
                        + "\n"
                    )

                    chunk_bbox = (lon_start, lat_start, lon_end, lat_end)

                    try:
                        if use_zarr:
                            emb, transform, crs = gtz.read_region(chunk_bbox, map_year)
                        else:
                            # Fall back to NPY: fetch tiles overlapping this chunk
                            tiles = gt.registry.load_blocks_for_region(chunk_bbox, map_year)
                            tiles = list(tiles)
                            if not tiles:
                                continue
                            tile_gen = gt.fetch_embeddings(tiles)
                            try:
                                _, _, _, emb, crs, transform = next(tile_gen)
                                emb = emb.astype(np.float32)
                            except StopIteration:
                                continue

                        if emb is None or emb.size == 0:
                            continue

                        h, w, c = emb.shape
                        # Flatten to (N, 128), predict, reshape
                        flat = emb.reshape(-1, c)

                        # Identify valid (non-NaN) pixels
                        nan_mask = np.isnan(flat).any(axis=1)
                        if is_classification:
                            predictions = np.zeros(flat.shape[0], dtype=np.uint8)
                        else:
                            # Real continuous values -- uint8 would truncate
                            # them and collide 0-as-nodata with 0 as a real
                            # height. NaN is the nodata sentinel instead.
                            predictions = np.full(flat.shape[0], np.nan, dtype=np.float32)

                        if (~nan_mask).sum() > 0:
                            valid_flat = flat[~nan_mask].astype(np.float32)
                            preds = clf.predict(valid_flat)
                            if is_classification:
                                # Class labels are 0-based from LabelEncoder.
                                # Store as 1-based (0 = nodata).
                                predictions[~nan_mask] = preds.astype(np.uint8) + 1
                            else:
                                predictions[~nan_mask] = preds.astype(np.float32)

                        predicted_2d = predictions.reshape(h, w)
                        chunk_results.append((predicted_2d, transform, crs, chunk_bbox))

                    except Exception as e:
                        logger.warning("Chunk %d failed: %s", chunk_counter, e)
                        yield (
                            json.dumps(
                                {
                                    "event": "status",
                                    "message": f"Chunk {chunk_counter} failed: {e}",
                                }
                            )
                            + "\n"
                        )
                        continue

            if not chunk_results:
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "message": f"No data found in map area {bbox_idx + 1}. Check embedding coverage.",
                        }
                    )
                    + "\n"
                )
                continue

            # Merge chunks into a single GeoTIFF
            yield (
                json.dumps(
                    {
                        "event": "status",
                        "message": f"Writing GeoTIFF for map area {bbox_idx + 1}...",
                    }
                )
                + "\n"
            )

            try:
                import rasterio
                import rasterio.io
                from rasterio.merge import merge as _rasterio_merge

                # Classification: 1-based class IDs, 0 = nodata, fits uint8.
                # Regression: real continuous values (e.g. heights), NaN =
                # nodata (0 is a valid real value, can't double as nodata).
                out_dtype = "uint8" if is_classification else "float32"
                out_nodata = 0 if is_classification else np.nan

                if len(chunk_results) == 1:
                    # Single chunk — write directly
                    predicted_2d, transform, crs, _ = chunk_results[0]
                    out_arr = predicted_2d
                    out_transform = transform
                    out_crs = crs
                else:
                    # Multiple chunks — merge using rasterio in-memory datasets
                    datasets = []
                    for predicted_2d, transform, crs, _ in chunk_results:
                        memfile = rasterio.io.MemoryFile()
                        ds = memfile.open(
                            driver="GTiff",
                            height=predicted_2d.shape[0],
                            width=predicted_2d.shape[1],
                            count=1,
                            dtype=out_dtype,
                            crs=crs,
                            transform=transform,
                            nodata=out_nodata,
                        )
                        ds.write(predicted_2d, 1)
                        datasets.append(ds)

                    mosaic, out_transform = _rasterio_merge(datasets, nodata=out_nodata)
                    out_arr = mosaic[0]
                    out_crs = chunk_results[0][2]

                    for ds in datasets:
                        ds.close()

                # Write final GeoTIFF
                map_name = f"map_{bbox_idx + 1}"
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".tif",
                    prefix=f"tee_map_{bbox_idx + 1}_",
                    delete=False,
                )
                with rasterio.open(
                    tmp.name,
                    "w",
                    driver="GTiff",
                    height=out_arr.shape[0],
                    width=out_arr.shape[1],
                    count=1,
                    dtype=out_dtype,
                    crs=out_crs,
                    transform=out_transform,
                    nodata=out_nodata,
                    compress="lz4",
                ) as dst:
                    dst.write(out_arr, 1)
                    # Store class names as tags (classification) -- empty for
                    # regression, which has no class taxonomy; tag the field
                    # name instead so a regression GeoTIFF is still self-describing.
                    tags = {f"class_{i + 1}": name for i, name in enumerate(class_names)}
                    tags["classifier"] = classifier_name
                    tags["train_year"] = str(train_year)
                    tags["map_year"] = str(map_year)
                    if not is_classification:
                        tags["field"] = cache["key"][0] if cache.get("key") else ""
                    dst.update_tags(**tags)

                _generated_maps[map_name] = tmp.name
                logger.info(
                    "GeoTIFF written: %s (%d x %d)", tmp.name, out_arr.shape[1], out_arr.shape[0]
                )

                yield (
                    json.dumps(
                        {
                            "event": "map_ready",
                            "name": map_name,
                            "bbox_idx": bbox_idx,
                            "download_url": f"/api/evaluation/download-map/{map_name}",
                            "width": out_arr.shape[1],
                            "height": out_arr.shape[0],
                            "n_classes": len(class_names),
                            "train_year": train_year,
                            "map_year": map_year,
                        }
                    )
                    + "\n"
                )

            except Exception as e:
                logger.error("GeoTIFF write failed: %s", e, exc_info=True)
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "message": f"Failed to write GeoTIFF: {e}",
                        }
                    )
                    + "\n"
                )
                continue

        _cancel_flag = None
        elapsed = time.time() - t0
        yield (
            json.dumps(
                {
                    "event": "done",
                    "elapsed_seconds": round(elapsed, 1),
                    "maps_available": list(_generated_maps.keys()),
                }
            )
            + "\n"
        )

    return Response(
        _padded(stream()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/evaluation/download-map/<name>", methods=["GET"])
def download_map(name):
    """Serve a generated GeoTIFF map file."""
    path = _generated_maps.get(name)
    if not path or not Path(path).exists():
        return jsonify({"error": f"No generated map '{name}'"}), 404
    return send_file(path, as_attachment=True, download_name=f"{name}.tif", mimetype="image/tiff")


@app.route("/health", methods=["GET"])
def health():
    """Health check — reports status, hosted server, and loaded data."""
    import socket

    gdf = _get_merged_gdf()
    return jsonify(
        {
            "status": "ok",
            "mode": "compute",
            "compute_host": socket.gethostname(),
            "hosted": _hosted_url,
            "version": _get_version(),
            "shapefiles": len(_uploaded_shapefiles),
            "features": len(gdf) if gdf is not None else 0,
            "models_available": list(_trained_models.keys()),
        }
    )


# ── Reverse proxy for everything else ──


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def proxy(path):
    """Forward all non-eval requests to the hosted server.

    Uses the shared _proxy_session (module-level requests.Session), not the
    top-level requests.request() -- that helper opens a fresh Session, and
    therefore a fresh TCP+TLS handshake, for every single call. A page load
    against a --hosted server is never just one request (HTML, several JS
    modules, CSS, auth/config/viewport-list API calls, ...), so paying a full
    handshake per request compounds badly: reported live as TEE's UI taking
    "several minutes" to even load through a gpu-box deploy (see
    deploy-compute.sh) sitting on the same network as the hosted server --
    geography wasn't the explanation, repeated handshakes were. Reusing one
    Session keeps the connection alive via requests' pooled HTTPAdapter.
    """
    if not _hosted_url:
        return jsonify({"error": "No --hosted URL configured"}), 502

    target = f"{_hosted_url}/{path}"
    if request.query_string:
        target += f"?{request.query_string.decode()}"

    # Forward headers (skip hop-by-hop)
    skip = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers if k.lower() not in skip}

    try:
        resp = _proxy_session.request(
            method=request.method,
            url=target,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            stream=True,
            timeout=300,
        )
    except requests.ConnectionError:
        return jsonify({"error": f"Cannot reach hosted server at {_hosted_url}"}), 502
    except requests.Timeout:
        return jsonify({"error": "Hosted server timed out"}), 504

    # Stream response back
    proxy_headers = {}
    for k, v in resp.headers.items():
        if k.lower() not in (
            "content-encoding",
            "content-length",
            "transfer-encoding",
            "connection",
        ):
            proxy_headers[k] = v

    return Response(
        resp.iter_content(chunk_size=8192),
        status=resp.status_code,
        headers=proxy_headers,
    )


# ── Helpers ──


def _get_version():
    try:
        from tessera_eval import __version__

        return __version__
    except Exception:
        return "unknown"


# ── CLI entry point ──


def main():
    global _hosted_url

    parser = argparse.ArgumentParser(
        description="TEE compute server — run ML evaluation locally, proxy data from hosted server",
    )
    parser.add_argument(
        "--hosted",
        default="https://tee.cl.cam.ac.uk",
        help="URL of the hosted TEE server for data/UI (default: https://tee.cl.cam.ac.uk)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to serve on (default: 8001)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in Flask debug mode (auto-reload, verbose errors)",
    )
    args = parser.parse_args()

    _hosted_url = args.hosted.rstrip("/")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("TEE Compute Server")
    logger.info("  Hosted server: %s", _hosted_url)
    logger.info("  Listening on:  http://%s:%d", args.host, args.port)
    logger.info("")
    logger.info("Open http://localhost:%d in your browser", args.port)

    if args.debug:
        app.run(host=args.host, port=args.port, debug=True)
    else:
        from waitress import serve

        serve(app, host=args.host, port=args.port, threads=4, channel_timeout=7200)


if __name__ == "__main__":
    main()
