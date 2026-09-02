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
import uuid
import zipfile
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import requests
from flask import Flask, Response, jsonify, request, send_file

from tessera_eval.classify import SPATIAL_MODELS

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


def _resolve_task(cache, task_override=None):
    """Decide classification vs regression for a map / final-model request.

    Precedence:
      1. An explicit "classification"/"regression" from the request body --
         the user forcing a task (e.g. a coarsely-binned continuous field
         they want mapped as regression). Anything else (None, "auto") is
         ignored here; task auto-detection happens once, in run_large_area.
      2. The task the evaluation run committed to the tile cache next to the
         vectors it applies to.
      3. A data-derived fallback: regression runs leave class_names empty,
         classification runs populate it from the LabelEncoder.

    (2) used to be a bare cache.get("_is_classification", True). That key was
    only written at the very end of run_large_area's response stream, so an
    evaluation cut short before its final event -- a client disconnect or
    cancel mid-run -- left it unset and every downstream map silently ran as
    classification: a classifier fit on continuous targets, uint8 output
    snapped onto the training values, class-palette preview. run_large_area
    now writes the key with the vectors; this fallback covers older cache
    entries and the no-key edge.
    """
    if task_override in ("classification", "regression"):
        return task_override == "classification"
    if cache.get("_is_classification") is not None:
        return bool(cache["_is_classification"])
    return bool(cache.get("class_names"))


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
_zarr_instance = None  # cached GeoTesseraZarr handle; False = tried and failed
_cancel_flag = None  # threading.Event, set when user cancels
# Shared across every proxy() call so the TCP+TLS connection to _hosted_url is
# kept alive and reused (requests' connection-pooling adapter), instead of a
# fresh handshake per proxied request -- see proxy()'s docstring for why this
# matters. Sharing one Session across threads is standard/safe for this: no
# concurrent mutation of cookies/state beyond urllib3's own thread-safe pools.
_proxy_session = requests.Session()

FLUSH_PAD = 18 * 1024  # pad NDJSON lines to force Waitress flush

ZARR_CACHE_MAX_BYTES = 20 * 1024**3  # bound the on-disk zarr chunk cache

# In-browser map preview (create_map): a small lat/lon PNG + legend the
# viewer drops on the map as an L.imageOverlay, so a map can be eyeballed
# without downloading the GeoTIFF and opening it in QGIS. The GeoTIFF is
# still the real deliverable -- the preview is best-effort and its failure
# never blocks a map.
_PREVIEW_MAX_PX = 1024
_CLASS_PREVIEW_PALETTE = (
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fab1c0",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#c8b900",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
)
_REG_PREVIEW_RAMP = ("#2b4abd", "#26b25c", "#ffe119", "#e63c3c")


def _get_cache_dir():
    """Return the cache directory, creating it if needed."""
    global _tile_disk_cache_dir
    if _tile_disk_cache_dir is None:
        _tile_disk_cache_dir = Path.home() / ".cache" / "tessera-eval"
    _tile_disk_cache_dir.mkdir(parents=True, exist_ok=True)
    return _tile_disk_cache_dir


def _get_zarr():
    """Return a cached GeoTesseraZarr handle, or None when zarr is unavailable.

    The store is opened once per process and the outcome is cached, failure
    included; callers fall back to the NPY tile path on None. Chunk reads
    are cached on disk alongside the NPY tile cache.
    """
    global _zarr_instance
    if _zarr_instance is None:
        try:
            from geotessera.store import GeoTesseraZarr

            inst = GeoTesseraZarr(
                cache_dir=str(_get_cache_dir() / "zarr"),
                cache_max_size=ZARR_CACHE_MAX_BYTES,
            )
            if getattr(inst, "years", None):
                logger.info("GeoTesseraZarr available: %s", inst.url)
                _zarr_instance = inst
            else:
                logger.info("Zarr store has no tiles; using NPY tiles")
                _zarr_instance = False
        except Exception as e:
            logger.info("Zarr store unavailable (%s); using NPY tiles", e)
            _zarr_instance = False
    return _zarr_instance or None


def _probe_zarr_coverage(gtz, bounds, year):
    """True when the zarr store has a valid embedding for *year* at the
    centre of *bounds* (west, south, east, north).

    geotessera's single-pixel probe distinguishes genuine coverage from
    water and from areas not yet produced; anything but a valid embedding
    sends the caller to the NPY tile path.
    """
    try:
        if year not in getattr(gtz, "years", []):
            return False
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        _vec, status = gtz.probe(cx, cy, year)
        return status == "valid"
    except Exception:
        return False


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
    seed=42,
):
    """Extract pixel-aligned 2D patches and optionally point samples from tiles.

    Uses zarr read_region() when available (roughly an order of magnitude
    faster on a cold cache than pulling whole NPY tiles), falling back to
    gt.fetch_embeddings() for NPY tile downloads.

    is_classification=False (regression): patches carry real continuous
    target values (via rasterize_shapefile_continuous, NaN = unlabelled)
    instead of LabelEncoder class IDs (via rasterize_shapefile, 0 =
    unlabelled) -- le/n_classes are unused in that case. spatial_mlp and
    spatial_mlp_5x5 do support regression (make_regressor recognizes both
    names directly, no "_reg" suffix -- see its docstring) -- the spatial
    labels returned here (all_spatial_labels_3x3/5x5) carry real values for
    regression too, not the 1-based-to-0-based-shifted class indices
    classification needs.

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

    rng = np.random.RandomState(seed)

    # Find tiles overlapping the shapefile
    bounds = gdf.total_bounds

    # Try zarr — but verify coverage with a single-pixel probe first,
    # since the zarr store only has 2025 for some regions.
    gtz = _get_zarr()
    use_zarr = gtz is not None and _probe_zarr_coverage(gtz, bounds, year)
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

            # The "- 1" shift converts label_patch's 1-based class IDs (0 =
            # unlabelled, already excluded by labelled_mask) to 0-based
            # indices make_classifier's models expect. Regression targets
            # are real continuous values (e.g. heights), not class IDs --
            # shifting them by 1 would silently corrupt every value.
            if needs_spatial_3x3:
                sf = gather_spatial_features_2d(emb_patch, radius=1, mask=labelled_mask)
                all_spatial_3x3.append(sf)
                lbls = label_patch[labelled_mask]
                all_spatial_labels_3x3.append(lbls - 1 if is_classification else lbls)
            if needs_spatial_5x5:
                sf = gather_spatial_features_2d(emb_patch, radius=2, mask=labelled_mask)
                all_spatial_5x5.append(sf)
                lbls = label_patch[labelled_mask]
                all_spatial_labels_5x5.append(lbls - 1 if is_classification else lbls)

        if logger:
            logger.info("  %d patches so far (%d from this tile)", len(unet_patches), n_pick)

    spatial_3x3 = (
        np.concatenate(all_spatial_3x3, axis=0).astype(np.float32) if all_spatial_3x3 else None
    )
    spatial_5x5 = (
        np.concatenate(all_spatial_5x5, axis=0).astype(np.float32) if all_spatial_5x5 else None
    )
    # int32 class IDs for classification; float32 real values for regression
    # -- same reasoning as unet_patches's label dtype above. This cast used
    # to be unconditionally int32 regardless of is_classification, which
    # truncated every regression target to its integer part (e.g. a height
    # of 3.7 silently became 3) -- a second, independent bug from the
    # classification-only "-1" shift a few lines up (that one corrupted the
    # value additively, this one corrupted it by truncation; both needed
    # fixing, neither implies the other).
    _spatial_label_dtype = np.int32 if is_classification else np.float32
    spatial_labels_3x3 = (
        np.concatenate(all_spatial_labels_3x3).astype(_spatial_label_dtype)
        if all_spatial_labels_3x3
        else None
    )
    spatial_labels_5x5 = (
        np.concatenate(all_spatial_labels_5x5).astype(_spatial_label_dtype)
        if all_spatial_labels_5x5
        else None
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
    cached_unet_patches,
    cached_sample_points,
    *,
    needs_spatial_3x3,
    needs_spatial_5x5,
    needs_unet,
    has_spatial_bboxes,
):
    """Does an in-memory _tile_cache hit (same key, vectors present) still
    need a fresh reload, because this specific request needs data the cached
    entry doesn't have?

    True when the cache was populated by an earlier request that didn't need
    spatial_mlp/spatial_mlp_5x5 features, didn't need U-Net patches, or
    didn't need a spatial train/test split (so never cached sample-point
    coordinates), and *this* request does. Whoever calls this with True must
    also invalidate the cache's own key (not just its own local `vectors`
    variable) -- confirmed live as a real bug otherwise: run_large_area's
    cache-hit branch used to set `vectors = None` here without touching
    `_tile_cache["key"]`, so the later "reload from GeoTessera" block
    (guarded by `_tile_cache["key"] != cache_key`) never triggered either,
    since the key still matched. vectors stayed None all the way through and
    crashed downstream at `len(vectors)` (TypeError: object of type
    'NoneType' has no len()) on a spatial_mlp request that hit a cache entry
    from a prior non-spatial run.

    The needs_unet check below is the same bug for U-Net, confirmed live
    (Louis Driver): running any plain pixel regressor first (field/year/
    sampling unchanged, no U-Net needed) cached unet_patches=[]. Checking
    U-Net next, with the same field/year/sampling, hit this in-memory cache
    -- and since this function didn't know about needs_unet, it reported
    "no reload needed", silently reusing the empty patch list. U-Net then
    got filtered out of active_models entirely (server.py's "no labelled
    patches found" skip), producing a 0-classifier run that finished
    suspiciously fast with no error at all.
    """
    if (needs_spatial_3x3 and cached_spatial_3x3 is None) or (
        needs_spatial_5x5 and cached_spatial_5x5 is None
    ):
        return True
    if needs_unet and not cached_unet_patches:
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


@app.route("/api/evaluation/list-shapefiles", methods=["GET"])
def list_shapefiles():
    """Names + feature counts of the shapefiles currently merged into the
    ground-truth set. Uploads accumulate (multi-shapefile merge, by design);
    the viewer shows this on entering Validation so an earlier upload that's
    still in the set is visible rather than a surprise."""
    return jsonify(
        {"files": [{"name": name, "features": int(len(gdf))} for name, gdf in _uploaded_shapefiles]}
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
    # "learning_curve" (default) or "kfold". k-fold cross-validates over all
    # labelled pixels (no train/test bboxes, no learning curve, pixel models
    # only) -- run_kfold_cv, previously CLI-only.
    eval_mode = body.get("eval_mode", "learning_curve")
    kfold_k = max(2, min(20, int(body.get("kfold_k", 5))))
    # One seed for the whole run: sample-point selection, tile-fetch order,
    # learning-curve resampling / k-fold splits, and every estimator's own
    # random_state (RF/XGBoost/MLP, U-Net). Settable from the UI / CLI.
    seed = int(body.get("seed", 42))

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

    # k-fold CV cross-validates over the pixel-embedding matrix directly --
    # run_kfold_cv has no neighbourhood/patch path -- so drop spatial MLP
    # and U-Net now, before their (expensive) feature extraction is set up.
    # The stream reports what was dropped once it starts.
    kfold_dropped = []
    if eval_mode == "kfold":
        _pixel = [n for n in model_names if _base_name(n) not in (*SPATIAL_MODELS, "unet")]
        kfold_dropped = [n for n in model_names if n not in _pixel]
        model_names = _pixel

    # A fixed test set (spatial bboxes, or a different test year) has no
    # neighbourhood features, so spatial models are skipped for such runs
    # and their (expensive) feature extraction is not worth doing.
    has_fixed_test_set = bool(train_bboxes or test_bboxes) or test_year != train_year

    # Determine which spatial features are needed (check base names)
    needs_spatial_3x3 = (
        any(_base_name(n) == "spatial_mlp" for n in model_names) and not has_fixed_test_set
    )
    needs_spatial_5x5 = (
        any(_base_name(n) == "spatial_mlp_5x5" for n in model_names) and not has_fixed_test_set
    )
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
        has_spatial_bboxes = has_fixed_test_set
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
            # Keep the cached task type in step with the vectors being
            # reused (a run cut short before this generator's final event
            # may have left it stale or unset).
            _tile_cache["_is_classification"] = is_classification
            _tile_cache["_seed"] = seed

            # See _cached_tiles_need_reload's docstring: setting vectors =
            # None alone (without also invalidating _tile_cache["key"]) used
            # to leave vectors None all the way through and crash downstream
            # at `len(vectors)` -- confirmed live on a spatial_mlp request
            # that hit a cache entry from a prior non-spatial run.
            if _cached_tiles_need_reload(
                spatial_3x3,
                spatial_5x5,
                unet_patches,
                all_sample_points,
                needs_spatial_3x3=needs_spatial_3x3,
                needs_spatial_5x5=needs_spatial_5x5,
                needs_unet=needs_unet,
                has_spatial_bboxes=has_spatial_bboxes,
            ):
                if (needs_spatial_3x3 and spatial_3x3 is None) or (
                    needs_spatial_5x5 and spatial_5x5 is None
                ):
                    logger.info("Spatial features needed but not cached — reloading tiles")
                elif needs_unet and not unet_patches:
                    logger.info("U-Net needed but no patches cached — reloading tiles")
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
                    # Commit the task type alongside the vectors it applies
                    # to -- see the identical key in the fetch path below.
                    "_is_classification": is_classification,
                    "_seed": seed,
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

                    sampling_rng = np.random.RandomState(seed)
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
                    sampling_rng = np.random.RandomState(seed)
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
                                seed=seed,
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
                        # Commit the task type alongside the vectors it
                        # applies to, so create_map()/train_models() can
                        # trust it even when this stream is cut short before
                        # its final event (client disconnect, cancel). The
                        # sole previous write was at the very end of this
                        # generator -- an interrupted regression run then
                        # left _is_classification unset and the map silently
                        # ran as classification. See
                        # test_map_task_type_persisted_early.
                        "_is_classification": is_classification,
                        "_seed": seed,
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
            if bn in SPATIAL_MODELS:
                if has_fixed_test_set:
                    yield (
                        json.dumps(
                            {
                                "event": "status",
                                "message": (
                                    f"{name} skipped — spatial models are not supported "
                                    "with a separate test region or test year"
                                ),
                            }
                        )
                        + "\n"
                    )
                    continue
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
            # "field_start" also carries the task, but it's only emitted on
            # a cache miss (gated by `_tile_cache["key"] != cache_key`,
            # further up) -- any run that hits the in-memory or disk cache
            # (e.g. re-running with different classifiers but the same
            # field/year/sampling, or after a prior identical run in the
            # same session) never sees it. The frontend used to rely on
            # field_start alone to set its task-tracking state, so a
            # cache-hit run silently kept whatever task the *previous* run
            # left it at (or null on the session's first run if something
            # upstream ever skipped field_start) -- confirmed live (Louis
            # Driver): R² stopped showing in the GUI (despite being logged
            # server-side/CLI) for every evaluation after the first one in
            # a session. "start" is unconditional regardless of cache
            # state, so carrying task here directly removes that ordering
            # dependency instead of just papering over one occurrence of it.
            "task": task,
            "classifiers": active_models,
            "classes": class_info if is_classification else [],
            "total_labelled_pixels": total_labelled,
            "confusion_matrix_labels": class_names if is_classification else [],
            "training_pcts": training_pcts,
            "stats": stats,
            "train_year": train_year,
            "test_year": test_year,
            "mode": eval_mode,
            "k": kfold_k,
            "seed": seed,
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
            seed=seed,
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

        if eval_mode == "kfold":
            # k-fold CV over all labelled pixels. No learning curve, no
            # train/test bboxes; pixel models only (run_kfold_cv has no
            # neighbourhood/patch path -- spatial MLP / U-Net were already
            # dropped from model_names above). Emits fold_result / aggregate
            # / confusion_matrices, which the frontend already understands.
            from tessera_eval.evaluate import run_kfold_cv

            for _n in kfold_dropped:
                yield (
                    json.dumps(
                        {
                            "event": "status",
                            "message": (
                                f"{_n} skipped — k-fold CV supports pixel models "
                                "only (k-NN, RF, XGBoost, MLP)"
                            ),
                        }
                    )
                    + "\n"
                )
            if not active_models:
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "message": "k-fold CV needs at least one pixel model (k-NN, RF, XGBoost, MLP).",
                        }
                    )
                    + "\n"
                )
                return
            for event in run_kfold_cv(
                vectors,
                labels,
                active_models,
                k=kfold_k,
                task=task,
                model_params=model_params,
                max_training_samples=max_train,
                seed=seed,
            ):
                if _cancelled():
                    logger.info("Evaluation cancelled during k-fold CV")
                    yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
                    return
                et = event["type"]
                if et == "fold_result":
                    yield (
                        json.dumps(
                            {
                                "event": "fold_result",
                                "fold": event["fold"],
                                "total_folds": kfold_k,
                                "models": event["models"],
                            }
                        )
                        + "\n"
                    )
                elif et == "aggregate":
                    yield json.dumps({"event": "aggregate", "models": event["models"]}) + "\n"
                elif et == "confusion_matrices":
                    yield (
                        json.dumps(
                            {
                                "event": "confusion_matrices",
                                "confusion_matrices": event["confusion_matrices"],
                            }
                        )
                        + "\n"
                    )
                elif et == "heartbeat":
                    yield json.dumps({"event": "heartbeat"}) + "\n"

        for event in (
            run_learning_curve(vectors, labels, active_models, training_pcts, **lc_kwargs)
            if eval_mode != "kfold"
            else ()
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
        # make_classifier or make_regressor, and with which seed, so stash
        # both here.
        _tile_cache["_is_classification"] = is_classification
        _tile_cache["_seed"] = seed

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
    # No request body here (Download Models is a bare POST) -- rely on the
    # cached task type, with a data-derived fallback. Same for the seed:
    # reuse whatever the evaluation run cached so the downloaded models
    # match the scored ones.
    is_classification = _resolve_task(cache, None)
    seed = int(cache.get("_seed", 42))

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
                elif _bn in SPATIAL_MODELS and not is_classification:
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
                    clf = make_classifier(name, model_params.get(name, {}), seed=seed)
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
                    clf = make_classifier(name, model_params.get(name, {}), seed=seed)
                    clf.fit(X_aug, y_aug)
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".joblib", prefix=f"{name}_model_", delete=False
                    )
                    joblib.dump({"model": clf, "class_names": valid_class_names}, tmp.name)
                    _trained_models[name] = tmp.name
                else:
                    clf = (
                        make_classifier(name, model_params.get(name, {}), seed=seed)
                        if is_classification
                        else make_regressor(name, model_params.get(name, {}), seed=seed)
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


def _predict_raster(clf, emb, is_classification, clip_range=None):
    """Predict every pixel of an (H, W, C) embedding block.

    Returns a 2D raster: 1-based class IDs with 0 as nodata for
    classification, or real values with NaN as nodata for regression.

    clip_range: optional (lo, hi) for regression. MLP/XGBoost regressors can
    extrapolate well past the training targets' span (negative heights,
    impossible biomass) on embeddings unlike anything they were trained on;
    a dense map of those is misleading, so clamp predictions to the observed
    range. None (the default) leaves predictions untouched.
    """
    h, w, c = emb.shape
    flat = emb.reshape(-1, c)
    nan_mask = np.isnan(flat).any(axis=1)
    if is_classification:
        predictions = np.zeros(flat.shape[0], dtype=np.uint8)
    else:
        predictions = np.full(flat.shape[0], np.nan, dtype=np.float32)
    if (~nan_mask).sum() > 0:
        preds = clf.predict(flat[~nan_mask].astype(np.float32))
        if is_classification:
            predictions[~nan_mask] = preds.astype(np.uint8) + 1
        else:
            preds = preds.astype(np.float32)
            if clip_range is not None:
                preds = np.clip(preds, clip_range[0], clip_range[1])
            predictions[~nan_mask] = preds
    return predictions.reshape(h, w)


def _crop_tile_to_bbox(emb, transform, crs, bbox):
    """Crop an (H, W, C) tile, on its native grid, to the part inside a
    lon/lat bounding box (west, south, east, north).

    Returns (cropped, transform), or (None, None) when the tile lies
    entirely outside the box.
    """
    import rasterio.errors
    import rasterio.windows
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    h, w = emb.shape[:2]
    west, south, east, north = bbox
    bounds = transform_bounds("EPSG:4326", crs, west, south, east, north)
    window = from_bounds(*bounds, transform=transform)
    try:
        window = window.intersection(Window(0, 0, w, h))
    except rasterio.errors.WindowError:
        return None, None
    r0 = max(0, int(window.row_off))
    c0 = max(0, int(window.col_off))
    r1 = min(h, int(np.ceil(window.row_off + window.height)))
    c1 = min(w, int(np.ceil(window.col_off + window.width)))
    if r1 <= r0 or c1 <= c0:
        return None, None
    cropped_window = Window(c0, r0, c1 - c0, r1 - r0)
    return emb[r0:r1, c0:c1], rasterio.windows.transform(cropped_window, transform)


def _reproject_prediction(arr, transform, crs, dst_crs, nodata):
    """Reproject one 2D prediction raster onto dst_crs.

    Nearest-neighbour resampling only: class IDs must not be blended, and
    regression values should stay actual model outputs rather than
    invented in-between values.
    """
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    h, w = arr.shape
    dst_transform, dst_w, dst_h = calculate_default_transform(
        crs, dst_crs, w, h, *array_bounds(h, w, transform)
    )
    out = np.full((dst_h, dst_w), nodata, dtype=arr.dtype)
    reproject(
        source=arr,
        destination=out,
        src_transform=transform,
        src_crs=crs,
        src_nodata=nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=nodata,
        resampling=Resampling.nearest,
    )
    return out, dst_transform


def _merge_prediction_rasters(rasters, out_dtype, out_nodata):
    """Merge per-block prediction rasters into one (array, transform, crs).

    Blocks arrive on the native grid they were predicted on, so a map area
    spanning more than one UTM zone yields blocks in different CRSs.
    Blocks in a minority CRS are reprojected -- predictions only, never
    embeddings -- onto the most common CRS before merging.
    """
    from collections import Counter

    import rasterio.io
    from rasterio.merge import merge as _rasterio_merge

    if len(rasters) == 1:
        return rasters[0]

    counts = Counter(str(crs) for _, _, crs in rasters)
    target_str = counts.most_common(1)[0][0]
    target_crs = next(crs for _, _, crs in rasters if str(crs) == target_str)

    datasets = []
    try:
        for arr, transform, crs in rasters:
            if str(crs) != target_str:
                arr, transform = _reproject_prediction(arr, transform, crs, target_crs, out_nodata)
            memfile = rasterio.io.MemoryFile()
            ds = memfile.open(
                driver="GTiff",
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype=out_dtype,
                crs=target_crs,
                transform=transform,
                nodata=out_nodata,
            )
            ds.write(arr.astype(out_dtype), 1)
            datasets.append(ds)
        mosaic, out_transform = _rasterio_merge(datasets, nodata=out_nodata)
    finally:
        for ds in datasets:
            ds.close()
    return mosaic[0], out_transform, target_crs


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _render_map_preview(arr, transform, crs, is_classification, class_names, reg_clip):
    """Best-effort lat/lon PNG preview of a prediction raster, plus a legend.

    The viewer drops this on the map as an L.imageOverlay so a map can be
    checked at a glance without downloading the GeoTIFF. Returns a dict
    ({"png": data-URL, "bounds": [[s,w],[n,e]], "legend": ..., ...}) or
    None -- any failure here is swallowed, the GeoTIFF is the real output.
    """
    try:
        import base64

        from rasterio.io import MemoryFile
        from rasterio.transform import array_bounds
        from rasterio.transform import from_bounds as _from_bounds
        from rasterio.warp import Resampling, reproject, transform_bounds

        h, w = arr.shape
        left, bottom, right, top = array_bounds(h, w, transform)
        west, south, east, north = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
        span_x, span_y = east - west, north - south
        if not (span_x > 0 and span_y > 0):
            return None

        if span_x >= span_y:
            dst_w = min(_PREVIEW_MAX_PX, w)
            dst_h = max(1, round(dst_w * span_y / span_x))
        else:
            dst_h = min(_PREVIEW_MAX_PX, h)
            dst_w = max(1, round(dst_h * span_x / span_y))

        dst_transform = _from_bounds(west, south, east, north, dst_w, dst_h)
        src_nodata = 0 if is_classification else float("nan")
        dst = np.full((dst_h, dst_w), src_nodata, dtype=arr.dtype)
        reproject(
            source=arr,
            destination=dst,
            src_transform=transform,
            src_crs=crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=src_nodata,
            resampling=Resampling.nearest,
        )

        rgba = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
        if is_classification:
            legend = []
            for cid in (int(v) for v in np.unique(dst)):
                if cid == 0:
                    continue
                hexc = _CLASS_PREVIEW_PALETTE[(cid - 1) % len(_CLASS_PREVIEW_PALETTE)]
                r, g, b = _hex_to_rgb(hexc)
                rgba[dst == cid] = (r, g, b, 255)
                label = class_names[cid - 1] if 0 <= cid - 1 < len(class_names) else f"Class {cid}"
                legend.append({"value": cid, "label": label, "color": hexc})
        else:
            finite = np.isfinite(dst)
            if reg_clip is not None:
                lo, hi = reg_clip
            elif finite.any():
                lo, hi = float(np.nanmin(dst)), float(np.nanmax(dst))
            else:
                lo, hi = 0.0, 1.0
            span = (hi - lo) or 1.0
            t = np.clip((np.nan_to_num(dst, nan=lo) - lo) / span, 0.0, 1.0)
            stops = np.array([_hex_to_rgb(c) for c in _REG_PREVIEW_RAMP], dtype=np.float64)
            pos = np.linspace(0.0, 1.0, len(stops))
            for c in range(3):
                rgba[..., c] = np.interp(t, pos, stops[:, c]).astype(np.uint8)
            rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
            legend = {"min": round(lo, 4), "max": round(hi, 4), "ramp": list(_REG_PREVIEW_RAMP)}

        import warnings

        import rasterio.errors

        with MemoryFile() as mf, warnings.catch_warnings():
            # A plain RGBA PNG has no geotransform -- that's the point here.
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            with mf.open(driver="PNG", width=dst_w, height=dst_h, count=4, dtype="uint8") as dpng:
                dpng.write(np.transpose(rgba, (2, 0, 1)))
            png_bytes = mf.read()

        return {
            "png": "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii"),
            "bounds": [[south, west], [north, east]],
            "legend": legend,
            "is_classification": is_classification,
        }
    except Exception as exc:  # preview is optional -- never break map generation
        logger.warning("map preview render failed (non-fatal): %s", exc)
        return None


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
    is_classification = _resolve_task(cache, body.get("task"))
    # Explicit seed wins; otherwise reuse the evaluation run's cached seed so
    # the map's model matches the one that was scored.
    seed = int(body.get("seed", cache.get("_seed", 42)))
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
    unsupported = SPATIAL_MODELS + ("unet",)
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

        _task_label = "classification" if is_classification else "regression"
        yield (
            json.dumps(
                {
                    "event": "status",
                    "message": (
                        f"Training {classifier_name} ({_task_label}) on all "
                        f"{len(vectors):,} labels..."
                    ),
                }
            )
            + "\n"
        )

        try:
            from tessera_eval.classify import make_classifier, make_regressor

            clf = (
                make_classifier(model_key, model_params.get(model_key, {}), seed=seed)
                if is_classification
                else make_regressor(model_key, model_params.get(model_key, {}), seed=seed)
            )
            clf.fit(vectors, labels)
        except Exception as e:
            yield (
                json.dumps({"event": "error", "message": f"Failed to train classifier: {e}"}) + "\n"
            )
            return

        # Regression maps are clamped to the training targets' span -- see
        # _predict_raster's docstring. Recorded in the GeoTIFF tags and
        # surfaced to the UI so a clamped map isn't mistaken for the model
        # genuinely predicting flat values at the extremes.
        reg_clip = None
        if not is_classification:
            reg_clip = (float(np.min(labels)), float(np.max(labels)))
            yield (
                json.dumps(
                    {
                        "event": "status",
                        "message": (
                            f"Regression output clamped to the training range "
                            f"[{reg_clip[0]:.4g}, {reg_clip[1]:.4g}]"
                        ),
                    }
                )
                + "\n"
            )

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

        # map_name (below) used to be "map_{bbox_idx+1}" alone -- identical
        # across every create_map() call for the same bbox slot, so
        # /api/evaluation/download-map/map_1's URL never changed between
        # generations. Confirmed harmless for the normal flow (the frontend
        # downloads immediately after each run's own "done" event, before
        # any later run's cleanup), but a real risk regardless: an
        # intermediate cache (browser, proxy) keying purely on URL has no
        # reason to know a *different* file now lives behind it, and could
        # serve a stale map. A short random suffix, unique per create_map()
        # call, means two generations never share a URL.
        run_id = uuid.uuid4().hex[:8]

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

            bbox_lonlat = (west, south, east, north)

            # Probe zarr coverage
            gtz = _get_zarr()
            use_zarr = gtz is not None and _probe_zarr_coverage(gtz, bbox_lonlat, map_year)

            yield (
                json.dumps(
                    {
                        "event": "status",
                        "message": f"Using {'zarr (fast)' if use_zarr else 'NPY tiles'} for predictions",
                    }
                )
                + "\n"
            )

            # Embeddings are predicted on whatever native grid they arrive
            # on; only the resulting prediction rasters are reprojected, at
            # merge time, if the area spans more than one UTM zone.
            chunk_results = []  # list of (predicted_2d, transform, crs)

            if use_zarr:
                # Split bbox into 0.1 deg chunks to manage memory. Chunks
                # additionally break at UTM zone edges (6-degree multiples):
                # the store serves a zone-straddling bbox from the centre
                # zone alone, silently clipping at the edge, which would
                # leave a nodata strip along the boundary.
                CHUNK_SIZE = 0.1
                chunk_lons = []
                lon = west
                while lon < east:
                    zone_edge = (np.floor(lon / 6.0) + 1) * 6.0
                    chunk_lons.append((lon, min(lon + CHUNK_SIZE, zone_edge, east)))
                    lon = chunk_lons[-1][1]
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
                            emb, transform, crs = gtz.read_region(chunk_bbox, map_year)
                            if emb is None or emb.size == 0:
                                continue
                            predicted_2d = _predict_raster(
                                clf, emb, is_classification, clip_range=reg_clip
                            )
                            chunk_results.append((predicted_2d, transform, crs))
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
            else:
                # NPY fallback: one tile at a time, on each tile's own
                # native UTM grid, cropped to the map area.
                try:
                    tiles = list(gt.registry.load_blocks_for_region(bbox_lonlat, map_year))
                except Exception as e:
                    logger.warning("Tile listing failed for map area %d: %s", bbox_idx + 1, e)
                    yield (
                        json.dumps(
                            {
                                "event": "error",
                                "message": f"Could not list tiles for map area {bbox_idx + 1}: {e}",
                            }
                        )
                        + "\n"
                    )
                    continue
                total_tiles = len(tiles)
                yield (
                    json.dumps(
                        {
                            "event": "status",
                            "message": f"Map area {bbox_idx + 1}: {total_tiles} tiles",
                        }
                    )
                    + "\n"
                )
                tiles_gen = gt.fetch_embeddings(tiles)

                for t_idx in range(total_tiles):
                    if _cancelled():
                        yield json.dumps({"event": "error", "message": "Cancelled"}) + "\n"
                        return

                    yield (
                        json.dumps(
                            {
                                "event": "map_progress",
                                "bbox_idx": bbox_idx,
                                "chunk": t_idx + 1,
                                "total_chunks": total_tiles,
                                "message": f"Predicting tile {t_idx + 1}/{total_tiles}",
                            }
                        )
                        + "\n"
                    )

                    try:
                        _, _, _, emb, crs, transform = next(tiles_gen)
                        cropped, crop_transform = _crop_tile_to_bbox(
                            emb, transform, crs, bbox_lonlat
                        )
                        if cropped is None:
                            continue
                        predicted_2d = _predict_raster(
                            clf,
                            np.asarray(cropped, dtype=np.float32),
                            is_classification,
                            clip_range=reg_clip,
                        )
                        chunk_results.append((predicted_2d, crop_transform, crs))
                    except StopIteration:
                        break
                    except Exception as e:
                        logger.warning("Tile %d failed: %s", t_idx + 1, e)
                        yield (
                            json.dumps(
                                {
                                    "event": "status",
                                    "message": f"Tile {t_idx + 1} failed: {e}",
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

                # Classification: 1-based class IDs, 0 = nodata, fits uint8.
                # Regression: real continuous values (e.g. heights), NaN =
                # nodata (0 is a valid real value, can't double as nodata).
                out_dtype = "uint8" if is_classification else "float32"
                out_nodata = 0 if is_classification else np.nan

                out_arr, out_transform, out_crs = _merge_prediction_rasters(
                    chunk_results, out_dtype, out_nodata
                )

                # Write final GeoTIFF. run_id makes this URL unique per
                # create_map() call -- see the comment where it's generated.
                map_name = f"map_{bbox_idx + 1}_{run_id}"
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
                    compress="deflate",
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
                        if reg_clip is not None:
                            tags["clamp_min"] = repr(reg_clip[0])
                            tags["clamp_max"] = repr(reg_clip[1])
                    dst.update_tags(**tags)

                _generated_maps[map_name] = tmp.name
                logger.info(
                    "GeoTIFF written: %s (%d x %d)", tmp.name, out_arr.shape[1], out_arr.shape[0]
                )

                preview = _render_map_preview(
                    out_arr, out_transform, out_crs, is_classification, class_names, reg_clip
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
                            "crs": str(out_crs),
                            # The task this map was actually generated as, so
                            # the viewer can show it and a mismatch with the
                            # evaluation run is visible rather than silent.
                            "task": "classification" if is_classification else "regression",
                            "n_classes": len(class_names),
                            "train_year": train_year,
                            "map_year": map_year,
                            "preview": preview,
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
    resp = send_file(path, as_attachment=True, download_name=f"{name}.tif", mimetype="image/tiff")
    # map_name is now unique per create_map() call (see run_id in create_map),
    # so this specific URL will never point at a different file in practice --
    # but explicit no-store is cheap insurance against any browser/proxy that
    # might otherwise cache a GET response by URL alone.
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
