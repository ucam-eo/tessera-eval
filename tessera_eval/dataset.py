"""Where tessera-eval reads TESSERA embeddings from.

Every embedding read goes through one reader: geotessera's GeoTesseraZarr
on the **v1.1-dclimate** dataset (the complete, wall-to-wall 2017-2025 run,
published as Zarr/Icechunk only -- there are no NPY tiles for it). NPY tiles
are deprecated in geotessera 0.11, and the dataset is named explicitly here
rather than left to geotessera's default, which has changed between
releases (0.11 made a bare ``GeoTessera()`` mean the sparse v1.1-cambridge
test run).

``ZarrClient`` presents the small part of the old ``GeoTessera`` client
interface the rest of tessera-eval uses (``registry.load_blocks_for_region``,
``sample_embeddings_at_points``, ``fetch_embeddings``), so callers are
unchanged while the data now comes from one Zarr store.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDINGS_DATASET_VERSION = "v1.1"
EMBEDDINGS_DATASET_VARIANT = "dclimate"

ZARR_CACHE_MAX_BYTES = 20 * 1024**3  # bound the on-disk chunk cache

TILE_DEG = 0.1  # TESSERA's tile grid: 0.1-degree cells centred on x.x5

# Reads go over the network; geotessera 0.11 reads Source Coop straight from
# the bucket, but a transient HTTP error (e.g. a 502) can still surface.
# Retry each tile / point batch a few times before giving up on it.
_READ_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0

# Points are sampled in batches so progress can be reported and a failure
# only costs one batch's retry, not the whole set.
_POINT_BATCH = 20_000


def zarr_store_url():
    """The Zarr store URL for the pinned dataset."""
    from geotessera.registry import zarr_store_url as _url

    return _url(EMBEDDINGS_DATASET_VERSION, EMBEDDINGS_DATASET_VARIANT)


def _with_retries(fn, what):
    for attempt in range(1, _READ_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == _READ_ATTEMPTS:
                raise
            logger.warning(
                "%s failed (attempt %d/%d): %s -- retrying",
                what,
                attempt,
                _READ_ATTEMPTS,
                e,
            )
            time.sleep(_RETRY_BACKOFF_S * attempt)


def tiles_for_bbox(bbox, year=None):
    """(year, lon, lat) for every 0.1-degree tile intersecting *bbox*.

    *bbox* is (west, south, east, north). Tile centres sit on the x.x5 grid,
    matching the old NPY tile registry's keys.
    """
    west, south, east, north = bbox
    eps = 1e-9  # 16.3 / 0.1 is 162.99999..., not 163
    i0, i1 = math.floor(west / TILE_DEG + eps), math.ceil(east / TILE_DEG - eps)
    j0, j1 = math.floor(south / TILE_DEG + eps), math.ceil(north / TILE_DEG - eps)
    return [
        (year, round((i + 0.5) * TILE_DEG, 2), round((j + 0.5) * TILE_DEG, 2))
        for i in range(i0, max(i1, i0 + 1))
        for j in range(j0, max(j1, j0 + 1))
    ]


class _TileGrid:
    """Stands in for the old NPY registry: lists tiles by geometry alone.

    Whether a tile actually has data is discovered when it is read (missing
    data arrives as NaN, which every caller already drops).
    """

    def __init__(self, store):
        self._store = store

    def load_blocks_for_region(self, bbox, year):
        years = getattr(self._store, "years", None)
        if years and year not in years:
            return []
        return tiles_for_bbox(bbox, year)


class ZarrClient:
    """GeoTessera-compatible reader over the pinned Zarr store."""

    def __init__(self, cache_dir=None, cache_max_size=ZARR_CACHE_MAX_BYTES, **_ignored):
        from geotessera.store import GeoTesseraZarr

        kwargs = {"store_url": zarr_store_url(), "cache_max_size": cache_max_size}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self.store = GeoTesseraZarr(**kwargs)
        self.registry = _TileGrid(self.store)
        self.dataset_version = EMBEDDINGS_DATASET_VERSION
        self.dataset_variant = EMBEDDINGS_DATASET_VARIANT

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        """(N, 128) float32 embeddings at (lon, lat) points; NaN rows where
        there is no data (water, outside coverage)."""
        points = [(float(p[0]), float(p[1])) for p in points]
        n = len(points)
        if n == 0:
            return np.empty((0, 128), np.float32)
        parts = []
        for start in range(0, n, _POINT_BATCH):
            batch = points[start : start + _POINT_BATCH]
            vecs = _with_retries(
                lambda b=batch: self.store.sample_points(b, year, progress=False),
                f"Sampling points {start:,}-{start + len(batch):,}",
            )
            parts.append(np.asarray(vecs, dtype=np.float32))
            if progress_callback:
                progress_callback(min(start + len(batch), n), n, "sampling")
        return np.concatenate(parts, axis=0)

    def read_tile(self, year, lon, lat):
        """(embedding (H, W, 128) float32, crs, transform) for one tile, on
        its zone's native UTM grid."""
        half = TILE_DEG / 2
        bbox = (lon - half, lat - half, lon + half, lat + half)
        emb, transform, crs = _with_retries(
            lambda: self.store.read_region(bbox, year), f"Reading tile ({lon:.2f}, {lat:.2f})"
        )
        return np.asarray(emb, dtype=np.float32), crs, transform

    def fetch_embeddings(self, tiles):
        """Yield (year, lon, lat, embedding, crs, transform) per tile, in order.

        A tile that still fails after retries yields embedding=None (and
        crs/transform None) rather than raising: callers pair results with
        their tile list by position, and an exception would end the
        generator, losing every remaining tile.
        """
        for year, lon, lat in tiles:
            try:
                emb, crs, transform = self.read_tile(year, lon, lat)
            except Exception as e:
                logger.warning("Tile (%.2f, %.2f) %s unavailable: %s", lon, lat, year, e)
                emb = crs = transform = None
            yield (year, lon, lat, emb, crs, transform)


def make_client(cache_dir=None):
    """The embeddings reader. Looked up at call time so tests can swap
    ``ZarrClient`` for a fake."""
    return ZarrClient(cache_dir=cache_dir)
