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

Reads are region reads, never per-point: points are grouped by tile and each
tile is read once. Tiles are cached on disk (geotessera's own Zarr cache
keeps only metadata, so without this every run re-downloads every tile), and
the dataset's tile registry decides which tiles have data at all, so tiles
over the sea or in coverage gaps are never read.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDINGS_DATASET_VERSION = "v1.1"
EMBEDDINGS_DATASET_VARIANT = "dclimate"

ZARR_CACHE_MAX_BYTES = 20 * 1024**3  # geotessera's own (metadata) cache
TILE_CACHE_MAX_BYTES = 40 * 1024**3  # our tile cache (~110 MB per tile)

TILE_DEG = 0.1  # TESSERA's tile grid: 0.1-degree cells centred on x.x5

# Reads go over the network; geotessera 0.11 reads Source Coop straight from
# the bucket, but a transient HTTP error (e.g. a 502) can still surface.
# Retry each read a few times before giving up on it.
_READ_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0

YEARS = range(2017, 2026)

# Create Map: when the map covers at least this share of a tile, read (and
# cache) the whole tile rather than just the map's part of it.
_CLIP_CACHE_FRACTION = 0.25


def default_cache_dir():
    return Path.home() / ".cache" / "tessera-eval" / "zarr"


def zarr_store_url():
    """The Zarr store URL for the pinned dataset."""
    from geotessera.registry import zarr_store_url as _url

    return _url(EMBEDDINGS_DATASET_VERSION, EMBEDDINGS_DATASET_VARIANT)


def _open_tile_registry(cache_dir):
    """The dataset's tile registry (which blocks have embeddings, per year),
    or None if it can't be opened -- callers then fall back to the plain
    0.1-degree grid. Separate function so tests can replace it."""
    try:
        from geotessera.icechunk import TileRegistry
        from geotessera.registry import find_dataset

        ds = find_dataset(EMBEDDINGS_DATASET_VERSION.lstrip("v"), EMBEDDINGS_DATASET_VARIANT)
        if ds is None or not ds.tile_registry:
            return None
        return TileRegistry(ds.tile_registry, cache_dir=str(cache_dir) if cache_dir else None)
    except Exception as e:
        logger.warning("Tile registry unavailable (%s); listing tiles by geometry", e)
        return None


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


def _dequantize(q, scales):
    """float32 embeddings from int8 values and per-pixel scales; works for
    a whole tile (H, W, 128) / (H, W) or picked pixels (N, 128) / (N,)."""
    q = np.asarray(q, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    if scales.ndim == q.ndim - 1:
        scales = scales[..., None]
    return q * scales


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


def _tile_box(lon, lat):
    h = TILE_DEG / 2
    return (lon - h, lat - h, lon + h, lat + h)


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class _TileGrid:
    """Stands in for the old NPY registry.

    Lists the 0.1-degree tiles in a bbox that overlap a block the dataset
    actually embedded that year (from its tile registry), so tiles with no
    data are never read. Falls back to every tile in the bbox when the
    registry can't be reached.
    """

    def __init__(self, client):
        self._client = client

    def load_blocks_for_region(self, bbox, year):
        years = getattr(self._client.store, "years", None)
        if years and year not in years:
            return []
        tiles = tiles_for_bbox(bbox, year)
        blocks = self._client.embedded_blocks(bbox, year)
        if blocks is None:
            return tiles
        return [t for t in tiles if any(_overlaps(_tile_box(t[1], t[2]), b) for b in blocks)]


class ZarrClient:
    """GeoTessera-compatible reader over the pinned Zarr store."""

    def __init__(
        self,
        cache_dir=None,
        cache_max_size=ZARR_CACHE_MAX_BYTES,
        tile_cache_max_bytes=TILE_CACHE_MAX_BYTES,
        **_ignored,
    ):
        from geotessera.store import GeoTesseraZarr

        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.store = GeoTesseraZarr(
            store_url=zarr_store_url(),
            cache_dir=str(self.cache_dir),
            cache_max_size=cache_max_size,
        )
        self.registry = _TileGrid(self)
        self.dataset_version = EMBEDDINGS_DATASET_VERSION
        self.dataset_variant = EMBEDDINGS_DATASET_VARIANT
        self._tile_dir = (
            self.cache_dir / f"tiles-{EMBEDDINGS_DATASET_VERSION}-{EMBEDDINGS_DATASET_VARIANT}"
        )
        self._tile_cache_max = tile_cache_max_bytes
        self._tile_cache_bytes = None  # running total, measured on first save
        self._tile_registry = None
        self._tile_registry_tried = False
        self._blocks = {}  # (bbox, year) -> list of embedded block bboxes

    # -- coverage --------------------------------------------------------

    def embedded_blocks(self, bbox, year=None):
        """Bboxes of the dataset's blocks that hold embeddings and overlap
        *bbox* -- for one year, or a {year: [...]} dict when *year* is None.
        None when the tile registry can't be reached."""
        if not self._tile_registry_tried:
            self._tile_registry_tried = True
            self._tile_registry = _open_tile_registry(self.cache_dir / "registry")
        if self._tile_registry is None:
            return None
        key = (tuple(round(v, 6) for v in bbox), year)
        if key not in self._blocks:
            try:
                df = _with_retries(
                    lambda: self._tile_registry.tiles(bbox=tuple(bbox), year=year),
                    "Reading the tile registry",
                )
            except Exception as e:
                logger.warning("Tile registry query failed (%s); listing tiles by geometry", e)
                return None
            df = df[df["embedded"].astype(bool)]
            boxes = list(
                zip(
                    df["year"].astype(int),
                    zip(df["bbox_west"], df["bbox_south"], df["bbox_east"], df["bbox_north"]),
                )
            )
            if year is None:
                by_year = {}
                for y, b in boxes:
                    by_year.setdefault(y, []).append(b)
                self._blocks[key] = by_year
            else:
                self._blocks[key] = [b for _, b in boxes]
        return self._blocks[key]

    # -- reading ---------------------------------------------------------

    def sample_embeddings_at_points(self, points, year=None, progress_callback=None):
        """(N, 128) float32 embeddings at (lon, lat) points; NaN rows where
        there is no data (water, outside coverage, or a tile that couldn't
        be read).

        Reads each tile the points fall in once and picks the points out
        locally, rather than per-point reads: a run samples up to hundreds
        of thousands of points inside a limited set of tiles, so the cost is
        one (cached) read per tile regardless of the point count. Progress
        is reported per tile.
        """
        from pyproj import Transformer
        from rasterio.transform import rowcol

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        n = len(pts)
        out = np.full((n, 128), np.nan, dtype=np.float32)
        if n == 0:
            return out

        # Group point indices by the 0.1-degree tile each falls in.
        ti = np.floor(pts[:, 0] / TILE_DEG + 1e-9).astype(np.int64)
        tj = np.floor(pts[:, 1] / TILE_DEG + 1e-9).astype(np.int64)
        groups = {}
        for idx, key in enumerate(zip(ti.tolist(), tj.tolist())):
            groups.setdefault(key, []).append(idx)

        total = len(groups)
        for done, ((i, j), idx_list) in enumerate(groups.items(), start=1):
            lon = round((i + 0.5) * TILE_DEG, 2)
            lat = round((j + 0.5) * TILE_DEG, 2)
            try:
                q, scales, crs, transform = self.read_tile_quantized(year, lon, lat)
            except Exception as e:
                logger.warning("Tile (%.2f, %.2f) %s unavailable: %s", lon, lat, year, e)
                q = None
            if q is not None:
                idx = np.asarray(idx_list)
                xs, ys = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
                    pts[idx, 0], pts[idx, 1]
                )
                rows, cols = (np.asarray(a) for a in rowcol(transform, xs, ys))
                h, w = q.shape[:2]
                ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
                r, c = rows[ok], cols[ok]
                # Dequantize only the picked pixels, not the whole tile.
                out[idx[ok]] = _dequantize(q[r, c], scales[r, c])
            if progress_callback:
                progress_callback(done, total, "tiles")
        return out

    def _tile_path(self, year, lon, lat):
        return self._tile_dir / str(year) / f"{lon:.2f}_{lat:.2f}.npz"

    def _load_cached_tile(self, path):
        from affine import Affine

        with np.load(path, allow_pickle=False) as f:
            q, scales = f["q"], f["scales"]
            transform = Affine(*f["transform"].tolist())
            crs = str(f["crs"])
        os.utime(path)  # mark as recently used for eviction
        return q, scales, crs, transform

    def _cached_files(self):
        return [p for p in self._tile_dir.rglob("*.npz") if ".tmp-" not in p.name]

    def _save_tile(self, path, q, scales, transform, crs):
        import threading

        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique per process and thread, so concurrent writers of the same
        # tile can't clobber each other's half-written file.
        tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}-{threading.get_ident()}.npz")
        # Uncompressed: a reload is a plain read, not a decompression.
        np.savez(
            tmp,
            q=q,
            scales=scales,
            transform=np.asarray(tuple(transform)[:6], dtype=np.float64),
            crs=np.asarray(str(crs)),
        )
        os.replace(tmp, path)
        if self._tile_cache_bytes is None:
            self._tile_cache_bytes = sum(p.stat().st_size for p in self._cached_files())
        else:
            self._tile_cache_bytes += path.stat().st_size
        if self._tile_cache_bytes > self._tile_cache_max:
            self._evict()

    def _evict(self):
        """Delete least-recently-used tiles until the cache is under its cap.
        Only runs when the running total says it's over, so a normal run
        never rescans the cache directory."""
        files = self._cached_files()
        sizes = {p: p.stat().st_size for p in files}
        total = sum(sizes.values())
        for p in sorted(files, key=lambda p: p.stat().st_mtime):
            if total <= self._tile_cache_max:
                break
            total -= sizes[p]
            p.unlink(missing_ok=True)
        self._tile_cache_bytes = total

    def read_tile_quantized(self, year, lon, lat):
        """(int8 embedding (H, W, 128), scales (H, W), crs, transform) for one
        tile, on its zone's native UTM grid. Cached on disk."""
        path = self._tile_path(year, lon, lat)
        if path.exists():
            try:
                return self._load_cached_tile(path)
            except Exception as e:
                logger.warning("Discarding unreadable cached tile %s: %s", path, e)
                path.unlink(missing_ok=True)
        q, scales, transform, crs = _with_retries(
            lambda: self.store.read_region_quantized(_tile_box(lon, lat), year),
            f"Reading tile ({lon:.2f}, {lat:.2f})",
        )
        try:
            self._save_tile(path, q, scales, transform, crs)
        except Exception as e:
            logger.warning("Could not cache tile %s: %s", path, e)
        return q, scales, str(crs), transform

    def read_tile(self, year, lon, lat):
        """(embedding (H, W, 128) float32, crs, transform) for one tile, on
        its zone's native UTM grid. Cached on disk."""
        q, scales, crs, transform = self.read_tile_quantized(year, lon, lat)
        return _dequantize(q, scales), crs, transform

    def read_clipped(self, year, lon, lat, clip_bbox):
        """Like read_tile, but for the part of the tile inside *clip_bbox*
        (west, south, east, north).

        Goes through the tile cache (reading and caching the whole tile)
        when the tile is already cached or the map covers a substantial part
        of it -- maps are typically re-run on the same area with different
        models, and the area usually overlaps tiles an evaluation already
        cached. Only a small corner of an uncached tile is fetched on its
        own (cheap, so not worth caching)."""
        tb = _tile_box(lon, lat)
        region = (
            max(tb[0], clip_bbox[0]),
            max(tb[1], clip_bbox[1]),
            min(tb[2], clip_bbox[2]),
            min(tb[3], clip_bbox[3]),
        )
        if region[0] >= region[2] or region[1] >= region[3]:
            return None, None, None
        fraction = ((region[2] - region[0]) * (region[3] - region[1])) / (TILE_DEG * TILE_DEG)
        if fraction >= _CLIP_CACHE_FRACTION or self._tile_path(year, lon, lat).exists():
            return self.read_tile(year, lon, lat)
        emb, transform, crs = _with_retries(
            lambda: self.store.read_region(region, year),
            f"Reading map area of tile ({lon:.2f}, {lat:.2f})",
        )
        return np.asarray(emb, dtype=np.float32), str(crs), transform

    def fetch_embeddings(self, tiles, clip_bbox=None):
        """Yield (year, lon, lat, embedding, crs, transform) per tile, in order.

        With *clip_bbox*, only the part of each tile inside it is read (Create
        Map). A tile that still fails after retries yields embedding=None
        (and crs/transform None) rather than raising: callers pair results
        with their tile list by position, and an exception would end the
        generator, losing every remaining tile.
        """
        for year, lon, lat in tiles:
            try:
                if clip_bbox is None:
                    emb, crs, transform = self.read_tile(year, lon, lat)
                else:
                    emb, crs, transform = self.read_clipped(year, lon, lat, clip_bbox)
            except Exception as e:
                logger.warning("Tile (%.2f, %.2f) %s unavailable: %s", lon, lat, year, e)
                emb = crs = transform = None
            yield (year, lon, lat, emb, crs, transform)


def year_coverage(client, bbox, years=YEARS):
    """{"year": n_tiles} for *bbox*: how many 0.1-degree tiles overlap a
    block the dataset embedded that year (0 = no data). One tile-registry
    query, no embedding reads. Falls back to the store's year list (every
    tile counted) if the registry can't be reached."""
    by_year = client.embedded_blocks(bbox, year=None)
    tiles = tiles_for_bbox(bbox)
    if by_year is None:
        store_years = set(getattr(client.store, "years", None) or [])
        return {str(y): (len(tiles) if y in store_years else 0) for y in years}
    return {
        str(y): sum(
            1 for t in tiles if any(_overlaps(_tile_box(t[1], t[2]), b) for b in by_year.get(y, []))
        )
        for y in years
    }


def make_client(cache_dir=None):
    """The embeddings reader. Looked up at call time so tests can swap
    ``ZarrClient`` for a fake."""
    return ZarrClient(cache_dir=cache_dir)
