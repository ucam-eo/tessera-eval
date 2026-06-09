# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.3]

### Fixed
- `__version__` is now read from the installed package metadata via
  `importlib.metadata` instead of a hardcoded literal (which had been left at
  `1.0.0` through 1.0.1/1.0.2). No functional change.

## [1.0.2]

### Fixed
- `zarr_utils.read_region_chunked`: correctly handle bboxes spanning **more than
  one UTM zone**. Each 0.1° chunk is now read in its own native zone and
  **reprojected into a single shared EPSG:4326 grid** (one resolution for the whole
  bbox, nearest-neighbour, NaN-preserving seam merge). This supersedes the v1.0.1
  NW-origin metre-offset merge (which could only place same-zone chunks) and also
  fixes geotessera's silent centre-zone clipping of small cross-zone bboxes. The
  small-region fast path now also routes through the merge when the bbox crosses a
  zone boundary.

## [1.0.1]

### Fixed
- `zarr_utils.read_region_chunked`: correct the multi-chunk merge. Chunks are now
  anchored at a north-west origin (`max .f` / `min .c`) instead of the first
  (south-west) chunk — the old anchor gave northern chunks a negative row offset,
  which landed them on an empty slice (crash) and under-sized the mosaic height.
  Also skip (with a warning) chunks whose CRS differs from the first chunk's,
  rather than mis-placing them on an incompatible metre grid.

## [1.0.0]

First public release.

### Added
- `data`: load + dequantize Tessera embeddings — GeoTessera int8 × per-pixel
  scale (`dequantize_int8`), TEE per-dim uint8 vector directories
  (`dequantize_uint8`, `load_tee_vectors`), and tile-by-tile loading for
  shapefile labels (`load_embeddings_for_shapefile`).
- `rasterize`: burn shapefile polygons onto a pixel grid with stable class IDs.
- `classify`: classifier/regressor factory (k-NN, random forest, MLP, spatial
  MLP, optional XGBoost) and spatial neighbourhood feature extraction.
- `evaluate`: streaming learning curves, k-fold cross-validation, spatial
  hold-out splits, classification (F1, confusion) and regression (R²/RMSE/MAE)
  metrics, and field-type detection.
- `unet`: optional PyTorch U-Net for sparse-label tile segmentation.
- `zarr_utils`: cached GeoTessera zarr access with chunked reads + EPSG:4326
  reprojection.
- `server` (`tee-compute`): local Flask compute server that runs ML locally and
  proxies data/UI to a hosted TEE server.
- README, data-format reference, API reference, tutorial, and compute-server
  guide; test suite; MIT license.
