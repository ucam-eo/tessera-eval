# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [1.5.1]

### Fixed
- "Download Models" (`train_models`/`/api/evaluation/train-models`) always
  called `make_classifier`, never `make_regressor` — clicking it after a
  regression run (e.g. kNN regression) failed every model with "Unknown
  classifier: nn_reg". This endpoint retrains a final model in a separate
  request after evaluation finishes, so it had no `is_classification`
  context of its own; the earlier regression fixes in `run_learning_curve`/
  `run_large_area` (1.3.1-1.4.1) didn't reach it. Now `run_large_area`
  stashes `is_classification` in the tile cache and `train_models` dispatches
  on it, for the plain pixel-classifier path and the U-Net path (using
  `train_unet_regressor_on_patches`). Spatial MLP (3x3/5x5) has no
  regressor variant yet, so it's skipped with a clear status message for
  regression rather than silently mis-training or crashing.

## [1.5.0]

### Added
- `create_map` accepts an optional `map_year`, independent of the model's
  training year — trains as normal on one year, then runs that
  already-trained model as pure inference against a *different* year's
  embeddings across the map area (e.g. train on 2025, map 2018, to look
  for change over time). No ground truth needed for `map_year`; nothing
  gets scored. Distinct from the train/test-year Validation feature, which
  evaluates against held-out ground truth at the same points rather than
  generating a map.

## [1.4.1]

### Fixed
- `run_large_area`'s sample-point generation now actually respects
  `max_training_samples` when a shapefile has more rows than the budget —
  `sample_points(size=N)` generates N points *per row*, and the "every row
  gets at least one point" floor had no corresponding cap, so a
  420,000-row shapefile against a 200,000-point budget generated ~420,000
  points regardless. Affects both classification and regression.

## [1.4.0]

### Added
- U-Net regression support. `rasterize_shapefile_continuous` burns real
  field values (not `LabelEncoder` ranks — U-Net patches previously always
  went through the same class-encoding as pixel classifiers, regardless of
  task), `train_unet_regressor_on_patches`/`predict_unet_tile_regression`
  train/predict a single-channel `TinyUNet` with a masked-MSE loss, and
  `run_learning_curve`'s U-Net branch reports R²/RMSE/MAE for regression
  the same way pixel regressors do (v1.3.1). Spatial MLP regression is
  still not supported (no regressor variant exists for it).

## [1.3.3]

### Fixed
- Fixed a `NameError` risk introduced by 1.3.2: requesting a spatial
  classifier (Spatial MLP/U-Net) together with regression mode would have
  crashed the SSE stream (`le` was left undefined for regression).

## [1.3.2]

### Fixed
- `run_large_area`'s regression targets were `LabelEncoder` rank integers
  (0, 1, 2, ...), not the real field values — a continuous field like tree
  height got silently discretized before v1.3.1's regressor fix ever saw
  it, so regressors were fitting against meaningless ranks the whole time
  despite producing plausible-looking R²/RMSE/MAE. Also the real mechanism
  behind the reported "25x too many sample points" — the per-class
  sampling floor that caused it only existed because regression was being
  treated as N-way classification. Regression now samples against a single
  combined point budget (no per-class weighting) and recovers each point's
  real field value directly.

## [1.3.1]

### Fixed
- `run_learning_curve` (used by `run-large-area`) now correctly runs pixel
  regressors (`nn_reg`/`rf_reg`/`xgboost_reg`/`mlp_reg`) — previously it had
  no classification/regression dispatch at all and always called
  `make_classifier`, so every regression request crashed with `ValueError:
  Unknown classifier: nn_reg` (etc.) the moment training started. Regression
  runs now also emit an `"aggregate"` event (largest-percentage R²/RMSE/MAE)
  that the frontend's regression display was already built to consume but
  never received. Not yet fixed: Spatial MLP and U-Net regression (no
  regressor variant exists for either yet).

## [1.3.0]

### Added
- CLI: `tessera-eval` command (`load`/`kfold`/`learning-curve`), with raster
  (GeoTIFF) support and a `learning-curve --test-year` option to score a
  classifier trained on one year's embeddings against a different year's
  embeddings from a separate `--test-data` file.
- `server.py`'s `/api/evaluation/run-large-area` accepts the equivalent
  `train_year`/`test_year` in the request body (in place of a single `year`;
  `train_year` falls back to `year` for compatibility, `test_year` defaults
  to `train_year`) to score a classifier trained on one year's embeddings
  against a different year's embeddings at the *same* locations. Both this
  and the CLI option feed `run_learning_curve`'s pre-existing `test_vectors`/
  `test_labels` fixed-test-set mechanism, which is unchanged.

## [1.2.0]

### Changed
- `zarr_utils` moved to its own package,
  [`tessera-zarr-utils`](https://github.com/ucam-eo/tessera-zarr-utils), so it can
  be used without the eval/ML stack. `server.py` imports it from there; it's a
  `[server]`-extra dependency. No API change for tessera-eval users.
- Require `geotessera>=0.9.0`.

### Removed
- `tessera_eval/zarr_utils.py` (now provided by `tessera-zarr-utils`).

## [1.1.0]

### Added
- `data.load_embeddings_for_shapefile_vq`: load labelled embeddings from a
  **VQ bolt-on** (or any client exposing `fetch_mosaic_for_region`), pulling
  *reconstructed* embeddings region-by-region. Splits the shapefile bbox into
  `<= max_km` chunks (the bolt-on caps bbox size), skips chunks no polygon
  touches and chunks with no VQ coverage, and returns the same
  `(vectors, labels, class_names, stats)` contract as
  `load_embeddings_for_shapefile`. Lets you evaluate downstream accuracy on
  VQ-reconstructed embeddings vs. the raw-tile reference. The VQ client
  (e.g. `tessera_vq.VQTessera`) is duck-typed and passed in — no tessera-vq
  dependency is added.

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
