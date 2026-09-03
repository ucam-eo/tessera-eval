# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.8.7]

### Added
- `run-large-area` accepts a **separate held-out test shapefile**. Upload
  one shapefile as the training ground truth and a different one (via
  `POST /api/evaluation/upload-shapefile` with `role=test`) as the test
  set; the evaluation samples the test file at `test_year` and uses it as
  the fixed test set. When a test file is present, any drawn train/test
  rectangles are ignored, and spatial models are skipped (a fixed test
  region has no neighbourhood features), same as the other fixed-test-set
  paths. Classification requires the test file's classes to be a subset of
  the training classes. Combined with the existing train/test years, this
  is what lets a repeat survey be evaluated for between-year transfer with
  real surface change in the data (Louis Driver). k-fold ignores it (it
  makes its own folds) with a status note. The `start` event carries
  `file_split: true` with `train_count` / `test_count`.
- `POST /api/evaluation/upload-shapefile` takes a `role` form field
  (`"train"` default / `"test"`); `GET /api/evaluation/list-shapefiles`
  now returns `test_files` alongside `files`; `POST
  /api/evaluation/clear-shapefiles` takes `{"role": "train"|"test"|"all"}`
  (default `"all"`).

## [1.8.6]

### Changed
- One random seed now keys every draw in an evaluation or map run, and it
  is settable. `run-large-area` and `create-map` accept a `seed` field
  (default 42); `learning-curve` gains `--seed` (`kfold` already had it).
  The seed threads through sample-point selection, tile-fetch order,
  learning-curve resampling, k-fold splits, and every estimator's own
  `random_state` (RF / XGBoost / MLP, and the U-Net's augmentation,
  DataLoader shuffle, and weight init via `torch.manual_seed`).
  `make_classifier` / `make_regressor` take a `seed=` argument instead of
  a hardcoded `random_state=42`; `run_learning_curve` takes `seed=`
  (its per-repeat RNGs are now `seed + repeat`, so the default-42 run's
  numbers shift from the old `RandomState(0..n)` sequence). The seed is
  cached with the run so Download Models and Create Map reuse it, and it
  is echoed on the `start` event.

## [1.8.5]

### Added
- `run-large-area` accepts `eval_mode: "kfold"` (with `kfold_k`, default 5,
  clamped to 2..20) alongside the default `"learning_curve"`. k-fold CV was
  previously CLI-only (`tessera-eval kfold`); this wires `run_kfold_cv` into
  the streaming endpoint so the Validation panel can run it. It
  cross-validates over all labelled pixels (no train/test bboxes, no
  learning curve) with pixel models only -- spatial MLP and U-Net are
  dropped before feature extraction, with a per-model status message. The
  `start` event carries `mode` and `k`; evaluation emits `fold_result`
  (per fold), `aggregate` (mean +/- std across folds), and, for
  classification, `confusion_matrices` (summed over folds).

## [1.8.4]

### Added
- `GET /api/evaluation/list-shapefiles` returns the names and feature
  counts of the shapefiles currently in the merged ground-truth set.
  Uploads accumulate (multi-shapefile merge), so the viewer shows this
  list on entering Validation -- an earlier upload still in the set is
  then visible rather than a surprise.

### Fixed
- `create_map` and `train_models` no longer silently fall back to
  classification. Both read the task type only from a tile-cache flag that
  `run_large_area` used to write once, at the very end of its response
  stream -- so an evaluation cut short before that final event (a client
  disconnect / throttled tab mid-run, or a cancel) left the flag unset and
  a regression map ran as classification: a classifier fit on the
  continuous targets, `uint8` predictions snapped onto the label values,
  and a discrete class palette in the preview. Confirmed live (Louis
  Driver): a tree-height map came out quantised to `1, 3, ... 65` despite
  the evaluation reporting R².
  - `run_large_area` now commits `_is_classification` to the tile cache
    *with* the vectors it applies to (both cache-population paths), so it
    is set before any training and survives an interrupted stream.
  - Task resolution is centralised in `_resolve_task()`: an explicit
    `task` in the request wins, then the cached flag, then a data-derived
    fallback (regression runs leave `class_names` empty) -- never a blind
    default to classification.
  - `create_map` accepts a `task` request field (the caller passes the
    task its evaluation ran as) and its `map_ready` event now reports the
    `task` the map was generated as, so a mismatch with the evaluation run
    is visible rather than silent.

## [1.8.3]

### Changed
- kNN classifier and regressor now run their neighbour query with
  `n_jobs=-1`. The search is brute-force in 128-d, so a full-map predict
  is millions of independent distance scans; using all cores is a large,
  free speedup on `create_map` with kNN selected.

## [1.8.2]

### Added
- `create_map`'s `map_ready` event carries a `preview`: a small EPSG:4326
  PNG of the prediction raster (`data:` URL, nearest-neighbour, capped at
  1024 px), its `bounds` as `[[south, west], [north, east]]`, and a
  `legend` (per-class name/colour for classification, `min`/`max`/`ramp`
  for regression). Lets the viewer show a map as an image overlay without
  the GeoTIFF round-trip. Best-effort -- `preview` is `null` if rendering
  fails, and the GeoTIFF is unaffected.

## [1.8.1]

### Changed
- Regression `create_map` output is now clamped to the training targets'
  observed span `[min, max]`. MLP/XGBoost regressors extrapolate freely
  (negative heights, impossible biomass) on embeddings unlike their
  training set, and a dense raster of those extremes is misleading. The
  clamp range is emitted as a status message and written to the GeoTIFF
  as `clamp_min` / `clamp_max` tags. kNN/RF are unaffected (they can't
  extrapolate). Scoring is *not* clamped.
- `run_learning_curve` regression results now carry `oor_frac` (fraction
  of the largest-percentage test predictions falling outside the training
  span) and `train_range` `[min, max]` per model, for the UI to surface.
  Predicted values and R²/RMSE/MAE are untouched.

## [1.8.0]

### Fixed
- `create_map` now predicts on the embeddings' native UTM grids and
  reprojects only the resulting prediction rasters (nearest-neighbour)
  when a map area spans more than one UTM zone. The NPY fallback used to
  fetch each chunk already reprojected to lon/lat, resampling every
  embedding vector before the model saw it; the zarr path failed with
  "CRS mismatch with source" on areas crossing a zone boundary.
- Extra sentinel nodata values (e.g. MS-NFI's 32766/32767) are removed
  from reference rasters *before* resampling. Bilinear resampling used to
  blend a sentinel with its neighbours first, producing large in-between
  values that survived as apparently valid regression targets.
- Spatial MLP models are skipped, with a status message, when the test
  set is a separate region or year. They used to fall back to a random
  split of their own training pixels and report optimistic scores next to
  honestly held-out ones.
- `evaluate()` converts its requested training sizes into percentages
  instead of misreading them as percentages, and `Results.summary()` no
  longer raises KeyError.
- Map GeoTIFFs are compressed with DEFLATE. The previous `lz4` setting is
  not a GeoTIFF compression method and GDAL silently wrote uncompressed
  files.
- `load_embeddings_for_shapefile_vq` chunks the shapefile's bounds in
  lon/lat degrees regardless of the input CRS or `target_crs`; a
  projected `target_crs` used to push metre coordinates through the
  degree-based chunk arithmetic.

### Changed
- Map GeoTIFFs are now georeferenced in the embeddings' native UTM CRS
  (the majority zone when an area spans more than one) instead of always
  EPSG:4326. Any reader that honours the file's own CRS is unaffected;
  the `map_ready` event now carries a `crs` field so consumers can tell.
- The compute server now uses geotessera's own `GeoTesseraZarr` interface
  directly, and the zarr fast path is enabled again — the external
  `tessera-zarr-utils` dependency (whose pinned release disabled zarr
  entirely) is removed. geotessera 0.10.1 fixes the UTM-zone-boundary bug
  and serves every published year, which is why the workaround package
  existed.
- The `geotessera` floor is now 0.10.1: older releases download from
  retired hosting that is being shut down.

## [1.7.3]

### Changed
- `kfold`'s per-class recall/confusion summary (classification only) is now
  always printed; the `--confusion`/`--no-confusion` flag that used to
  toggle it is removed.
- Added `--confusion-matrix`, which additionally prints the full raw-count
  confusion matrix (rows=true, cols=predicted) alongside the summary.
  (sasormunen, #3)

## [1.7.2]

### Fixed
- The `"start"` event now carries `"task"` directly. Previously the
  frontend's only source for "is this run classification or regression"
  was the `"field_start"` event — but that's gated by the tile cache key
  changing, so it's only emitted on a cache miss. Re-running with the same
  field/year/sampling (e.g. just changing which classifiers are checked)
  hits the in-memory cache and skips `field_start` entirely, leaving the
  frontend's task-tracking state stuck at whatever the *previous* run left
  it at. Confirmed live, Louis Driver: R² stopped showing in the GUI
  (despite being logged server-side/in the CLI) for every evaluation after
  the first one in a session, and the learning curve failed to build.
  `"start"` is unconditional regardless of cache state, so carrying task
  there removes the ordering dependency instead of relying on
  `field_start` having fired first.

## [1.7.1]

### Fixed
- `create_map`'s NPY fallback path raised `rasterio.errors.RasterioError:
  CRS mismatch with source` for large map areas. Root cause: it called
  `gt.registry.load_blocks_for_region()` + `gt.fetch_embeddings()` and took
  only the *first* tile via `next(tile_gen)` — for a chunk spanning
  multiple embedding tiles, this silently dropped the rest, and different
  chunks ended up carrying whatever native UTM CRS their (arbitrarily
  first) tile happened to be in. `rasterio.merge.merge()` requires every
  source dataset to share one CRS. This was previously masked for large
  areas because zarr's `read_region` already reprojected everything to a
  shared EPSG:4326 grid before this code path was even hit — it only
  surfaced once zarr was disabled (1.6.1, the UTM-boundary-bug fix) and
  the NPY fallback became the only path. Likely also explains a related
  report of small maps sometimes coming out slightly askew N/S in QGIS
  (usually within one UTM zone, so no hard crash, but still unreprojected).
  Fixed by calling `gt.fetch_mosaic_for_region(chunk_bbox, target_crs=
  "EPSG:4326")` instead — geotessera's own purpose-built method for dense
  raster prediction, which merges every overlapping tile *and* reprojects
  to a common CRS internally. Confirmed live, Louis Driver.

## [1.7.0]

### Added
- Spatial MLP regression (`spatial_mlp`, `spatial_mlp_5x5`) — previously
  crashed the entire evaluation stream (`ValueError: Unknown regressor:
  spatial_mlp`), killing every other classifier's results in the same run
  too, not just spatial_mlp's own. Confirmed live, Louis Driver.
  `make_regressor` now recognizes both names directly (deliberately no
  `_reg` suffix, unlike every other regressor — see its docstring for why).
  Two independent, previously-unfixed data-pipeline bugs in
  `_extract_tile_patches` are fixed alongside this, both silently
  corrupting regression targets rather than crashing: an unconditional
  1-based-to-0-based `-1` shift (meaningful for class IDs, wrong for
  continuous values), and an unconditional `int32` cast on the assembled
  spatial-label arrays (silently truncating e.g. a height of 3.7 to 3).
  Both are now conditional on `is_classification`, matching the pattern
  already used for `unet_patches`'s own label dtype.
  Create Map still doesn't support spatial_mlp for either task (dense
  per-pixel neighbourhood features are too expensive for full-map
  prediction) — that's an unrelated, pre-existing, permanent limitation,
  not something this touches. Download Models (`train_models`) also still
  skips spatial_mlp for regression with a clear message rather than
  training it — a deliberate scope boundary for this change, tracked
  separately, since its existing spatial_mlp *classification* handling
  pairs `spatial_3x3`/`spatial_5x5` with the plain per-point `labels`
  array rather than the patch-derived `spatial_labels_3x3`/
  `spatial_labels_5x5` run_learning_curve uses, which needs its own
  investigation before extending to regression.

## [1.6.1]

### Fixed
- Bumped `tessera-zarr-utils` pin to v0.4.0, which disables zarr entirely
  (`get_zarr()` now returns `None` unconditionally) until a UTM-zone-
  boundary bug still open upstream is actually fixed -- it's geographic,
  not year-specific, so 1.5.4's `RELIABLE_ZARR_YEARS` restriction (2024
  only) wasn't broad enough. No tessera-eval code change needed -- it
  already falls back to NPY whenever `get_zarr()` returns `None`.

## [1.6.0]

### Added
- Predicted-vs-actual scatter points for regression evaluations, requested
  by Louis Driver ("a scatterplot/heatmap of the prediction vs actual data
  along with the model results"). `run_learning_curve`'s `"aggregate"`
  event now carries `"scatter": {"y_true": [...], "y_pred": [...]}` on
  each model that had at least one successful fit at the largest training
  percentage -- up to 1000 actual-vs-predicted pairs (`_MAX_SCATTER_POINTS`),
  randomly subsampled so a large evaluation doesn't turn into an unbounded
  SSE payload. Works for both plain pixel regressors and U-Net regression.
  server.py already forwards the `"aggregate"` event's `"models"` dict
  verbatim, so no server.py change was needed.

## [1.5.5]

### Fixed
- `create_map`'s download URL (`map_name`) was identical across every call
  for the same bbox slot (`map_1`, `map_2`, ...), regardless of when or
  what was generated. Harmless for the normal frontend flow (it downloads
  immediately after each run's own `done` event, before any later run's
  cleanup), but a real risk regardless: any cache keying on URL alone
  (browser, proxy) has no way to know a *different* file now lives behind
  it, and could serve a stale map. `map_name` now includes a short random
  suffix unique per `create_map()` call, and `download-map` responses now
  send `Cache-Control: no-store` as well.

## [1.5.4]

### Fixed
- Bumped `tessera-zarr-utils` pin to v0.3.1, which restricts
  `probe_zarr_coverage` to years actually known reliable (currently just
  2024). The zarr store's own metadata advertises 2017-2025 as populated,
  but only 2024 is genuinely trustworthy right now; other years returned
  real-looking, non-NaN, but incorrect data, which caused `create_map`'s
  `map_year` (added in 1.5.0) to silently use zarr for e.g. `map_year=2018`
  and produce a map identical to the training year's, instead of falling
  back to the NPY path (which does fetch correct, year-varying
  embeddings). No code change needed here -- tessera-eval already falls
  back to NPY whenever `probe_zarr_coverage` returns `False`.

## [1.5.3]

### Fixed
- U-Net regression could silently run with 0 classifiers and finish
  suspiciously fast, with no error: `_cached_tiles_need_reload` (the
  in-memory tile-cache staleness check) knew about `spatial_mlp`/
  `spatial_mlp_5x5` needing a reload when their features weren't cached,
  but never checked U-Net. Running any plain pixel regressor first (same
  field/year/sampling) cached `unet_patches=[]`; selecting U-Net next, with
  the same cache key, silently reused that empty patch list instead of
  reloading tiles — U-Net then got filtered out of `active_models`
  entirely. Root-caused against a real shapefile (205k polygons, Louis
  Driver) after confirming the tile-fetch and patch-extraction logic
  itself was correct in isolation.

## [1.5.2]

### Fixed
- `create_map` ("Create Map" GeoTIFF generation) had never been adapted for
  regression — it unconditionally trained via `make_classifier` and wrote
  predictions as `uint8` with `nodata=0`. XGBoost's classifier validates
  class labels strictly and crashed outright ("Invalid classes inferred
  from unique values of y") the moment continuous values (e.g. heights)
  were passed as `y`. k-NN/RF/MLP don't validate that, so they silently
  "succeeded" — training as an enormous multi-class classifier over
  continuous values treated as arbitrary class IDs, then truncating real
  predictions to `uint8` and colliding a real value of 0 with the nodata
  sentinel. Now dispatches `make_classifier`/`make_regressor` via the
  cached task (`_is_classification`, same mechanism as 1.5.1's Download
  Models fix) and a UI-name → `_reg`-suffixed lookup (`_CLF_TO_REG`,
  hoisted to module level), and writes regression output as `float32` with
  NaN nodata instead of `uint8`/0.

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
