# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.14.2]

### Changed
- **Zarr fast path temporarily disabled (`_ZARR_DISABLED = True`), forcing
  every embedding fetch onto the NPY tile path.** Keshav (2026-09-23): the
  zarr fast path is using a smaller chunk size than intended, and every
  read still hits the network regardless of the on-disk cache — the same
  symptom Moustafa Eweda reported and forwarded to Anil (geotessera's own
  `GeoTesseraZarr`, not `tessera_eval` code). `_get_zarr()` now
  short-circuits to `None` before even attempting to open the store, so
  both call sites (`_extract_tile_patches`, `create_map`'s tile loop)
  already fall back to NPY tiles cleanly (existing `gtz is not None and
  ...` guards, unchanged). `_get_zarr()`'s own real connection/caching
  logic is untouched, just bypassed — flip `_ZARR_DISABLED` back to
  `False` once geotessera's chunk-size/caching behaviour is confirmed
  fixed upstream. 1 new test (`test_get_zarr_disabled_returns_none_
  without_even_trying_to_connect`); the 2 existing tests that exercise
  `_get_zarr()`'s real connect logic now explicitly re-enable it via the
  file's `_fresh_cache` fixture. Full suite 275 passed (was 274).

## [1.14.1]

### Fixed
- **A model that failed to train reported a real 0.0 score with no visible
  cause, indistinguishable from "genuinely the worst model".** Confirmed
  live (Keshav, testing `spatial_kfold`): `deep_mlp` reported macro F1 =
  0.0 for every fold, with no explanation in the UI — the actual cause was
  a missing optional dependency (PyTorch not installed on that machine),
  logged server-side only (`logger.warning`) and silently zero-filled
  everywhere else. `run_learning_curve` and `run_kfold_cv` now yield a
  `{"type": "classifier_status", "message": "<name> failed to train: <exc>"}`
  event the *first* time a given model fails (deduped — a hard dependency
  failure repeats identically at every pct/repeat/fold, so one message per
  model per run, not one per attempt), and mark that model's metrics dict
  with `"failed": True` from then on — the numeric fields stay a real 0.0
  fallback (so aggregation math needs no special case), but a consumer can
  now tell "didn't run" apart from "scored zero" without parsing logs.
  **A second, more severe bug found while building this fix, also fixed
  here**: in `run_learning_curve` (not `run_kfold_cv`, which already had
  this right), `make_classifier()`/`make_regressor()` were called *outside*
  the per-model `try`/`except` — a classifier whose *construction* can
  raise (`deep_mlp` without torch calls `_require_torch()` in its
  `__init__`; `xgboost`/`xgboost_reg` do a lazy `from xgboost import ...`
  that raises `ImportError` if xgboost isn't installed) crashed the *entire
  learning-curve run* uncaught, taking every other model's results down
  with it — not just failing that one model, as `run_kfold_cv` already
  correctly did. Both constructor calls moved inside their `try` blocks.
  9 new tests (`tests/test_classifier_failure_reporting.py`): failure
  deduplication (once per model, not once per fold/repeat) in both
  functions, the `failed` flag appearing on progress/fold_result/aggregate
  events, independence from other (working) models in the same run,
  regression coverage, and a direct reproduction of the deep_mlp-without-
  torch construction-time crash (monkeypatches `deep_mlp._HAS_TORCH` rather
  than depending on the test environment's own torch install). Full suite
  274 passed (was 265).

## [1.14.0]

### Added
- **`spatial_kfold`: a real geographic split for k-fold cross-validation.**
  Ordinary k-fold here (`run_kfold_cv`) shuffles points at random, with no
  notion of location at all — two points from neighbouring fields, or even
  the same field, can land on opposite sides of a fold, the same
  spatial-autocorrelation optimism a plain random split has (documented in
  the user guide's own "K-fold is not a spatial split" section). The
  learning curve already has a real geographic split (Spatial Train/Test
  Split, drawn train/test bounding boxes) — k-fold had no equivalent until
  this. New opt-in request flag `spatial_kfold` (off by default, same
  rationale as `group_by_field`: it changes every reported k-fold score).
  Mechanism: new `server.py` helper `_spatial_block_groups()` assigns each
  sampled point to one of ~k geographic blocks via a quantile grid over its
  (lon, lat) (quantile edges rather than equal-width degree bins, so block
  *point counts* stay roughly balanced even over a lopsided AOI), then
  feeds that as `groups` into the exact same `StratifiedGroupKFold`/
  `GroupKFold` machinery `group_by_field` already uses (v1.13.0) — the
  splitting mechanism needed zero changes, only the grouping key is new.
  k-fold-only (the learning curve already has its own geographic split);
  takes precedence over `group_by_field` when both are requested (only one
  grouping can drive a single split); falls back to plain k-fold with a
  clear status message when sample-point coordinates aren't available
  (cached result) or points are too clustered to form k distinct blocks.
  12 new tests: `tests/test_spatial_block_groups.py` (grouping-function
  properties — balance, determinism, degenerate single-location/single-line
  inputs) and `tests/test_spatial_kfold_server.py` (request-flag wiring —
  activates in k-fold mode, ignored outside it, takes precedence over
  `group_by_field` with a status message, `group_by_field` alone still
  works unchanged). Full suite 265 passed (was 253).

## [1.13.1]

### Fixed
- **"Download Models" and Create Map could silently go dark mid-training,
  surfacing to the user as a browser "network error" even though the
  server-side training was still running (or had already finished).**
  `train_models()` ("Download Models", deferred from evaluation) and
  `create_map()`'s own from-scratch refit both trained every model with a
  single raw, blocking call — `train_unet_on_patches(...)` or `clf.fit(...)`
  directly — with no heartbeat at all, unlike `run_learning_curve`/
  `run_kfold_cv`, which already got `evaluate.py`'s `_fit_with_heartbeat`
  fix for exactly this failure mode (a slow enough fit leaves the SSE
  stream silent for its whole duration, and whatever's carrying the
  connection reads that as dead and drops it). Confirmed live (Moustafa
  Eweda): a U-Net "Download Models" run that had completed successfully
  server-side surfaced in the browser as "Training error: network error" —
  the download endpoint had gone completely silent for the whole run.
  Fixed by a new `_fit_with_wire_heartbeat()` (translates
  `_fit_with_heartbeat`'s events into this module's own JSON-string wire
  format), wrapping all 6 blocking training calls across both endpoints:
  U-Net classification and regression, `spatial_mlp`/`spatial_mlp_5x5`,
  and the generic pixel-classifier/regressor branch in each. The generic
  branch also protects `deep_mlp`'s own download-model and create-map
  paths the same way — a 150-epoch fit on a large training set is exactly
  the kind of long, silent call these two endpoints never guarded against
  before v1.12.0 added it as an option.
  4 new tests (`tests/test_train_models_heartbeat.py`): isolated coverage
  of `_fit_with_wire_heartbeat` (fast fit → no heartbeats; slow fit →
  heartbeats + correct return value; exception still propagates after
  heartbeats), plus an integration test reproducing the actual bug —
  `train_models()` with a slow classifier fit must emit real heartbeat
  events, not go silent. Full suite 253 passed (was 249).

## [1.13.0]

### Added
- **Group-by-field split (`groups` in `run_learning_curve`/`run_kfold_cv`,
  opt-in `group_by_field` request flag) — fixes a real optimism bug in
  every existing TEE evaluation, not just Louis's Austrian-crop case.**
  Continuing the same investigation as v1.12.0's `deep_mlp`: Tessera
  embeddings are highly spatially autocorrelated within a field, so TEE's
  existing splits (`StratifiedKFold` for k-fold, per-class percentage
  sampling for the learning curve) let pixels from the *same* shapefile
  polygon land on both sides of train/test — a classifier can then partly
  "recognize the field" instead of learning the habitat, inflating the
  reported score. Verified directly on real data (not synthetic): fetched
  ~2M real per-pixel embeddings across a 25km×25km slice of Austria (6,674
  fields, all 17 classes — genuinely more labelled pixels than Louis's own
  177K-pixel run), subsampled to 40K, and compared the *same* `mlp`/
  `deep_mlp` models under a naive `StratifiedKFold` vs. an honest
  `StratifiedGroupKFold` (fields never split across folds): **naive macro
  F1 was 0.11–0.14 higher** than the honest number for both models (mlp:
  0.744 vs. 0.607; deep_mlp: 0.747 vs. 0.638) — a bigger effect than
  `deep_mlp`'s own architecture edge over plain `mlp` by an order of
  magnitude. (One open puzzle, not yet resolved: the *naive* number on this
  compact 25km test region lands close to Frank's reported paper number,
  and the *honest* number lands close to Louis's own real (much larger,
  more geographically dispersed) run — direction not fully understood yet,
  noted for follow-up rather than papered over.)
  Mechanism: `_sample_points_within_budget` (server.py) already computed
  `row_index` — which source shapefile polygon each sampled point came
  from — for every point sampled for pixel-classifier training; it was
  simply discarded (`_row_idx`) at every call site. Now captured as
  `sample_groups`, threaded through to `vectors`/`labels` construction
  (same order, same NaN-coverage filtering) and passed as `groups` to
  `run_learning_curve` (new `groups`/`group_test_fraction` params: carves
  out a fixed pool of whole groups as the test set via
  `StratifiedGroupKFold`, then behaves exactly like the existing
  spatial-split fixed-test-set mode from there) and `run_kfold_cv` (new
  `groups` param: swaps the *pixel*-model splitter to
  `StratifiedGroupKFold`/`GroupKFold`; spatial_mlp/spatial_mlp_5x5 keep
  their own separate patch-derived split, unaffected).
  **Opt-in, not the new default** (`group_by_field` request flag, off by
  default): flipping this on changes every reported score, usually
  downward, so it needs an explicit decision rather than silently
  invalidating comparisons against a user's past evaluations. Ignored (with
  a clear status message) whenever a fixed test set is already active
  (spatial/year/file split takes precedence), when the on-disk result cache
  is hit (predates storing group ids — the in-memory cache carries them,
  the disk one doesn't yet), or for regression (not yet supported).
  Classification-only for now, same scoping rationale as `deep_mlp`.
  7 new tests (`tests/test_group_holdout.py`), including two direct
  regression tests against synthetic leaky data (a naive split must score
  ≥0.1 macro F1 higher than the group-holdout split, same data, same
  model) and a fold-membership check that no group ever appears in more
  than one `StratifiedGroupKFold` test fold. Full suite 249 passed (was
  242 after v1.12.0).

## [1.12.0]

### Added
- **`deep_mlp`: a PyTorch MLP matching the Tessera paper's own downstream-eval
  architecture.** Continuing the Louis Driver Austrian-crop F1-gap
  investigation (v1.11.4's changelog entry): Frank, the paper's own
  downstream-eval author, described his setup directly (2026-09-16) --
  128→512→256→17, each hidden layer **Linear+BatchNorm1d+ReLU+Dropout(0.3)**,
  trained with **AdamW**(1e-3, weight_decay 0.01), batch 8192, 150 epochs,
  **best checkpoint kept by validation weighted-F1** -- and flagged,
  unprompted, that "Louis's v1/v2/v3 don't seem to have BatchNorm and that
  really matters." sklearn's `MLPClassifier` (classify.py's existing `mlp`,
  even after v1.11.4's `StandardScaler` fix) has no BatchNorm/Dropout
  equivalent and no checkpoint selection at all -- confirmed the gap
  survives regardless of training-data volume (Louis's own learning-curve
  numbers: best sklearn variant tops out at macro F1 0.598 at 80% training
  data, vs. Frank's 0.7248 at 30%). New `tessera_eval/deep_mlp.py`
  (`DeepMLPClassifier`, a plain fit/predict/predict_proba wrapper around a
  small `torch.nn.Sequential`) reproduces that architecture; wired into
  `make_classifier`'s new `"deep_mlp"` branch (hyperparameters: `hidden_layers`,
  `dropout`, `lr`, `weight_decay`, `batch_size`, `epochs`) and
  `available_classifiers()`. Classification-only -- that's the only task
  this was diagnosed against.
  Optional dependency, same convention as U-Net (`unet.py`'s `_HAS_TORCH`
  guard): not in `requirements.txt`/`pyproject.toml`, install manually
  (`pip install torch`); `available_classifiers()` only advertises it when
  torch is importable, and requesting it without torch installed raises a
  clear `RuntimeError` rather than an opaque one.
  Interestingly, Frank *doesn't* scale his input features either --
  BatchNorm1d right after the first `Linear` does that job per-batch, which
  is also why v1.11.4's scaling fix alone only closed ~1-2 of the ~20+ point
  gap: it was fixing the same symptom sklearn's MLP has for a different
  reason (no BatchNorm to do it implicitly).
  **Honest verification, not yet a solved gap**: fit/predict/predict_proba
  tested on synthetic separable data and wired end-to-end through
  `make_classifier` (`tests/test_deep_mlp.py`, 10 tests, skipped when torch
  isn't installed — mirrors `test_unet_regression.py`'s own skip pattern).
  Head-to-head against sklearn's `mlp` (Louis's own best variant,
  `256,128,64`) on real Tessera embeddings (2,040 points, 17 balanced
  classes, sampled across the Austria extent via `geotessera` directly) via
  `run_learning_curve` at 30%/80% training splits: `deep_mlp` beat `mlp` by
  a small, real margin (macro F1 +0.0075 at 30%, +0.0073 at 80%) -- real,
  but nowhere close to Frank's reported gap-closing number. Most likely
  explanation: that 2,040-point sample is tiny next to Frank's actual run
  (a 78k-pixel downsampled raster, tens of thousands of training pixels
  from a field-level area-stratified split) -- a ~200K-parameter net with
  BatchNorm needs real data volume to show its advantage, and this repo has
  no local equivalent of Frank's full labelled raster to test against. The
  fair test is Louis's own next full-scale run (his real ~177K-pixel
  dataset) with `deep_mlp` added to his classifier list, not a small
  synthetic-scale rehearsal of it here. The bulk of the F1 gap remains open.

## [1.11.4]

### Fixed
- **MLP classifiers/regressors trained on unscaled embeddings.**
  `make_classifier`/`make_regressor`'s `mlp`, `spatial_mlp`, and
  `spatial_mlp_5x5` branches built a bare `MLPClassifier`/`MLPRegressor`
  with no feature scaling. sklearn's MLP (Adam, default
  `learning_rate_init=0.001`) is sensitive to input scale, and real
  Tessera embeddings are not zero-mean/unit-variance (checked directly:
  per-dimension means range roughly -2.5 to +5, stds roughly 1.0-2.2)
  — so training converged to a visibly worse optimum within a fixed
  `max_iter` budget. Found while investigating a large macro-F1 gap
  between TEE and the Tessera paper's own Austrian-crop numbers (Louis
  Driver, 2026-09-15): a larger MLP sometimes scored *worse* than a
  smaller one at the same training percentage, and the gap widened
  rather than narrowed with more training data — both point at
  undertraining rather than a data or task-difficulty limit. Fixed by
  wrapping every MLP/spatial-MLP construction site in a shared
  `_make_mlp()` helper that builds `Pipeline(StandardScaler, MLP*)`
  instead of a bare estimator (`StandardScaler` fits fresh per training
  call, so there's no leakage across train/test). Verified real,
  positive, and low-variance on genuine embeddings (6,168 points sampled
  across the full Austrian crop dataset, 10 seeds each): macro F1
  +2.05 points at a 10% training split (48.03→50.08), +0.75 points at
  70% (59.54→60.29) — a real but modest effect, not by itself the
  explanation for the much larger gap still under investigation.
  `RandomForestClassifier`/`XGBClassifier`/`KNeighborsClassifier` are
  scale-invariant (or already scale-sensitive by design, for kNN) and
  are left untouched — this fix is scoped to the MLP family only.

## [1.11.3]

### Fixed
- **Download Models never actually produced a real Spatial MLP model.**
  `_tile_cache["spatial_3x3"]`/`["spatial_5x5"]` are always written `None`
  by design (that key means "same-request cache-hit", and spatial features
  are deliberately always re-extracted fresh within a request rather than
  trusted from a stale hit) — but `train_models()` read that same key, so
  its spatial branches never ran. A "Download Models" click for
  `spatial_mlp`/`spatial_mlp_5x5` silently fell through to the generic
  branch and trained a plain, non-windowed `MLPClassifier` on the raw
  pixel vectors, saved under the spatial name, no error at all — a
  downloaded model that didn't match what the evaluation scored. Fixed by
  stashing the run's real spatial features *and their own labels* under
  new `_spatial_3x3`/`_spatial_5x5`/`_spatial_labels_3x3`/
  `_spatial_labels_5x5` keys at the end of a successful run (mirroring the
  pre-existing `_unet_patches` stash), and having `train_models()` read
  those instead. Also fixes a related bug the dead code was hiding: the
  spatial branches paired `spatial_3x3` (patch-derived, its own point
  count) with the *pixel* `labels` array instead of `spatial_labels_3x3`
  — now fixed, and a `train_models()` request for a spatial name whose
  cached data is ever missing skips cleanly with a status message instead
  of falling through to the generic branch.
- **`augment_spatial` gains a `cap`** (default 50,000 rows, existing
  callers unaffected below that) that subsamples before the 4x flip
  expansion. `train_models()`'s two spatial call sites had no size bound
  at all — the same shape of bug as 1.11.2's U-Net fix, just via
  `MLPClassifier.fit()`'s eager array instead of `np.stack`. One shared
  cap in `augment_spatial` itself protects every call site (8 across the
  codebase) without each needing its own capping logic, and additionally
  bounds the learning curve's own spatial branches at very large scale
  (previously unbounded at pct=80%, now capped there too).

## [1.11.2]

### Fixed
- **U-Net final-model training (`train_models()` / "Download Models") could
  OOM the whole compute server**, even when the learning curve for the same
  run completed all its percentages fine. `train_unet_on_patches` /
  `train_unet_regressor_on_patches` built their 16x data augmentation (4
  rotations x {as-is, +noise} x {as-is, h-flipped}) as one eager `np.stack`
  before training started: `len(patches) * 16 * dim * 256 * 256 * 4` bytes,
  all at once. Confirmed live — "Unable to allocate 74.5 GiB for an array
  with shape (2384, 128, 256, 256)" at the default 500-patch cap, "50.0 GiB"
  at 100 patches (Moustafa Eweda). The learning curve survives because each
  percentage step additionally caps itself at 20 patches internally; the
  final-model path has no such cap and hands the whole cached patch set to
  the same eagerly-augmenting function — so "Max spatial/U-Net patches"
  looked like it wasn't being respected for that step, because for that
  step it effectively wasn't.
  Fixed by generating the 16 variants lazily, one `DataLoader` batch at a
  time (`_AugmentedPatches`, a proper `torch.utils.data.Dataset`), instead
  of materializing all of them up front. Peak memory is now
  `O(batch_size)` instead of `O(len(patches) * 16)`. Same 16 deterministic
  variants per patch, same total dataset size and training behaviour;
  Gaussian noise is now independently seeded per sample rather than drawn
  from one sequential stream, so exact U-Net numbers at the same seed shift
  slightly (still fully reproducible: same seed → same result).

## [1.11.1]

### Changed
- Docs only. README quickstart now runs verbatim against the bundled
  `examples/austria_crops.geojson`; notes that `run_kfold_cv` covers
  regression (R²/RMSE/MAE + pooled predicted-vs-actual scatter) and the
  Spatial MLP models; CLI install lines use the PyPI package. First
  release published to PyPI via the trusted-publishing workflow.

## [1.11.0]

### Added
- **`create-map` takes a `clamp` flag** (default `true`, unchanged
  behaviour). Set `clamp: false` in the request body to write the raw
  regression predictions instead of clamping them to the training-target
  span — useful for seeing where a model extrapolates. No effect on
  classification maps. The `map_ready` event now carries `clamped`
  (bool), the status line says whether the output was clamped, and
  `clamp_min` / `clamp_max` GeoTIFF tags are written only when it was
  (Louis Driver — wanted a toggle).

## [1.10.0]

### Added
- **k-fold regression now emits a predicted-vs-actual scatter.**
  `run_kfold_cv` pools the held-out predictions across all folds (every
  sampled point is a test point in exactly one fold), subsamples for
  display, and attaches `{"scatter": {"y_true": [...], "y_pred": [...]}}`
  to each model's entry in the `aggregate` event — the same shape
  `run_learning_curve` emits, so the viewer's scatter plot and the CSV
  export pick it up with no frontend change. Previously k-fold regression
  runs had no scatter in the viewer or the results JSON (Louis Driver).
- **`run_kfold_cv` logs per-fold progress** — a header line (k, task,
  point count, models) and one `Fold i/k done in N.Ns — <scores>` line
  per fold, matching `run_learning_curve`'s terminal output. Makes a
  headless k-fold run followable (Louis Driver).

### Fixed
- **`run-large-area` learning-curve runs with no models left to evaluate
  now return a clear error instead of silently "completing".** Selecting
  only Spatial MLP / U-Net together with a spatial split (or a different
  test year, or a separate test file) drops every model — a fixed test
  set has no neighbourhood features — leaving nothing to run; the learning
  curve then looped through the training percentages doing nothing and
  finished with "0 classifiers" (Louis Driver). The k-fold branch already
  guarded this; the learning-curve branch now does too, and names the
  cause (spatial models + a fixed test set), pointing at the pixel models
  or k-fold (which supports Spatial MLP).

## [1.9.0]

### Added
- **k-fold cross-validation now supports the Spatial MLP models**
  (`spatial_mlp` 3×3, `spatial_mlp_5x5` 5×5). `run_kfold_cv` takes optional
  `spatial_vectors` / `spatial_vectors_5x5` / `spatial_labels` / `dim`
  arguments: pixel models cross-validate over the embedding matrix as
  before, while spatial models cross-validate over their own
  neighbourhood-feature points (a separate, generally larger set drawn
  from the downloaded tile crops) using an independent k-fold split seeded
  identically. Training folds get the same 4× flip augmentation as the
  learning curve. `run-large-area` with `eval_mode="kfold"` extracts the
  spatial features (it no longer treats drawn rectangles / a differing
  test year / a test file as a fixed test set, since k-fold ignores all of
  those) and passes them through. Per-fold F1/R² tables, the mean ± std
  aggregate, and the confusion matrices all include the spatial models.
- U-Net remains unavailable in k-fold mode — it trains on 256×256 image
  patches, not points, so there is no point-based fold split for it. It is
  dropped from a k-fold run with a status note pointing at the learning
  curve. The "needs at least one model" guard message now mentions Spatial
  MLP.

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
