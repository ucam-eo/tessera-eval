# Backend refactor: make `server.py` a thin wrapper over the library

**Status:** planned, not started. Written 2026-09-08. Do this *after* the teaching-term
crunch — it is a maintainability investment, not a bug fix, and it carries
short-term regression risk that is only worth taking when you can watch it settle.

## Goal

`tessera_eval/server.py` is ~3,560 lines. Most of it is orchestration logic
(tile fetching, sampling, caching, the evaluation pipeline, map generation)
tangled into three giant Flask `stream()` closures. Move that logic into the
importable library so:

- the Flask server is just: parse request → call a library generator → stream JSON;
- `tessera_eval/cli.py` and blore's `scripts/tee_evaluate.py` call the **same**
  code path (headless spatial k-fold and headless map creation come for free);
- there is one implementation to test and fix, not three parallel ones.

Target: `server.py` ≈ 500–700 lines.

## Non-goals

- No change to the HTTP API. **The `/api/evaluation/*` wire contract is frozen** —
  blore's Django proxies it verbatim and the frontend parses specific NDJSON
  event shapes. This is behaviour-preserving code movement.
- No new features. (Spatial k-fold in the CLI falls out of stage 5, but that is
  a consequence, not the point.)
- No change to the on-disk cache format or the GeoTIFF output.

## Hard constraint & safety net

Before moving any stream, add a **wire-snapshot test** for that endpoint:
drive it through `app.test_client()` with a fake `GeoTessera`, capture the full
ordered list of NDJSON events, assert it byte-for-byte (modulo timing fields)
after the move. The existing suite (~199 tests, many already endpoint-level) is
the backstop; the snapshots make "did the refactor change the wire?" a one-line
answer per stage.

## Current structure (what moves where)

| Currently in `server.py` | ~lines | Goes to |
|---|---|---|
| `_extract_tile_patches` (tile download + 3×3/5×5 windows + U-Net patches) | 341 | `tessera_eval/tiles.py` |
| `_sample_points_within_budget` + equal/proportional/sqrt allocation | ~120 | `tessera_eval/tiles.py` |
| `_get_zarr`, `_probe_zarr_coverage` | ~50 | `tessera_eval/tiles.py` |
| `_load_cached_result`, `_save_cached_result`, `_gdf_hash`, `_result_cache_path`, `_cached_tiles_need_reload` | ~120 | `tessera_eval/resultcache.py` |
| `_predict_raster`, `_crop_tile_to_bbox`, `_reproject_prediction`, `_merge_prediction_rasters`, `_render_map_preview`, `_hex_to_rgb`, palettes | ~260 | `tessera_eval/rastermap.py` |
| `run_large_area.stream()` body | ~1,380 | `tessera_eval/pipeline.py` → `evaluate_large_area()` |
| `train_models.stream()` body | ~150 | `tessera_eval/pipeline.py` → `train_final_models()` |
| `create_map.stream()` body | ~465 | `tessera_eval/pipeline.py` → `create_map()` |

**Stays in `server.py`:** route decorators, `request`/`jsonify`/`Response`/`send_file`,
CORS, `upload_shapefile` (multipart + zip extraction), `list/clear-shapefiles`,
`cancel`, `finish-classifier`, `download-model/<name>`, `download-map/<name>`,
`/health`, the reverse `proxy`, `main()` + waitress, and the session-state holders.

### Design of the `pipeline.py` generators

Each is a generator yielding plain dict events (no Flask). It takes **explicit
parameters** instead of reaching for module globals:

```python
def evaluate_large_area(
    gdf, field, *, train_year, test_year, classifiers, classifier_params,
    sampling, max_patches, eval_mode, kfold_k, seed, task,
    train_bboxes, test_bboxes, test_gdf, test_field, max_training_samples,
    gt,                 # a GeoTessera / GeoTesseraZarr handle, caller-owned
    cache=None,         # a TileCache object (server holds one; CLI passes None)
    cancel=lambda: False,  # Callable[[], bool]
) -> Iterator[dict]: ...
```

- **`cache`** — wrap the current `_tile_cache` dict in a small class
  (`TileCache`) with typed accessors. The server constructs one at startup and
  passes it every call; the CLI passes `None` (a fresh throwaway) so nothing is
  reused between invocations.
- **`cancel`** — the server passes `lambda: _cancel_flag.is_set()`; the CLI
  passes the default. Replaces every `if _cancelled():` check.
- **`gt`** — the server keeps its cached `_geotessera_instance` (registry init is
  10–30 s); the CLI builds one per run. The generator never constructs it.

The Flask route then reads:

```python
@app.route("/api/evaluation/run-large-area", methods=["POST"])
def run_large_area():
    body = request.get_json()
    gdf = _get_merged_gdf()
    if gdf is None:
        return jsonify({"error": "No shapefile uploaded."}), 400
    gen = pipeline.evaluate_large_area(
        gdf, body["field"], gt=_get_geotessera(), cache=_TILE_CACHE,
        cancel=lambda: _cancel_flag.is_set(),
        **_evaluate_kwargs_from_body(body),
    )
    return Response(_padded(_json_lines(gen)), mimetype="application/x-ndjson")
```

## Staging

Each stage is a self-contained commit/tag, suite green, wire snapshots
unchanged. Stop after any stage and the tree is consistent.

### Stage 1 — `tiles.py` (~1 day)
Move `_extract_tile_patches`, `_sample_points_within_budget`, the sampling
allocation, and the zarr helpers as **pure functions**. `server.py` imports them;
zero behaviour change. Add unit tests that call `extract_tile_patches` directly
with a fake `GeoTessera` (today it is only exercised through the endpoint).

### Stage 2 — `resultcache.py` + `rastermap.py` (~½ day)
Move the disk-cache helpers and the raster/reproject/preview helpers. Mechanical.
`_cached_tiles_need_reload` already has good test coverage; keep it.

### Stage 3 — `pipeline.evaluate_large_area` (~2 days, riskiest)
Extract the `run_large_area.stream()` body into the generator above.
- First: add the wire-snapshot test for `/run-large-area` (classification,
  regression, k-fold, spatial split, year split, file split — one snapshot each).
- Then: lift the body verbatim, threading `cache` / `cancel` / `gt` as params.
- The route shrinks to ~40 lines.
- Watch: the in-memory-cache-hit fast path, the disk-cache shortcut, and the
  three split constructions — those are where state coupling hides.

### Stage 4 — `pipeline.train_final_models` + `pipeline.create_map` (~1–1.5 days)
Same treatment. `create_map` also needs its GeoTIFF-output and preview snapshot
pinned first (open the written file, assert dtype/CRS/nodata/tags).

### Stage 5 — CLI convergence (~1 day)
- Add `tessera-eval evaluate` (and optionally `create-map`) commands that call
  the pipeline generators.
- Rewrite blore `scripts/tee_evaluate.py` to build a kwargs dict from the config
  and call `pipeline.evaluate_large_area`.
- Widen `tee_evaluate_config_v1`: carry `spatial_models` and `max_patches`;
  drop the `valid_clf -= {"spatial_mlp", "spatial_mlp_5x5"}` exclusion in
  `validate_config` (spatial k-fold now works headless because the tile
  extractor is importable).
- Update `blore/public/user_guide.md`: the CLI section no longer says
  "pixel models only".

### Stage 6 — cleanup (~½ day)
Delete dead code. Make `cli.py`'s existing `kfold` / `learning_curve` commands
thin shims over the pipeline (or leave them — they are point-only and already
tested; converging them is optional).

## Rollout (per stage that ships)

Stages 1–2 and 6 are `tessera_eval`-internal — no wire change, but still:

1. `tessera-eval`: bump `pyproject.toml` (`1.9.0` → `1.10.0` after stage 1, etc.),
   CHANGELOG entry, commit, `git tag vX.Y.0`, push `main` + tag over HTTPS
   (`gh auth setup-git`; `git push https://github.com/ucam-eo/tessera-eval.git main`).
2. `blore`: bump the `requirements.txt` pin to the new tag, commit, push
   `https://github.com/ucam-eo/TEE.git main`.
3. Docker: `gh workflow run docker-build.yml --repo ucam-eo/TEE`, watch to green
   (`gh run watch <id> --repo ucam-eo/TEE --exit-status`). Rebuilds `sk818/tee:stable`.
4. Local: `pip install -e ~/code/tessera-eval` then `./scripts/deploy-compute.sh --local`;
   run one eval end to end.
5. **tee.cl**: your step — `sudo bash manage.sh` → 7 (pulls the image, restarts).
   tee.cl is UI-only, so this ships only the frontend half.
6. **Compute server(s)**: whoever runs `tee-compute` (you, and anyone following
   the user-guide self-host path) needs `git pull && pip install --upgrade -r
   requirements.txt` + restart to pick up the new `tessera_eval`. Verify with
   `curl <host>/api/evaluation/health` → `"version": "X.Y.0"`.

Stages 3–5 additionally: run every wire-snapshot test before tagging, and after
deploying give it 2–3 days of real use before starting the next stage.

## Risk register

| Risk | Mitigation |
|---|---|
| Silent wire regression in a stream (event order, a dropped field) | Wire-snapshot test per endpoint *before* the move; diff after. |
| `_tile_cache` coupling — a fast-path that mutates shared state | Model as an explicit `TileCache` object; grep every `_tile_cache[` before stage 3 and enumerate the accesses in the PR description. |
| Cancellation stops working | Keep `/cancel` route + `_cancel_flag`; only the *check* moves (to the `cancel` callable). Test: start a run, POST `/cancel`, assert an `error: Cancelled` event. |
| GeoTessera registry re-init per request (perf regression) | `gt` is a parameter; server passes its cached instance. Assert in a test that the route reuses one handle. |
| Half-finished migration if you stop mid-stage | Stages are independently shippable; never leave a stage's branch merged half-done. |

## Minimal fallback (if you never get to the full thing)

Do **stage 1 only**: lift `_extract_tile_patches` + sampling into `tiles.py` as
pure functions, no behaviour change. That alone makes the tile extractor
importable, so a future "spatial k-fold in the CLI" is a small follow-up instead
of a blocked one. Everything else can stay as-is indefinitely — it works and it
is tested.
