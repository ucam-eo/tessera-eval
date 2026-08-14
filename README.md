# tessera-eval

Evaluate land-cover / habitat classifiers on **Tessera** satellite embeddings.

[Tessera](https://github.com/ucam-eo/tessera) is a geospatial foundation model
that produces a **128-dimensional embedding for every ~10 m pixel** of the Earth's
surface, per year. `tessera-eval` is a small, framework-independent Python library
for the question that immediately follows: *how well can you map a category of
interest (a habitat, crop, or land-cover class) from those embeddings, given some
labelled polygons?*

It handles the unglamorous-but-fiddly parts end to end:

- **Loading + dequantizing** embeddings from the formats Tessera tooling emits
  (GeoTessera `int8 × per-pixel-scale` tiles, and TEE per-dim `uint8` vector
  directories).
- **Rasterizing** a labelled shapefile/GeoJSON onto the embedding pixel grid with
  stable class IDs.
- **Training + scoring** a panel of classifiers/regressors (k-NN, random forest,
  MLP, spatial MLP, optional XGBoost, optional U-Net) with **learning curves**,
  **k-fold cross-validation**, and **spatial hold-out** splits.
- An optional **local compute server** (`tee-compute`) so you can run the ML on
  your own machine while pulling tiles/UI from a hosted service.

> The library core (`data`, `rasterize`, `classify`, `evaluate`) is pure NumPy /
> scikit-learn / rasterio and has no web-framework or hosting dependency. GeoTessera
> zarr access lives in the separate
> [`tessera-zarr-utils`](https://github.com/ucam-eo/tessera-zarr-utils) package
> (used by the compute server).

## Install

```bash
pip install tessera-eval                 # core library
pip install "tessera-eval[geotessera]"   # + tile access (load_embeddings_for_shapefile)
pip install "tessera-eval[server]"       # + the tee-compute local server
pip install "tessera-eval[all]"          # geotessera + xgboost + matplotlib
```

Optional extras: `geotessera` (fetch tiles), `xgboost` (gradient-boosted models),
`torch` (the U-Net), `plot` (matplotlib), `server` (Flask compute server),
`dev` (pytest/ruff/mypy). Python ≥ 3.10.

## Quickstart

Cross-validate a classifier on labelled polygons, pulling embeddings tile-by-tile:

```python
import geopandas as gpd
from geotessera import GeoTessera
from tessera_eval import load_embeddings_for_shapefile, run_kfold_cv

# 1. Labelled polygons (any CRS — reprojected internally) with a class column.
gdf = gpd.read_file("habitats.geojson")

# 2. Pull a 128-d embedding for every pixel under the polygons (memory-bounded:
#    one GeoTessera tile at a time, keeping only labelled pixels).
gt = GeoTessera()
vectors, labels, class_names, stats = load_embeddings_for_shapefile(
    gdf, field="habitat", year=2024, gt_instance=gt
)
print(f"{stats['total_pixels']:,} labelled pixels over {stats['n_classes']} classes")

# 3. 5-fold cross-validation of a random forest and a nearest-neighbour baseline.
for event in run_kfold_cv(vectors, labels, ["rf", "nn"], k=5):
    if event["type"] == "aggregate":
        for name, m in event["models"].items():
            print(f"{name:>4}: macro-F1 {m['mean_f1']:.3f} ± {m['std_f1']:.3f}")
```

Already have a TEE vector directory on disk? Load it directly:

```python
from tessera_eval import load_tee_vectors
vectors, coords, metadata = load_tee_vectors("/path/to/vectors/aoi/2024")
# vectors: float32 (N, 128); coords: int32 (N, 2) pixel (x, y); metadata: dict
```

See the [tutorial](docs/tutorial.md) for the full workflow (labels → learning
curve → confusion matrix → interpretation).

## Command-line interface

The workflow covered in the [tutorial](docs/tutorial.md) can also be run through the command line.

First, install the `geotessera` package needed for the `load` step below:

```
pip install -e ".[geotessera]"
```

Optional installation for using xgboost:

```
pip install -e ".[xgboost]"
```

Download the Tessera embeddings for your labelled ground truth, and save the result to a file (vectors.npz by default, change the name with argument `--output`). `--data` accepts either a shapefile/GeoJSON of labelled polygons or a GeoTIFF of an already-rasterized reference layer.

For a shapefile/GeoJSON, `--field` is the column holding the class or target values (e.g. `habitat`):

```bash
tessera-eval load --data /path/to/habitats.geojson --field habitat --year 2024
```

For a GeoTIFF, `--bbox` is required, in EPSG:4326 (longitude,latitude in degrees), since a raster has no natural area boundary the way labelled polygons do. `--nodata` marks any missing-value codes:

```bash
tessera-eval load --data site_type.tif --bbox 27.1,67.75,27.2,67.85 --year 2024 --nodata 32766,32767
```

`kfold` and `learning-curve` reuse the cached `vectors.npz` automatically. Pass `--vectors <path>` to use a different cached file instead.

Run k-fold cross-validation and print accuracy per model.

```bash
tessera-eval kfold --models rf,nn,mlp      # for classification
tessera-eval kfold --models rf_reg,nn_reg  # for regression
```

Optional arguments:

```bash
# --k: number of cross-validation folds (default: 5)
# --seed: random seed for reproducible fold splits (default: 42)
# --confusion / --no-confusion: show or hide the confusion summary (for classification only)
# --vectors: path to a different cached .npz (default: vectors.npz from `load`)
# --max-samples: cap the training set size per fold (random, not stratified by class) -
#   usually needed for raster-derived data, which labels every pixel, not a hand-picked subset
tessera-eval kfold --models rf --k 10 --seed 1 --no-confusion --max-samples 50000
```

Investigate how accuracy changes using different fractions of training labels (currently only for classification task).
```bash
tessera-eval learning-curve --models rf --training-pcts 1,5,10,30,50,80 --repeats 5  # --training-pcts: % of labels per step (default: 1,5,10,30,50,80); --repeats: random repeats per step (default: 5)
```



Since neighbouring pixels are usually very similar, a random split can overstate how accurate the model really is (see [tutorial](docs/tutorial.md)). For a more reliable estimate, train on one geographic half of your area and test on the other, splitting by longitude:

```bash
tessera-eval learning-curve --models rf --spatial-holdout
```

You can also choose exactly which region to hold out, by passing a bounding box (for a shapefile/GeoJSON, in the same CRS as your data; for a GeoTIFF, always EPSG:4326):


```bash
tessera-eval learning-curve --models rf --spatial-holdout --test-bbox 27.16,67.77,27.23,67.82
```

You can also use a completely separate file as the test set - a different region, a different year, or both. If you are evaluating on a different year, pass `--test-year` along with `--test-data`.

```bash
tessera-eval learning-curve --models rf --test-data /path/to/other_region.geojson --test-year 2023
```

For a GeoTIFF, `--test-bbox` is also required, defining the test region within that file:

```bash
tessera-eval learning-curve --test-data site_type_2023.tif --test-bbox 27.1,67.75,27.2,67.85 --test-year 2023 --models rf
```

This reuses the cached `vectors.npz` as the training data by default. To use a different training area or year instead, pass `--data`/`--bbox`/`--year` explicitly:

```bash
tessera-eval learning-curve --data site_type_2019.tif --bbox 27.1,67.75,27.2,67.85 --year 2019 --test-data site_type_2023.tif --test-bbox 27.1,67.75,27.2,67.85 --test-year 2023 --models rf
```

Run any command with `--help` for a list of all possible arguments. 



## Documentation

- **[Data formats](docs/data-formats.md)** — the Tessera embedding formats this
  library reads and the exact dequantization maths. *Start here if you're wiring in
  your own data.*
- **[API reference](docs/api-reference.md)** — every public function, with array
  shapes and dtypes.
- **[Tutorial](docs/tutorial.md)** — an end-to-end worked example.
- **[Compute server](docs/compute-server.md)** — running `tee-compute` (local ML,
  hosted data).

## What's in the box

| Module | Purpose |
|---|---|
| `tessera_eval.data` | Load + dequantize embeddings (`load_tee_vectors`, `dequantize_int8`, `dequantize_uint8`, `load_embeddings_for_shapefile`, `load_embeddings_for_shapefile_vq`, `load_embeddings_for_raster`). |
| `tessera_eval.rasterize` | Burn shapefile polygons onto a pixel grid with stable, 1-based class IDs|
| `tessera_eval.classify` | Classifier/regressor factory + spatial neighbourhood features. |
| `tessera_eval.evaluate` | Learning curves, k-fold CV, spatial split, metrics, field-type detection. |
| `tessera_eval.unet` | Optional PyTorch U-Net for sparse-label tile segmentation. |
| `tessera_eval.server` | `tee-compute`: local Flask compute server, proxies data/UI to a hosted TEE. |
| `tessera_eval.cli` | `tessera-eval` command-line interface: `load`, `kfold`, `learning-curve`. |

GeoTessera zarr access (`get_zarr`, `probe_zarr_coverage`, `read_region_chunked`) now
lives in [`tessera-zarr-utils`](https://github.com/ucam-eo/tessera-zarr-utils); the
compute server depends on it.

Available models: `nn`, `rf`, `mlp`, `spatial_mlp`, `spatial_mlp_5x5`, `xgboost`
(if installed), `unet` (if torch installed); regressors `nn_reg`, `rf_reg`,
`mlp_reg`, `xgboost_reg`. See `available_classifiers()` / `available_regressors()`.

## Design notes

- **Class imbalance is expected and fine.** Macro-F1 is reported alongside
  weighted-F1 precisely so rare classes are visible.
- **Determinism.** Estimators use `random_state=42`; evaluation takes an explicit
  `seed`. Same inputs → same numbers.
- **Spatial leakage.** For honest accuracy on contiguous habitats, prefer the
  spatial hold-out (`run_learning_curve(..., test_vectors=, test_labels=)`) over a
  random pixel split — neighbouring pixels are highly autocorrelated.
- **Memory.** `load_embeddings_for_shapefile` streams one tile at a time and keeps
  only labelled pixels, so county/country-scale shapefiles are tractable.

## Development

```bash
git clone https://github.com/ucam-eo/tessera-eval && cd tessera-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[server,dev]"
ruff check . && ruff format --check . && pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

If this is useful in academic work, please cite the Tessera model and link back to
this repository. (A `CITATION.cff` will be added alongside the Tessera paper
reference.)

## License

[MIT](LICENSE).
