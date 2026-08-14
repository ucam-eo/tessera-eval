"""
Tessera-eval command-line interface
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import typer

from tessera_eval import (
    detect_field_type,
    load_embeddings_for_raster,
    load_embeddings_for_shapefile,
    run_kfold_cv,
    run_learning_curve,
)

# Default cache file for vectors/labels, in the current working directory —
# same convention GeoTessera itself uses for its own tile mirror.
DEFAULT_VECTORS_PATH = Path("vectors.npz")


def _bboxes_overlap(a, b):
    a_minx, a_miny, a_maxx, a_maxy = a
    b_minx, b_miny, b_maxx, b_maxy = b
    return a_minx < b_maxx and b_minx < a_maxx and a_miny < b_maxy and b_miny < a_maxy


def _detect_source_kind(path: str) -> str:
    """Guess whether a ground-truth file is a raster or a vector
    (shapefile/GeoJSON), from its extension."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        return "raster"
    if ext in (".geojson", ".json", ".shp", ".gpkg"):
        return "shapefile"
    raise typer.BadParameter(
        f"Could not determine file type for {path} (unrecognized extension "
        f"'{ext}'). Expected .tif/.tiff for rasters, or .geojson/.shp/.gpkg "
        "for vector data."
    )


app = typer.Typer()


def _get_vectors_and_labels(
    data: str | None,
    field: str | None,
    year: int,
    vectors_path: str | None,
    bbox: str | None = None,
    band: int = 1,
    nodata: str | None = None,
):
    """Load vectors, labels, task, and class names either from a cached
    .npz, or fresh from a shapefile/GeoJSON or GeoTIFF.

    Args:
        data: path to a shapefile/GeoJSON or GeoTIFF, or None to load from a cache
        field: class/target column (required if data is a shapefile/GeoJSON)
        year: Tessera embedding year, used only when loading fresh from data
        vectors_path: path to a cached .npz, or None to use the default cache
            path or fall back to data
        bbox: minx,miny,maxx,maxy in EPSG:4326, required if data is a raster
        band: 1-based band index to read (raster only)
        nodata: comma-separated extra sentinel nodata values (raster only)

    Returns:
        vectors, labels, class_names, task — see load/learning_curve docstrings
    """
    if vectors_path is None and data is None:
        if not DEFAULT_VECTORS_PATH.exists():
            raise typer.BadParameter("No cached vectors found — run `load` first, or pass --data.")
        vectors_path = str(DEFAULT_VECTORS_PATH)

    if vectors_path:
        print(f"Loading cached vectors from {vectors_path}...")
        npz = np.load(vectors_path, allow_pickle=True)
        task = str(npz["task"]) if "task" in npz.files else "classification"
        return npz["vectors"], npz["labels"], list(npz["class_names"]), task

    if not data:
        raise typer.BadParameter("Provide either --vectors, or --data.")
    if year is None:
        raise typer.BadParameter("--year is required when --data is given directly.")

    source_kind = _detect_source_kind(data)

    try:
        from geotessera import GeoTessera
    except ImportError as exc:
        raise typer.BadParameter(
            'Loading from --data requires the geotessera extra: pip install -e ".[geotessera]"'
        ) from exc

    gt = GeoTessera()

    if source_kind == "shapefile":
        if not field:
            raise typer.BadParameter("--field is required when --data is a shapefile/GeoJSON.")
        gdf = gpd.read_file(data)
        task = detect_field_type(gdf, field)
        print(f"Loading embeddings from {data} (field={field}, year={year})...")
        print(f"Detected task: {task}")
        vectors, labels, class_names, stats = load_embeddings_for_shapefile(
            gdf,
            field=field,
            year=year,
            gt_instance=gt,
            callback=lambda i, n: print(f"  tile {i}/{n}", end="\r"),
        )
        print(f"\n{stats['total_pixels']:,} pixels, {stats['n_classes']} classes: {class_names}")
        return vectors, labels, class_names, task

    else:  # raster
        if not bbox:
            raise typer.BadParameter(
                "--bbox is required when --data is a raster, since a raster has "
                "no natural area boundary the way labelled polygons do."
            )
        try:
            minx, miny, maxx, maxy = (float(v) for v in bbox.split(","))
        except ValueError as exc:
            raise typer.BadParameter(
                "--bbox must be four comma-separated numbers: minx,miny,maxx,maxy"
            ) from exc
        nodata_values = [float(v) for v in nodata.split(",")] if nodata else None

        print(f"Loading embeddings for {data} (bbox={bbox}, year={year})...")
        vectors, labels, class_names, stats, task = load_embeddings_for_raster(
            data,
            bbox=(minx, miny, maxx, maxy),
            year=year,
            gt_instance=gt,
            band=band,
            nodata_values=nodata_values,
            callback=lambda i, n: print(f"  tile {i}/{n}", end="\r"),
        )
        print(f"\nDetected task: {task}")
        print(
            f"{stats['total_pixels']:,} pixels across {stats['tiles_with_data']}/{stats['tile_count']} tiles"
        )
        return vectors, labels, class_names, task


@app.command()
def version():
    """Print the tessera-eval library version."""
    from tessera_eval import __version__

    print(__version__)


@app.command()
def load(
    data: str = typer.Option(..., help="Path to a shapefile/GeoJSON or GeoTIFF ground-truth file"),
    field: str = typer.Option(None, help="Class/target column (required for shapefile/GeoJSON)"),
    bbox: str = typer.Option(
        None, help="minx,miny,maxx,maxy, EPSG:4326 — required for raster input"
    ),
    year: int = typer.Option(None, help="Tessera embedding year (required)"),
    band: int = typer.Option(1, help="1-based band index to read (raster only)"),
    nodata: str = typer.Option(
        None, help="Comma-separated extra sentinel nodata values (raster only)"
    ),
    task: str = typer.Option(
        None, help="'classification' or 'regression' (default: auto-detected)"
    ),
    output: str = typer.Option(None, help=f"Output .npz path (default: {DEFAULT_VECTORS_PATH})"),
):
    """Download embeddings for labelled ground truth and cache them for reuse.

    Accepts either a shapefile/GeoJSON of labelled polygons, or a GeoTIFF of
    an already-rasterized reference layer — detected automatically from the
    file extension.

    Args:
        data: path to a shapefile/GeoJSON or GeoTIFF ground-truth file
        field: class/target label column (required for shapefile/GeoJSON)
        bbox: minx,miny,maxx,maxy in EPSG:4326 (required for raster input)
        year: Tessera embedding year (required)
        band: 1-based band index to read (raster only)
        nodata: comma-separated extra sentinel values to treat as missing (raster only)
        task: "classification" or "regression"; auto-detected if not given
        output: .npz path to write; defaults to DEFAULT_VECTORS_PATH if not given
    """
    if year is None:
        raise typer.BadParameter("--year is required.")

    source_kind = _detect_source_kind(data)
    output_path = Path(output) if output else DEFAULT_VECTORS_PATH

    if source_kind == "shapefile":
        if not field:
            raise typer.BadParameter("--field is required when --data is a shapefile/GeoJSON.")
        vectors, labels, class_names, task_out = _get_vectors_and_labels(data, field, year, None)
        np.savez(
            output_path,
            vectors=vectors,
            labels=labels,
            class_names=np.array(class_names, dtype=object),
            source_data=data,
            source_field=field,
            source_year=year,
            source_kind="shapefile",
            task=task_out,
        )
    else:  # raster
        if not bbox:
            raise typer.BadParameter(
                "--bbox is required when --data is a raster, since a raster has "
                "no natural area boundary the way labelled polygons do."
            )
        try:
            minx, miny, maxx, maxy = (float(v) for v in bbox.split(","))
        except ValueError as exc:
            raise typer.BadParameter(
                "--bbox must be four comma-separated numbers: minx,miny,maxx,maxy"
            ) from exc
        if not (
            -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90
        ):
            raise typer.BadParameter(
                f"--bbox values look out of range for EPSG:4326 degrees: {bbox}. "
                "Did you pass coordinates in meters instead?"
            )

        nodata_values = [float(v) for v in nodata.split(",")] if nodata else None

        try:
            from geotessera import GeoTessera
        except ImportError as exc:
            raise typer.BadParameter(
                'Loading rasters requires the geotessera extra: pip install -e ".[geotessera]"'
            ) from exc

        gt = GeoTessera()
        print(f"Loading embeddings for {data} (bbox={bbox}, year={year})...")
        vectors, labels, class_names, stats, task_out = load_embeddings_for_raster(
            data,
            bbox=(minx, miny, maxx, maxy),
            year=year,
            gt_instance=gt,
            band=band,
            task=task,
            nodata_values=nodata_values,
            callback=lambda i, n: print(f"  tile {i}/{n}", end="\r"),
        )
        print(f"\nDetected task: {task_out}")
        print(
            f"{stats['total_pixels']:,} pixels across {stats['tiles_with_data']}/{stats['tile_count']} tiles"
        )

        np.savez(
            output_path,
            vectors=vectors,
            labels=labels,
            class_names=np.array(class_names, dtype=object),
            source_data=data,
            source_field="",
            source_year=year,
            source_kind="raster",
            source_bbox=bbox,
            source_band=band,
            source_nodata=nodata or "",
            task=task_out,
        )

    print(f"Saved to {output_path}")


@app.command()
def kfold(
    data: str = typer.Option(
        None, help="Path to a shapefile/GeoJSON or GeoTIFF (skip if using --vectors)"
    ),
    field: str = typer.Option(None, help="Class/target label column (shapefile/GeoJSON only)"),
    bbox: str = typer.Option(
        None, help="minx,miny,maxx,maxy, EPSG:4326 — required if --data is a raster"
    ),
    band: int = typer.Option(1, help="1-based band index to read (raster only)"),
    nodata: str = typer.Option(
        None, help="Comma-separated extra sentinel nodata values (raster only)"
    ),
    year: int = typer.Option(
        None,
        help="Tessera embedding year. Required if --data is given directly; "
        "inferred from the cache otherwise.",
    ),
    vectors_path: str = typer.Option(
        None, "--vectors", help=f"Path to a cached .npz (default: {DEFAULT_VECTORS_PATH})"
    ),
    models: str = typer.Option(
        "rf,nn", help="Comma-separated model/regressor names (e.g. rf,nn or rf_reg,nn_reg)"
    ),
    k: int = typer.Option(5, help="Number of cross-validation folds"),
    task: str = typer.Option(
        None, help="'classification' or 'regression' (default: auto-detected)"
    ),
    seed: int = typer.Option(42, help="Random seed for reproducible fold splits"),
    max_samples: int = typer.Option(
        None, help="Cap the training set size per fold (random, not stratified by class)"
    ),
    confusion: bool = typer.Option(
        True, help="Print per-class recall / confusion summary (classification only)"
    ),
):
    """Cross-validate classifiers/regressors on labelled data and print results.

    Args:
        data: path to a shapefile/GeoJSON or GeoTIFF (skip if using vectors_path)
        field: class/target label column (shapefile/GeoJSON only)
        bbox: minx,miny,maxx,maxy, EPSG:4326; required if data is a raster
        band: 1-based band index to read (raster only)
        nodata: comma-separated extra sentinel nodata values (raster only)
        year: Tessera embedding year; required if data is given directly
        vectors_path: path to a cached .npz from `load`
        models: comma-separated model/regressor names passed to run_kfold_cv
        k: number of cross-validation folds
        task: "classification" or "regression"; auto-detected if not given
        seed: random seed for reproducible fold splits
        max_samples: optional cap on training set size per fold
        confusion: whether to print a per-class recall summary (classification only)
    """
    model_list = models.split(",")
    vectors, labels, class_names, detected_task = _get_vectors_and_labels(
        data,
        field,
        year,
        vectors_path,
        bbox=bbox,
        band=band,
        nodata=nodata,
    )
    task = task or detected_task
    if task not in ("classification", "regression"):
        raise typer.BadParameter("--task must be 'classification' or 'regression'")

    print(f"\nRunning {k}-fold cross-validation ({task}, seed={seed})...")
    confusion_matrices = {}
    for event in run_kfold_cv(
        vectors, labels, model_list, k=k, task=task, seed=seed, max_training_samples=max_samples
    ):
        if event["type"] == "fold_result":
            print(f"  [fold {event['fold']}] done")
        elif event["type"] == "aggregate":
            print("\nResults:")
            for name, m in event["models"].items():
                if task == "regression":
                    print(
                        f"  {name:>8}: R² {m['mean_r2']:.3f} | "
                        f"RMSE {m['mean_rmse']:.3f} | MAE {m['mean_mae']:.3f}"
                    )
                else:
                    print(f"  {name:>8}: macro-F1 {m['mean_f1']:.3f} ± {m['std_f1']:.3f}")
        elif event["type"] == "confusion_matrices":
            confusion_matrices = event["confusion_matrices"]

    if confusion and task == "classification":
        for name, cm_raw in confusion_matrices.items():
            cm = np.array(cm_raw)
            recall = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            print(f"\nConfusion summary [{name}]:")
            for i, cname in enumerate(class_names):
                off_diag = [(j, recall[i, j]) for j in range(len(class_names)) if j != i]
                most_confused_idx, most_confused_val = max(off_diag, key=lambda x: x[1])
                print(
                    f"  {cname:>20}: recall {recall[i, i]:.2f} "
                    f"(most confused with {class_names[most_confused_idx]})"
                )


@app.command()
def learning_curve(
    data: str = typer.Option(
        None, help="Path to a shapefile/GeoJSON or GeoTIFF (skip if using --vectors)"
    ),
    field: str = typer.Option(None, help="Class/target label column (shapefile/GeoJSON only)"),
    bbox: str = typer.Option(
        None,
        help="Training region, EPSG:4326. Required if --data is a raster. For "
        "--spatial-holdout with a shapefile, this is optional (defaults to the "
        "polygons' own extent).",
    ),
    band: int = typer.Option(1, help="1-based band index to read (raster only)"),
    nodata: str = typer.Option(
        None, help="Comma-separated extra sentinel nodata values (raster only)"
    ),
    year: int = typer.Option(
        None,
        help="Tessera embedding year for training. Required if --data is given "
        "directly; inferred from the cache if using --spatial-holdout/--test-data "
        "without --data.",
    ),
    vectors_path: str = typer.Option(
        None, "--vectors", help=f"Path to a cached .npz (default: {DEFAULT_VECTORS_PATH})"
    ),
    spatial_holdout: bool = typer.Option(
        False,
        help="Hold out a separate test region, splitting --data itself if --test-data isn't given",
    ),
    test_bbox: str = typer.Option(
        None,
        help="Held-out test region, EPSG:4326. Required if the test source (--test-data, "
        "or --data under --spatial-holdout) is a raster and doesn't already have its own bbox. "
        "If omitted under plain --spatial-holdout, --data's region is split in half by longitude.",
    ),
    test_data: str = typer.Option(
        None,
        help="A separate shapefile/GeoJSON or GeoTIFF to use as the test set — a "
        "different region, a different year (with --test-year), or both. "
        "Required whenever --test-year is given.",
    ),
    test_field: str = typer.Option(
        None, help="Class/target label column in --test-data (default: same as --field)"
    ),
    test_year: int = typer.Option(
        None,
        help="Year for the test set. Requires --test-data — a different year always "
        "needs its own ground-truth file, since labels may have changed.",
    ),
    models: str = typer.Option("rf", help="Comma-separated model/regressor names"),
    training_pcts: str = typer.Option(
        "1,5,10,30,50,80", help="Comma-separated training percentages"
    ),
    repeats: int = typer.Option(5, help="Repeats per training percentage"),
    task: str = typer.Option(
        None,
        help="'classification' or 'regression' (default: auto-detected). "
        "Note: run_learning_curve does not currently support regression.",
    ),
):
    """See how model performance changes as training-label percentage increases.

    Held-out test sets, in order of precedence:
    --test-data alone (a separate region and/or year, fully independent);
    --spatial-holdout (splits --data itself into train/test regions, same
    year, unless --test-year + --test-data are also given); otherwise a
    random split of --data/--vectors.

    Args:
        data: path to a shapefile/GeoJSON or GeoTIFF (skip if using vectors_path)
        field: class/target label column (shapefile/GeoJSON only)
        bbox: training region, EPSG:4326; required if data is a raster
        band: 1-based band index to read (raster only)
        nodata: comma-separated extra sentinel nodata values (raster only)
        year: Tessera embedding year for training; required if data is given directly
        vectors_path: path to a cached .npz from `load`
        spatial_holdout: hold out a separate test region from data
        test_bbox: held-out test region, EPSG:4326
        test_data: separate shapefile/GeoJSON or GeoTIFF for the test set;
            required if test_year is given
        test_field: class/target label column in test_data (default: same as field)
        test_year: year for the test set; requires test_data
        models: comma-separated model/regressor names passed to run_learning_curve
        training_pcts: comma-separated training-set percentages to evaluate at
        repeats: repeats per training percentage
        task: "classification" or "regression"; auto-detected if not given
    """
    model_list = models.split(",")
    pcts = [int(p) for p in training_pcts.split(",")]
    test_vectors, test_labels = None, None

    if test_year is not None and not test_data:
        raise typer.BadParameter(
            "--test-year requires --test-data — a different year always needs "
            "its own ground-truth file."
        )

    # Fill in --data/--field/--bbox/--year from the cached .npz's saved
    # metadata, but only for whichever the user didn't explicitly pass.
    if (spatial_holdout or test_data) and not data:
        if not DEFAULT_VECTORS_PATH.exists():
            raise typer.BadParameter("No cached vectors found — run `load` first, or pass --data.")
        npz = np.load(DEFAULT_VECTORS_PATH, allow_pickle=True)
        data = str(npz["source_data"])
        source_kind = str(npz["source_kind"]) if "source_kind" in npz.files else "shapefile"
        if source_kind == "raster":
            bbox = bbox or str(npz["source_bbox"])
            band = int(npz["source_band"]) if "source_band" in npz.files else band
            if not nodata and "source_nodata" in npz.files:
                nodata = str(npz["source_nodata"]) or None
        else:
            field = field or str(npz["source_field"])
        if year is None:
            year = int(npz["source_year"])
        if task is None and "task" in npz.files:
            task = str(npz["task"])
        print(f"Using cached source: data={data}, field={field}, year={year}")

    if data and year is None:
        raise typer.BadParameter(
            "--year is required when --data is given directly (no cached .npz to infer it from)."
        )
    eval_year = test_year if test_year is not None else year

    try:
        from geotessera import GeoTessera
    except ImportError as exc:
        raise typer.BadParameter(
            'Loading embeddings requires the geotessera extra: pip install -e ".[geotessera]"'
        ) from exc

    def _load_side(src_path, src_bbox, src_field, src_year, label):
        """Load one side (train or test) of the split, auto-detecting type."""
        kind = _detect_source_kind(src_path)
        if kind == "shapefile":
            if not src_field:
                raise typer.BadParameter(f"--field is required for {label} (shapefile/GeoJSON).")
            gdf = gpd.read_file(src_path)
            if len(gdf) == 0:
                raise typer.BadParameter(f"{label} file {src_path} contains no polygons.")
            gt_local = GeoTessera()
            print(f"Loading {label} data from {src_path} ({src_year})...")
            vecs, labs, cnames, _ = load_embeddings_for_shapefile(
                gdf, field=src_field, year=src_year, gt_instance=gt_local
            )
            return vecs, labs, cnames, "classification" if task != "regression" else task
        else:  # raster
            if not src_bbox:
                raise typer.BadParameter(f"--bbox is required for {label} (raster).")
            try:
                minx, miny, maxx, maxy = (float(x) for x in src_bbox.split(","))
            except ValueError as exc:
                raise typer.BadParameter(
                    f"bbox for {label} must be four comma-separated numbers: minx,miny,maxx,maxy"
                ) from exc
            nodata_values = [float(x) for x in nodata.split(",")] if nodata else None
            gt_local = GeoTessera()
            print(f"Loading {label} data from {src_path} ({src_year})...")
            vecs, labs, cnames, _, t = load_embeddings_for_raster(
                src_path,
                bbox=(minx, miny, maxx, maxy),
                year=src_year,
                gt_instance=gt_local,
                band=band,
                task=task,
                nodata_values=nodata_values,
            )
            return vecs, labs, cnames, t

    if test_data:
        vectors, labels, class_names, task = _load_side(data, bbox, field, year, "training")
        if task == "regression":
            raise typer.BadParameter("learning-curve does not currently support regression")
        test_vectors, test_labels, test_class_names, _ = _load_side(
            test_data, test_bbox, test_field or field, eval_year, "test"
        )
        if list(test_class_names) != list(class_names):
            raise typer.BadParameter(
                f"Class mismatch between train ({class_names}) and test "
                f"({test_class_names}) — both must contain the same classes."
            )

    elif spatial_holdout:
        if not data:
            raise typer.BadParameter("--spatial-holdout requires --data.")
        source_kind = _detect_source_kind(data)

        if source_kind == "raster":
            if not bbox:
                raise typer.BadParameter("--spatial-holdout with a raster --data requires --bbox.")
            try:
                minx, miny, maxx, maxy = (float(v) for v in bbox.split(","))
            except ValueError as exc:
                raise typer.BadParameter(
                    "--bbox must be four comma-separated numbers: minx,miny,maxx,maxy"
                ) from exc
            if test_bbox:
                try:
                    test_minx, test_miny, test_maxx, test_maxy = (
                        float(v) for v in test_bbox.split(",")
                    )
                except ValueError as exc:
                    raise typer.BadParameter(
                        "--test-bbox must be four comma-separated numbers: minx,miny,maxx,maxy"
                    ) from exc
                train_region = (minx, miny, maxx, maxy)
                test_region = (test_minx, test_miny, test_maxx, test_maxy)
                if _bboxes_overlap(train_region, test_region):
                    raise typer.BadParameter(
                        f"--bbox {bbox} and --test-bbox {test_bbox} overlap — they must "
                        "be disjoint, since --bbox defines the training region and "
                        "shared pixels would leak between train and test."
                    )
            else:
                mid = (minx + maxx) / 2
                train_region = (minx, miny, mid, maxy)
                test_region = (mid, miny, maxx, maxy)
            print(f"Raster spatial split: train {train_region}, test {test_region}")

            nodata_values = [float(v) for v in nodata.split(",")] if nodata else None
            gt = GeoTessera()
            print(f"Loading training data ({year})...")
            vectors, labels, class_names, _, task = load_embeddings_for_raster(
                data,
                bbox=train_region,
                year=year,
                gt_instance=gt,
                band=band,
                task=task,
                nodata_values=nodata_values,
            )
            if task == "regression":
                raise typer.BadParameter("learning-curve does not currently support regression")
            print(f"Loading held-out test data ({eval_year})...")
            test_vectors, test_labels, test_class_names, _, _ = load_embeddings_for_raster(
                data,
                bbox=test_region,
                year=eval_year,
                gt_instance=gt,
                band=band,
                task=task,
                nodata_values=nodata_values,
            )
            if list(test_class_names) != list(class_names):
                raise typer.BadParameter(
                    f"Class mismatch between train ({class_names}) and test "
                    f"({test_class_names}) — try a different bbox or check coverage."
                )

        else:  # shapefile
            if not field:
                raise typer.BadParameter(
                    "--spatial-holdout with a shapefile --data requires --field."
                )
            gdf = gpd.read_file(data)
            if len(gdf) == 0:
                raise typer.BadParameter(f"--data {data} contains no polygons.")
            if task is None:
                task = detect_field_type(gdf, field)
            if task == "regression":
                raise typer.BadParameter("learning-curve does not currently support regression")

            centroids = gdf.centroid
            if test_bbox:
                try:
                    minx, miny, maxx, maxy = (float(v) for v in test_bbox.split(","))
                except ValueError as exc:
                    raise typer.BadParameter(
                        "--test-bbox must be four comma-separated numbers: minx,miny,maxx,maxy"
                    ) from exc
                in_box = (
                    (centroids.x >= minx)
                    & (centroids.x <= maxx)
                    & (centroids.y >= miny)
                    & (centroids.y <= maxy)
                )
                train_gdf, test_gdf = gdf[~in_box], gdf[in_box]
                print(
                    f"Bounding-box split: {len(train_gdf)} training polygons, "
                    f"{len(test_gdf)} held-out polygons inside {test_bbox}"
                )
                if len(test_gdf) == 0:
                    raise typer.BadParameter(
                        f"No polygons found inside --test-bbox {test_bbox}. Check that "
                        f"the coordinates are in the same CRS as --data (currently {gdf.crs})."
                    )
                if len(train_gdf) == 0:
                    raise typer.BadParameter(
                        f"--test-bbox {test_bbox} covers all polygons in --data — "
                        "nothing left to train on."
                    )
            else:
                minx, miny, maxx, maxy = gdf.total_bounds
                mid = (minx + maxx) / 2
                train_gdf = gdf[centroids.x < mid]
                test_gdf = gdf[centroids.x >= mid]
                print(
                    f"Spatial split: {len(train_gdf)} polygons west, "
                    f"{len(test_gdf)} polygons east of {mid:.4f}"
                )
                if len(test_gdf) == 0 or len(train_gdf) == 0:
                    raise typer.BadParameter(
                        "Spatial split produced an empty train or test set — your "
                        "polygons may all share the same location, or check --data's CRS."
                    )

            gt = GeoTessera()
            print(f"Loading training data ({year})...")
            vectors, labels, class_names, _ = load_embeddings_for_shapefile(
                train_gdf, field=field, year=year, gt_instance=gt
            )
            print(f"Loading held-out test data ({eval_year})...")
            test_vectors, test_labels, test_class_names, _ = load_embeddings_for_shapefile(
                test_gdf, field=field, year=eval_year, gt_instance=gt
            )
            if task == "classification" and list(test_class_names) != list(class_names):
                raise typer.BadParameter(
                    f"Class mismatch between train ({class_names}) and test "
                    f"({test_class_names}) — try a different split or check label coverage."
                )

    else:
        vectors, labels, class_names, detected_task = _get_vectors_and_labels(
            data,
            field,
            year,
            vectors_path,
            bbox=bbox,
            band=band,
            nodata=nodata,
        )
        task = task or detected_task

    if task not in ("classification", "regression"):
        raise typer.BadParameter("--task must be 'classification' or 'regression'")
    if task == "regression":
        raise typer.BadParameter("learning-curve does not currently support regression")

    mode = "spatial hold-out" if test_vectors is not None else "random split"
    print(f"\nRunning learning curve over {pcts}% of labels ({mode}, {task})...")
    for event in run_learning_curve(
        vectors,
        labels,
        classifier_names=model_list,
        training_pcts=pcts,
        repeats=repeats,
        test_vectors=test_vectors,
        test_labels=test_labels,
        task=task,
    ):
        if event["type"] == "progress":
            for name in model_list:
                m = event["classifiers"][name]
                if task == "regression":
                    print(
                        f"  {event['pct']:>3}% [{name}] R² {m['mean_r2']:.3f} | "
                        f"RMSE {m['mean_rmse']:.3f} | MAE {m['mean_mae']:.3f}"
                    )
                else:
                    print(f"  {event['pct']:>3}% [{name}] macro-F1 {m['mean_f1']:.3f}")


def main():
    app()


if __name__ == "__main__":
    main()
