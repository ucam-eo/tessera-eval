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


app = typer.Typer()


def _get_vectors_and_labels(
    data: str | None, field: str | None, year: int, vectors_path: str | None
):
    """Load vectors, labels, and task type from a cached .npz, or fresh from a shapefile.

    Falls back to the default cache file (DEFAULT_VECTORS_PATH) in the current
    directory if neither vectors_path nor data is given.

    Args:
        data: path to a shapefile/GeoJSON with labelled polygons, or None to
            load from a cache instead
        field: column name in data containing the class/target label
        year: Tessera embedding year, used only when loading fresh from data
        vectors_path: path to a cached .npz previously written by `load`, or
            None to use the default cache path or fall back to data

    Returns:
        vectors: float32 array, shape (N, 128), one embedding per labelled pixel
        labels: int array, shape (N,) for classification, or float32 array,
            shape (N,) for regression
        class_names: list of str, class names in label-index order
            (meaningless for regression, kept for a consistent return shape)
        task: str, "classification" or "regression"
    """

    # No explicit --vectors and no --data given: fall back to the default
    # cache file in the current directory, if it exists.
    if vectors_path is None and data is None:
        if not DEFAULT_VECTORS_PATH.exists():
            raise typer.BadParameter("No cached vectors found — run `load` first, or pass --data.")
        vectors_path = str(DEFAULT_VECTORS_PATH)

    if vectors_path:
        print(f"Loading cached vectors from {vectors_path}...")
        npz = np.load(vectors_path, allow_pickle=True)
        task = str(npz["task"]) if "task" in npz.files else "classification"
        return npz["vectors"], npz["labels"], list(npz["class_names"]), task

    if not data or not field:
        raise typer.BadParameter("Provide either --vectors, or both --data and --field.")

    try:
        from geotessera import GeoTessera
    except ImportError as exc:
        raise typer.BadParameter(
            'Loading from --data requires the geotessera extra: pip install -e ".[geotessera]"'
        ) from exc

    gdf = gpd.read_file(data)
    task = detect_field_type(gdf, field)
    gt = GeoTessera()  # defaults embeddings_dir to the current working directory
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


@app.command()
def version():
    """Print the tessera-eval library version."""
    from tessera_eval import __version__

    print(__version__)


@app.command()
@app.command()
def load(
    data: str = typer.Option(..., help="Path to a shapefile/GeoJSON with labelled polygons"),
    field: str = typer.Option(..., help="Column name containing the class/target label"),
    year: int = typer.Option(2024, help="Tessera embedding year"),
    output: str = typer.Option(None, help=f"Output .npz path (default: {DEFAULT_VECTORS_PATH})"),
):
    """Download embeddings once and cache vectors/labels/task to disk for reuse.

    Args:
        data: path to a shapefile/GeoJSON with labelled polygons
        field: column name in data containing the class/target label
        year: Tessera embedding year
        output: .npz path to write; defaults to DEFAULT_VECTORS_PATH if not given
    """
    output_path = Path(output) if output else DEFAULT_VECTORS_PATH

    vectors, labels, class_names, task = _get_vectors_and_labels(data, field, year, None)
    np.savez(
        output_path,
        vectors=vectors,
        labels=labels,
        class_names=np.array(class_names, dtype=object),
        source_data=data,
        source_field=field,
        source_year=year,
        source_kind="shapefile",
        task=task,
    )
    print(f"Saved to {output_path}")


@app.command(name="load-raster")
@app.command(name="load-raster")
def load_raster(
    raster: str = typer.Option(..., help="Path to a GeoTIFF (or other rasterio-readable raster)"),
    bbox: str = typer.Option(
        ...,
        help="minx,miny,maxx,maxy in EPSG:4326 (longitude,latitude in degrees) — "
        "required, since a raster may cover a much larger area than you want "
        "to process",
    ),
    year: int = typer.Option(2024, help="Tessera embedding year"),
    band: int = typer.Option(1, help="1-based band index to read"),
    nodata: str = typer.Option(
        None,
        help="Comma-separated extra sentinel values to treat as missing "
        "(e.g. 32766,32767), beyond the raster's own declared nodata",
    ),
    task: str = typer.Option(
        None, help="'classification' or 'regression' (default: auto-detected)"
    ),
    output: str = typer.Option(None, help=f"Output .npz path (default: {DEFAULT_VECTORS_PATH})"),
):
    """Download embeddings for pixels covered by an already-rasterized reference
    layer (e.g. a forest-inventory GeoTIFF), and cache them for reuse.

    Args:
        raster: path to a GeoTIFF (or other rasterio-readable raster)
        bbox: minx,miny,maxx,maxy in EPSG:4326, bounding the area to process
        year: Tessera embedding year
        band: 1-based band index to read
        nodata: comma-separated extra sentinel values to treat as missing
        task: "classification" or "regression"; auto-detected if not given
        output: .npz path to write; defaults to DEFAULT_VECTORS_PATH if not given
    """
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
            'load-raster requires the geotessera extra: pip install -e ".[geotessera]"'
        ) from exc

    gt = GeoTessera()
    print(f"Loading embeddings for {raster} (bbox={bbox}, year={year})...")
    vectors, labels, class_names, stats, detected_task = load_embeddings_for_raster(
        raster,
        bbox=(minx, miny, maxx, maxy),
        year=year,
        gt_instance=gt,
        band=band,
        task=task,
        nodata_values=nodata_values,
        callback=lambda i, n: print(f"  tile {i}/{n}", end="\r"),
    )
    task = task or detected_task
    print(f"\nDetected task: {task}")
    print(
        f"{stats['total_pixels']:,} pixels across {stats['tiles_with_data']}/{stats['tile_count']} tiles"
    )

    output_path = Path(output) if output else DEFAULT_VECTORS_PATH
    np.savez(
        output_path,
        vectors=vectors,
        labels=labels,
        class_names=np.array(class_names, dtype=object),
        source_data=raster,
        source_field="",
        source_year=year,
        source_kind="raster",
        source_bbox=bbox,
        source_band=band,
        source_nodata=nodata or "",
        task=task,
    )
    print(f"Saved to {output_path}")


@app.command()
def kfold(
    data: str = typer.Option(None, help="Path to a shapefile/GeoJSON (skip if using --vectors)"),
    field: str = typer.Option(None, help="Class/target label column (skip if using --vectors)"),
    year: int = typer.Option(2024, help="Tessera embedding year"),
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
        None,
        help="Cap the training set size per fold — recommended for dense, "
        "wall-to-wall data (e.g. from load-raster), where every pixel in the "
        "area is a labelled example rather than a sparse hand-labelled subset",
    ),
    confusion: bool = typer.Option(
        True, help="Print per-class recall / confusion summary (classification only)"
    ),
):
    """Cross-validate classifiers/regressors on labelled data and print results.

    Args:
        data: path to a shapefile/GeoJSON (skip if using vectors_path)
        field: class/target label column in data (skip if using vectors_path)
        year: Tessera embedding year, used only when loading fresh from data
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
        data, field, year, vectors_path
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
    data: str = typer.Option(None, help="Path to a shapefile/GeoJSON (skip if using --vectors)"),
    field: str = typer.Option(None, help="Class/target label column (skip if using --vectors)"),
    raster: str = typer.Option(
        None, help="Path to a GeoTIFF instead of --data (skip if using --vectors)"
    ),
    year: int = typer.Option(
        None,
        help="Tessera embedding year (default: 2024, or the cached year with --spatial-holdout)",
    ),
    vectors_path: str = typer.Option(
        None, "--vectors", help=f"Path to a cached .npz (default: {DEFAULT_VECTORS_PATH})"
    ),
    spatial_holdout: bool = typer.Option(
        False, help="Hold out a separate test region from --data/--raster"
    ),
    bbox: str = typer.Option(
        None,
        help="Training/overall region. For --data: optional, defaults to the "
        "polygons' own extent. For --raster: required — the area to load "
        "(minx,miny,maxx,maxy, EPSG:4326). If --test-bbox is omitted, this "
        "region is split in half by longitude for train/test.",
    ),
    test_bbox: str = typer.Option(
        None,
        help="Held-out test region (minx,miny,maxx,maxy, EPSG:4326). If "
        "omitted, --bbox (or the polygons' own extent, for --data) is split "
        "in half by longitude for train/test instead.",
    ),
    band: int = typer.Option(1, help="1-based band index to read (--raster only)"),
    nodata: str = typer.Option(
        None, help="Comma-separated extra sentinel nodata values (--raster only)"
    ),
    test_data: str = typer.Option(
        None,
        help="Path to a separate shapefile/GeoJSON to use as a fixed, independent "
        "test set. Takes precedence over --spatial-holdout/--bbox/--test-bbox. "
        "Not supported for --raster.",
    ),
    test_field: str = typer.Option(
        None, help="Class/target label column in --test-data (default: same as --field)"
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

    Three ways to get a spatially separate test set: --test-data (a separate
    shapefile/region), --spatial-holdout with --test-bbox (you choose the
    held-out region), or --spatial-holdout alone (automatic east/west split
    of --data's extent, or of --bbox for --raster).

    Args:
        data: path to a shapefile/GeoJSON (skip if using vectors_path or raster)
        field: class/target label column in data (skip if using vectors_path)
        raster: path to a GeoTIFF instead of data (skip if using vectors_path)
        year: Tessera embedding year; falls back to the cached year with
            spatial_holdout, or 2024 otherwise
        vectors_path: path to a cached .npz from `load`/`load-raster`
        spatial_holdout: hold out a separate test region from data/raster
        bbox: training/overall region; required for raster, optional for data
        test_bbox: held-out test region; if omitted, bbox (or data's own
            extent) is split in half by longitude instead
        band: 1-based band index to read (raster only)
        nodata: comma-separated extra sentinel nodata values (raster only)
        test_data: path to a separate shapefile/GeoJSON for a fixed,
            independent test set; not supported for raster
        test_field: class/target label column in test_data (default: same as field)
        models: comma-separated model/regressor names passed to run_learning_curve
        training_pcts: comma-separated training-set percentages to evaluate at
        repeats: repeats per training percentage
        task: "classification" or "regression"; auto-detected if not given
    """
    model_list = models.split(",")
    pcts = [int(p) for p in training_pcts.split(",")]

    test_vectors, test_labels = None, None
    source_kind = "raster" if raster else ("shapefile" if data else None)

    # Fill in source info from the cached .npz's saved metadata, but only
    # for whichever the user didn't explicitly pass.
    if (spatial_holdout or test_data) and not data and not raster:
        if not DEFAULT_VECTORS_PATH.exists():
            raise typer.BadParameter(
                "No cached vectors found — run `load`/`load-raster` first, or pass --data/--raster."
            )
        npz = np.load(DEFAULT_VECTORS_PATH, allow_pickle=True)
        source_kind = str(npz["source_kind"]) if "source_kind" in npz.files else "shapefile"
        if source_kind == "raster":
            raster = str(npz["source_data"])
            bbox = bbox or str(npz["source_bbox"])
            band = int(npz["source_band"]) if "source_band" in npz.files else band
            if not nodata and "source_nodata" in npz.files:
                nodata = str(npz["source_nodata"]) or None
        else:
            data = str(npz["source_data"])
            field = field or str(npz["source_field"])
        if year is None:
            year = int(npz["source_year"])
        if task is None and "task" in npz.files:
            task = str(npz["task"])
        print(
            f"Using cached source: kind={source_kind}, "
            f"data={data or raster}, field={field}, year={year}"
        )

    if year is None:
        year = 2024

    if test_data:
        if source_kind == "raster":
            raise typer.BadParameter("--test-data is not supported with --raster.")
        if not data or not field:
            raise typer.BadParameter(
                "--test-data requires --data and --field for the training side."
            )

        try:
            from geotessera import GeoTessera
        except ImportError as exc:
            raise typer.BadParameter(
                'Loading from --data requires the geotessera extra: pip install -e ".[geotessera]"'
            ) from exc

        train_gdf = gpd.read_file(data)
        if len(train_gdf) == 0:
            raise typer.BadParameter(f"--data {data} contains no polygons.")
        if task is None:
            task = detect_field_type(train_gdf, field)
        if task == "regression":
            raise typer.BadParameter("learning-curve does not currently support regression")

        gt = GeoTessera()
        print(f"Loading training data from {data}...")
        vectors, labels, class_names, _ = load_embeddings_for_shapefile(
            train_gdf, field=field, year=year, gt_instance=gt
        )
        print(f"Loading independent test data from {test_data}...")
        test_gdf = gpd.read_file(test_data)
        if len(test_gdf) == 0:
            raise typer.BadParameter(f"--test-data {test_data} contains no polygons.")
        test_vectors, test_labels, test_class_names, _ = load_embeddings_for_shapefile(
            test_gdf, field=test_field or field, year=year, gt_instance=gt
        )
        if task == "classification" and list(test_class_names) != list(class_names):
            raise typer.BadParameter(
                f"Class mismatch between train ({class_names}) and test ({test_class_names}) — "
                "both must contain the same classes."
            )

    elif spatial_holdout and source_kind == "raster":
        if not raster or not bbox:
            raise typer.BadParameter(
                "--spatial-holdout with --raster requires --bbox (the training/overall region)."
            )
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
                    f"--bbox {bbox} and --test-bbox {test_bbox} overlap — they must be "
                    "disjoint, since --bbox defines the training region and shared "
                    "pixels would leak between train and test."
                )
        else:
            # No --test-bbox given: auto-split --bbox in half by longitude,
            # matching the shapefile east/west default.
            mid = (minx + maxx) / 2
            train_region = (minx, miny, mid, maxy)
            test_region = (mid, miny, maxx, maxy)

        print(f"Raster spatial split: train {train_region}, test (held-out) {test_region}")

        try:
            from geotessera import GeoTessera
        except ImportError as exc:
            raise typer.BadParameter(
                'Spatial hold-out requires the geotessera extra: pip install -e ".[geotessera]"'
            ) from exc

        nodata_values = [float(v) for v in nodata.split(",")] if nodata else None
        gt = GeoTessera()
        print("Loading training data...")
        vectors, labels, class_names, _, task = load_embeddings_for_raster(
            raster,
            bbox=train_region,
            year=year,
            gt_instance=gt,
            band=band,
            task=task,
            nodata_values=nodata_values,
        )
        if task == "regression":
            raise typer.BadParameter("learning-curve does not currently support regression")
        print("Loading held-out test data...")
        test_vectors, test_labels, test_class_names, _, _ = load_embeddings_for_raster(
            raster,
            bbox=test_region,
            year=year,
            gt_instance=gt,
            band=band,
            task=task,
            nodata_values=nodata_values,
        )
        if list(test_class_names) != list(class_names):
            raise typer.BadParameter(
                f"Class mismatch between train ({class_names}) and test ({test_class_names}) "
                "regions — try a different bbox or explicit --test-bbox."
            )

    elif spatial_holdout:
        if not data or not field:
            raise typer.BadParameter("--spatial-holdout requires --data and --field.")

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

        try:
            from geotessera import GeoTessera
        except ImportError as exc:
            raise typer.BadParameter(
                'Spatial hold-out requires the geotessera extra: pip install -e ".[geotessera]"'
            ) from exc

        gt = GeoTessera()
        print("Loading training data...")
        vectors, labels, class_names, _ = load_embeddings_for_shapefile(
            train_gdf, field=field, year=year, gt_instance=gt
        )
        print("Loading held-out test data...")
        test_vectors, test_labels, test_class_names, _ = load_embeddings_for_shapefile(
            test_gdf, field=field, year=year, gt_instance=gt
        )
        if task == "classification" and list(test_class_names) != list(class_names):
            raise typer.BadParameter(
                f"Class mismatch between train ({class_names}) and test ({test_class_names}) "
                "regions — try a different split or check label coverage on both sides."
            )

    else:
        vectors, labels, class_names, detected_task = _get_vectors_and_labels(
            data, field, year, vectors_path
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
