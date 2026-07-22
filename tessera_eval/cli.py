"""
Tessera-eval command-line interface
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import typer

from tessera_eval import (
    detect_field_type,
    load_embeddings_for_shapefile,
    run_kfold_cv,
    run_learning_curve,
)

# Default cache file for vectors/labels, in the current working directory —
# same convention GeoTessera itself uses for its own tile mirror.
DEFAULT_VECTORS_PATH = Path("vectors.npz")

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
    for event in run_kfold_cv(vectors, labels, model_list, k=k, task=task, seed=seed):
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
                worst = np.argsort(recall[i])[::-1]
                second = worst[1] if len(worst) > 1 else worst[0]
                print(
                    f"  {cname:>20}: recall {recall[i, i]:.2f} "
                    f"(most confused with {class_names[second]})"
                )


@app.command()
def learning_curve(
    data: str = typer.Option(None, help="Path to a shapefile/GeoJSON (skip if using --vectors)"),
    field: str = typer.Option(None, help="Class/target label column (skip if using --vectors)"),
    year: int = typer.Option(
        None,
        help="Tessera embedding year (default: 2024, or the cached year with --spatial-holdout)",
    ),
    vectors_path: str = typer.Option(
        None, "--vectors", help=f"Path to a cached .npz (default: {DEFAULT_VECTORS_PATH})"
    ),
    spatial_holdout: bool = typer.Option(
        False, help="Split --data by an east/west bounding-box midpoint for a spatial hold-out"
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
    """Investigate how model performance changes as training-label percentage increases.

    With spatial_holdout, trains on the west half of data and tests on the east
    half - a more honest accuracy estimate than a random pixel split, since
    neighbouring pixels are highly spatially correlated.

    Args:
        data: path to a shapefile/GeoJSON (skip if using vectors_path)
        field: class/target label column in data (skip if using vectors_path)
        year: Tessera embedding year; falls back to the cached year with
            spatial_holdout, or 2024 otherwise
        vectors_path: path to a cached .npz from `load`
        spatial_holdout: split data by an east/west bounding-box midpoint
            instead of using a cached/random split
        models: comma-separated model/regressor names passed to run_learning_curve
        training_pcts: comma-separated training-set percentages to evaluate at
        repeats: repeats per training percentage
        task: "classification" or "regression"; auto-detected if not given
    """
    model_list = models.split(",")
    pcts = [int(p) for p in training_pcts.split(",")]

    test_vectors, test_labels = None, None

    # Fill in --data/--field/--year from the cached .npz's saved source info,
    # but only for whichever of them the user didn't explicitly pass.
    if spatial_holdout and not data:
        if not DEFAULT_VECTORS_PATH.exists():
            raise typer.BadParameter("No cached vectors found — run `load` first, or pass --data.")
        npz = np.load(DEFAULT_VECTORS_PATH, allow_pickle=True)
        data = str(npz["source_data"])
        field = field or str(npz["source_field"])
        if year is None:
            year = int(npz["source_year"])
        if task is None and "task" in npz.files:
            task = str(npz["task"])
        print(f"Using cached source: data={data}, field={field}, year={year}")

    # Fallback default if nothing above set a year (e.g. --data passed
    # directly alongside --spatial-holdout, with no --year and no cache).
    if year is None:
        year = 2024

    if spatial_holdout:
        if not data or not field:
            raise typer.BadParameter("--spatial-holdout requires --data and --field.")

        gdf = gpd.read_file(data)
        if task is None:
            task = detect_field_type(gdf, field)

        if task == "regression":
            raise typer.BadParameter("learning-curve does not currently support regression")

        minx, miny, maxx, maxy = gdf.total_bounds
        mid = (minx + maxx) / 2
        west = gdf[gdf.centroid.x < mid]
        east = gdf[gdf.centroid.x >= mid]
        print(f"Spatial split: {len(west)} polygons west, {len(east)} polygons east of {mid:.4f}")

        try:
            from geotessera import GeoTessera
        except ImportError as exc:
            raise typer.BadParameter(
                'Spatial hold-out requires the geotessera extra: pip install -e ".[geotessera]"'
            ) from exc

        gt = GeoTessera()
        print("Loading training half (west)...")
        vectors, labels, class_names, _ = load_embeddings_for_shapefile(
            west, field=field, year=year, gt_instance=gt
        )
        print("Loading held-out half (east)...")
        test_vectors, test_labels, test_class_names, _ = load_embeddings_for_shapefile(
            east, field=field, year=year, gt_instance=gt
        )
        if task == "classification" and list(test_class_names) != list(class_names):
            raise typer.BadParameter(
                f"Class mismatch between west ({class_names}) and east ({test_class_names}) halves — "
                "try a different split or check label coverage on both sides."
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
