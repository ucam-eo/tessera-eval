"""Classifier and regressor factory, plus spatial feature extraction."""

import re

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Models that train on precomputed neighbourhood features rather than plain
# per-pixel vectors. The single source of truth for every "is this a spatial
# model" decision (feature extraction, fixed-test-set skips, map support).
SPATIAL_MODELS = ("spatial_mlp", "spatial_mlp_5x5")


def _make_mlp(estimator_cls, hidden, max_iter, seed):
    """Build an MLP (classifier or regressor) preceded by feature scaling.

    sklearn's MLPClassifier/MLPRegressor use the Adam solver with a fixed
    default learning_rate_init=0.001 and no scaling of their own. Tessera
    embeddings aren't guaranteed to already be zero-mean/unit-variance, and
    an unscaled MLP under those defaults converges to a visibly worse
    optimum within a fixed max_iter budget -- diagnosed from a real macro-F1
    gap vs. the Tessera paper's own Austrian-crop numbers (Louis Driver,
    2026-09-15): TEE's F1 plateaued ~20-26 points below the paper's at every
    training percentage, widening rather than narrowing with more data (the
    signature of an optimization ceiling, not a data-availability one), and
    a *larger* MLP sometimes scored *worse* than a smaller one at the same
    percentage -- both point at undertraining, not model capacity or task
    difficulty. StandardScaler is fit fresh inside the Pipeline on each
    training call, so there's no leakage between train/test folds; every
    other classifier/regressor here (RF, XGBoost, kNN's Euclidean metric
    aside) doesn't need it and is left as-is to keep this fix scoped to the
    actual suspect.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", estimator_cls(hidden_layer_sizes=hidden, max_iter=max_iter, random_state=seed)),
        ]
    )


def _strip_variant_suffix(name):
    """Strip hyperparameter variant suffix (e.g., 'mlp_v2' -> 'mlp').

    Variant names are created by the server when a classifier has multiple
    parameter sets. The base name is used for classifier lookup.
    """
    return re.sub(r"_v\d+$", "", name)


def available_classifiers():
    """Return list of available classifier names."""
    names = ["nn", "rf", "mlp", "spatial_mlp", "spatial_mlp_5x5"]
    try:
        import xgboost  # noqa: F401

        names.append("xgboost")
    except ImportError:
        pass
    try:
        import torch  # noqa: F401

        names.append("deep_mlp")
    except ImportError:
        pass
    return names


def make_classifier(name, params=None, seed=42):
    """Create a classifier instance by name with optional hyperparameters.

    Args:
        name: Classifier name — one of 'nn', 'rf', 'xgboost', 'mlp',
              'deep_mlp', 'spatial_mlp', 'spatial_mlp_5x5'.
              May include a variant suffix (e.g., 'mlp_v2') which is
              stripped before lookup.
        params: Optional dict of hyperparameters
        seed: Random seed for every estimator that takes one (RF, XGBoost,
            MLP weight init). Threaded from the CLI --seed / the web
            request so a whole run is reproducible from one number.

    Returns:
        scikit-learn compatible classifier (fit/predict interface)

    Raises:
        ValueError: If name is unknown
        ImportError: If xgboost is requested but not installed
    """
    base_name = _strip_variant_suffix(name)
    p = params or {}
    if base_name == "nn":
        # n_jobs=-1: the neighbour query is brute-force in 128-d (no tree
        # helps at that dimensionality), so a full-map predict is millions
        # of independent distance scans -- parallelising it across cores is
        # a large, free speedup. Louis Driver: "producing maps using kNN is
        # very slow".
        return KNeighborsClassifier(
            n_neighbors=int(p.get("n_neighbors", 5)),
            weights=p.get("weights", "uniform"),
            metric="euclidean",
            n_jobs=-1,
        )
    elif base_name == "rf":
        max_depth = p.get("max_depth")
        if max_depth is not None:
            max_depth = int(max_depth)
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=max_depth,
            n_jobs=-1,
            random_state=seed,
        )
    elif base_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=int(p.get("max_depth", 6)),
            learning_rate=float(p.get("learning_rate", 0.3)),
            n_jobs=-1,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        )
    elif base_name == "mlp":
        layers_str = p.get("hidden_layers", "64,32")
        if isinstance(layers_str, str):
            hidden = tuple(int(x) for x in layers_str.split(","))
        else:
            hidden = (64, 32)
        return _make_mlp(MLPClassifier, hidden, int(p.get("max_iter", 200)), seed)
    elif base_name == "deep_mlp":
        # PyTorch MLP matching the Tessera paper's own downstream-eval
        # architecture (BatchNorm + Dropout + AdamW + val-F1 checkpoint
        # selection) -- see deep_mlp.py's module docstring for the full
        # diagnosis. Optional (requires torch); raises a clear error via
        # _require_torch() if it's requested without torch installed,
        # rather than available_classifiers() silently hiding it from a
        # caller that already knows the name.
        from tessera_eval.deep_mlp import DeepMLPClassifier

        layers_str = p.get("hidden_layers", "512,256")
        if isinstance(layers_str, str):
            hidden = tuple(int(x) for x in layers_str.split(","))
        else:
            hidden = (512, 256)
        return DeepMLPClassifier(
            hidden=hidden,
            dropout=float(p.get("dropout", 0.3)),
            lr=float(p.get("lr", 1e-3)),
            weight_decay=float(p.get("weight_decay", 0.01)),
            batch_size=int(p.get("batch_size", 8192)),
            epochs=int(p.get("epochs", 150)),
            seed=seed,
        )
    elif base_name in SPATIAL_MODELS:
        default_layers = "256,128" if base_name == "spatial_mlp" else "512,256"
        default_iter = 300 if base_name == "spatial_mlp" else 400
        layers_str = p.get("hidden_layers", default_layers)
        if isinstance(layers_str, str):
            hidden = tuple(int(x) for x in layers_str.split(","))
        else:
            hidden = tuple(int(x) for x in default_layers.split(","))
        return _make_mlp(MLPClassifier, hidden, int(p.get("max_iter", default_iter)), seed)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def gather_spatial_features(vectors, coords, width, height, radius=1, subset_mask=None):
    """Build (2r+1)^2 neighbourhood features from (N, dim) vectors on a regular grid.

    For each pixel, concatenates its own embedding with those of its
    neighbours in a (2*radius+1) x (2*radius+1) window. Missing neighbours
    are zero-filled.

    Args:
        vectors: float32 array, shape (N, dim)
        coords: int32 array, shape (N, 2) — pixel (x, y) coordinates
        width: Grid width in pixels
        height: Grid height in pixels
        radius: Neighbourhood radius (1 = 3x3, 2 = 5x5)
        subset_mask: Optional bool array, shape (N,) — only compute for
                     these pixels (saves memory)

    Returns:
        float32 array, shape (M, window*window*dim) where M = sum(subset_mask) or N
    """
    dim = vectors.shape[1]
    window = 2 * radius + 1
    grid = np.full((height, width), -1, dtype=np.int32)
    grid[coords[:, 1], coords[:, 0]] = np.arange(len(coords))

    if subset_mask is not None:
        sub_coords = coords[subset_mask]
    else:
        sub_coords = coords

    offsets = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)]
    spatial = np.zeros((len(sub_coords), window * window * dim), dtype=np.float32)

    for i, (dr, dc) in enumerate(offsets):
        nr = sub_coords[:, 1] + dr
        nc = sub_coords[:, 0] + dc
        valid = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
        idx = np.where(valid, grid[np.clip(nr, 0, height - 1), np.clip(nc, 0, width - 1)], -1)
        has_neighbour = valid & (idx >= 0)
        spatial[has_neighbour, i * dim : (i + 1) * dim] = vectors[idx[has_neighbour]]

    return spatial


def gather_spatial_features_2d(tile_emb, radius=1, mask=None):
    """Extract spatial neighbourhood features from a contiguous 2D tile.

    For each pixel, concatenates the embeddings of all pixels in a
    (2*radius+1) x (2*radius+1) window centered on it. Edge pixels
    are zero-padded.

    Args:
        tile_emb: float32 array, shape (H, W, dim) — contiguous tile embeddings
        radius: neighbourhood radius (1 for 3x3, 2 for 5x5)
        mask: optional bool array, shape (H, W) — if provided, only extract
            features for True pixels (much faster for sparse labels)

    Returns:
        If mask is None: float32 array, shape (H, W, window*window*dim)
        If mask is provided: float32 array, shape (N, window*window*dim)
            where N = mask.sum()
    """
    H, W, dim = tile_emb.shape
    window = 2 * radius + 1
    padded = np.pad(tile_emb, ((radius, radius), (radius, radius), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window, window, dim))
    # windows shape: (H, W, 1, window, window, dim)
    if mask is not None:
        # Only materialize features for masked pixels (avoids H*W*window²*dim allocation)
        # windows is a view — indexing with mask only copies selected pixels
        masked = windows[mask]  # shape: (N, 1, window, window, dim)
        return masked.reshape(masked.shape[0], window * window * dim).astype(np.float32)
    return windows.reshape(H, W, window * window * dim).astype(np.float32)


# augment_spatial's 4x expansion is materialized eagerly (scikit-learn's
# .fit() needs the whole array up front, unlike PyTorch's lazily-batched
# DataLoader -- see unet.py's _AugmentedPatches for that version of this
# problem). A 5x5-window, 128-dim point is 12.8 KB; above this many input
# points the augmented output starts costing real memory (50_000 points ->
# 2.4 GB augmented; the default is chosen to stay well clear of a crash on
# a typical machine). This is every augment_spatial call's *only* size
# bound unless the caller does its own additional subsampling first (the
# learning curve samples a pct-of-data slice; run_kfold_cv applies
# max_training_samples) -- callers that don't (train_models(), which used
# to pass its full, uncapped cached point set straight through) rely on
# this default entirely. Confirmed live for the analogous U-Net path:
# "Unable to allocate 74.5 GiB for an array with shape (2384, 128, 256,
# 256)" (Moustafa Eweda) -- same eager-augmentation shape, different
# function. See CHANGELOG.
DEFAULT_AUGMENT_CAP = 50_000


def augment_spatial(X, y, window, dim, cap=DEFAULT_AUGMENT_CAP, seed=42):
    """4x data augmentation via horizontal/vertical flips of spatial patches.

    Subsamples X/y down to at most `cap` rows first (deterministic, seeded)
    when X is larger than that, so the returned arrays never exceed
    `min(len(X), cap) * 4` rows regardless of how large the caller's X is --
    every call site gets this protection without needing its own capping
    logic. Pass cap=None to disable it (the caller is asserting X is
    already sized appropriately, e.g. a deliberately small test fixture).

    Args:
        X: float32 array, shape (N, window*window*dim)
        y: array, shape (N,) -- int class labels or float regression targets
        window: Spatial window size (e.g., 3 or 5)
        dim: Embedding dimension (e.g., 128)
        cap: Maximum input rows before augmenting (default DEFAULT_AUGMENT_CAP,
            currently 50,000); None disables the cap.
        seed: Subsampling seed, used only when len(X) > cap.

    Returns:
        Tuple of (X_aug, y_aug) with 4x min(len(X), cap) samples
    """
    if cap is not None and len(X) > cap:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=cap, replace=False)
        X, y = X[idx], y[idx]

    n = len(X)
    patches = X.reshape(n, window, window, dim)
    augmented = [
        X,
        patches[:, :, ::-1, :].copy().reshape(n, -1),
        patches[:, ::-1, :, :].copy().reshape(n, -1),
        patches[:, ::-1, ::-1, :].copy().reshape(n, -1),
    ]
    return np.concatenate(augmented, axis=0), np.tile(y, 4)


def available_regressors():
    """Return list of available regressor names."""
    # spatial_mlp/spatial_mlp_5x5 keep their unsuffixed names here too --
    # see make_regressor's comment on why they don't get "_reg".
    names = ["nn_reg", "rf_reg", "mlp_reg", "spatial_mlp", "spatial_mlp_5x5"]
    try:
        import xgboost  # noqa: F401

        names.append("xgboost_reg")
    except ImportError:
        pass
    return names


def make_regressor(name, params=None, seed=42):
    """Create a regressor instance by name with optional hyperparameters.

    Args:
        name: Regressor name — one of 'nn_reg', 'rf_reg', 'mlp_reg', 'xgboost_reg'.
              May include a variant suffix (e.g., 'rf_reg_v2') which is
              stripped before lookup.
        params: Optional dict of hyperparameters
        seed: Random seed for every estimator that takes one (RF, XGBoost,
            MLP weight init).

    Returns:
        scikit-learn compatible regressor (fit/predict interface)

    Raises:
        ValueError: If name is unknown
        ImportError: If xgboost_reg is requested but not installed
    """
    base_name = _strip_variant_suffix(name)
    p = params or {}
    if base_name == "nn_reg":
        return KNeighborsRegressor(
            n_neighbors=int(p.get("n_neighbors", 5)),
            weights=p.get("weights", "uniform"),
            metric="euclidean",
            n_jobs=-1,  # brute-force 128-d query; parallelise it (see make_classifier)
        )
    elif base_name == "rf_reg":
        max_depth = p.get("max_depth")
        if max_depth is not None:
            max_depth = int(max_depth)
        return RandomForestRegressor(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=max_depth,
            n_jobs=-1,
            random_state=seed,
        )
    elif base_name == "xgboost_reg":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=int(p.get("max_depth", 6)),
            learning_rate=float(p.get("learning_rate", 0.3)),
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
    elif base_name == "mlp_reg":
        layers_str = p.get("hidden_layers", "64,32")
        if isinstance(layers_str, str):
            hidden = tuple(int(x) for x in layers_str.split(","))
        else:
            hidden = (64, 32)
        return _make_mlp(MLPRegressor, hidden, int(p.get("max_iter", 200)), seed)
    elif base_name in SPATIAL_MODELS:
        # Deliberately no "_reg" suffix, unlike every other regressor here --
        # "spatial" describes which precomputed feature array (3x3 or 5x5
        # neighbourhood-augmented embeddings) this trains on, which is
        # entirely a server.py/evaluate.py data-selection concern keyed on
        # this exact literal name (needs_spatial_3x3, the base_clf_name
        # branch in run_learning_curve, etc.) -- renaming it per task would
        # mean updating every one of those call sites too, for no benefit
        # (the caller already knows whether to build a classifier or
        # regressor; the name doesn't need to encode that here).
        default_layers = "256,128" if base_name == "spatial_mlp" else "512,256"
        default_iter = 300 if base_name == "spatial_mlp" else 400
        layers_str = p.get("hidden_layers", default_layers)
        if isinstance(layers_str, str):
            hidden = tuple(int(x) for x in layers_str.split(","))
        else:
            hidden = tuple(int(x) for x in default_layers.split(","))
        return _make_mlp(MLPRegressor, hidden, int(p.get("max_iter", default_iter)), seed)
    else:
        raise ValueError(f"Unknown regressor: {name}")
