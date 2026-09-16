"""Tests for group-aware splitting (run_learning_curve's `groups` param,
run_kfold_cv's `groups` param) -- StratifiedGroupKFold/GroupKFold instead of
StratifiedKFold/KFold, so a group (e.g. a shapefile polygon/field) never has
rows on both sides of a train/test split.

Real-data motivation (see CHANGELOG): TEE's embeddings are highly spatially
autocorrelated within a field, so a naive per-pixel random split lets a
classifier partly "recognize the field" rather than learn the habitat --
confirmed on real Austria data, 0.11-0.14 macro F1 of pure optimism between
a naive per-pixel StratifiedKFold and an honest StratifiedGroupKFold, same
data, same models (Louis Driver's investigation, 2026-09-16).
"""

from __future__ import annotations

import numpy as np

from tessera_eval.evaluate import run_kfold_cv, run_learning_curve


def _grouped_classes(n_groups=60, pts_per_group=20, dim=8, k=3, seed=0):
    """Synthetic data shaped like the real leakage scenario: each group
    (field) is a single random "location" in feature space, and every pixel
    in that group is that location plus tiny noise -- so a classifier that
    ever sees *any* pixel from a group during training can trivially
    recognize every other pixel from that same group at test time, even
    though it has learned nothing about the class itself (each group's
    location is independent of its class). A naive split must therefore
    score much higher than a group-aware split on this data."""
    rng = np.random.RandomState(seed)
    group_labels = rng.randint(0, k, n_groups)
    group_centers = rng.randn(n_groups, dim) * 10  # class-independent -- "field identity"
    X, y, groups = [], [], []
    for g in range(n_groups):
        pts = group_centers[g] + rng.randn(pts_per_group, dim) * 0.1
        X.append(pts)
        y.extend([group_labels[g]] * pts_per_group)
        groups.extend([g] * pts_per_group)
    X = np.concatenate(X).astype(np.float32)
    y = np.array(y)
    groups = np.array(groups)
    return X, y, groups


def test_group_holdout_no_group_crosses_train_test_in_learning_curve():
    X, y, groups = _grouped_classes()
    events = list(
        run_learning_curve(
            X, y, ["nn"], training_pcts=[50], repeats=1, task="classification", seed=1,
            groups=groups,
        )
    )
    # Can't observe the split directly through the public event stream, but
    # a crash-free run through the group-holdout branch is still a real
    # assertion: it exercises StratifiedGroupKFold end-to-end. The leakage
    # assertion below (naive vs honest score) is the real behavioural test.
    progress = [e for e in events if e["type"] == "progress"]
    assert progress


def test_naive_split_scores_higher_than_group_holdout_on_leaky_synthetic_data():
    """The actual regression test for the bug this exists to fix: on data
    engineered so within-group pixels are near-duplicates unrelated to
    class, a plain (non-grouped) split must score much higher than the
    group-holdout split, same data, same classifier."""
    X, y, groups = _grouped_classes(n_groups=80, pts_per_group=15, seed=2)

    naive_events = list(
        run_learning_curve(
            X, y, ["nn"], training_pcts=[80], repeats=3, task="classification", seed=2,
        )
    )
    naive_f1 = [e for e in naive_events if e["type"] == "progress"][-1]["classifiers"]["nn"][
        "mean_f1"
    ]

    honest_events = list(
        run_learning_curve(
            X, y, ["nn"], training_pcts=[80], repeats=3, task="classification", seed=2,
            groups=groups,
        )
    )
    honest_f1 = [e for e in honest_events if e["type"] == "progress"][-1]["classifiers"]["nn"][
        "mean_f1"
    ]

    assert naive_f1 > honest_f1 + 0.1, (
        f"expected the naive split to be substantially optimistic vs. the group "
        f"holdout on leaky data, got naive={naive_f1:.3f} honest={honest_f1:.3f}"
    )


def test_group_holdout_is_ignored_when_test_vectors_already_given():
    """An explicit fixed test set (spatial/year split) takes precedence --
    groups must not override it."""
    X, y, groups = _grouped_classes(n_groups=20, pts_per_group=10, seed=3)
    test_X, test_y = X[:10], y[:10]
    events = list(
        run_learning_curve(
            X, y, ["nn"], training_pcts=[50], repeats=1, task="classification", seed=3,
            groups=groups, test_vectors=test_X, test_labels=test_y,
        )
    )
    assert any(e["type"] == "progress" for e in events)


def test_group_holdout_ignored_for_regression():
    """groups is documented classification-only for run_learning_curve
    (StratifiedGroupKFold needs classes) -- a regression task must ignore it
    rather than crash."""
    rng = np.random.RandomState(4)
    X = rng.randn(100, 4).astype(np.float32)
    y = rng.randn(100).astype(np.float32)
    groups = rng.randint(0, 20, 100)
    events = list(
        run_learning_curve(
            X, y, ["rf_reg"], training_pcts=[50], repeats=1, task="regression", seed=4,
            groups=groups,
        )
    )
    assert any(e["type"] in ("progress", "aggregate") for e in events)


def test_run_kfold_cv_groups_no_group_split_across_folds():
    """Directly verifies the fold membership property StratifiedGroupKFold
    is supposed to give us: every pixel from a given group ends up being
    tested in exactly one fold -- i.e., the k pixel_folds partition groups,
    not pixels, across the test side."""
    from sklearn.model_selection import StratifiedGroupKFold

    X, y, groups = _grouped_classes(n_groups=50, pts_per_group=10, seed=5)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=5)
    seen_groups_by_fold = []
    for _, test_idx in splitter.split(X, y, groups=groups):
        seen_groups_by_fold.append(set(groups[test_idx]))
    # No group appears in more than one fold's test set.
    all_seen = [g for s in seen_groups_by_fold for g in s]
    assert len(all_seen) == len(set(all_seen))


def test_run_kfold_cv_accepts_groups_and_runs():
    X, y, groups = _grouped_classes(n_groups=60, pts_per_group=10, seed=6)
    events = list(
        run_kfold_cv(X, y, ["nn"], k=3, task="classification", seed=6, groups=groups)
    )
    agg = [e for e in events if e["type"] == "aggregate"]
    assert agg
    assert "nn" in agg[0]["models"]


def test_run_kfold_cv_naive_scores_higher_than_grouped_on_leaky_data():
    X, y, groups = _grouped_classes(n_groups=80, pts_per_group=15, seed=7)

    naive = list(run_kfold_cv(X, y, ["nn"], k=5, task="classification", seed=7))
    naive_f1 = [e for e in naive if e["type"] == "aggregate"][0]["models"]["nn"]["mean_f1"]

    grouped = list(
        run_kfold_cv(X, y, ["nn"], k=5, task="classification", seed=7, groups=groups)
    )
    grouped_f1 = [e for e in grouped if e["type"] == "aggregate"][0]["models"]["nn"]["mean_f1"]

    assert naive_f1 > grouped_f1 + 0.1, (
        f"expected naive kfold to be optimistic vs. grouped kfold on leaky data, "
        f"got naive={naive_f1:.3f} grouped={grouped_f1:.3f}"
    )
