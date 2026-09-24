"""Regression test: run_learning_curve's confusion matrix for a spatial
model (spatial_mlp/spatial_mlp_5x5) must include classes that are present
in spatial_labels but absent from the pixel labels array -- not silently
drop them.

Background: n_classes (used to size the confusion matrix) was computed from
the pixel labels array only. A class this run's random per-class pixel
sampling happened to miss entirely (a small/rare class, say) would shrink
n_classes below its true count -- and sklearn's confusion_matrix() silently
drops any sample whose true/predicted label isn't in the `labels=` list it's
given, rather than erroring. That drops whichever class has the *highest*
label-encoder index from every model's confusion matrix this run, not just
the pixel classifiers that genuinely lacked training data for it -- a
spatial model, whose points come from a separate patch-derived sampling
process, can have real data (and a real, correct prediction) for exactly
that class and still show a fully zero row.

Confirmed live (Moustafa Eweda): "Inland rock outcrop and scree" showed a
fully zero row (0% recall, including its own diagonal) in a Spatial MLP 3x3
learning-curve confusion matrix, despite the model correctly classifying
every pixel of it (100% recall) in a previous run, and the map itself
looking fine -- a reporting bug, not a real regression.

run_kfold_cv already had the matching fix (size the CM for the union of
pixel and spatial label domains); run_learning_curve never got it until now.
"""

from __future__ import annotations

import numpy as np

from tessera_eval.evaluate import run_learning_curve


def test_spatial_model_confusion_matrix_includes_a_class_missing_from_pixel_labels():
    rng = np.random.RandomState(0)
    dim = 4
    window = 3  # spatial_mlp's own window

    # Pixel data: only classes 0 and 1 -- class 2 is deliberately absent
    # entirely, mirroring a rare class this run's pixel sampling missed.
    vectors = np.concatenate(
        [
            rng.randn(30, dim).astype(np.float32) + np.array([5, 0, 0, 0]),
            rng.randn(30, dim).astype(np.float32) + np.array([-5, 0, 0, 0]),
        ]
    )
    labels = np.array([0] * 30 + [1] * 30)

    # Spatial data: all three classes, well-separated so a real classifier
    # can learn class 2 easily -- if the bug is present, class 2's row in
    # the confusion matrix is zero regardless of how well it's classified.
    n_per_class = 40
    spatial_labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    centers = {0: 5.0, 1: -5.0, 2: 50.0}  # class 2 far from the other two
    spatial_vectors = np.concatenate(
        [
            (rng.randn(n_per_class, window * window * dim).astype(np.float32) * 0.5 + centers[c])
            for c in (0, 1, 2)
        ]
    )

    events = list(
        run_learning_curve(
            vectors,
            labels,
            ["spatial_mlp"],
            training_pcts=[80],
            repeats=1,
            spatial_vectors=spatial_vectors,
            spatial_labels=spatial_labels,
            task="classification",
            seed=0,
        )
    )

    cm_events = [e for e in events if e["type"] == "confusion_matrices"]
    assert cm_events, "no confusion_matrices event produced"
    cm = np.array(cm_events[0]["confusion_matrices"]["spatial_mlp"])

    # The bug: class 2 (absent from pixel `labels`) gets silently dropped,
    # so the matrix would only be 2x2 (or a 3x3 matrix with row 2 always
    # zero, depending on exactly which class index truncation drops -- see
    # this test module's docstring). Either way, row 2 not being real,
    # nonzero data is the failure this guards against.
    assert cm.shape == (3, 3), f"confusion matrix should cover all 3 classes, got shape {cm.shape}"
    assert cm[2].sum() > 0, "class 2's row is entirely zero -- it was dropped from the matrix"
    # Well-separated synthetic data for class 2 -- expect it to be classified
    # correctly essentially every time, i.e. concentrated on the diagonal.
    assert cm[2, 2] / cm[2].sum() > 0.9, f"class 2 should be classified correctly, got row {cm[2]}"


def test_pixel_only_model_unaffected_by_the_wider_confusion_matrix():
    """A plain pixel classifier ('nn') has no spatial_labels-derived data
    for the missing class -- its own row for that class should just stay
    legitimately empty (no samples ever existed for it), not error or
    misbehave now that the matrix is sized for the union."""
    rng = np.random.RandomState(1)
    dim = 4
    vectors = np.concatenate(
        [
            rng.randn(30, dim).astype(np.float32) + np.array([5, 0, 0, 0]),
            rng.randn(30, dim).astype(np.float32) + np.array([-5, 0, 0, 0]),
        ]
    )
    labels = np.array([0] * 30 + [1] * 30)

    n_per_class = 40
    window = 3
    spatial_labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    centers = {0: 5.0, 1: -5.0, 2: 50.0}
    spatial_vectors = np.concatenate(
        [
            (rng.randn(n_per_class, window * window * dim).astype(np.float32) * 0.5 + centers[c])
            for c in (0, 1, 2)
        ]
    )

    events = list(
        run_learning_curve(
            vectors,
            labels,
            ["nn"],
            training_pcts=[80],
            repeats=1,
            spatial_vectors=spatial_vectors,
            spatial_labels=spatial_labels,
            task="classification",
            seed=1,
        )
    )
    cm_events = [e for e in events if e["type"] == "confusion_matrices"]
    assert cm_events
    cm = np.array(cm_events[0]["confusion_matrices"]["nn"])
    assert cm.shape == (3, 3)
    # 'nn' never saw class 2 at all (no pixel samples for it) -- its row is
    # legitimately all zero, which is correct, not the bug.
    assert cm[2].sum() == 0
