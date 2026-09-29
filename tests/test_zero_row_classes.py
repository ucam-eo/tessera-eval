"""Unit tests for _zero_row_classes -- identifies classes with zero true
test examples in a confusion matrix (a zero row), distinct from a class the
classifier merely never predicts (a zero column with a non-zero row, which
this deliberately does NOT flag -- see the function's own docstring).

Also covers _zero_row_advice -- the per-classifier-type advice text that
goes with a zero-row warning. Its spatial-vs-pixel split exists because
"increase Max patches" turned out to be the wrong advice for spatial
models (see its own docstring for the live investigation that found
this): they draw a fixed 5 patches per tile regardless of Max patches, so
a different seed is what actually helps, not a bigger patch budget.
"""

from __future__ import annotations

from tessera_eval.server import _zero_row_advice, _zero_row_classes


def test_no_zero_rows_returns_empty():
    matrix = [
        [5, 1, 0],
        [0, 4, 1],
        [1, 0, 6],
    ]
    assert _zero_row_classes(matrix, ["a", "b", "c"]) == []


def test_a_single_zero_row_is_flagged():
    matrix = [
        [5, 1, 0],
        [0, 0, 0],  # class "b" never appears as a true label
        [1, 0, 6],
    ]
    assert _zero_row_classes(matrix, ["a", "b", "c"]) == ["b"]


def test_multiple_zero_rows_are_all_flagged_in_order():
    matrix = [
        [0, 0, 0],
        [1, 4, 1],
        [0, 0, 0],
    ]
    assert _zero_row_classes(matrix, ["a", "b", "c"]) == ["a", "c"]


def test_zero_column_with_nonzero_row_is_not_flagged():
    """A class with real test examples that the classifier simply never
    predicts (column all zero, row not) is a genuine "can't recognise this
    class" finding -- a different thing from missing test coverage, and
    must not be explained away the same way."""
    matrix = [
        [5, 0, 2],  # class "a" has real test examples (row sum 7)
        [3, 0, 1],  # class "b" likewise (row sum 4) -- but column "b" is
        [0, 0, 6],  # all zero: never predicted as "b" by anyone
    ]
    assert _zero_row_classes(matrix, ["a", "b", "c"]) == []


def test_diagonal_zero_alone_is_not_enough():
    """A class scored on but never correctly classified (0 on the
    diagonal) still has real test examples elsewhere in its row -- not the
    same as zero test examples."""
    matrix = [
        [0, 3, 2],  # class "a": 0 correct, but 5 real test examples
        [1, 4, 0],
        [0, 1, 5],
    ]
    assert _zero_row_classes(matrix, ["a", "b", "c"]) == []


def test_numpy_array_input_works_the_same_as_nested_lists():
    import numpy as np

    matrix = np.array([[0, 0], [2, 3]])
    assert _zero_row_classes(matrix, ["rare", "common"]) == ["rare"]


def test_class_names_shorter_than_matrix_is_handled_defensively():
    """Shouldn't happen in practice, but a mismatched class_names list
    must not IndexError -- rows within range are still resolved and
    flagged normally; only rows beyond class_names' length are skipped
    (nothing to name them with)."""
    matrix = [
        [0, 0, 0],  # index 0: resolvable, zero row -> flagged
        [1, 2, 0],  # index 1: out of range, but not zero anyway
        [0, 0, 0],  # index 2: out of range -- skipped, can't be named
    ]
    assert _zero_row_classes(matrix, ["only_one"]) == ["only_one"]


def test_empty_matrix_returns_empty():
    assert _zero_row_classes([], []) == []


def test_zero_row_advice_for_spatial_mlp_does_not_recommend_max_patches():
    advice = _zero_row_advice("spatial_mlp")
    assert "rarely helps" in advice
    assert "different seed" in advice


def test_zero_row_advice_for_spatial_mlp_5x5_does_not_recommend_max_patches():
    advice = _zero_row_advice("spatial_mlp_5x5")
    assert "rarely helps" in advice


def test_zero_row_advice_for_spatial_mlp_variant_suffix_still_recognised():
    """A hyperparameter-sweep variant name (spatial_mlp_v2) must still be
    recognised as a spatial model -- variant suffixes are stripped the
    same way _base_name does it elsewhere in server.py."""
    advice = _zero_row_advice("spatial_mlp_v2")
    assert "rarely helps" in advice
    assert "different seed" in advice


def test_zero_row_advice_for_pixel_classifier_suggests_max_pixel_samples():
    for name in ("rf", "mlp", "deep_mlp", "xgboost", "nn"):
        advice = _zero_row_advice(name)
        assert "Max pixel samples" in advice, f"unexpected advice for {name}: {advice}"
