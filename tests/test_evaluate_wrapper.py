"""evaluate() must honour its training_sizes argument.

run_learning_curve moved from absolute training sizes to percentages, but
evaluate() kept passing sizes straight through -- a size of 10000 was read
as 10000 percent, so every requested size above ~100 collapsed to the same
80% training split.  Results.summary() also still read a 'size' key that
progress events no longer carry, and raised KeyError.
"""

import numpy as np

from tessera_eval.evaluate import evaluate


def _separable_data(n=200, dim=8, seed=0):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, 2, size=n)
    centers = np.stack([np.zeros(dim), np.full(dim, 5.0)])
    vectors = (centers[labels] + rng.randn(n, dim) * 0.1).astype(np.float32)
    return vectors, labels.astype(np.int32)


def test_training_sizes_control_the_number_of_training_pixels():
    vectors, labels = _separable_data()
    results = evaluate(vectors, labels, classifiers=["nn"], training_sizes=[20, 100], repeats=1)

    counts = [ev["pixel_train_count"] for ev in results.progress]
    assert len(counts) == 2
    assert abs(counts[0] - 20) <= 2
    assert abs(counts[1] - 100) <= 2


def test_oversized_training_size_is_capped_not_misread_as_percentage():
    vectors, labels = _separable_data()
    results = evaluate(vectors, labels, classifiers=["nn"], training_sizes=[10**6], repeats=1)

    counts = [ev["pixel_train_count"] for ev in results.progress]
    assert len(counts) == 1
    assert counts[0] <= 0.8 * len(labels) + 2


def test_summary_renders_without_error():
    vectors, labels = _separable_data()
    results = evaluate(vectors, labels, classifiers=["nn"], training_sizes=[20, 100], repeats=1)

    text = results.summary()
    assert "nn" in text
    assert len(text.splitlines()) >= 4
