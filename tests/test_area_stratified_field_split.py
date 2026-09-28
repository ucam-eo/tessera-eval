"""Unit tests for _area_stratified_field_split -- the per-class, per-field
area-quota split reproducing Frank Feng's (TESSERA paper co-author)
Austrian-crop methodology: "30% of each class's area -> train; rest 1/7
val, 6/7 test". See tests/test_area_stratified_split_server.py for the
request-flag wiring (server.py's area_stratified_split) built on top of
this.
"""

from __future__ import annotations

import numpy as np

from tessera_eval.server import _area_stratified_field_split


def _totals(field_ids, class_ids, areas, bucket):
    """Sum area per (class, bucket) for easy assertions."""
    out = {}
    for fid, cls, area in zip(field_ids, class_ids, areas):
        out.setdefault(cls, {"train": 0.0, "val": 0.0, "test": 0.0})[bucket[fid]] += area
    return out


def test_train_fraction_is_close_to_requested_per_class():
    rng = np.random.RandomState(0)
    n_per_class = 200
    field_ids = np.arange(2 * n_per_class)
    class_ids = np.array([0] * n_per_class + [1] * n_per_class)
    # Lots of small, similar-sized fields per class so the greedy
    # cumulative-area walk can land close to the requested fraction --
    # a single huge field would make an exact 30% impossible.
    areas = rng.uniform(0.8, 1.2, size=len(field_ids))

    bucket = _area_stratified_field_split(field_ids, class_ids, areas, seed=1)

    totals = _totals(field_ids, class_ids, areas, bucket)
    for cls in (0, 1):
        t = totals[cls]
        grand_total = t["train"] + t["val"] + t["test"]
        assert abs(t["train"] / grand_total - 0.30) < 0.02
        assert abs(t["val"] / grand_total - 0.10) < 0.02
        assert abs(t["test"] / grand_total - 0.60) < 0.02


def test_every_field_is_assigned_exactly_one_bucket():
    rng = np.random.RandomState(2)
    field_ids = np.arange(150)
    class_ids = rng.randint(0, 5, size=150)
    areas = rng.uniform(1, 10, size=150)

    bucket = _area_stratified_field_split(field_ids, class_ids, areas, seed=3)

    assert set(bucket.keys()) == set(field_ids.tolist())
    assert set(bucket.values()) <= {"train", "val", "test"}


def test_rare_class_still_gets_a_nonempty_train_bucket():
    """A class with only a handful of fields must not be starved just
    because it's tiny -- this is the whole point of the feature (a fixed
    train *share of the class's own area*, not competing for a shared
    global budget the way TEE's `sampling` strategies do)."""
    field_ids = np.arange(6)
    # 5 fields of a common class, 1 field of a rare class.
    class_ids = np.array([0, 0, 0, 0, 0, 1])
    areas = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 3.0])

    bucket = _area_stratified_field_split(field_ids, class_ids, areas, seed=0)

    # The rare class's single field must land somewhere real -- with only
    # one field, the 30% cumulative threshold still rounds up to include
    # it (searchsorted + 1), so it goes to train rather than being dropped
    # or shut out because the class is small.
    assert bucket[5] == "train"


def test_class_with_one_field_puts_it_in_train():
    field_ids = np.array([0])
    class_ids = np.array([7])
    areas = np.array([42.0])

    bucket = _area_stratified_field_split(field_ids, class_ids, areas, seed=0)

    assert bucket[0] == "train"


def test_deterministic_given_same_seed():
    rng = np.random.RandomState(5)
    field_ids = np.arange(80)
    class_ids = rng.randint(0, 4, size=80)
    areas = rng.uniform(1, 5, size=80)

    b1 = _area_stratified_field_split(field_ids, class_ids, areas, seed=99)
    b2 = _area_stratified_field_split(field_ids, class_ids, areas, seed=99)

    assert b1 == b2


def test_different_seeds_can_pick_different_fields():
    rng = np.random.RandomState(6)
    field_ids = np.arange(80)
    class_ids = rng.randint(0, 4, size=80)
    areas = rng.uniform(1, 5, size=80)

    b1 = _area_stratified_field_split(field_ids, class_ids, areas, seed=1)
    b2 = _area_stratified_field_split(field_ids, class_ids, areas, seed=2)

    # Same aggregate fractions (both honour the 30/10/60 rule), but not
    # necessarily the exact same fields in each bucket.
    assert b1 != b2


def test_custom_train_and_val_fractions():
    rng = np.random.RandomState(7)
    field_ids = np.arange(300)
    class_ids = np.zeros(300, dtype=int)
    areas = rng.uniform(0.9, 1.1, size=300)

    bucket = _area_stratified_field_split(
        field_ids, class_ids, areas, train_frac=0.5, val_frac=0.0, seed=0
    )

    totals = _totals(field_ids, class_ids, areas, bucket)[0]
    grand_total = sum(totals.values())
    assert abs(totals["train"] / grand_total - 0.5) < 0.02
    assert totals["val"] == 0.0
