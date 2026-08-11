"""Tests for evaluate._fit_with_heartbeat and its use in run_learning_curve.

Regression coverage: a single classifier .fit() call used to block
run_learning_curve's generator -- and therefore the whole SSE response --
with zero output for as long as training took. Confirmed live: a
spatial_mlp fit running 20+ minutes left the stream completely silent for
its whole duration, and the connection (an SSH tunnel, in the reported
case) was dropped as idle partway through, surfacing as a "network error"
moments before the fit would have finished successfully.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from tessera_eval.evaluate import _fit_with_heartbeat, run_learning_curve


def _drain(gen):
    """Run a generator to completion, collecting yields and the return value
    (PEP 380: a StopIteration raised from a generator carries its `return`
    value in .value)."""
    yielded = []
    try:
        while True:
            yielded.append(next(gen))
    except StopIteration as stop:
        return yielded, stop.value


def test_fast_fit_yields_no_heartbeats_and_returns_result():
    yielded, result = _drain(_fit_with_heartbeat(lambda: 42, interval=1.0))
    assert yielded == []
    assert result == 42


def test_slow_fit_yields_heartbeats_and_still_returns_result():
    def slow():
        time.sleep(0.22)
        return "done"

    yielded, result = _drain(_fit_with_heartbeat(slow, interval=0.05))
    assert result == "done"
    assert len(yielded) >= 2  # ~0.22s / 0.05s interval, some slack for scheduling
    assert all(ev == {"type": "heartbeat"} for ev in yielded)


def test_fit_exception_is_reraised_after_heartbeats():
    def slow_then_raise():
        time.sleep(0.11)
        raise ValueError("boom")

    gen = _fit_with_heartbeat(slow_then_raise, interval=0.05)
    yielded = []
    with pytest.raises(ValueError, match="boom"):
        while True:
            yielded.append(next(gen))
    assert len(yielded) >= 1  # got at least one heartbeat before the raise


def test_fast_fit_exception_is_reraised_with_no_heartbeats():
    def fails():
        raise RuntimeError("nope")

    gen = _fit_with_heartbeat(fails, interval=1.0)
    with pytest.raises(RuntimeError, match="nope"):
        next(gen)


# ── Integration: run_learning_curve still works with the wrapped fit ──


@pytest.fixture
def classification_data():
    rng = np.random.RandomState(0)
    dim = 8
    n_classes = 3
    vectors, labels = [], []
    for cls in range(n_classes):
        center = rng.randn(dim) * 3
        vectors.append(center + rng.randn(60, dim) * 0.3)
        labels.extend([cls] * 60)
    return np.vstack(vectors).astype(np.float32), np.array(labels)


def test_run_learning_curve_still_produces_progress_events(classification_data):
    """Wiring _fit_with_heartbeat into the classifier-fit call site must not
    change run_learning_curve's ordinary (fast-fit) behaviour or output."""
    vectors, labels = classification_data
    events = list(
        run_learning_curve(
            vectors,
            labels,
            ["nn"],
            training_pcts=[50, 80],
            repeats=1,
        )
    )
    progress_events = [e for e in events if e["type"] == "progress"]
    assert len(progress_events) == 2
    assert all("nn" in e["classifiers"] for e in progress_events)
    assert all(e["classifiers"]["nn"]["mean_f1"] is not None for e in progress_events)


def test_run_learning_curve_emits_heartbeats_for_a_slow_classifier(
    classification_data, monkeypatch
):
    """End-to-end: a classifier whose .fit() is slow enough to cross the
    heartbeat interval must actually produce heartbeat events in
    run_learning_curve's own output, not just in the isolated helper."""
    from sklearn.neighbors import KNeighborsClassifier

    # tessera_eval/__init__.py re-exports a function *also* named `evaluate`
    # from this submodule, which shadows `tessera_eval.evaluate` (the
    # submodule) for `import tessera_eval.evaluate as x` -- go via
    # sys.modules directly to reliably get the real module.
    evaluate_mod = sys.modules["tessera_eval.evaluate"]
    monkeypatch.setattr(evaluate_mod, "_HEARTBEAT_INTERVAL_S", 0.05)

    real_fit = KNeighborsClassifier.fit

    def slow_fit(self, X, y):
        time.sleep(0.16)
        return real_fit(self, X, y)

    monkeypatch.setattr(KNeighborsClassifier, "fit", slow_fit)

    vectors, labels = classification_data
    events = list(run_learning_curve(vectors, labels, ["nn"], training_pcts=[80], repeats=1))
    heartbeats = [e for e in events if e["type"] == "heartbeat"]
    progress_events = [e for e in events if e["type"] == "progress"]
    assert len(heartbeats) >= 1
    assert len(progress_events) == 1  # the fit still completed and produced a real result
