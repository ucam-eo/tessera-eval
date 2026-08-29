"""The compute server's zarr fast path uses geotessera's GeoTesseraZarr
directly: a handle opened once per process with the outcome cached (failure
included), and a coverage probe that trusts geotessera's own probe statuses.
"""

import numpy as np
import pytest

import tessera_eval.server as srv


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(srv, "_zarr_instance", None)


class _FakeStore:
    def __init__(self, years=(2024, 2025), status="valid"):
        self.years = list(years)
        self.url = "fake://store"
        self._status = status
        self.probed = []

    def probe(self, lon, lat, year):
        self.probed.append((lon, lat, year))
        if self._status == "valid":
            return np.ones(8, dtype=np.float32), "valid"
        return None, self._status


def test_get_zarr_returns_instance_and_caches_it(monkeypatch):
    calls = []

    def _make(*args, **kwargs):
        calls.append(kwargs)
        return _FakeStore()

    monkeypatch.setattr("geotessera.store.GeoTesseraZarr", _make)
    first = srv._get_zarr()
    second = srv._get_zarr()
    assert first is second is not None
    assert len(calls) == 1
    assert calls[0].get("cache_max_size", 0) > 0, (
        "the on-disk chunk cache must be bounded for a long-lived server"
    )


def test_get_zarr_returns_none_when_store_cannot_open(monkeypatch):
    calls = []

    def _make(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("no network")

    monkeypatch.setattr("geotessera.store.GeoTesseraZarr", _make)
    assert srv._get_zarr() is None
    assert srv._get_zarr() is None
    assert len(calls) == 1, "a failed open must be cached, not retried per call"


def test_get_zarr_returns_none_for_a_store_with_no_years(monkeypatch):
    monkeypatch.setattr("geotessera.store.GeoTesseraZarr", lambda **kw: _FakeStore(years=()))
    assert srv._get_zarr() is None


def test_probe_coverage_true_only_for_valid_embeddings():
    assert srv._probe_zarr_coverage(_FakeStore(status="valid"), (0.0, 50.0, 0.1, 50.1), 2024)
    assert not srv._probe_zarr_coverage(_FakeStore(status="water"), (0.0, 50.0, 0.1, 50.1), 2024)
    assert not srv._probe_zarr_coverage(_FakeStore(status="nodata"), (0.0, 50.0, 0.1, 50.1), 2024)


def test_probe_coverage_false_for_a_year_the_store_lacks():
    store = _FakeStore(years=(2024,))
    assert not srv._probe_zarr_coverage(store, (0.0, 50.0, 0.1, 50.1), 2018)
    assert store.probed == []


def test_probe_coverage_probes_the_centre_of_bounds():
    store = _FakeStore()
    srv._probe_zarr_coverage(store, (0.0, 50.0, 0.2, 50.2), 2024)
    assert store.probed == [(0.1, 50.1, 2024)]


def test_probe_coverage_false_when_probe_raises():
    class _Broken(_FakeStore):
        def probe(self, lon, lat, year):
            raise RuntimeError("store went away")

    assert not srv._probe_zarr_coverage(_Broken(), (0.0, 50.0, 0.1, 50.1), 2024)
