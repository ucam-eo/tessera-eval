"""Tests for tee-compute's reverse proxy (tessera_eval.server.proxy).

proxy() forwards every non-eval request to --hosted (the real TEE server).
Regression coverage for the fix: it used to call the top-level
requests.request(), which opens and discards a fresh requests.Session --
and therefore a fresh TCP+TLS handshake -- on every single call. A page
load is never one request (HTML, several JS modules, CSS, several API
calls), so that compounded into serious real-world latency even with the
hosted server on the same network. proxy() now reuses one module-level
requests.Session (_proxy_session) across every call instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

import tessera_eval.server as srv


@pytest.fixture
def client():
    srv.app.config["TESTING"] = True
    return srv.app.test_client()


def _fake_response(body=b"ok", status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"Content-Type": "text/plain"}
    resp.iter_content = lambda chunk_size: iter([body])
    return resp


def test_proxy_session_is_a_real_requests_session():
    """The module-level session proxy() shares must actually be a
    requests.Session -- the whole point is connection pooling/keep-alive,
    which only a real Session provides."""
    assert isinstance(srv._proxy_session, requests.Session)


def test_proxy_reuses_the_same_session_object_across_requests(client, monkeypatch):
    """Regression: two separate proxied requests must go through the exact
    same Session object (identity check), not a fresh one each time --
    that's what actually gives connection reuse instead of a handshake per
    request."""
    old_hosted = srv._hosted_url
    session_before = srv._proxy_session
    mock_request = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(srv._proxy_session, "request", mock_request)
    try:
        srv._hosted_url = "https://tee.cl.cam.ac.uk"

        resp1 = client.get("/some/ui/path")
        resp2 = client.get("/some/other/path")

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert mock_request.call_count == 2
        # Same Session object used both times -- no new Session was created
        # in between (which would defeat connection pooling entirely).
        assert srv._proxy_session is session_before
    finally:
        srv._hosted_url = old_hosted


def test_proxy_forwards_method_path_and_query_string(client, monkeypatch):
    old_hosted = srv._hosted_url
    mock_request = MagicMock(return_value=_fake_response(b"hello"))
    monkeypatch.setattr(srv._proxy_session, "request", mock_request)
    try:
        srv._hosted_url = "https://tee.cl.cam.ac.uk"
        resp = client.get("/viewports/list?year=2024")
        assert resp.status_code == 200
        assert resp.data == b"hello"
        _args, kwargs = mock_request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == "https://tee.cl.cam.ac.uk/viewports/list?year=2024"
    finally:
        srv._hosted_url = old_hosted


def test_proxy_returns_502_with_no_hosted_url_configured(client):
    old_hosted = srv._hosted_url
    try:
        srv._hosted_url = None
        resp = client.get("/anything")
        assert resp.status_code == 502
    finally:
        srv._hosted_url = old_hosted


def test_proxy_returns_502_on_connection_error(client, monkeypatch):
    old_hosted = srv._hosted_url
    monkeypatch.setattr(
        srv._proxy_session, "request", MagicMock(side_effect=requests.ConnectionError("no route"))
    )
    try:
        srv._hosted_url = "https://tee.cl.cam.ac.uk"
        resp = client.get("/anything")
        assert resp.status_code == 502
    finally:
        srv._hosted_url = old_hosted


def test_proxy_returns_504_on_timeout(client, monkeypatch):
    old_hosted = srv._hosted_url
    monkeypatch.setattr(
        srv._proxy_session, "request", MagicMock(side_effect=requests.Timeout("too slow"))
    )
    try:
        srv._hosted_url = "https://tee.cl.cam.ac.uk"
        resp = client.get("/anything")
        assert resp.status_code == 504
    finally:
        srv._hosted_url = old_hosted
