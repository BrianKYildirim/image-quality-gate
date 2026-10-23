import asyncio

import pytest
from fastapi import HTTPException

from app.core.body_limit import BodySizeLimitMiddleware

LIMIT = 1000


def http_scope(headers: list[tuple[bytes, bytes]]) -> dict:
    return {"type": "http", "method": "POST", "path": "/", "headers": headers}


class Downstream:
    """An ASGI app that reads the whole body, then answers 200."""

    def __init__(self) -> None:
        self.called = False
        self.body = b""

    async def __call__(self, scope, receive, send) -> None:
        self.called = True
        while True:
            message = await receive()
            self.body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def run(app, scope, chunks):
    pulled = []
    sent = []

    async def receive():
        chunk = chunks[len(pulled)]
        pulled.append(chunk)
        more = len(pulled) < len(chunks)
        return {"type": "http.request", "body": chunk, "more_body": more}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return pulled, sent


def test_declared_oversized_body_is_refused_without_reading_it():
    downstream = Downstream()
    app = BodySizeLimitMiddleware(downstream, max_bytes=lambda: LIMIT)
    scope = http_scope([(b"content-length", str(LIMIT + 1).encode())])
    pulled, sent = run(app, scope, [b"x" * (LIMIT + 1)])
    assert pulled == [] and downstream.called is False
    assert sent[0]["status"] == 413
    assert (b"content-type", b"application/json") in sent[0]["headers"]
    assert sent[1]["body"] == b'{"detail":"File too large"}'


def test_bodies_within_the_limit_pass_through_unchanged():
    downstream = Downstream()
    app = BodySizeLimitMiddleware(downstream, max_bytes=lambda: LIMIT)
    scope = http_scope([(b"content-length", str(LIMIT).encode())])
    _, sent = run(app, scope, [b"a" * 600, b"b" * 400])
    assert downstream.body == b"a" * 600 + b"b" * 400
    assert sent[0]["status"] == 200


def test_streamed_body_is_cut_off_once_it_passes_the_limit():
    # No Content-Length (chunked upload): stop reading as soon as the count
    # passes the limit, without pulling the remaining chunks.
    downstream = Downstream()
    app = BodySizeLimitMiddleware(downstream, max_bytes=lambda: LIMIT)
    chunks = [b"z" * 300] * 100
    with pytest.raises(HTTPException) as excinfo:
        run(app, http_scope([]), chunks)
    assert (excinfo.value.status_code, excinfo.value.detail) == (413, "File too large")
    assert len(downstream.body) == 900  # the 4th chunk (1200 bytes total) raised


def test_unparseable_content_length_falls_back_to_counting():
    app = BodySizeLimitMiddleware(Downstream(), max_bytes=lambda: LIMIT)
    with pytest.raises(HTTPException):
        run(app, http_scope([(b"content-length", b"abc")]), [b"z" * 600] * 3)


def test_limit_is_read_per_request():
    limits = iter([LIMIT, 10])
    downstream = Downstream()
    app = BodySizeLimitMiddleware(downstream, max_bytes=lambda: next(limits))
    run(app, http_scope([(b"content-length", b"50")]), [b"z" * 50])
    _, sent = run(app, http_scope([(b"content-length", b"50")]), [b"z" * 50])
    assert sent[0]["status"] == 413


def test_non_http_scopes_are_passed_through():
    seen = []

    async def lifespan_app(scope, receive, send):
        seen.append(scope["type"])

    app = BodySizeLimitMiddleware(lifespan_app, max_bytes=lambda: 0)
    asyncio.run(app({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]
