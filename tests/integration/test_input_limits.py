"""API-level tests for the input limits: pixel count, request body size, and
truncated or incomplete files."""

import asyncio
from io import BytesIO
from pathlib import Path

import pytest
import starlette.formparsers
from fastapi.testclient import TestClient
from PIL import Image, ImageFile

from app.main import app

SHARP = (Path(__file__).parent.parent / "assets" / "sharp_small.jpg").read_bytes()
INVALID = (400, b'{"detail":"Invalid image data"}')
TOO_LARGE = (413, b'{"detail":"File too large"}')
ENDPOINTS = ("/quality", "/quality/diagnostics")


def post(client, data, path="/quality", content_type="image/jpeg", **kw):
    return client.post(path, files={"file": ("upload", data, content_type)}, **kw)


def status_and_body(resp):
    return resp.status_code, resp.content


@pytest.fixture
def load_calls(monkeypatch):
    calls = []
    real_load = ImageFile.ImageFile.load

    def recording_load(self):
        calls.append(self.size)
        return real_load(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load", recording_load)
    return calls


# --- Pixel limit -------------------------------------------------------------


def test_default_pixel_limit_rejects_large_dimensions(
    settings_env, make_image, claim_dimensions, load_calls
):
    s = settings_env()
    assert s.max_image_pixels == 25_000_000
    client = TestClient(app)
    buf = BytesIO()
    Image.new("L", (6000, 5000), 128).save(buf, format="PNG")
    oversized = {
        "png_30mp_compressed": buf.getvalue(),
        "png_header_claims_64mp": claim_dimensions(
            make_image(16, 16, "RGB", "PNG"), 8000, 8000
        ),
        "jpeg_header_claims_64mp": claim_dimensions(make_image(16, 16), 8000, 8000),
    }
    for name, data in oversized.items():
        for path in ENDPOINTS:
            assert status_and_body(post(client, data, path)) == INVALID, name
    assert load_calls == []  # nothing was decoded


def test_images_under_the_default_limit_are_scored(settings_env, make_image):
    settings_env()
    resp = post(TestClient(app), make_image(3000, 2000))
    assert resp.status_code == 200
    assert (resp.json()["width"], resp.json()["height"]) == (1600, 1066)


def test_configured_pixel_limit_and_its_boundary(settings_env, make_image):
    settings_env(env={"MAX_IMAGE_PIXELS": "10000"})
    client = TestClient(app)
    assert post(client, make_image(100, 100)).status_code == 200
    assert status_and_body(post(client, make_image(100, 101))) == INVALID
    settings = client.get("/quality/config").json()["settings"]
    assert settings["max_image_pixels"] == {"value": 10000, "source": "env"}


# --- Request body size -------------------------------------------------------


def test_oversized_request_is_refused_before_the_body_is_parsed(
    settings_env, monkeypatch
):
    settings_env(env={"MAX_UPLOAD_MB": "1"})
    parsed = []
    real_parse = starlette.formparsers.MultiPartParser.parse

    async def recording_parse(self):
        parsed.append(True)
        return await real_parse(self)

    monkeypatch.setattr(starlette.formparsers.MultiPartParser, "parse", recording_parse)
    client = TestClient(app)
    for path in ENDPOINTS:
        resp = post(
            client,
            b"\x00" * (2 * 1024 * 1024),
            path,
            headers={"Origin": "https://example.com", "X-Request-ID": "big-1"},
        )
        assert status_and_body(resp) == TOO_LARGE
        # The 413 still passes through the CORS and request-ID middleware.
        assert resp.headers["access-control-allow-origin"] == "*"
        assert resp.headers["x-request-id"] == "big-1"
    assert parsed == []


def test_requests_within_the_limit_reach_the_handler(settings_env):
    settings_env(env={"MAX_UPLOAD_MB": "1"})
    client = TestClient(app)
    # Exactly the file limit: passes both limits, then fails as image data.
    assert status_and_body(post(client, b"\x00" * (1024 * 1024))) == INVALID
    # One byte over: the handler's check on the file itself answers.
    assert status_and_body(post(client, b"\x00" * (1024 * 1024 + 1))) == TOO_LARGE
    assert client.get("/health").status_code == 200


def test_chunked_upload_is_cut_off_at_the_limit(settings_env):
    # No Content-Length, so only the running byte count can stop it. Drive the
    # real app over ASGI so the body really arrives in chunks.
    s = settings_env(env={"MAX_UPLOAD_MB": "1"})
    boundary = "limit-test"
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="big.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    chunk = b"\x00" * 65536
    total_chunks = 200  # about 13 MB if it were all read
    pulled = 0
    sent = []

    async def receive():
        nonlocal pulled
        pulled += 1
        body = head if pulled == 1 else chunk
        return {
            "type": "http.request",
            "body": body,
            "more_body": pulled < total_chunks,
        }

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/quality",
        "raw_path": b"/quality",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"transfer-encoding", b"chunked"),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    assert (start["status"], body) == TOO_LARGE
    # Reading stopped at the first chunk that crossed the limit, out of 200.
    before_last = len(head) + (pulled - 2) * len(chunk)
    assert before_last <= s.max_request_bytes < before_last + len(chunk)
    assert pulled < total_chunks


# --- Truncated and incomplete files ------------------------------------------


def test_truncated_jpeg_is_rejected_and_valid_files_are_scored(
    settings_env, make_image
):
    settings_env()
    client = TestClient(app)
    png = make_image(300, 200, "RGB", "PNG")
    for path in ENDPOINTS:
        assert status_and_body(post(client, SHARP[: len(SHARP) // 2], path)) == INVALID
        assert (
            status_and_body(post(client, png[: len(png) // 2], path, "image/png"))
            == INVALID
        )
        assert post(client, SHARP, path).status_code == 200
        assert post(client, png, path, "image/png").status_code == 200
