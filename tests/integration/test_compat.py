"""
POST /quality must stay byte-for-byte compatible with the code before
diagnostics were added, except for the input-safety changes made on purpose
(see test_intentional_changes). legacy_app below is a verbatim copy of that
route and pipeline; both apps run with the same settings in the same
environment, so library versions cancel out.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image, ImageFile, ImageOps
from pydantic import BaseModel

from app.main import app

ASSETS = Path(__file__).parent.parent / "assets"


class QualityResponse(BaseModel):
    blur_score: float
    brightness: float
    width: int
    height: int
    is_ok: bool
    thresholds: dict[str, float | int]


@dataclass(frozen=True)
class LegacyThresholds:
    blur_min: float
    bright_min: int
    bright_max: int


def legacy_app(settings) -> FastAPI:
    def load_and_normalize(image_bytes: bytes, max_dim: int):
        # The original module set this globally at import; scope it to this copy.
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            opened = Image.open(BytesIO(image_bytes))
            img = cast(Image.Image, ImageOps.exif_transpose(opened).convert("RGB"))
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
        w, h = img.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr, img.width, img.height

    def to_gray(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def assess(gray, thr: LegacyThresholds) -> dict:
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(gray.mean())
        is_ok = (blur_score >= thr.blur_min) and (
            thr.bright_min <= bright <= thr.bright_max
        )
        return {"blur_score": blur_score, "brightness": bright, "is_ok": is_ok}

    legacy = FastAPI()

    @legacy.post("/quality", response_model=QualityResponse)
    async def quality(file: UploadFile = File(...)) -> QualityResponse:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Unsupported media type")
        content = await file.read()
        byte_limit = settings.max_upload_mb * 1024 * 1024
        if len(content) > byte_limit:
            raise HTTPException(status_code=413, detail="File too large")
        try:
            bgr, w, h = load_and_normalize(content, settings.resize_max_dim)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image data")
        gray = to_gray(bgr)
        res = assess(
            gray,
            LegacyThresholds(
                settings.blur_min, settings.bright_min, settings.bright_max
            ),
        )
        return QualityResponse(
            **res,
            width=w,
            height=h,
            thresholds={
                "blur_min": settings.blur_min,
                "bright_min": settings.bright_min,
                "bright_max": settings.bright_max,
            },
        )

    return legacy


def corpus(make_image):
    cases = [
        ("asset_" + p.name, p.read_bytes(), "image/jpeg")
        for p in sorted(ASSETS.glob("*.jpg"))
    ]
    for o in (None, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        cases.append(
            (f"exif_{o}", make_image(300, 200, "RGB", "JPEG", o), "image/jpeg")
        )
    cases += [
        ("resized_exif6", make_image(2000, 1000, "RGB", "JPEG", 6), "image/jpeg"),
        ("rgba", make_image(120, 90, "RGBA", "PNG"), "image/png"),
        ("palette_png", make_image(120, 90, "P", "PNG"), "image/png"),
        ("palette_gif", make_image(120, 90, "P", "GIF"), "image/gif"),
        ("gray", make_image(120, 90, "L", "JPEG"), "image/jpeg"),
        ("gray_alpha", make_image(120, 90, "LA", "PNG"), "image/png"),
        ("cmyk", make_image(120, 90, "CMYK", "JPEG"), "image/jpeg"),
        ("webp", make_image(120, 90, "RGB", "WEBP"), "image/webp"),
        ("one_pixel", make_image(1, 1, "RGB", "PNG"), "image/png"),
        ("thin", make_image(1601, 10, "RGB", "PNG"), "image/png"),
    ]
    jpeg = make_image(300, 200, "RGB", "JPEG")
    cases += [
        ("jpeg_trailing_bytes", jpeg + b"\x00extra", "image/jpeg"),
        ("png_without_iend", make_image(120, 90, "RGB", "PNG")[:-12], "image/png"),
        ("garbage", b"definitely not an image", "image/jpeg"),
        ("empty", b"", "image/png"),
        ("not_an_image_type", jpeg, "text/plain"),
        ("uppercase_type", jpeg, "IMAGE/JPEG"),
        ("too_large", b"\xff" * (1024 * 1024 + 1), "image/jpeg"),
        ("far_too_large", b"\xff" * (2 * 1024 * 1024), "image/jpeg"),
        ("exactly_the_limit", b"\xff" * (1024 * 1024), "image/jpeg"),
    ]
    return cases


def assert_same(new, old):
    assert new.status_code == old.status_code
    assert new.content == old.content
    for header in ("content-type", "content-length"):
        assert new.headers.get(header) == old.headers.get(header)


def test_quality_responses_are_byte_identical(settings_env, make_image):
    s = settings_env(env={"MAX_UPLOAD_MB": "1"})
    new, old = TestClient(app), TestClient(legacy_app(s))
    statuses = set()
    for name, data, content_type in corpus(make_image):
        files = {"file": (f"{name}.bin", data, content_type)}
        r_new = new.post("/quality", files=files)
        r_old = old.post("/quality", files=files)
        assert_same(r_new, r_old)
        statuses.add(r_new.status_code)
    assert statuses == {200, 400, 413, 415}


def test_intentional_changes(settings_env, make_image):
    # Inputs the old code scored but that are now rejected with its own
    # existing 400 response: incomplete files, and images over MAX_IMAGE_PIXELS.
    s = settings_env(env={"MAX_IMAGE_PIXELS": "10000"})
    new, old = TestClient(app), TestClient(legacy_app(s))
    sharp = (ASSETS / "sharp_small.jpg").read_bytes()
    small_jpeg = make_image(100, 100, "RGB", "JPEG")
    cases = [
        ("truncated_jpeg", sharp[: len(sharp) // 2]),
        ("jpeg_without_eoi", small_jpeg[:-2]),
        ("over_pixel_limit", make_image(101, 100, "RGB", "JPEG")),
    ]
    for name, data in cases:
        files = {"file": (f"{name}.jpg", data, "image/jpeg")}
        assert old.post("/quality", files=files).status_code == 200, name
        r_new = new.post("/quality", files=files)
        assert (r_new.status_code, r_new.content) == (
            400,
            b'{"detail":"Invalid image data"}',
        ), name
    files = {"file": ("ok.jpg", small_jpeg, "image/jpeg")}
    assert_same(new.post("/quality", files=files), old.post("/quality", files=files))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # no multipart body at all
        {"files": {"upload": ("a.jpg", b"x", "image/jpeg")}},  # wrong field name
        {"data": {"file": "not a file"}},
    ],
)
def test_validation_errors_are_byte_identical(settings_env, kwargs):
    s = settings_env()
    r_new = TestClient(app).post("/quality", **kwargs)
    r_old = TestClient(legacy_app(s)).post("/quality", **kwargs)
    assert r_new.status_code == 422
    assert_same(r_new, r_old)


def test_quality_openapi_is_unchanged(settings_env):
    s = settings_env()
    new_spec = app.openapi()
    old_spec = legacy_app(s).openapi()
    assert new_spec["paths"]["/quality"] == old_spec["paths"]["/quality"]
    new_schemas, old_schemas = (
        new_spec["components"]["schemas"],
        old_spec["components"]["schemas"],
    )
    assert (
        new_schemas["Body_quality_quality_post"]
        == old_schemas["Body_quality_quality_post"]
    )
    assert new_schemas["QualityResponse"] == old_schemas["QualityResponse"]


def test_default_response_fields_and_types(settings_env):
    # Values are compared byte-for-byte above; this pins the field order and types.
    settings_env()
    body = (
        TestClient(app)
        .post(
            "/quality",
            files={
                "file": (
                    "s.jpg",
                    (ASSETS / "sharp_small.jpg").read_bytes(),
                    "image/jpeg",
                )
            },
        )
        .json()
    )
    assert list(body) == [
        "blur_score",
        "brightness",
        "width",
        "height",
        "is_ok",
        "thresholds",
    ]
    assert [type(body[k]) for k in body] == [float, float, int, int, bool, dict]
    assert (body["width"], body["height"], body["is_ok"]) == (256, 256, True)
    assert body["thresholds"] == {
        "blur_min": 140.0,
        "bright_min": 20,
        "bright_max": 235,
    }
    assert [type(v) for v in body["thresholds"].values()] == [float, int, int]
