import os
import struct
import sys
import pathlib
import zlib
from io import BytesIO

import numpy as np
import pytest

repo_root = (
    pathlib.Path(__file__).resolve().parents[1]
)  # go up from tests/ to repo root
sys.path.insert(0, str(repo_root))

from PIL import ExifTags, Image  # noqa: E402

from app.core import config  # noqa: E402
from app.core.config import Settings  # noqa: E402


@pytest.fixture
def settings_env(tmp_path, monkeypatch):
    """
    Isolate settings from the developer's environment and .env: work in an
    empty temp dir with no settings variables set. Returns install(), which
    builds Settings from given env vars / .env text and puts them in effect
    for the API (routes and middleware read config.settings per request).
    """
    monkeypatch.chdir(tmp_path)
    fields = set(Settings.model_fields)
    for key in list(os.environ):
        if key.lower() in fields:
            monkeypatch.delenv(key)

    def install(env: dict[str, str] | None = None, dotenv: str | None = None):
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        if dotenv is not None:
            (tmp_path / ".env").write_text(dotenv)
        s = Settings()
        monkeypatch.setattr(config, "settings", s)
        return s

    return install


def _claim_dimensions(data: bytes, width: int, height: int) -> bytes:
    """
    Rewrite the dimensions a PNG (IHDR) or baseline JPEG (SOF0) header declares,
    leaving the pixel data as it is: a small file claiming a huge image.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        ihdr = b"IHDR" + struct.pack(">II", width, height) + data[24:29]
        crc = struct.pack(">I", zlib.crc32(ihdr))
        return data[:12] + ihdr + crc + data[33:]
    sof = data.index(b"\xff\xc0")
    dims = struct.pack(">HH", height, width)
    return data[: sof + 5] + dims + data[sof + 9 :]


@pytest.fixture
def claim_dimensions():
    """claim_dimensions(png_or_jpeg_bytes, width, height) -> bytes"""
    return _claim_dimensions


def _pattern(width: int, height: int) -> np.ndarray:
    # Asymmetric content, so every flip/rotation gives different pixels.
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            xx * 255 // max(1, width - 1),
            yy * 255 // max(1, height - 1),
            (xx * yy) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    rgb[height // 8 : height // 3, width // 10 : width // 2] = (250, 20, 20)
    return rgb


@pytest.fixture
def make_image():
    """make_image(width, height, mode="RGB", fmt="JPEG", orientation=None) -> bytes"""

    def make(
        width: int,
        height: int,
        mode: str = "RGB",
        fmt: str = "JPEG",
        orientation: int | None = None,
    ) -> bytes:
        img = Image.fromarray(_pattern(width, height))
        if mode == "P":
            img = img.quantize(colors=64)
        elif mode != "RGB":
            img = img.convert(mode)
        save: dict = {}
        if orientation is not None:
            exif = Image.Exif()
            exif[ExifTags.Base.Orientation] = orientation
            save["exif"] = exif.tobytes()
        buf = BytesIO()
        img.save(buf, format=fmt, **save)
        return buf.getvalue()

    return make
