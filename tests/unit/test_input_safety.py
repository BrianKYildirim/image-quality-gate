from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFile

import app.main  # noqa: F401  (import the whole app: nothing may re-enable truncation)
from app.core.timing import StageTimer
from app.services.preprocess import (
    ImageTooLarge,
    load_and_normalize,
    load_and_normalize_traced,
)

SHARP = (Path(__file__).parent.parent / "assets" / "sharp_small.jpg").read_bytes()


def encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


@pytest.fixture
def load_calls(monkeypatch):
    """Records every Pillow ImageFile.load() call (where pixel decoding happens)."""
    calls = []
    real_load = ImageFile.ImageFile.load

    def recording_load(self):
        calls.append(self.size)
        return real_load(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load", recording_load)
    return calls


# --- Truncated and incomplete files ------------------------------------------


def test_truncated_images_are_not_enabled_anywhere_in_the_app():
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def test_the_truncated_jpeg_is_rejected():
    # The first half of a real asset: it used to decode with the missing rows
    # filled in and was scored as a pass.
    with pytest.raises(OSError, match="truncated"):
        load_and_normalize(SHARP[: len(SHARP) // 2], 1600)


@pytest.mark.parametrize(
    "fmt, kwargs, keep",
    [
        ("JPEG", {"quality": 90}, 0.9),
        ("JPEG", {"quality": 90, "progressive": True}, 0.6),
        ("PNG", {}, 0.5),
        ("GIF", {}, 0.5),
        ("TIFF", {}, 0.5),
    ],
)
def test_incomplete_files_are_rejected(make_image, fmt, kwargs, keep):
    img = Image.open(BytesIO(make_image(256, 192, "RGB", "PNG")))
    data = encode(img.quantize(64) if fmt == "GIF" else img, fmt, **kwargs)
    load_and_normalize(data, 1600)  # complete file: fine
    with pytest.raises(OSError):
        load_and_normalize(data[: int(len(data) * keep)], 1600)


def test_jpeg_missing_only_its_end_marker_is_rejected(make_image):
    # Strict on purpose: the decoder reports the data ended early and cannot
    # tell this apart from a file cut off right at the end of its scan data.
    data = make_image(256, 192, "RGB", "JPEG")
    assert data.endswith(b"\xff\xd9")
    with pytest.raises(OSError, match="truncated"):
        load_and_normalize(data[:-2], 1600)


@pytest.mark.parametrize(
    "fmt, damage",
    [
        ("JPEG", lambda d: d + b"\x00trailing bytes"),
        ("PNG", lambda d: d[:-12]),  # without the IEND chunk
    ],
)
def test_complete_pixel_data_with_damaged_trailers_still_scores_the_same(
    make_image, fmt, damage
):
    data = make_image(256, 192, "RGB", fmt)
    reference, _, _ = load_and_normalize(data, 1600)
    damaged, _, _ = load_and_normalize(damage(data), 1600)
    assert np.array_equal(damaged, reference)


def test_valid_jpeg_and_png_load(make_image):
    for data in (SHARP, make_image(300, 200, "RGB", "PNG")):
        bgr, w, h = load_and_normalize(data, 1600)
        assert bgr.shape == (h, w, 3)


# --- Pixel limit -------------------------------------------------------------


def test_pixel_limit_is_inclusive(make_image):
    _, w, h = load_and_normalize(make_image(100, 100), 1600, max_pixels=10_000)
    assert (w, h) == (100, 100)
    with pytest.raises(ImageTooLarge):
        load_and_normalize(make_image(101, 100), 1600, max_pixels=10_000)


def test_pixel_limit_is_checked_before_decoding(make_image, load_calls):
    timer = StageTimer()
    with pytest.raises(ImageTooLarge, match="101x100 exceeds 10000 pixels"):
        load_and_normalize_traced(make_image(101, 100), 1600, timer, 10_000)
    assert [s.name for s in timer.stages] == ["open"]
    assert load_calls == []
    load_and_normalize(make_image(100, 100), 1600, max_pixels=10_000)
    # The spy does see decodes (later load() calls are no-ops once loaded).
    assert load_calls and set(load_calls) == {(100, 100)}


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_small_file_claiming_huge_dimensions_is_rejected_before_decoding(
    make_image, claim_dimensions, load_calls, fmt
):
    # A few hundred bytes that claim 8000x8000 (64 MP): above the 25 MP
    # default, below Pillow's own 89.5 MP warning threshold.
    data = claim_dimensions(make_image(16, 16, "RGB", fmt), 8000, 8000)
    assert len(data) < 2000
    assert Image.open(BytesIO(data)).size == (8000, 8000)
    with pytest.raises(ImageTooLarge):
        load_and_normalize(data, 1600, max_pixels=25_000_000)
    assert load_calls == []


def test_highly_compressible_large_image_is_rejected_before_decoding(load_calls):
    # A real, valid 6000x5000 (30 MP) image that compresses to a few kilobytes.
    data = encode(Image.new("L", (6000, 5000), 128), "PNG")
    assert len(data) < 100_000
    with pytest.raises(ImageTooLarge):
        load_and_normalize(data, 1600, max_pixels=25_000_000)
    assert load_calls == []


def test_pillow_still_refuses_extreme_dimensions_itself(make_image, claim_dimensions):
    data = claim_dimensions(make_image(16, 16, "RGB", "PNG"), 20000, 20000)
    with pytest.raises(Image.DecompressionBombError):
        load_and_normalize(data, 1600, max_pixels=10**12)


def test_no_pixel_limit_unless_one_is_given(make_image):
    _, w, h = load_and_normalize(make_image(300, 200), 1600)
    assert (w, h) == (300, 200)
