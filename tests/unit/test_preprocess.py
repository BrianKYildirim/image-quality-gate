from io import BytesIO
from typing import cast

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from app.core.timing import StageTimer
from app.services.preprocess import load_and_normalize, load_and_normalize_traced


def legacy_load_and_normalize(image_bytes: bytes, max_dim: int):
    """The implementation before diagnostics were added, verbatim."""
    opened = Image.open(BytesIO(image_bytes))
    img = cast(Image.Image, ImageOps.exif_transpose(opened).convert("RGB"))
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, img.width, img.height


def traced(data: bytes, max_dim: int = 1600):
    timer = StageTimer()
    bgr, info = load_and_normalize_traced(data, max_dim, timer)
    return bgr, info, timer


CASES = [
    *[(300, 200, "RGB", "JPEG", o) for o in (None, 1, 2, 3, 4, 5, 6, 7, 8)],
    (2000, 1000, "RGB", "JPEG", 6),
    (1700, 900, "RGB", "PNG", None),
    (1601, 10, "RGB", "PNG", None),
    (120, 90, "RGBA", "PNG", None),
    (120, 90, "P", "PNG", None),
    (120, 90, "P", "GIF", None),
    (120, 90, "L", "JPEG", 8),
    (120, 90, "LA", "PNG", None),
    (120, 90, "CMYK", "JPEG", None),
    (120, 90, "RGB", "WEBP", None),
    (1, 1, "RGB", "PNG", None),
]


@pytest.mark.parametrize("width, height, mode, fmt, orientation", CASES)
def test_same_pixels_as_the_previous_implementation(
    make_image, width, height, mode, fmt, orientation
):
    data = make_image(width, height, mode, fmt, orientation)
    legacy_bgr, legacy_w, legacy_h = legacy_load_and_normalize(data, 1600)
    bgr, w, h = load_and_normalize(data, 1600)
    assert (w, h) == (legacy_w, legacy_h)
    assert bgr.dtype == legacy_bgr.dtype
    assert np.array_equal(bgr, legacy_bgr)


@pytest.mark.parametrize("orientation", [6, 8])
def test_rotating_orientations_swap_dimensions(make_image, orientation):
    _, info, _ = traced(make_image(300, 200, "RGB", "PNG", orientation))
    assert info.exif_orientation == orientation
    assert info.orientation_applied is True
    assert (info.original_width, info.original_height) == (300, 200)
    assert (info.oriented_width, info.oriented_height) == (200, 300)
    assert (info.width, info.height) == (200, 300)


@pytest.mark.parametrize("orientation, k", [(3, 2), (6, 3), (8, 1)])
def test_orientation_is_applied_to_the_pixels(make_image, orientation, k):
    upright, _, _ = traced(make_image(64, 48, "RGB", "PNG", 1))
    rotated, info, _ = traced(make_image(64, 48, "RGB", "PNG", orientation))
    # Pillow ROTATE_180/270/90 are counter-clockwise, as is np.rot90.
    assert np.array_equal(rotated, np.rot90(upright, k))
    assert info.orientation_applied is True


@pytest.mark.parametrize("orientation, reported", [(None, None), (1, 1), (9, 9)])
def test_untransposed_orientations(make_image, orientation, reported):
    _, info, _ = traced(make_image(300, 200, "RGB", "JPEG", orientation))
    assert info.exif_orientation == reported
    assert info.orientation_applied is False
    assert (info.oriented_width, info.oriented_height) == (300, 200)


@pytest.mark.parametrize(
    "size, max_dim, resized, scale, processed",
    [
        ((2000, 1000), 1600, True, 0.8, (1600, 800)),
        ((1600, 1200), 1600, False, 1.0, (1600, 1200)),
        ((800, 600), 1600, False, 1.0, (800, 600)),
        ((1000, 3000), 1500, True, 0.5, (500, 1500)),
    ],
)
def test_resize_is_reported(make_image, size, max_dim, resized, scale, processed):
    bgr, info, timer = traced(make_image(*size, "RGB", "PNG"), max_dim)
    assert info.resized is resized
    assert info.scale == scale
    assert (info.width, info.height) == processed
    assert bgr.shape[:2] == (processed[1], processed[0])
    assert ("resize" in [s.name for s in timer.stages]) is resized


def test_scale_and_processed_size_follow_the_pipeline_arithmetic(make_image):
    # The pipeline truncates: 10 * (1600 / 1601) = 9.99... becomes 9.
    _, info, _ = traced(make_image(1601, 10, "RGB", "PNG"))
    assert info.scale == 1600 / 1601
    assert (info.width, info.height) == (int(1601 * info.scale), 9)


def test_rotation_happens_before_the_resize(make_image):
    _, info, _ = traced(make_image(2000, 1000, "RGB", "JPEG", 6))
    assert (info.oriented_width, info.oriented_height) == (1000, 2000)
    assert info.scale == 0.8
    assert (info.width, info.height) == (800, 1600)


@pytest.mark.parametrize(
    "mode, fmt",
    [
        ("RGB", "JPEG"),
        ("RGBA", "PNG"),
        ("P", "PNG"),
        ("P", "GIF"),
        ("L", "PNG"),
        ("LA", "PNG"),
        ("CMYK", "JPEG"),
        ("RGB", "WEBP"),
    ],
)
def test_decoded_format_and_mode_are_reported(make_image, mode, fmt):
    _, info, _ = traced(make_image(40, 30, mode, fmt))
    assert (info.format, info.mode) == (fmt, mode)


def test_stages_run_in_order(make_image):
    _, _, timer = traced(make_image(2000, 1000, "RGB", "JPEG"))
    assert [s.name for s in timer.stages] == [
        "open",
        "decode",
        "exif_transpose",
        "convert_rgb",
        "resize",
        "to_array",
        "rgb_to_bgr",
    ]
    for stage in timer.stages:
        assert isinstance(stage.duration_ns, int) and stage.duration_ns >= 0


def test_invalid_bytes_raise_from_open_with_no_stage_recorded():
    timer = StageTimer()
    with pytest.raises(Exception):
        load_and_normalize_traced(b"not an image", 1600, timer)
    assert timer.stages == []
