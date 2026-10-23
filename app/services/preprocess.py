import cv2
import numpy as np
from dataclasses import dataclass
from io import BytesIO
from typing import cast

from PIL import ExifTags, Image, ImageOps

from app.core.timing import StageTimer

# Truncated files are rejected: with Pillow's default
# ImageFile.LOAD_TRUNCATED_IMAGES = False, load() raises OSError when the data
# ends before the image is complete, instead of filling in the missing part.

# Pillow 10+ uses the Resampling enum; fall back gracefully for older versions.
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older Pillow
    RESAMPLE_LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

# Orientation values ImageOps.exif_transpose has a transpose for; others are left as-is.
TRANSPOSED_ORIENTATIONS = (2, 3, 4, 5, 6, 7, 8)


class ImageTooLarge(ValueError):
    """The image's dimensions exceed the configured pixel limit."""


@dataclass(frozen=True)
class ImageInfo:
    """What the pipeline observed while normalizing one image."""

    format: str | None
    mode: str
    original_width: int
    original_height: int
    exif_orientation: int | None
    orientation_applied: bool
    oriented_width: int
    oriented_height: int
    resized: bool
    scale: float
    width: int
    height: int


def load_and_normalize(
    image_bytes: bytes, max_dim: int, max_pixels: int | None = None
) -> tuple[np.ndarray, int, int]:
    """
    Load image bytes, correct EXIF orientation, cap size,
    return BGR array for OpenCV and width/height.
    """
    bgr, info = load_and_normalize_traced(
        image_bytes, max_dim, StageTimer(), max_pixels
    )
    return bgr, info.width, info.height


def load_and_normalize_traced(
    image_bytes: bytes,
    max_dim: int,
    timer: StageTimer,
    max_pixels: int | None = None,
) -> tuple[np.ndarray, ImageInfo]:
    """
    The load_and_normalize pipeline, timing each step where its work actually
    happens and returning what it observed along the way. Raises ImageTooLarge
    if the image has more than max_pixels pixels.
    """
    with timer.stage("open"):
        # Parses the header only; pixel data is decoded by load().
        opened = Image.open(BytesIO(image_bytes))
    if max_pixels is not None and opened.width * opened.height > max_pixels:
        # Checked against the header's dimensions, before anything is decoded.
        raise ImageTooLarge(
            f"{opened.width}x{opened.height} exceeds {max_pixels} pixels"
        )
    with timer.stage("decode"):
        # exif_transpose() starts with load() itself; calling it first only
        # moves the decode into its own stage (load() is idempotent).
        opened.load()
    with timer.stage("exif_transpose"):
        # The same cached Exif object exif_transpose() reads the tag from.
        orientation = opened.getexif().get(ExifTags.Base.Orientation)
        oriented = cast(Image.Image, ImageOps.exif_transpose(opened))
    with timer.stage("convert_rgb"):
        img = oriented.convert("RGB")
    # Release the transposed copy now, as the one-expression original did, so
    # later allocations can reuse its memory instead of faulting in new pages.
    oriented_width, oriented_height = oriented.size
    del oriented

    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        with timer.stage("resize"):
            img = img.resize((int(w * scale), int(h * scale)), RESAMPLE_LANCZOS)

    with timer.stage("to_array"):
        arr = np.asarray(img, dtype=np.uint8)  # RGB uint8
    with timer.stage("rgb_to_bgr"):
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    info = ImageInfo(
        format=opened.format,
        mode=opened.mode,
        original_width=opened.width,
        original_height=opened.height,
        exif_orientation=orientation if isinstance(orientation, int) else None,
        orientation_applied=orientation in TRANSPOSED_ORIENTATIONS,
        oriented_width=oriented_width,
        oriented_height=oriented_height,
        resized=scale < 1.0,
        scale=scale,
        width=img.width,
        height=img.height,
    )
    return bgr, info


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
