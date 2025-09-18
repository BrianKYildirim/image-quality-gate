import cv2
import numpy as np
from io import BytesIO
from typing import cast

from PIL import Image, ImageOps, ImageFile

# Safety for odd/truncated inputs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Pillow 10+ uses the Resampling enum; fall back gracefully for older versions.
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older Pillow
    RESAMPLE_LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


def load_and_normalize(image_bytes: bytes, max_dim: int) -> tuple[np.ndarray, int, int]:
    """
    Load image bytes, correct EXIF orientation, cap size,
    return BGR array for OpenCV and width/height.
    """
    opened = Image.open(BytesIO(image_bytes))
    # After exif_transpose + convert, the result is a real Image.Image; help mypy with cast.
    img = cast(Image.Image, ImageOps.exif_transpose(opened).convert("RGB"))

    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), RESAMPLE_LANCZOS)

    arr = np.asarray(img, dtype=np.uint8)  # RGB uint8
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, img.width, img.height


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
