import cv2, numpy as np
from PIL import Image, ImageOps
from io import BytesIO

def load_and_normalize(image_bytes: bytes, max_dim: int):
    # Load image bytes, correct EXIF orientation, cap size, return BGR array and width/height.
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    arr = np.array(img)            # RGB uint8
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, img.width, img.height

def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
