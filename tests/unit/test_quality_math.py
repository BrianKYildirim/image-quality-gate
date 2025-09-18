import numpy as np
from app.services.quality import assess, Thresholds

def test_bright_sharp_image_is_ok():
    img = np.zeros((256,256), dtype=np.uint8)
    img[64:192,64:192] = 255
    thr = Thresholds(blur_min=50, bright_min=10, bright_max=245)
    res = assess(img, thr)
    assert res["is_ok"] is True
    assert res["blur_score"] > 50

def test_dark_or_blurry_rejected():
    img = np.full((256,256), 5, dtype=np.uint8)
    thr = Thresholds(blur_min=50, bright_min=10, bright_max=245)
    res = assess(img, thr)
    assert res["is_ok"] is False
