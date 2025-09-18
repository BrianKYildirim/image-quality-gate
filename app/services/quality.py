import cv2
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    blur_min: float
    bright_min: int
    bright_max: int


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def assess(gray: np.ndarray, thr: Thresholds) -> dict:
    blur_score = variance_of_laplacian(gray)
    bright = brightness(gray)
    is_ok = (blur_score >= thr.blur_min) and (
        thr.bright_min <= bright <= thr.bright_max
    )
    return {
        "blur_score": blur_score,
        "brightness": bright,
        "is_ok": is_ok,
    }
