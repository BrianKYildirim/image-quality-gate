import cv2
import numpy as np
from dataclasses import dataclass
from typing import Literal

from app.core.timing import StageTimer


@dataclass(frozen=True)
class Thresholds:
    blur_min: float
    bright_min: int
    bright_max: int


@dataclass(frozen=True)
class Check:
    """One threshold comparison; is_ok is the conjunction of all checks."""

    name: str
    measurement: str
    comparator: Literal[">=", "<="]
    threshold: float
    value: float
    passed: bool


@dataclass(frozen=True)
class Assessment:
    blur_score: float
    brightness: float
    checks: tuple[Check, ...]
    is_ok: bool
    laplacian: np.ndarray | None  # the array blur_score is the variance of, if kept


def laplacian(gray: np.ndarray) -> np.ndarray:
    return cv2.Laplacian(gray, cv2.CV_64F)


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(laplacian(gray).var())


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def evaluate(blur_score: float, bright: float, thr: Thresholds) -> tuple[Check, ...]:
    return (
        Check(
            "blur_min",
            "blur_score",
            ">=",
            thr.blur_min,
            blur_score,
            blur_score >= thr.blur_min,
        ),
        Check(
            "bright_min",
            "brightness",
            ">=",
            thr.bright_min,
            bright,
            thr.bright_min <= bright,
        ),
        Check(
            "bright_max",
            "brightness",
            "<=",
            thr.bright_max,
            bright,
            bright <= thr.bright_max,
        ),
    )


def assess_traced(
    gray: np.ndarray, thr: Thresholds, timer: StageTimer, keep_laplacian: bool = False
) -> Assessment:
    with timer.stage("laplacian"):
        lap = laplacian(gray)
    with timer.stage("statistics"):
        blur_score = float(lap.var())
        kept = lap if keep_laplacian else None
        # Release the (large, float64) array as soon as the original code did:
        # keeping it longer changes how memory is reused and measurably slows
        # the next allocations.
        del lap
        bright = brightness(gray)
    with timer.stage("decision"):
        checks = evaluate(blur_score, bright, thr)
        is_ok = all(check.passed for check in checks)
    return Assessment(blur_score, bright, checks, is_ok, kept)


def assess(gray: np.ndarray, thr: Thresholds) -> dict:
    result = assess_traced(gray, thr, StageTimer())
    return {
        "blur_score": result.blur_score,
        "brightness": result.brightness,
        "is_ok": result.is_ok,
    }
