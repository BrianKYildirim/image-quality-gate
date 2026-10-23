"""
Assembles quality diagnostics from the values one pipeline run produced.
Nothing here re-runs the pipeline; extra statistics and the histogram are
computed from the same gray and Laplacian arrays the decision used.
"""

import platform
from typing import Any

import cv2
import numpy as np
import PIL
from PIL import Image, features

from app.core.config import Settings
from app.core.timing import Stage
from app.models.schemas import (
    CheckResult,
    Decision,
    Diagnostics,
    EffectiveSettings,
    ImageMetadata,
    LaplacianHistogram,
    Measurements,
    QualityConfigResponse,
    SettingValue,
    StageTiming,
    Timings,
    Versions,
)
from app.services.preprocess import ImageInfo
from app.services.quality import Assessment, Check

DECISION_RULE = (
    "is_ok = (blur_score >= blur_min) AND (bright_min <= brightness) "
    "AND (brightness <= bright_max)"
)


def _feature_version(name: str) -> str | None:
    try:
        return features.version(name)
    except ValueError:  # pragma: no cover - feature unknown to this Pillow
        return None


# Versions of the libraries that produce the numbers, as loaded in this process.
_LIBRARY_VERSIONS: dict[str, Any] = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "opencv": cv2.__version__,
    "pillow": PIL.__version__,
    "libjpeg": _feature_version("jpg"),
    "libjpeg_turbo": _feature_version("libjpeg_turbo"),
}


def versions(s: Settings) -> Versions:
    return Versions(app=s.app_version, **_LIBRARY_VERSIONS)


def effective_settings(s: Settings) -> EffectiveSettings:
    def configured(name: str) -> SettingValue:
        return SettingValue(value=getattr(s, name), source=s.source_of(name))

    return EffectiveSettings(
        blur_min=configured("blur_min"),
        bright_min=configured("bright_min"),
        bright_max=configured("bright_max"),
        resize_max_dim=configured("resize_max_dim"),
        max_upload_mb=configured("max_upload_mb"),
        max_upload_bytes=SettingValue(value=s.max_upload_bytes, source="derived"),
        max_request_bytes=SettingValue(value=s.max_request_bytes, source="derived"),
        max_image_pixels=configured("max_image_pixels"),
        pillow_max_image_pixels=SettingValue(
            value=Image.MAX_IMAGE_PIXELS, source="pillow"
        ),
    )


def config(s: Settings) -> QualityConfigResponse:
    return QualityConfigResponse(
        schema_version=1,
        decision_rule=DECISION_RULE,
        settings=effective_settings(s),
        versions=versions(s),
    )


def measurements(result: Assessment, gray: np.ndarray, lap: np.ndarray) -> Measurements:
    return Measurements(
        blur_score=result.blur_score,
        brightness=result.brightness,
        brightness_std=float(gray.std()),
        gray_min=int(gray.min()),
        gray_max=int(gray.max()),
        laplacian_mean=float(lap.mean()),
        laplacian_min=float(lap.min()),
        laplacian_max=float(lap.max()),
        pixel_count=int(gray.size),
    )


def laplacian_histogram(lap: np.ndarray, bins: int) -> LaplacianHistogram:
    # Signed values, no clipping or scaling; equal-width bins over [min, max].
    counts, edges = np.histogram(lap, bins=bins)
    return LaplacianHistogram(
        bins=bins, bin_edges=edges.tolist(), counts=counts.tolist()
    )


def _check_result(check: Check) -> CheckResult:
    # Distance from the threshold on the passing side; >= 0 exactly when passed.
    if check.comparator == ">=":
        margin = check.value - check.threshold
    else:
        margin = check.threshold - check.value
    return CheckResult(
        name=check.name,
        measurement=check.measurement,
        comparator=check.comparator,
        threshold=check.threshold,
        value=check.value,
        passed=check.passed,
        margin=margin,
    )


def decision(result: Assessment) -> Decision:
    return Decision(
        rule=DECISION_RULE,
        combine="all",
        checks=[_check_result(check) for check in result.checks],
        is_ok=result.is_ok,
    )


def image_metadata(
    info: ImageInfo, upload_bytes: int, content_type: str
) -> ImageMetadata:
    return ImageMetadata(
        upload_bytes=upload_bytes,
        content_type=content_type,
        format=info.format,
        mode=info.mode,
        original_width=info.original_width,
        original_height=info.original_height,
        exif_orientation=info.exif_orientation,
        orientation_applied=info.orientation_applied,
        oriented_width=info.oriented_width,
        oriented_height=info.oriented_height,
        resized=info.resized,
        scale=info.scale,
        processed_width=info.width,
        processed_height=info.height,
    )


def timings(stages: list[Stage], handler_total_ns: int) -> Timings:
    return Timings(
        clock="perf_counter_ns",
        stages=[
            StageTiming(name=s.name, offset_ns=s.offset_ns, duration_ns=s.duration_ns)
            for s in stages
        ],
        stages_sum_ns=sum(s.duration_ns for s in stages),
        handler_total_ns=handler_total_ns,
    )


def server_timing(stages: list[Stage], handler_total_ns: int) -> str:
    # Server-Timing durations are in milliseconds.
    entries = [f"{s.name};dur={s.duration_ns / 1e6:.3f}" for s in stages]
    entries.append(f"handler_total;dur={handler_total_ns / 1e6:.3f}")
    return ", ".join(entries)


def build(
    *,
    s: Settings,
    request_id: str | None,
    result: Assessment,
    measured: Measurements,
    info: ImageInfo,
    upload_bytes: int,
    content_type: str,
    stages: list[Stage],
    handler_total_ns: int,
    histogram: LaplacianHistogram | None,
) -> Diagnostics:
    return Diagnostics(
        schema_version=1,
        request_id=request_id,
        measurements=measured,
        decision=decision(result),
        settings=effective_settings(s),
        image=image_metadata(info, upload_bytes, content_type),
        timings=timings(stages, handler_total_ns),
        versions=versions(s),
        laplacian_histogram=histogram,
    )
