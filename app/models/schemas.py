from typing import Literal

from pydantic import BaseModel


class QualityResponse(BaseModel):
    blur_score: float
    brightness: float
    width: int
    height: int
    is_ok: bool
    thresholds: dict[str, float | int]


# Diagnostics (POST /quality/diagnostics, GET /quality/config).
# Field categories are documented in the README "Diagnostics" section.


class SettingValue(BaseModel):
    value: float | int | None
    source: Literal["default", "env", "dotenv", "init", "derived", "pillow"]


class EffectiveSettings(BaseModel):
    blur_min: SettingValue
    bright_min: SettingValue
    bright_max: SettingValue
    resize_max_dim: SettingValue
    max_upload_mb: SettingValue
    max_upload_bytes: SettingValue
    max_request_bytes: SettingValue
    max_image_pixels: SettingValue
    pillow_max_image_pixels: SettingValue


class Versions(BaseModel):
    app: str
    python: str
    numpy: str
    opencv: str
    pillow: str
    libjpeg: str | None
    libjpeg_turbo: str | None


class Measurements(BaseModel):
    blur_score: float
    brightness: float
    brightness_std: float
    gray_min: int
    gray_max: int
    laplacian_mean: float
    laplacian_min: float
    laplacian_max: float
    pixel_count: int


class CheckResult(BaseModel):
    name: str
    measurement: str
    comparator: Literal[">=", "<="]
    threshold: float | int
    value: float
    passed: bool
    margin: float


class Decision(BaseModel):
    rule: str
    combine: Literal["all"]
    checks: list[CheckResult]
    is_ok: bool


class ImageMetadata(BaseModel):
    upload_bytes: int
    content_type: str
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
    processed_width: int
    processed_height: int


class StageTiming(BaseModel):
    name: str
    offset_ns: int
    duration_ns: int


class Timings(BaseModel):
    clock: Literal["perf_counter_ns"]
    stages: list[StageTiming]
    stages_sum_ns: int
    handler_total_ns: int


class LaplacianHistogram(BaseModel):
    bins: int
    bin_edges: list[float]
    counts: list[int]


class Diagnostics(BaseModel):
    schema_version: Literal[1]
    request_id: str | None
    measurements: Measurements
    decision: Decision
    settings: EffectiveSettings
    image: ImageMetadata
    timings: Timings
    versions: Versions
    laplacian_histogram: LaplacianHistogram | None


class QualityDiagnosticsResponse(QualityResponse):
    diagnostics: Diagnostics


class QualityConfigResponse(BaseModel):
    schema_version: Literal[1]
    decision_rule: str
    settings: EffectiveSettings
    versions: Versions
