"""
The FastAPI-free pipeline behind POST /quality and POST /quality/diagnostics.
The routes keep only the HTTP parts (reading the upload, headers, logging);
everything from the size check to the response body happens here. It is also
called in-process by tests/parity and by the portfolio's browser demo
(Pyodide), so all three run the same code.
"""

from dataclasses import dataclass

import numpy as np

from app.core.config import Settings
from app.core.timing import Stage, StageTimer
from app.models.schemas import Decision, QualityDiagnosticsResponse
from app.services import diagnostics, quality
from app.services.preprocess import ImageInfo, load_and_normalize_traced, to_gray
from app.services.quality import Assessment, Thresholds, assess_traced


class PipelineError(Exception):
    """A rejected upload: the HTTP status and detail the service answers with."""

    status: int
    detail: str


class UnsupportedMediaType(PipelineError):
    status = 415
    detail = "Unsupported media type"


class UploadTooLarge(PipelineError):
    status = 413
    detail = "File too large"


class InvalidImage(PipelineError):
    """The bytes could not be turned into an image; __cause__ says why."""

    status = 400
    detail = "Invalid image data"


@dataclass(frozen=True)
class Run:
    content_type: str
    upload_bytes: int
    image: ImageInfo
    gray: np.ndarray
    result: Assessment


def check_content_type(content_type: str | None) -> None:
    if not content_type or not content_type.startswith("image/"):
        raise UnsupportedMediaType


def read_upload(
    data: bytes | bytearray | memoryview, s: Settings, timer: StageTimer
) -> bytes:
    """
    The in-memory counterpart of the route's await file.read(max_upload_bytes + 1):
    copies at most one byte past the limit, timed as the same "read_upload" stage.
    """
    with timer.stage("read_upload"):
        return bytes(memoryview(data)[: s.max_upload_bytes + 1])


def run(
    content: bytes,
    content_type: str,
    s: Settings,
    timer: StageTimer,
    keep_laplacian: bool = False,
) -> Run:
    """Size check, decode, grayscale and decision for one upload's bytes."""
    if len(content) > s.max_upload_bytes:
        raise UploadTooLarge
    try:
        bgr, info = load_and_normalize_traced(
            content, s.resize_max_dim, timer, s.max_image_pixels
        )
    except Exception as e:
        raise InvalidImage from e
    with timer.stage("grayscale"):
        gray = to_gray(bgr)
    result = assess_traced(
        gray, Thresholds(s.blur_min, s.bright_min, s.bright_max), timer, keep_laplacian
    )
    return Run(content_type, len(content), info, gray, result)


def quality_fields(run: Run, s: Settings) -> dict:
    return dict(
        blur_score=run.result.blur_score,
        brightness=run.result.brightness,
        is_ok=run.result.is_ok,
        width=run.image.width,
        height=run.image.height,
        thresholds={
            "blur_min": s.blur_min,
            "bright_min": s.bright_min,
            "bright_max": s.bright_max,
        },
    )


def diagnostics_response(
    run: Run,
    s: Settings,
    timer: StageTimer,
    bins: int | None,
    request_id: str | None,
) -> tuple[QualityDiagnosticsResponse, list[Stage], int]:
    """
    The /quality/diagnostics body for a run kept with its Laplacian, plus the
    stages and total time the route logs and puts in Server-Timing.
    """
    lap = run.result.laplacian
    assert lap is not None  # run with keep_laplacian=True
    with timer.stage("extra_statistics"):
        measured = diagnostics.measurements(run.result, run.gray, lap)
    histogram = None
    if bins is not None:
        with timer.stage("laplacian_histogram"):
            histogram = diagnostics.laplacian_histogram(lap, bins)
    total_ns = timer.elapsed_ns()
    stages = timer.stages
    response = QualityDiagnosticsResponse(
        **quality_fields(run, s),
        diagnostics=diagnostics.build(
            s=s,
            request_id=request_id,
            result=run.result,
            measured=measured,
            info=run.image,
            upload_bytes=run.upload_bytes,
            content_type=run.content_type,
            stages=stages,
            handler_total_ns=total_ns,
            histogram=histogram,
        ),
    )
    return response, stages, total_ns


def diagnose_bytes(
    data: bytes | bytearray | memoryview,
    content_type: str,
    s: Settings,
    bins: int | None = None,
    request_id: str | None = None,
) -> QualityDiagnosticsResponse:
    """POST /quality/diagnostics for bytes already in memory, same stages."""
    timer = StageTimer()
    check_content_type(content_type)
    content = read_upload(data, s, timer)
    result = run(content, content_type, s, timer, keep_laplacian=True)
    response, _, _ = diagnostics_response(result, s, timer, bins, request_id)
    return response


def outcome(
    data: bytes | bytearray | memoryview,
    content_type: str,
    s: Settings,
    bins: int | None = None,
) -> dict[str, object]:
    """
    What the service would answer, without raising for a rejected upload:
    {"status": 200, "response_json": ...} or {"status", "detail", "cause"},
    where cause is the class name of the error behind a 400 (never its message).
    """
    try:
        response = diagnose_bytes(data, content_type, s, bins)
    except PipelineError as e:
        cause = type(e.__cause__).__name__ if e.__cause__ is not None else None
        return {"status": e.status, "detail": e.detail, "cause": cause}
    return {"status": 200, "response_json": response.model_dump_json()}


def reevaluate(blur_score: float, brightness: float, s: Settings) -> Decision:
    """The decision the service would make for these scores under s's thresholds."""
    checks = quality.evaluate(
        blur_score, brightness, Thresholds(s.blur_min, s.bright_min, s.bright_max)
    )
    return diagnostics.decision(
        Assessment(blur_score, brightness, checks, quality.combine(checks), None)
    )
