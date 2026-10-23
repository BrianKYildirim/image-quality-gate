import logging
from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request, Response
from app.core import config
from app.core.config import Settings
from app.core.timing import Stage, StageTimer
from app.services import diagnostics
from app.services.preprocess import ImageInfo, load_and_normalize_traced, to_gray
from app.services.quality import Assessment, assess_traced, Thresholds
from app.models.schemas import (
    QualityConfigResponse,
    QualityDiagnosticsResponse,
    QualityResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Run:
    content_type: str
    upload_bytes: int
    image: ImageInfo
    gray: np.ndarray
    result: Assessment


async def _run_quality(
    file: UploadFile, s: Settings, timer: StageTimer, keep_laplacian: bool = False
) -> _Run:
    """The pipeline behind both /quality and /quality/diagnostics."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type")
    byte_limit = s.max_upload_bytes
    with timer.stage("read_upload"):
        # One byte past the limit is enough to detect an oversized upload
        # without pulling all of it into memory.
        content = await file.read(byte_limit + 1)
    if len(content) > byte_limit:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        bgr, info = load_and_normalize_traced(
            content, s.resize_max_dim, timer, s.max_image_pixels
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")
    with timer.stage("grayscale"):
        gray = to_gray(bgr)
    result = assess_traced(
        gray, Thresholds(s.blur_min, s.bright_min, s.bright_max), timer, keep_laplacian
    )
    return _Run(file.content_type, len(content), info, gray, result)


def _quality_fields(run: _Run, s: Settings) -> dict:
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


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _log_timings(
    request: Request, route: str, run: _Run, stages: list[Stage], total_ns: int
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    rid = _request_id(request)
    logger.debug(
        "quality timings",
        extra={
            "extra": {
                "request_id": rid[:128] if rid else None,
                "route": route,
                "is_ok": run.result.is_ok,
                "handler_total_ns": total_ns,
                "stages_ns": {s.name: s.duration_ns for s in stages},
            }
        },
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/quality", response_model=QualityResponse)
async def quality(request: Request, file: UploadFile = File(...)) -> QualityResponse:
    timer = StageTimer()
    s = config.settings
    run = await _run_quality(file, s, timer)
    if logger.isEnabledFor(logging.DEBUG):
        _log_timings(request, "/quality", run, timer.stages, timer.elapsed_ns())
    return QualityResponse(**_quality_fields(run, s))


@router.post("/quality/diagnostics", response_model=QualityDiagnosticsResponse)
async def quality_diagnostics(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    laplacian_histogram_bins: int | None = Query(
        None,
        ge=2,
        le=256,
        description="Include a histogram of the Laplacian with this many bins.",
    ),
) -> QualityDiagnosticsResponse:
    timer = StageTimer()
    s = config.settings
    run = await _run_quality(file, s, timer, keep_laplacian=True)
    lap = run.result.laplacian
    assert lap is not None  # kept above
    with timer.stage("extra_statistics"):
        measured = diagnostics.measurements(run.result, run.gray, lap)
    histogram = None
    if laplacian_histogram_bins is not None:
        with timer.stage("laplacian_histogram"):
            histogram = diagnostics.laplacian_histogram(lap, laplacian_histogram_bins)
    total_ns = timer.elapsed_ns()
    stages = timer.stages
    _log_timings(request, "/quality/diagnostics", run, stages, total_ns)
    response.headers["Server-Timing"] = diagnostics.server_timing(stages, total_ns)
    return QualityDiagnosticsResponse(
        **_quality_fields(run, s),
        diagnostics=diagnostics.build(
            s=s,
            request_id=_request_id(request),
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


@router.get("/quality/config", response_model=QualityConfigResponse)
def quality_config() -> QualityConfigResponse:
    return diagnostics.config(config.settings)
