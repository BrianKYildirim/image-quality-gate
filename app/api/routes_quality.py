import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request, Response
from app.core import config
from app.core.config import Settings
from app.core.timing import Stage, StageTimer
from app.services import diagnostics, pipeline
from app.models.schemas import (
    QualityConfigResponse,
    QualityDiagnosticsResponse,
    QualityResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


async def _read_upload(
    file: UploadFile, s: Settings, timer: StageTimer
) -> tuple[bytes, str]:
    """The HTTP half of the pipeline (app/services/pipeline.py does the rest)."""
    pipeline.check_content_type(file.content_type)
    assert file.content_type is not None  # checked above
    with timer.stage("read_upload"):
        # One byte past the limit is enough to detect an oversized upload
        # without pulling all of it into memory.
        content = await file.read(s.max_upload_bytes + 1)
    return content, file.content_type


def _http_error(e: pipeline.PipelineError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=e.detail)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _log_timings(
    request: Request, route: str, run: pipeline.Run, stages: list[Stage], total_ns: int
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
    try:
        content, content_type = await _read_upload(file, s, timer)
        run = pipeline.run(content, content_type, s, timer)
    except pipeline.PipelineError as e:
        raise _http_error(e) from e
    if logger.isEnabledFor(logging.DEBUG):
        _log_timings(request, "/quality", run, timer.stages, timer.elapsed_ns())
    return QualityResponse(**pipeline.quality_fields(run, s))


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
    try:
        content, content_type = await _read_upload(file, s, timer)
        run = pipeline.run(content, content_type, s, timer, keep_laplacian=True)
        body, stages, total_ns = pipeline.diagnostics_response(
            run, s, timer, laplacian_histogram_bins, _request_id(request)
        )
    except pipeline.PipelineError as e:
        raise _http_error(e) from e
    _log_timings(request, "/quality/diagnostics", run, stages, total_ns)
    response.headers["Server-Timing"] = diagnostics.server_timing(stages, total_ns)
    return body


@router.get("/quality/config", response_model=QualityConfigResponse)
def quality_config() -> QualityConfigResponse:
    return diagnostics.config(config.settings)
