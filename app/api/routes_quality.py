from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.config import settings
from app.services.preprocess import load_and_normalize, to_gray
from app.services.quality import assess, Thresholds
from app.models.schemas import QualityResponse

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/quality", response_model=QualityResponse)
async def quality(file: UploadFile = File(...)) -> QualityResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type")
    content = await file.read()
    byte_limit = settings.max_upload_mb * 1024 * 1024
    if len(content) > byte_limit:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        bgr, w, h = load_and_normalize(content, settings.resize_max_dim)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")
    gray = to_gray(bgr)
    res = assess(
        gray, Thresholds(settings.blur_min, settings.bright_min, settings.bright_max)
    )
    return QualityResponse(
        **res,
        width=w,
        height=h,
        thresholds={
            "blur_min": settings.blur_min,
            "bright_min": settings.bright_min,
            "bright_max": settings.bright_max,
        }
    )
