from pydantic import BaseModel


class QualityResponse(BaseModel):
    blur_score: float
    brightness: float
    width: int
    height: int
    is_ok: bool
    thresholds: dict[str, float | int]
