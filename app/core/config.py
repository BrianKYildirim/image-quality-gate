from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "image-quality-gate"
    app_version: str = "0.0.0"
    blur_min: float = Field(140.0, ge=0)
    bright_min: int = Field(20, ge=0, le=255)
    bright_max: int = Field(235, ge=0, le=255)
    resize_max_dim: int = Field(1600, ge=256)
    max_upload_mb: int = Field(6, ge=1, le=25)
    log_level: str = "INFO"
    log_json: bool = True
    prometheus_enabled: bool = True

    class Config:
        env_file = ".env"


# mypy can misinterpret BaseSettings __init__ signature; ignore that call-arg check.
settings: Settings = Settings()  # type: ignore[call-arg]
