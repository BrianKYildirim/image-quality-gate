from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "image-quality-gate"
    app_version: str = "0.0.0"
    blur_min: float = Field(160.0, ge=0)
    bright_min: int = Field(40, ge=0, le=255)
    bright_max: int = Field(180, ge=0, le=255)
    resize_max_dim: int = Field(1600, ge=256)
    max_upload_mb: int = Field(6, ge=1, le=25)
    log_level: str = "INFO"
    log_json: bool = True
    prometheus_enabled: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
