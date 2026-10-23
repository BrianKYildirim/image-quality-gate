import os
from typing import Any, Literal

from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings

SettingSource = Literal["default", "env", "dotenv", "init"]

# Room for the multipart framing around the uploaded file (boundaries, part
# headers, file name) when limiting the whole request body.
MULTIPART_ALLOWANCE_BYTES = 64 * 1024


class Settings(BaseSettings):
    app_name: str = "image-quality-gate"
    app_version: str = "0.0.0"
    blur_min: float = Field(140.0, ge=0)
    bright_min: int = Field(20, ge=0, le=255)
    bright_max: int = Field(235, ge=0, le=255)
    resize_max_dim: int = Field(1600, ge=256)
    max_upload_mb: int = Field(6, ge=1, le=25)
    max_image_pixels: int = Field(25_000_000, ge=1)
    log_level: str = "INFO"
    log_json: bool = True
    prometheus_enabled: bool = True

    _sources: dict[str, SettingSource] = PrivateAttr(default_factory=dict)

    class Config:
        env_file = ".env"

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        # Record where each value came from at load time, following
        # pydantic-settings' precedence (init kwargs > environment > .env >
        # default) and its case-insensitive environment lookup. Fields not
        # explicitly set kept their default; set fields not passed in or found
        # in the environment can only have come from the .env file.
        init_keys = {key for key in values if not key.startswith("_")}
        env_keys = {key.lower() for key in os.environ}
        for name in type(self).model_fields:
            if name not in self.model_fields_set:
                self._sources[name] = "default"
            elif name in init_keys:
                self._sources[name] = "init"
            elif name in env_keys:
                self._sources[name] = "env"
            else:
                self._sources[name] = "dotenv"

    def source_of(self, name: str) -> SettingSource:
        return self._sources[name]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_request_bytes(self) -> int:
        return self.max_upload_bytes + MULTIPART_ALLOWANCE_BYTES


# mypy can misinterpret BaseSettings __init__ signature; ignore that call-arg check.
settings: Settings = Settings()  # type: ignore[call-arg]
