from app.core.config import Settings

THRESHOLDS = ("blur_min", "bright_min", "bright_max")


def test_defaults_are_reported_as_defaults(settings_env):
    s = settings_env()
    assert (s.blur_min, s.bright_min, s.bright_max) == (140.0, 20, 235)
    assert {s.source_of(name) for name in Settings.model_fields} == {"default"}


def test_environment_overrides_are_reported(settings_env):
    s = settings_env(env={"BLUR_MIN": "55.5", "bright_max": "200"})
    assert (s.blur_min, s.source_of("blur_min")) == (55.5, "env")
    # pydantic-settings matches variable names case-insensitively.
    assert (s.bright_max, s.source_of("bright_max")) == (200, "env")
    assert s.source_of("bright_min") == "default"


def test_dotenv_values_are_reported(settings_env):
    s = settings_env(dotenv="BRIGHT_MIN=12\nRESIZE_MAX_DIM=800\n")
    assert (s.bright_min, s.source_of("bright_min")) == (12, "dotenv")
    assert (s.resize_max_dim, s.source_of("resize_max_dim")) == (800, "dotenv")
    assert s.source_of("blur_min") == "default"


def test_environment_wins_over_dotenv(settings_env):
    s = settings_env(env={"BLUR_MIN": "70"}, dotenv="BLUR_MIN=90\nBRIGHT_MIN=10\n")
    assert (s.blur_min, s.source_of("blur_min")) == (70.0, "env")
    assert (s.bright_min, s.source_of("bright_min")) == (10, "dotenv")


def test_value_equal_to_default_still_reports_its_source(settings_env):
    s = settings_env(env={"BLUR_MIN": "140"})
    assert (s.blur_min, s.source_of("blur_min")) == (140.0, "env")


def test_init_arguments_are_reported(settings_env):
    settings_env(env={"BLUR_MIN": "70"})
    s = Settings(blur_min=1.0)
    assert (s.blur_min, s.source_of("blur_min")) == (1.0, "init")


def test_sources_are_fixed_when_settings_load(settings_env, monkeypatch):
    s = settings_env()
    monkeypatch.setenv("BLUR_MIN", "5")
    assert (s.blur_min, s.source_of("blur_min")) == (140.0, "default")


def test_max_upload_bytes(settings_env):
    assert settings_env(env={"MAX_UPLOAD_MB": "2"}).max_upload_bytes == 2 * 1024 * 1024
