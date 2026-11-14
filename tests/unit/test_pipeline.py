import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.timing import StageTimer
from app.main import app
from app.services import diagnostics, pipeline

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "tests" / "assets"
SHARP = (ASSETS / "sharp_small.jpg").read_bytes()
TRUNCATED = SHARP[: len(SHARP) // 2]
STAGES = [
    "read_upload",
    "open",
    "decode",
    "exif_transpose",
    "convert_rgb",
    "to_array",
    "rgb_to_bgr",
    "grayscale",
    "laplacian",
    "statistics",
    "decision",
    "extra_statistics",
]


def over_limit_png(make_image, claim_dimensions) -> bytes:
    # 6000 x 6000 = 36 MP claimed in the header, over the 25 MP default.
    return claim_dimensions(make_image(16, 16, "RGB", "PNG"), 6000, 6000)


def post(client, data, content_type="image/jpeg", **kw):
    files = {"file": ("upload", data, content_type)}
    return client.post("/quality/diagnostics", files=files, **kw)


def without_run_specifics(body: dict) -> dict:
    # Timings and the request ID differ between any two runs.
    body = json.loads(json.dumps(body))
    del body["diagnostics"]["timings"], body["diagnostics"]["request_id"]
    return body


def stage_names(body: dict) -> list[str]:
    return [s["name"] for s in body["diagnostics"]["timings"]["stages"]]


# --- Rejections ---------------------------------------------------------------


@pytest.mark.parametrize("content_type", [None, "", "text/plain"])
def test_check_content_type_rejects_non_images(content_type):
    with pytest.raises(pipeline.UnsupportedMediaType) as caught:
        pipeline.check_content_type(content_type)
    assert (caught.value.status, caught.value.detail) == (
        415,
        "Unsupported media type",
    )


def test_check_content_type_accepts_images():
    assert pipeline.check_content_type("image/png") is None


def test_run_rejections_carry_the_route_answer_and_the_cause(
    settings_env, make_image, claim_dimensions
):
    s = settings_env()
    oversized = b"\x00" * (s.max_upload_bytes + 1)
    with pytest.raises(pipeline.UploadTooLarge) as caught:
        pipeline.run(oversized, "image/jpeg", s, StageTimer())
    assert (caught.value.status, caught.value.detail) == (413, "File too large")

    with pytest.raises(pipeline.InvalidImage) as caught:
        pipeline.run(TRUNCATED, "image/jpeg", s, StageTimer())
    assert (caught.value.status, caught.value.detail) == (400, "Invalid image data")
    assert isinstance(caught.value.__cause__, OSError)

    png = over_limit_png(make_image, claim_dimensions)
    with pytest.raises(pipeline.InvalidImage) as caught:
        pipeline.run(png, "image/png", s, StageTimer())
    assert type(caught.value.__cause__).__name__ == "ImageTooLarge"


def test_read_upload_copies_at_most_one_byte_past_the_limit(settings_env):
    s = settings_env(env={"MAX_UPLOAD_MB": "1"})
    for data in (bytearray(SHARP), memoryview(SHARP)):
        timer = StageTimer()
        assert pipeline.read_upload(data, s, timer) == SHARP
        assert [st.name for st in timer.stages] == ["read_upload"]
    big = bytearray(s.max_upload_bytes + 10)
    copied = pipeline.read_upload(big, s, StageTimer())
    assert type(copied) is bytes and len(copied) == s.max_upload_bytes + 1


# --- Agreement with the API ---------------------------------------------------


def test_diagnose_bytes_gives_the_route_response(settings_env):
    s = settings_env()
    client = TestClient(app)
    for bins, extra in ((None, []), (64, ["laplacian_histogram"])):
        params = {} if bins is None else {"laplacian_histogram_bins": bins}
        route = post(client, SHARP, params=params).json()
        ours = json.loads(
            pipeline.diagnose_bytes(SHARP, "image/jpeg", s, bins).model_dump_json()
        )
        assert without_run_specifics(ours) == without_run_specifics(route)
        assert stage_names(ours) == stage_names(route) == STAGES + extra


def test_outcome_gives_the_route_answer_for_every_input(
    settings_env, make_image, claim_dimensions
):
    s = settings_env(env={"MAX_UPLOAD_MB": "1"})
    client = TestClient(app)
    cases = [
        (SHARP, "image/jpeg", None),
        (SHARP, "text/plain", None),
        (b"\x00" * (1024 * 1024 + 1), "image/jpeg", None),
        (TRUNCATED, "image/jpeg", "OSError"),
        (over_limit_png(make_image, claim_dimensions), "image/png", "ImageTooLarge"),
        (b"not an image", "image/jpeg", "UnidentifiedImageError"),
    ]
    for data, content_type, cause in cases:
        resp = post(client, data, content_type)
        out = pipeline.outcome(data, content_type, s)
        assert out["status"] == resp.status_code, content_type
        if resp.status_code == 200:
            assert set(out) == {"status", "response_json"}
            ours = json.loads(str(out["response_json"]))
            assert without_run_specifics(ours) == without_run_specifics(resp.json())
        else:
            assert out == {
                "status": resp.status_code,
                "detail": resp.json()["detail"],
                "cause": cause,
            }


def test_outcome_reports_a_memory_error_while_decoding(settings_env, monkeypatch):
    s = settings_env()

    def out_of_memory(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(pipeline, "load_and_normalize_traced", out_of_memory)
    resp = post(TestClient(app), SHARP)
    assert (resp.status_code, resp.json()) == (400, {"detail": "Invalid image data"})
    assert pipeline.outcome(SHARP, "image/jpeg", s) == {
        "status": 400,
        "detail": "Invalid image data",
        "cause": "MemoryError",
    }


# --- Threshold re-evaluation --------------------------------------------------


def test_reevaluate_is_the_upstream_decision(settings_env):
    s = settings_env()
    result = pipeline.run(SHARP, "image/jpeg", s, StageTimer()).result
    decision = pipeline.reevaluate(result.blur_score, result.brightness, s)
    assert decision == diagnostics.decision(result)

    strict = pipeline.reevaluate(
        result.blur_score, result.brightness, Settings(blur_min=1e9)
    )
    assert strict.is_ok is False and strict.checks[0].passed is False


def test_reevaluate_values_at_the_thresholds_pass(settings_env):
    s = settings_env()
    for brightness in (float(s.bright_min), float(s.bright_max)):
        decision = pipeline.reevaluate(s.blur_min, brightness, s)
        assert [c.passed for c in decision.checks] == [True, True, True]
        assert decision.is_ok is True
    assert decision.checks[0].margin == 0.0


# --- Isolation from the web framework -----------------------------------------


def test_pipeline_imports_no_web_framework():
    code = (
        "import sys, app.services.pipeline; print(sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('fastapi', 'starlette') or m.startswith('app.api')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"
