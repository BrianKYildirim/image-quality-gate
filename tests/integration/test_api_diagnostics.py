import logging
import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import PIL
import pytest
import starlette.datastructures
from fastapi.testclient import TestClient

from app.api import routes_quality
from app.main import app
from app.services.preprocess import load_and_normalize, to_gray

ASSETS = Path(__file__).parent.parent / "assets"
SHARP = (ASSETS / "sharp_small.jpg").read_bytes()
BLURRY = (ASSETS / "blurry_small.jpg").read_bytes()
SETTINGS_KEYS = {
    "blur_min",
    "bright_min",
    "bright_max",
    "resize_max_dim",
    "max_upload_mb",
    "max_upload_bytes",
    "max_request_bytes",
    "max_image_pixels",
    "pillow_max_image_pixels",
}
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


@pytest.fixture
def client(settings_env):
    settings_env()
    return TestClient(app)


def post(
    client, data=SHARP, path="/quality/diagnostics", content_type="image/jpeg", **kw
):
    return client.post(path, files={"file": ("upload.jpg", data, content_type)}, **kw)


def diagnose(client, data=SHARP, **kw):
    resp = post(client, data, **kw)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body, body.pop("diagnostics")


# --- Equality with the default response -------------------------------------


def test_default_response_has_no_diagnostics(client):
    resp = post(client, path="/quality")
    assert "diagnostics" not in resp.json()
    assert "server-timing" not in resp.headers


def test_diagnostics_add_to_the_default_response_without_changing_it(
    client, make_image
):
    images = [SHARP, BLURRY, make_image(300, 200, orientation=6)]
    images.append(make_image(2000, 1000, "RGB", "PNG"))
    for data in images:
        plain = post(client, data, path="/quality").json()
        body, _ = diagnose(client, data)
        assert list(body) == list(plain)
        assert body == plain
        with_hist, _ = diagnose(client, data, params={"laplacian_histogram_bins": 64})
        assert with_hist == plain


def test_diagnostic_values_are_the_values_the_decision_used(client):
    for data in (SHARP, BLURRY):
        body, diag = diagnose(client, data)
        m, decision = diag["measurements"], diag["decision"]
        assert m["blur_score"] == body["blur_score"]
        assert m["brightness"] == body["brightness"]
        blur, bright_min, bright_max = decision["checks"]
        assert (blur["name"], blur["comparator"]) == ("blur_min", ">=")
        assert (bright_min["name"], bright_min["comparator"]) == ("bright_min", ">=")
        assert (bright_max["name"], bright_max["comparator"]) == ("bright_max", "<=")
        assert blur["value"] == body["blur_score"]
        assert bright_min["value"] == bright_max["value"] == body["brightness"]
        assert {c["name"]: c["threshold"] for c in decision["checks"]} == body[
            "thresholds"
        ]
        for check in decision["checks"]:
            sign = 1 if check["comparator"] == ">=" else -1
            assert check["margin"] == sign * (check["value"] - check["threshold"])
            assert check["passed"] is (check["margin"] >= 0)
        assert decision["combine"] == "all"
        assert decision["is_ok"] is body["is_ok"]
        assert body["is_ok"] is all(c["passed"] for c in decision["checks"])
    assert diagnose(client, BLURRY)[1]["decision"]["checks"][0]["passed"] is False


def test_extra_measurements_describe_the_scored_arrays(client, make_image):
    data = make_image(500, 300, orientation=8)
    _, diag = diagnose(client, data, params={"laplacian_histogram_bins": 32})
    m = diag["measurements"]
    bgr, _, _ = load_and_normalize(data, 1600)
    gray = to_gray(bgr)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    assert m["blur_score"] == float(lap.var())
    assert m["brightness_std"] == float(gray.std())
    assert (m["gray_min"], m["gray_max"]) == (int(gray.min()), int(gray.max()))
    assert m["laplacian_mean"] == float(lap.mean())
    assert (m["laplacian_min"], m["laplacian_max"]) == (lap.min(), lap.max())
    assert m["pixel_count"] == gray.size == 300 * 500

    hist = diag["laplacian_histogram"]
    counts, edges = np.histogram(lap, bins=32)
    assert hist == {"bins": 32, "bin_edges": edges.tolist(), "counts": counts.tolist()}
    assert sum(hist["counts"]) == m["pixel_count"]
    assert (hist["bin_edges"][0], hist["bin_edges"][-1]) == (
        m["laplacian_min"],
        m["laplacian_max"],
    )


def test_histogram_is_opt_in_and_bounded(client):
    assert diagnose(client)[1]["laplacian_histogram"] is None
    for bins in (2, 256):
        hist = diagnose(client, params={"laplacian_histogram_bins": bins})[1][
            "laplacian_histogram"
        ]
        assert len(hist["counts"]) == bins and len(hist["bin_edges"]) == bins + 1
    for bad in (1, 257, "many"):
        assert post(client, params={"laplacian_histogram_bins": bad}).status_code == 422


# --- Decision transparency through the API ----------------------------------


def test_blur_exactly_at_the_threshold_passes_and_just_above_fails(settings_env):
    settings_env()
    client = TestClient(app)
    score = post(client, path="/quality").json()["blur_score"]

    settings_env(env={"BLUR_MIN": repr(score)})
    body, diag = diagnose(client)
    blur = diag["decision"]["checks"][0]
    assert (blur["threshold"], blur["value"], blur["margin"]) == (score, score, 0.0)
    assert blur["passed"] is True and body["is_ok"] is True

    settings_env(env={"BLUR_MIN": repr(math.nextafter(score, math.inf))})
    body, diag = diagnose(client)
    blur = diag["decision"]["checks"][0]
    assert blur["passed"] is False and blur["margin"] < 0
    assert body["is_ok"] is False


# --- Settings in effect ------------------------------------------------------


def test_settings_and_their_sources(settings_env):
    settings_env(env={"BLUR_MIN": "70"}, dotenv="BRIGHT_MIN=10\nMAX_UPLOAD_MB=2\n")
    client = TestClient(app)
    body, diag = diagnose(client)
    settings = diag["settings"]
    assert set(settings) == SETTINGS_KEYS
    assert settings["blur_min"] == {"value": 70.0, "source": "env"}
    assert settings["bright_min"] == {"value": 10, "source": "dotenv"}
    assert settings["bright_max"] == {"value": 235, "source": "default"}
    assert settings["resize_max_dim"] == {"value": 1600, "source": "default"}
    assert settings["max_upload_mb"] == {"value": 2, "source": "dotenv"}
    assert settings["max_upload_bytes"] == {"value": 2097152, "source": "derived"}
    assert settings["max_request_bytes"] == {
        "value": 2097152 + 65536,
        "source": "derived",
    }
    assert settings["max_image_pixels"] == {"value": 25_000_000, "source": "default"}
    assert settings["pillow_max_image_pixels"] == {
        "value": PIL.Image.MAX_IMAGE_PIXELS,
        "source": "pillow",
    }
    assert body["thresholds"] == {"blur_min": 70.0, "bright_min": 10, "bright_max": 235}
    assert [c["threshold"] for c in diag["decision"]["checks"]] == [70.0, 10, 235]
    assert client.get("/quality/config").json()["settings"] == settings


def test_config_endpoint(settings_env):
    settings_env(env={"APP_VERSION": "1.2.3"})
    client = TestClient(app)
    config = client.get("/quality/config").json()
    assert set(config) == {"schema_version", "decision_rule", "settings", "versions"}
    assert config["schema_version"] == 1
    assert set(config["settings"]) == SETTINGS_KEYS
    versions = config["versions"]
    assert versions["app"] == "1.2.3"
    assert versions["python"] == platform.python_version()
    assert versions["numpy"] == np.__version__
    assert versions["opencv"] == cv2.__version__
    assert versions["pillow"] == PIL.__version__
    assert diagnose(client)[1]["versions"] == versions


# --- Image metadata ----------------------------------------------------------


def test_image_metadata_for_a_rotated_upload(client, make_image):
    data = make_image(300, 200, "RGB", "JPEG", orientation=6)
    body, diag = diagnose(client, data)
    assert diag["image"] == {
        "upload_bytes": len(data),
        "content_type": "image/jpeg",
        "format": "JPEG",
        "mode": "RGB",
        "original_width": 300,
        "original_height": 200,
        "exif_orientation": 6,
        "orientation_applied": True,
        "oriented_width": 200,
        "oriented_height": 300,
        "resized": False,
        "scale": 1.0,
        "processed_width": 200,
        "processed_height": 300,
    }
    assert (body["width"], body["height"]) == (200, 300)


def test_image_metadata_for_a_resized_upload(client, make_image):
    data = make_image(2000, 1000, "RGBA", "PNG", orientation=8)
    body, diag = diagnose(client, data, content_type="image/png")
    image = diag["image"]
    assert (image["format"], image["mode"], image["content_type"]) == (
        "PNG",
        "RGBA",
        "image/png",
    )
    assert (image["exif_orientation"], image["orientation_applied"]) == (8, True)
    assert (image["oriented_width"], image["oriented_height"]) == (1000, 2000)
    assert (image["resized"], image["scale"]) == (True, 0.8)
    assert (image["processed_width"], image["processed_height"]) == (800, 1600)
    assert (body["width"], body["height"]) == (800, 1600)


# --- Timings -----------------------------------------------------------------


def test_timings_cover_the_stages_in_execution_order(client, make_image):
    cases = [
        (SHARP, {}, STAGES),
        (make_image(2000, 1000), {}, STAGES[:5] + ["resize"] + STAGES[5:]),
        (SHARP, {"laplacian_histogram_bins": 8}, STAGES + ["laplacian_histogram"]),
    ]
    for data, params, expected in cases:
        start = time.perf_counter_ns()
        resp = post(client, data, params=params)
        outside_ns = time.perf_counter_ns() - start
        timings = resp.json()["diagnostics"]["timings"]
        assert timings["clock"] == "perf_counter_ns"
        assert [s["name"] for s in timings["stages"]] == expected
        end = 0
        for stage in timings["stages"]:
            assert type(stage["offset_ns"]) is int and type(stage["duration_ns"]) is int
            assert stage["duration_ns"] >= 0
            assert stage["offset_ns"] >= end  # sequential, no overlap
            end = stage["offset_ns"] + stage["duration_ns"]
        durations = sum(s["duration_ns"] for s in timings["stages"])
        assert timings["stages_sum_ns"] == durations
        assert durations <= end <= timings["handler_total_ns"] <= outside_ns


def test_server_timing_header_matches_the_body(client):
    resp = post(client, params={"laplacian_histogram_bins": 4})
    timings = resp.json()["diagnostics"]["timings"]
    entries = [e.split(";dur=") for e in resp.headers["server-timing"].split(", ")]
    expected = [(s["name"], s["duration_ns"]) for s in timings["stages"]]
    expected.append(("handler_total", timings["handler_total_ns"]))
    assert [name for name, _ in entries] == [name for name, _ in expected]
    for (_, dur), (_, ns) in zip(entries, expected):
        assert dur == f"{ns / 1e6:.3f}"


def test_request_id_is_the_one_in_the_header(client):
    resp = post(client, headers={"X-Request-ID": "demo-123"})
    assert resp.json()["diagnostics"]["request_id"] == "demo-123"
    resp = post(client)
    generated = resp.headers["x-request-id"]
    assert generated.startswith("rid-")
    assert resp.json()["diagnostics"]["request_id"] == generated


# --- Errors and security -----------------------------------------------------


def test_errors_match_the_default_endpoint(settings_env):
    settings_env(env={"MAX_UPLOAD_MB": "1"})
    client = TestClient(app)
    cases = [
        (SHARP, "text/plain", 415, b'{"detail":"Unsupported media type"}'),
        (
            b"\x00" * (1024 * 1024 + 1),
            "image/jpeg",
            413,
            b'{"detail":"File too large"}',
        ),
        (b"not an image", "image/jpeg", 400, b'{"detail":"Invalid image data"}'),
        (b"", "image/png", 400, b'{"detail":"Invalid image data"}'),
    ]
    for data, content_type, status, content in cases:
        for path in ("/quality", "/quality/diagnostics"):
            resp = post(client, data, path=path, content_type=content_type)
            assert (resp.status_code, resp.content) == (status, content)
            assert "server-timing" not in resp.headers
    missing = [client.post(p) for p in ("/quality", "/quality/diagnostics")]
    assert [r.status_code for r in missing] == [422, 422]
    assert missing[0].content == missing[1].content


def test_rejections_happen_before_decoding(settings_env, monkeypatch):
    settings_env(env={"MAX_UPLOAD_MB": "1"})
    client = TestClient(app)
    calls = []
    real = routes_quality.load_and_normalize_traced

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(routes_quality, "load_and_normalize_traced", spy)
    for path in ("/quality", "/quality/diagnostics"):
        assert (
            post(client, SHARP, path=path, content_type="text/plain").status_code == 415
        )
        oversized = b"\xff\xd8" + b"\x00" * (1024 * 1024)
        assert post(client, oversized, path=path).status_code == 413
    assert calls == []
    assert post(client, SHARP).status_code == 200
    assert len(calls) == 1


def test_oversized_uploads_are_not_read_into_memory_whole(settings_env, monkeypatch):
    settings_env(env={"MAX_UPLOAD_MB": "1"})
    reads = []
    real_read = starlette.datastructures.UploadFile.read

    async def recording_read(self, size=-1):
        data = await real_read(self, size)
        reads.append((size, len(data)))
        return data

    monkeypatch.setattr(starlette.datastructures.UploadFile, "read", recording_read)
    # Over the file limit but within the request allowance, so the handler's
    # own check is the one that answers (larger bodies stop at the middleware).
    data = b"\x00" * (1024 * 1024 + 32 * 1024)
    resp = post(TestClient(app), data, path="/quality")
    assert resp.status_code == 413
    assert reads == [(1024 * 1024 + 1, 1024 * 1024 + 1)]


def test_no_secrets_or_paths_in_diagnostic_output(settings_env, monkeypatch, tmp_path):
    monkeypatch.setenv("IQG_TEST_SECRET", "s3cr3t-token-value")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:hunter2@db/prod")
    settings_env(env={"BLUR_MIN": "70"}, dotenv="BRIGHT_MIN=10\nAPP_VERSION=9.9.9\n")
    client = TestClient(app)
    files = {"file": ("/home/someone/private/holiday-photo.jpg", SHARP, "image/jpeg")}
    resp = client.post("/quality/diagnostics", files=files)
    outputs = [
        resp.text,
        resp.headers["server-timing"],
        client.get("/quality/config").text,
    ]
    repo = str(Path(__file__).resolve().parents[2])
    needles = ["s3cr3t-token-value", "hunter2", "postgres://", "holiday-photo"]
    needles += [str(tmp_path), os.getcwd(), repo, "/home/", "Traceback"]
    for text in outputs:
        for needle in needles:
            assert needle not in text
    assert set(resp.json()["diagnostics"]["settings"]) == SETTINGS_KEYS


# --- Logging and concurrency -------------------------------------------------


def test_timing_log_is_structured_bounded_and_correlated(client, caplog):
    logger_name = routes_quality.logger.name
    with caplog.at_level(logging.INFO, logger=logger_name):
        post(client, path="/quality")
    assert [r for r in caplog.records if r.name == logger_name] == []

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        for path in ("/quality", "/quality/diagnostics"):
            post(client, path=path, headers={"X-Request-ID": "log-" + "x" * 500})
    records = [r for r in caplog.records if r.name == logger_name]
    assert [r.extra["route"] for r in records] == ["/quality", "/quality/diagnostics"]
    for record in records:
        extra = record.extra
        assert set(extra) == {
            "request_id",
            "route",
            "is_ok",
            "handler_total_ns",
            "stages_ns",
        }
        assert extra["request_id"] == ("log-" + "x" * 500)[:128]
        assert set(extra["stages_ns"]) <= set(STAGES) | {
            "resize",
            "laplacian_histogram",
        }
        assert all(type(v) is int for v in extra["stages_ns"].values())


def test_concurrent_requests_keep_their_own_diagnostics(client, make_image):
    images = {
        "a": make_image(300, 200, orientation=6),
        "b": make_image(640, 480, "L", "PNG"),
        "c": SHARP,
        "d": BLURRY,
    }
    expected = {k: diagnose(client, v)[1]["measurements"] for k, v in images.items()}
    jobs = [(f"{k}-{i}", k) for i in range(4) for k in images]

    with TestClient(app) as shared:

        def run(job):
            rid, key = job
            return (
                rid,
                key,
                diagnose(shared, images[key], headers={"X-Request-ID": rid}),
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, jobs))
    for rid, key, (_, diag) in results:
        assert diag["request_id"] == rid
        assert diag["measurements"] == expected[key]
