#!/usr/bin/env python3
"""
bench_quality.py — Compare POST /quality latency between this checkout and a
baseline checkout, and measure the cost of POST /quality/diagnostics.

Two measurements:
1. End to end: one uvicorn server per variant (one worker each, default
   settings) on 127.0.0.1, so no endpoint is affected by the memory state
   another left behind. Every round sends one request to each variant in a
   shuffled order over keep-alive connections, so drift and background load
   hit all variants alike; the paired per-round differences are reported.
2. Pipeline A/B: the baseline's preprocess.py/quality.py against the pipeline
   /quality now runs, in one process without HTTP, alternating call by call.

Usage (examples):
  git worktree add ../iqg-baseline main      # or: git archive main | tar -x -C DIR
  python scripts/bench_quality.py --baseline ../iqg-baseline
  python scripts/bench_quality.py --baseline ../iqg-baseline --rounds 300 --json out/bench.json

Notes:
- Uses tests/assets plus two generated photo-sized JPEGs (1600x1200, and
  4032x3024 with EXIF orientation 6), identical bytes for every variant.
- Servers run with default settings: an empty working directory (no .env),
  settings environment variables removed, LOG_LEVEL=WARNING.
"""
import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETTINGS_ENV = {
    "app_name",
    "app_version",
    "blur_min",
    "bright_min",
    "bright_max",
    "resize_max_dim",
    "max_upload_mb",
    "log_level",
    "log_json",
    "prometheus_enabled",
}
BOUNDARY = "benchboundary7d9f2c"


def make_photo(width: int, height: int, orientation: int | None = None) -> bytes:
    """Deterministic photo-like JPEG: gradients, shapes and mild sensor noise."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.stack(
        [xx * 200 // width + 30, yy * 180 // height + 40, (xx + yy) // 32 % 64 + 96],
        axis=-1,
    ).astype(np.int16)
    for _ in range(40):
        x0, y0 = int(rng.integers(0, width - 50)), int(rng.integers(0, height - 50))
        x1 = x0 + int(rng.integers(20, width // 4))
        y1 = y0 + int(rng.integers(20, height // 4))
        arr[y0:y1, x0:x1] = rng.integers(0, 256, 3)
    arr += rng.normal(0, 4, arr.shape).astype(np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    save: dict = {"quality": 90}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        save["exif"] = exif.tobytes()
    buf = BytesIO()
    img.save(buf, format="JPEG", **save)
    return buf.getvalue()


def multipart(name: str, data: bytes) -> bytes:
    head = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    return head + data + f"\r\n--{BOUNDARY}--\r\n".encode()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(app_dir: Path, workdir: Path, env: dict) -> tuple:
    port = free_port()
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", str(app_dir)]
    cmd += ["--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    # Photo rounds can leave a connection idle longer than the 5 s default.
    cmd += ["--timeout-keep-alive", "600"]
    proc = subprocess.Popen(cmd, cwd=workdir, env=env)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            if resp.status == 200:
                return proc, conn
        except OSError:
            time.sleep(0.2)
    proc.kill()
    sys.exit(f"server for {app_dir} did not start")


def timed_post(conn: http.client.HTTPConnection, path: str, body: bytes):
    headers = {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"}
    t0 = time.perf_counter_ns()
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    content = resp.read()
    elapsed = time.perf_counter_ns() - t0
    if resp.status != 200:
        sys.exit(f"{path}: HTTP {resp.status} {content[:200]!r}")
    return elapsed, content


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_pipeline(baseline: Path, assets: list, pairs: int) -> dict:
    """
    In-process A/B of the image pipeline without HTTP: the baseline's
    preprocess.py/quality.py (they import nothing from app.) against the
    sequence /quality now runs, alternating call by call.
    """
    sys.path.insert(0, str(REPO))
    from app.core.timing import StageTimer
    from app.services.preprocess import load_and_normalize_traced, to_gray
    from app.services.quality import Thresholds, assess_traced

    old_pre = load_module(
        "baseline_preprocess", baseline / "app/services/preprocess.py"
    )
    old_q = load_module("baseline_quality", baseline / "app/services/quality.py")
    old_thr = old_q.Thresholds(140.0, 20, 235)
    thr = Thresholds(140.0, 20, 235)

    def old(data: bytes) -> None:
        bgr, _, _ = old_pre.load_and_normalize(data, 1600)
        old_q.assess(old_pre.to_gray(bgr), old_thr)

    def new(data: bytes) -> None:
        timer = StageTimer()
        bgr, _ = load_and_normalize_traced(data, 1600, timer)
        with timer.stage("grayscale"):
            gray = to_gray(bgr)
        assess_traced(gray, thr, timer)

    results = {}
    for asset in assets:
        data = asset["data"]
        n = max(5, pairs // 20) if asset["photo"] else pairs
        old(data), new(data)  # warm up
        olds, news, deltas = [], [], []
        for i in range(n):
            first, second = (old, new) if i % 2 == 0 else (new, old)
            t0 = time.perf_counter_ns()
            first(data)
            t1 = time.perf_counter_ns()
            second(data)
            t2 = time.perf_counter_ns()
            a, b = (t1 - t0, t2 - t1) if first is old else (t2 - t1, t1 - t0)
            olds.append(a)
            news.append(b)
            deltas.append(b - a)
        results[asset["name"]] = {
            "n": n,
            "baseline_median_us": statistics.median(olds) / 1e3,
            "current_median_us": statistics.median(news) / 1e3,
            "paired_delta_median_us": statistics.median(deltas) / 1e3,
            "paired_delta_p25_us": quantile(deltas, 25) / 1e3,
            "paired_delta_p75_us": quantile(deltas, 75) / 1e3,
        }
    return results


def quantile(values: list, q: float) -> float:
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def summarize(values_ns: list) -> dict:
    ms = [v / 1e6 for v in values_ns]
    return {
        "n": len(ms),
        "median_ms": statistics.median(ms),
        "p25_ms": quantile(ms, 25),
        "p75_ms": quantile(ms, 75),
        "p5_ms": quantile(ms, 5),
        "p95_ms": quantile(ms, 95),
    }


def environment_info() -> dict:
    import cv2
    import fastapi
    import numpy
    import PIL
    import pydantic
    import starlette
    import uvicorn

    cpu = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "fastapi": fastapi.__version__,
        "starlette": starlette.__version__,
        "pydantic": pydantic.VERSION,
        "uvicorn": uvicorn.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", type=Path, help="Checkout to compare against.")
    ap.add_argument("--rounds", type=int, default=200, help="Per small asset.")
    ap.add_argument("--photo-rounds", type=int, default=20, help="Per photo.")
    ap.add_argument("--pairs", type=int, default=400, help="Pipeline A/B pairs.")
    ap.add_argument("--json", type=Path, help="Optional path to write results.")
    args = ap.parse_args()
    if args.baseline and not (args.baseline / "app" / "main.py").exists():
        sys.exit(f"--baseline has no app/main.py: {args.baseline}")

    small = sorted((REPO / "tests" / "assets").glob("*.jpg"))
    assets = [
        {"name": p.name, "data": p.read_bytes(), "photo": False} for p in small
    ] + [
        {"name": "photo_1600x1200.jpg", "data": make_photo(1600, 1200), "photo": True},
        {
            "name": "photo_4032x3024_exif6.jpg",
            "data": make_photo(4032, 3024, orientation=6),
            "photo": True,
        },
    ]

    load_start = os.getloadavg() if hasattr(os, "getloadavg") else None
    reference = "baseline /quality" if args.baseline else "current /quality"
    report: dict = {"results": {}}
    with tempfile.TemporaryDirectory() as tmp:
        env = {k: v for k, v in os.environ.items() if k.lower() not in SETTINGS_ENV}
        env.update(LOG_LEVEL="WARNING", PYTHONWARNINGS="ignore")
        variants = []
        if args.baseline:
            variants.append(("baseline /quality", args.baseline.resolve(), "/quality"))
        variants += [
            ("current /quality", REPO, "/quality"),
            ("current /quality/diagnostics", REPO, "/quality/diagnostics"),
            (
                "current /quality/diagnostics+hist64",
                REPO,
                "/quality/diagnostics?laplacian_histogram_bins=64",
            ),
        ]
        # One server per variant: an endpoint's measurements must not depend on
        # what another endpoint left behind in the same process (memory state).
        servers = {
            label: start_server(app_dir, Path(tmp), env)
            for label, app_dir, _ in variants
        }
        rng = random.Random(0)
        try:
            for asset in assets:
                body = multipart(asset["name"], asset["data"])
                rounds = args.photo_rounds if asset["photo"] else args.rounds
                lat: dict = {label: [] for label, _, _ in variants}
                handler: dict = {label: [] for label, _, _ in variants}
                digests: dict = {}
                for i in range(rounds + 3):  # first 3 rounds warm up
                    order = variants[:]
                    rng.shuffle(order)
                    for label, _, path in order:
                        elapsed, content = timed_post(servers[label][1], path, body)
                        digests.setdefault(label, hashlib.sha256(content).hexdigest())
                        if i < 3:
                            continue
                        lat[label].append(elapsed)
                        if "diagnostics" in path:
                            timings = json.loads(content)["diagnostics"]["timings"]
                            handler[label].append(timings["handler_total_ns"])
                report["results"][asset["name"]] = {
                    "bytes": len(asset["data"]),
                    "variants": {},
                }
                for label, _, _ in variants:
                    deltas = [a - b for a, b in zip(lat[label], lat[reference])]
                    report["results"][asset["name"]]["variants"][label] = {
                        **summarize(lat[label]),
                        "paired_delta_median_ms": statistics.median(deltas) / 1e6,
                        "paired_delta_p25_ms": quantile(deltas, 25) / 1e6,
                        "paired_delta_p75_ms": quantile(deltas, 75) / 1e6,
                        "handler_total_median_ms": (
                            statistics.median(handler[label]) / 1e6
                            if handler[label]
                            else None
                        ),
                    }
                if args.baseline:
                    report["results"][asset["name"]]["quality_body_identical"] = (
                        digests["baseline /quality"] == digests["current /quality"]
                    )
                print(f"{asset['name']} done", file=sys.stderr)
        finally:
            for proc, conn in servers.values():
                conn.close()
                proc.terminate()
                proc.wait()

    if args.baseline:
        report["pipeline_ab"] = paired_pipeline(
            args.baseline.resolve(), assets, args.pairs
        )
    info = environment_info()
    if load_start is not None:
        info["loadavg_1m_start_end"] = f"{load_start[0]:.2f}/{os.getloadavg()[0]:.2f}"
    report["environment"] = info
    print_report(report, reference)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.json}")


def print_report(report: dict, reference: str) -> None:
    info = report["environment"]
    print("\nEnvironment: " + ", ".join(f"{k}={v}" for k, v in info.items()))
    print(f"\nEnd to end over HTTP (ms). Paired Δ = variant - {reference}, per round.")
    for name, res in report["results"].items():
        print(f"\n{name} ({res['bytes']} bytes)")
        print(
            f"  {'variant':36s} {'n':>4s} {'median':>9s} {'p25':>9s} {'p75':>9s} "
            f"{'p5':>9s} {'p95':>9s} {'paired Δ [p25, p75]':>28s} {'handler':>8s}"
        )
        for label, s in res["variants"].items():
            delta = (
                f"{s['paired_delta_median_ms']:+.3f} "
                f"[{s['paired_delta_p25_ms']:+.3f}, {s['paired_delta_p75_ms']:+.3f}]"
            )
            handler = s["handler_total_median_ms"]
            handler_text = "-" if handler is None else f"{handler:.3f}"
            print(
                f"  {label:36s} {s['n']:4d} {s['median_ms']:9.3f} {s['p25_ms']:9.3f} "
                f"{s['p75_ms']:9.3f} {s['p5_ms']:9.3f} {s['p95_ms']:9.3f} "
                f"{delta:>28s} {handler_text:>8s}"
            )
        if "quality_body_identical" in res:
            same = "yes" if res["quality_body_identical"] else "NO"
            print(f"  /quality body identical to baseline: {same}")
    print(
        "'handler' is the median server-reported handler_total_ns (diagnostics only)."
    )
    if "pipeline_ab" in report:
        print("\nPipeline A/B (in-process, no HTTP; baseline vs current /quality path)")
        print(
            f"  {'asset':28s} {'pairs':>5s} {'baseline':>10s} {'current':>10s} "
            f"{'Δ median':>9s} {'Δ p25':>9s} {'Δ p75':>9s}"
        )
        for name, r in report["pipeline_ab"].items():
            print(
                f"  {name:28s} {r['n']:5d} {r['baseline_median_us']:10.1f} "
                f"{r['current_median_us']:10.1f} {r['paired_delta_median_us']:+9.1f} "
                f"{r['paired_delta_p25_us']:+9.1f} {r['paired_delta_p75_us']:+9.1f}"
            )
        print("  Times in µs; Δ = current - baseline, per alternating pair.")


if __name__ == "__main__":
    main()
