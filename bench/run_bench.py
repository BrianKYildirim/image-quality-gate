"""Reproducible benchmark runner for image-quality-gate: one command per environment.

  python bench/run_bench.py --env windows-native   (native Windows; asyncio loop)
  python bench/run_bench.py --env ubuntu-wsl2      (Ubuntu under WSL2, run from a
                                                    WSL-native clone; uvloop)

Run it with the bench venv's python (CPython 3.11, packages pinned by
bench/constraints-py311.txt; see that file). Each run extracts the baseline commit
(--baseline-ref, default 8e47d74: the code before diagnostics and the input limits)
with `git archive` into a temporary directory, then

  1. runs --sessions sessions (default 5) of scripts/bench_quality.py against it.
     Every session starts fresh single-worker uvicorn servers on 127.0.0.1, one per
     variant (baseline /quality, current /quality, current /quality/diagnostics and
     current /quality/diagnostics+hist64), and keeps every post-warm-up request;
  2. runs bench/limits_probe.py: every row of the README input-limits table, before
     (the baseline) and after (this tree), --limits-runs fresh servers per case;
  3. runs the test suite (pytest -q).

It writes

  bench/results/<YYYY-MM-DD>-<sha7>[-<tag>]/<env>/
      raw/           session-NN.json, session-NN.txt and session-NN/ (per-request
                     nanosecond arrays), limits.json, limits.txt, tests.txt
      meta.json      bench-meta/1: hardware, OS, power, load, Python and packages,
                     event loop, workload, timer, validity, raw sha256
      summary.json   bench-summary/1, computed only from raw/ and meta.json

and prints the environment directory as its last line. It refuses a dirty tree
unless --allow-dirty (recorded in meta.json source.dirty) and never overwrites an
existing environment directory.

  python bench/run_bench.py --resummarize bench/results/<session>/<env>
      rebuild summary.json from raw/ and meta.json alone

Labels (BENCH-05/06): per-request latency percentiles are per_op_latency (every
request is timed on its own, client side, send to body read). Paired differences
and single-request input-limit times are request_wall_time. The /quality/diagnostics
variants carry build instrumented:iqg-diagnostics-endpoint; /quality on the baseline
and on this tree is uninstrumented. Whole sessions are marked invalid (kept in raw/,
left out of summary values) when the machine is on battery, when CPU busy at their
start stays above --max-busy-pct after waiting, or when the harness fails. No
individual sample is ever dropped. Standard library only (Python 3.11).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

sys.dont_write_bytecode = True  # no bench/__pycache__ in the measured tree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmeta  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROJECT = "image-quality-gate"
REPO_SLUG = "BrianKYildirim/image-quality-gate"
ENVIRONMENTS = ("windows-native", "ubuntu-wsl2")
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BASELINE_REF = "8e47d744b7cc0db618fb95c35707e5a258b897cd"
# The harness changes do not touch app/: its tree must equal the app/ of this commit.
APP_REFERENCE_COMMIT = "e61761fb933d03f8d484ce124d708fedb8932152"
DIAG_BUILD = "instrumented:iqg-diagnostics-endpoint"
REFERENCE_KEY = "baseline_quality"
VARIANTS = (
    "baseline_quality",
    "current_quality",
    "current_diagnostics",
    "current_diagnostics_hist64",
)
LATENCY_PERCENTILES = ((50, "p50"), (95, "p95"), (99, "p99"))
LIMIT_VARIANTS = ("before", "after")
WARMUP_ROUNDS = 3  # bench_quality.py: the first 3 rounds per asset are not kept
REDACTION_NOTE = (
    "raw text: absolute repo, temporary and home paths replaced by <repo>, <tmp> "
    "and <home>; line endings normalized to LF"
)
WARMUP_POLICY = (
    "bench_quality.py: 3 warm-up rounds per asset and session, not kept; every "
    "session starts fresh servers; pipeline A/B: one warm-up call per side and "
    "asset. limits_probe.py: one GET /health per fresh server before the request"
)
OUTLIER_POLICY = "no samples dropped; whole runs marked invalid on precondition failure"
MEASURES_POLICY = (
    "per_op_latency: percentiles of individually timed requests (client side, send "
    "to body read); values are the per-session percentiles, pooled is the same "
    "percentile over every kept request of the valid sessions. request_wall_time: "
    "per-session medians of paired per-round differences (variant - "
    "baseline_quality, in-process baseline vs current), and single-request times "
    "of the input-limits probe. heap_bytes: growth of the server process's peak "
    "resident memory over one request (PeakWorkingSetSize on Windows, VmHWM on "
    "Linux), not heap accounting"
)


def log(msg: str) -> None:
    print(f"[run_bench] {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"error: {msg}")
    sys.exit(code)


def rel(path: str | Path, base: Path = REPO) -> str:
    """A path shown relative to base with forward slashes (never absolute)."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), base.resolve())).as_posix()
    except ValueError:  # another drive on Windows
        return f"<outside-repo>/{Path(path).name}"


# ---------- Raw output: path redaction and LF ----------


def redactor(paths: list[tuple[Path, str]]) -> Callable[[bytes], bytes]:
    """Replaces absolute paths in captured text (both slash styles and the
    JSON-escaped double-backslash form), longest first across all paths."""
    forms: dict[str, str] = {}
    for path, token in paths:
        p = str(Path(path).resolve())
        for form in (
            p,
            p.replace("\\", "/"),
            p.replace("/", "\\"),
            p.replace("/", "\\").replace("\\", "\\\\"),
        ):
            if len(form) > 3:
                forms.setdefault(form, token)
    flags = re.IGNORECASE if os.name == "nt" else 0
    subs = [
        (re.compile(re.escape(form), flags), token)
        for form, token in sorted(forms.items(), key=lambda kv: -len(kv[0]))
    ]

    def apply(data: bytes) -> bytes:
        text = data.replace(b"\r\n", b"\n").decode("utf-8", errors="surrogateescape")
        for rx, token in subs:
            text = rx.sub(token, text)
        return text.encode("utf-8", errors="surrogateescape")

    return apply


def redact_tree(path: Path, redact: Callable[[bytes], bytes]) -> None:
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for f in files:
        if f.is_file():
            f.write_bytes(redact(f.read_bytes()))


# ---------- Validity gate ----------


def gate(step: str, args: argparse.Namespace, env: str) -> dict[str, Any]:
    """Precondition sample before a measured step: AC power and CPU busy. Waits up
    to --quiet-timeout for quiet; returns the record kept in meta validity.runs."""
    start = time.monotonic()
    while True:
        busy = benchmeta.cpu_busy_pct(1.0)
        power = benchmeta.power_status()
        reasons = []
        if power["ac_line"] is False:
            reasons.append("on battery (AC line offline)")
        if busy is not None and busy > args.max_busy_pct:
            reasons.append(f"CPU busy {busy}% > {args.max_busy_pct}% at start")
        waited = round(time.monotonic() - start, 1)
        if not reasons or waited >= args.quiet_timeout:
            break
        log(f"{step}: waiting for quiet ({'; '.join(reasons)}; waited {waited} s)")
        time.sleep(4)
    if reasons:
        reasons[-1] += f" after waiting {waited} s"
    source = (
        "WSL VM /proc/stat (not the Windows host)"
        if env == "ubuntu-wsl2"
        else "GetSystemTimes"
    )
    return {
        "step": step,
        "cpu_busy_pct": busy,
        "ac_line": power["ac_line"],
        "waited_s": waited,
        "busy_source": source,
        "valid": not reasons,
        "reasons": reasons,
    }


def run_step(argv: list[str], cwd: Path = REPO) -> tuple[int, bytes]:
    """(exit status, combined stdout+stderr bytes); a failure is returned. Child
    stdio is UTF-8: bench_quality.py prints non-ASCII text (Δ, µs), which a pipe's
    cp1252 default on Windows cannot encode."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env,
        check=False,
    )
    return p.returncode, p.stdout or b""


# ---------- Parsing ----------


def parse_pytest(text: str) -> dict[str, int] | None:
    """{ passed, failed, skipped, total } from the pytest -q summary line."""
    lines = [ln for ln in text.splitlines() if re.search(r" in [0-9.]+s", ln)]
    if not lines:
        return None
    counts = {
        word: int(n)
        for n, word in re.findall(r"(\d+) ([a-z]+)", lines[-1].replace("=", " "))
    }
    passed = counts.get("passed", 0) + counts.get("xpassed", 0)
    failed = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    skipped = counts.get("skipped", 0) + counts.get("xfailed", 0)
    if not passed + failed + skipped:
        return None
    total = passed + failed + skipped
    return {"passed": passed, "failed": failed, "skipped": skipped, "total": total}


# ---------- Statistics ----------


def quantile(values: list[float], q: float) -> float:
    """Inclusive percentile q (1..99), as scripts/bench_quality.py computes it."""
    if len(values) < 2:
        return float(values[0])
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def stats(
    values: list[float], unit: str, measure: str, build: str, **extra: Any
) -> dict[str, Any]:
    """Stats object: run-level values in run order; median and inclusive quartiles
    as the portfolio checker recomputes them."""
    vals = [float(v) for v in values]
    if len(vals) == 1:
        med = p25 = p75 = vals[0]
    else:
        med = statistics.median(vals)
        q = statistics.quantiles(vals, n=4, method="inclusive")
        p25, p75 = q[0], q[2]
    return {
        "n": len(vals),
        "values": vals,
        "median": med,
        "min": min(vals),
        "max": max(vals),
        "p25": p25,
        "p75": p75,
        "unit": unit,
        "measure": measure,
        "build": build,
        **extra,
    }


def paired(values_per_run: list[list[float]], unit: str, build: str, **extra: Any):
    """request_wall_time stats over per-run medians, plus pooled median and IQR."""
    pooled = [v for run in values_per_run for v in run]
    return stats(
        [statistics.median(run) for run in values_per_run],
        unit,
        "request_wall_time",
        build,
        pooled_median=statistics.median(pooled),
        pooled_p25=quantile(pooled, 25),
        pooled_p75=quantile(pooled, 75),
        samples=len(pooled),
        **extra,
    )


def build_of(variant: str) -> str:
    return DIAG_BUILD if "diagnostics" in variant else "uninstrumented"


def ns_to_ms(values: list[int]) -> list[float]:
    return [v / 1e6 for v in values]


def summarize_http(
    env_dir: Path, sessions: list[dict[str, Any]], used: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(http, pipeline) sections from raw/session-NN/ of the valid sessions."""
    per: dict[str, dict[str, list[dict[str, Any]]]] = {}
    pipe: dict[str, list[dict[str, Any]]] = {}
    for run in sessions:
        root = env_dir / run["dir"]
        for f in sorted(root.glob("*/*.json")):
            doc = json.loads(f.read_text(encoding="utf-8"))
            used.append(f.relative_to(env_dir).as_posix())
            if f.parent.name == "pipeline":
                pipe.setdefault(doc["asset"], []).append(doc)
            else:
                per.setdefault(doc["asset"], {}).setdefault(f.stem, []).append(doc)
    http: dict[str, Any] = {}
    for asset, variants in sorted(per.items()):
        entry: dict[str, Any] = {"bytes": next(iter(variants.values()))[0]["bytes"]}
        for key in VARIANTS:
            docs = variants.get(key)
            if not docs:
                continue
            build = build_of(key)
            runs_ms = [ns_to_ms(d["latency_ns"]) for d in docs]
            pooled = [v for run in runs_ms for v in run]
            latency = {
                name: stats(
                    [quantile(run, pct) for run in runs_ms],
                    "ms",
                    "per_op_latency",
                    build,
                    percentile=pct,
                    pooled=quantile(pooled, pct),
                    samples=len(pooled),
                )
                for pct, name in LATENCY_PERCENTILES
            }
            v: dict[str, Any] = {"latency_ms": latency, "path": docs[0]["path"]}
            if key != REFERENCE_KEY:
                v["paired_delta_ms"] = paired(
                    [ns_to_ms(d["paired_delta_ns"]) for d in docs],
                    "ms",
                    build,
                    reference=REFERENCE_KEY,
                    definition=f"per round: {key} - {REFERENCE_KEY} request time, "
                    "both sent in the same shuffled round",
                )
            if any(d["handler_total_ns"] for d in docs):
                v["handler_total_ms"] = {
                    "unit": "ms",
                    "values": [
                        statistics.median(ns_to_ms(d["handler_total_ns"])) for d in docs
                    ],
                    "note": "per-session median of the server-reported "
                    "handler_total_ns (diagnostics response); context, not a "
                    "client-side measurement",
                }
            entry[key] = v
        if REFERENCE_KEY in variants and "current_quality" in variants:
            entry["quality_body_identical"] = [
                a["body_sha256"] == b["body_sha256"]
                for a, b in zip(variants[REFERENCE_KEY], variants["current_quality"])
            ]
        http[asset] = entry
    pipeline: dict[str, Any] = {}
    for asset, docs in sorted(pipe.items()):
        pipeline[asset] = {
            "pairs": docs[0]["pairs"],
            "baseline_ms": paired(
                [ns_to_ms(d["baseline_ns"]) for d in docs], "ms", "uninstrumented"
            ),
            "current_ms": paired(
                [ns_to_ms(d["current_ns"]) for d in docs], "ms", "uninstrumented"
            ),
            "delta_ms": paired(
                [ns_to_ms(d["paired_delta_ns"]) for d in docs],
                "ms",
                "uninstrumented",
                definition="per alternating pair: current - baseline in-process "
                "pipeline time (no HTTP)",
            ),
        }
    return http, pipeline


def summarize_limits(doc: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for case, info in doc["cases"].items():
        entry: dict[str, Any] = {}
        for variant in LIMIT_VARIANTS:
            ms = [
                m
                for m in doc["measurements"]
                if m["case"] == case and m["variant"] == variant
            ]
            if not ms:
                continue
            ms.sort(key=lambda m: m["run"])
            v: dict[str, Any] = {
                "status": {
                    "unit": "HTTP status",
                    "values": [m["status"] for m in ms],
                    "note": "status code per run; null when no response was read",
                },
                "elapsed_ms": stats(
                    [m["elapsed_ns"] / 1e6 for m in ms],
                    "ms",
                    "request_wall_time",
                    "uninstrumented",
                    definition="client side, from the first byte sent to the "
                    "response body read; one request per fresh server",
                ),
            }
            growth = [m["peak_growth_bytes"] for m in ms]
            if all(g is not None for g in growth):
                v["rss_growth_mib"] = stats(
                    [g / 2**20 for g in growth],
                    "MiB",
                    "heap_bytes",
                    "uninstrumented",
                    definition="server peak resident memory after the request "
                    "minus before it (Windows PeakWorkingSetSize, Linux VmHWM)",
                )
            entry[variant] = v
        entry["input_bytes"] = info["input_bytes"]
        entry["description"] = info["description"]
        out[case] = entry
    return out


def build_summary(env_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """summary.json from raw/ and meta.json only (also what --resummarize runs)."""
    runs = meta["validity"]["runs"]
    wl = meta["workload"]
    used: list[str] = []
    summary: dict[str, Any] = {
        "schema": "bench-summary/1",
        "project": meta["project"],
        "environment_id": meta["environment_id"],
        "source_sha": meta["source"]["sha"],
        "policy": {
            "quantile_method": "inclusive",
            "runs": wl["sessions"],
            "runs_by_harness": {
                "bench_quality_sessions": wl["sessions"],
                "limits_probe_runs": wl["limits_runs"],
            },
            "warmup": WARMUP_POLICY,
            "outliers": OUTLIER_POLICY,
            "values": "run-level: one value per bench_quality.py session (http, "
            "pipeline) or per fresh-server probe run (limits), in run order",
            "measures": MEASURES_POLICY,
            "invalid_runs_excluded": [r["file"] for r in runs if not r["valid"]],
        },
    }
    if (env_dir / "raw/tests.txt").is_file():
        tests = parse_pytest((env_dir / "raw/tests.txt").read_text(encoding="utf-8"))
        if tests:
            summary["tests"] = tests
            used.append("raw/tests.txt")
    sessions = [r for r in runs if r["step"] == "session" and r["valid"]]
    if sessions:
        http, pipeline = summarize_http(env_dir, sessions, used)
        if http:
            summary["http"] = http
        if pipeline:
            summary["pipeline"] = pipeline
    probe = next((r for r in runs if r["step"] == "limits_probe"), None)
    if probe and probe["valid"]:
        doc = json.loads((env_dir / probe["file"]).read_text(encoding="utf-8"))
        summary["limits"] = summarize_limits(doc)
        used.append(probe["file"])
    summary["generated_from"] = sorted(used)
    return summary


# ---------- Run ----------


def parse_args(argv: list[str]) -> argparse.Namespace:
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a = p.add_argument
    a("--env", choices=ENVIRONMENTS, help="environment id (unless --resummarize)")
    a("--out", help="results root (default bench/results; relative to the cwd)")
    a("--session-date", default=today, help="YYYY-MM-DD (default: today, UTC)")
    a("--tag", help="optional session directory suffix: <date>-<sha7>-<tag>")
    a("--allow-dirty", action="store_true", help="run with uncommitted changes")
    a(
        "--smoke",
        action="store_true",
        help="quick check: 20 rounds, 3 photo rounds, 20 pairs, 1 limits run",
    )
    a("--sessions", type=int, default=5, help="bench_quality.py sessions (5)")
    a("--rounds", type=int, help="rounds per small asset (300; smoke 20)")
    a("--photo-rounds", type=int, help="rounds per photo (60; smoke 3)")
    a("--pairs", type=int, help="pipeline A/B pairs (400; smoke 20)")
    a("--limits-runs", type=int, help="fresh servers per limits case (5; smoke 1)")
    a("--baseline-ref", default=BASELINE_REF, help="baseline commit (8e47d74)")
    a(
        "--uvicorn-loop",
        choices=("auto", "asyncio"),
        default="auto",
        help="uvicorn --loop for every server (auto: uvloop where installed)",
    )
    a("--max-busy-pct", type=float, default=10.0, help="CPU busy limit (10)")
    a("--quiet-timeout", type=float, default=120.0, help="seconds to wait (120)")
    a("--resummarize", metavar="ENVDIR", help="rebuild ENVDIR/summary.json, exit")
    args = p.parse_args(argv)
    if not args.resummarize and not args.env:
        p.error("--env is required")
    if args.tag is not None and not TAG_RE.match(args.tag):
        p.error(f"--tag must match {TAG_RE.pattern}")
    try:
        datetime.date.fromisoformat(args.session_date)
    except ValueError:
        p.error("--session-date must be YYYY-MM-DD")
    if args.sessions < 1:
        p.error("--sessions must be at least 1")
    smoke = args.smoke
    args.rounds = args.rounds or (20 if smoke else 300)
    args.photo_rounds = args.photo_rounds or (3 if smoke else 60)
    args.pairs = args.pairs or (20 if smoke else 400)
    if args.limits_runs is None:
        args.limits_runs = 1 if smoke else 5
    return args


def shown_command(argv: list[str]) -> list[str]:
    """The documented command with every path argument made repo-relative."""
    out, nxt = ["python", "bench/run_bench.py"], False
    path_flags = ("--out", "--resummarize")
    for a in argv:
        if nxt:
            out.append(rel(a) if os.path.isabs(a) or os.sep in a or "/" in a else a)
            nxt = False
        elif a in path_flags:
            out.append(a)
            nxt = True
        elif any(a.startswith(f + "=") for f in path_flags):
            k, v = a.split("=", 1)
            out.append(f"{k}={rel(v)}")
        else:
            out.append(a)
    return out


def check_platform(env: str) -> None:
    if env == "windows-native" and os.name != "nt":
        die("--env windows-native must run on native Windows")
    if env == "ubuntu-wsl2" and not benchmeta.is_wsl():
        die("--env ubuntu-wsl2 must run inside WSL2")


def constraint_mismatches(packages: dict[str, str]) -> list[str]:
    """Pins in bench/constraints-py311.txt that apply here but are not installed."""
    try:
        from packaging.markers import Marker
    except ImportError:  # packaging comes with pytest; without it, skip markers
        Marker = None  # type: ignore[assignment,misc]
    out = []
    text = (REPO / "bench" / "constraints-py311.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        spec, _, marker = (s.strip() for s in line.partition(";"))
        if marker and Marker is not None and not Marker(marker).evaluate():
            continue
        name, _, version = spec.partition("==")
        key = re.sub(r"[-_.]+", "-", name.split("[")[0]).lower()
        if packages.get(key) != version:
            out.append(f"{key}=={version} pinned, {packages.get(key)} installed")
    return out


def resummarize(env_dir_arg: str) -> None:
    env_dir = Path(env_dir_arg)
    meta = json.loads((env_dir / "meta.json").read_text(encoding="utf-8"))
    benchmeta.write_json(env_dir / "summary.json", build_summary(env_dir, meta))
    print(env_dir.as_posix())


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.resummarize:
        resummarize(args.resummarize)
        return 0
    env = args.env
    check_platform(env)
    source = benchmeta.git_source(REPO, REPO_SLUG)
    if source["dirty"] and not args.allow_dirty:
        die(
            f"tracked files have uncommitted changes ({', '.join(source['modified'])});"
            " commit them, or pass --allow-dirty to run anyway (recorded as dirty)"
        )
    session = f"{args.session_date}-{source['sha'][:7]}"
    session += f"-{args.tag}" if args.tag else ""
    out_root = Path(args.out).resolve() if args.out else REPO / "bench" / "results"
    env_dir = out_root / session / env
    if env_dir.exists():
        die(
            f"refusing to overwrite existing results directory {env_dir.as_posix()};"
            " use another --tag or --session-date",
            3,
        )
    with tempfile.TemporaryDirectory(prefix="iqg-bench-") as tmp_name:
        tmp = Path(tmp_name)
        baseline = tmp / "baseline"
        baseline_sha = benchmeta.extract_ref(REPO, args.baseline_ref, baseline)
        if not (baseline / "app" / "main.py").is_file():
            die(f"baseline {args.baseline_ref} has no app/main.py")
        log(f"baseline {baseline_sha[:7]} extracted with git archive")
        (env_dir / "raw").mkdir(parents=True, exist_ok=False)
        measure(args, argv, env, env_dir, source, baseline, baseline_sha, tmp)
    print(env_dir.as_posix())
    return 0


def measure(
    args: argparse.Namespace,
    argv: list[str],
    env: str,
    env_dir: Path,
    source: dict[str, Any],
    baseline: Path,
    baseline_sha: str,
    tmp: Path,
) -> None:
    raw = env_dir / "raw"
    redact = redactor([(REPO, "<repo>"), (tmp, "<tmp>"), (Path.home(), "<home>")])
    py = sys.executable
    loop = ["--uvicorn-loop", args.uvicorn_loop]
    size = [
        "--rounds",
        str(args.rounds),
        "--photo-rounds",
        str(args.photo_rounds),
        "--pairs",
        str(args.pairs),
    ]
    limits = ["--runs", str(args.limits_runs)]
    pytest = ["-m", "pytest", "-q", "-p", "no:cacheprovider"]
    app_head = benchmeta.tree_hash(REPO, "HEAD", "app")
    app_ref = benchmeta.tree_hash(REPO, APP_REFERENCE_COMMIT, "app")
    workload: dict[str, Any] = {
        "smoke": args.smoke,
        "sessions": args.sessions,
        "rounds": args.rounds,
        "photo_rounds": args.photo_rounds,
        "pairs": args.pairs,
        "warmup_rounds": WARMUP_ROUNDS,
        "limits_runs": args.limits_runs,
        "baseline_ref": args.baseline_ref,
        "baseline_sha": baseline_sha,
        "baseline_method": "git archive <baseline_sha> extracted with tarfile into "
        "a temporary directory (<baseline>)",
        "uvicorn_loop": args.uvicorn_loop,
        "app_tree": {
            "head": app_head,
            APP_REFERENCE_COMMIT: app_ref,
            "equal": app_head is not None and app_head == app_ref,
        },
        "steps": [
            ["python", "scripts/bench_quality.py", "--baseline", "<baseline>", *size]
            + ["--json", "raw/session-NN.json", "--raw-dir", "raw/session-NN", *loop],
            ["python", "bench/limits_probe.py", "--baseline", "<baseline>", *limits]
            + ["--out", "raw/limits.json", *loop],
            ["python", *pytest],
        ],
        "validity": {
            "max_busy_pct": args.max_busy_pct,
            "quiet_timeout_s": args.quiet_timeout,
        },
    }
    command = shown_command(argv)
    started = benchmeta.collect(PROJECT, env, command, workload)
    log(f"collected environment ({started['hardware']['cpu_model']})")
    log(f"measuring into {env_dir.as_posix()}")

    runs: list[dict[str, Any]] = []
    for i in range(1, args.sessions + 1):
        name = f"session-{i:02d}"
        g = gate("session", args, env)
        g.update(run=i, file=f"raw/{name}.json", dir=f"raw/{name}")
        log(f"{name} of {args.sessions}: scripts/bench_quality.py {' '.join(size)}")
        argv_i = [py, "scripts/bench_quality.py", "--baseline", str(baseline), *size]
        argv_i += ["--json", str(raw / f"{name}.json"), "--raw-dir", str(raw / name)]
        code, text = run_step(argv_i + loop)
        (raw / f"{name}.txt").write_bytes(redact(text))
        for p in (raw / f"{name}.json", raw / name):
            if p.exists():
                redact_tree(p, redact)
        if code != 0 or not (raw / f"{name}.json").is_file():
            g["valid"] = False
            g["reasons"] = g["reasons"] + [f"bench_quality.py exit status {code}"]
        runs.append(g)

    if args.limits_runs > 0:
        g = gate("limits_probe", args, env)
        g["file"] = "raw/limits.json"
        log(f"bench/limits_probe.py --runs {args.limits_runs}")
        argv_l = [py, "bench/limits_probe.py", "--baseline", str(baseline), *limits]
        code, text = run_step(argv_l + ["--out", str(raw / "limits.json"), *loop])
        (raw / "limits.txt").write_bytes(redact(text))
        if (raw / "limits.json").is_file():
            redact_tree(raw / "limits.json", redact)
        if code != 0 or not (raw / "limits.json").is_file():
            g["valid"] = False
            g["reasons"] = g["reasons"] + [f"limits_probe.py exit status {code}"]
        runs.append(g)

    log("pytest -q")
    code, text = run_step([py, *pytest])
    (raw / "tests.txt").write_bytes(redact(text))
    tests = parse_pytest(redact(text).decode("utf-8", "replace"))
    log(f"tests: {tests}")

    ended = benchmeta.collect(PROJECT, env, command, workload)
    reasons = [f"{r['file']}: {x}" for r in runs for x in r["reasons"]]
    if code != 0 or not tests or tests["failed"]:
        reasons.append(f"pytest failed (exit status {code}; {tests})")
    packages = started["toolchain"]["packages"]
    mismatches = constraint_mismatches(packages)
    reasons += [f"not the bench/constraints-py311.txt pin: {m}" for m in mismatches]
    if not workload["app_tree"]["equal"]:
        reasons.append(f"app/ differs from app/ at {APP_REFERENCE_COMMIT[:7]}")

    loop_used = benchmeta.event_loop(args.uvicorn_loop)
    notes = [REDACTION_NOTE]
    if env == "windows-native":
        notes.append(
            "event loop asyncio: uvicorn's standard extra does not install uvloop on "
            "Windows (RESEARCH Pitfall 2); ubuntu-wsl2 runs uvloop unless "
            "--uvicorn-loop asyncio"
        )
    else:
        notes.append(
            "cpu_busy_pct and loadavg describe the WSL VM only, not Windows host "
            "activity (RESEARCH Pitfall 4)"
        )
    if source["dirty"]:
        notes.append(
            f"dirty tree (--allow-dirty): modified {source['modified']}, "
            f"untracked {source['untracked']}"
        )
    if args.smoke:
        notes.append("smoke run: reduced rounds, pairs and limits runs; not for claims")
    if started["power"]["ac_line"] != ended["power"]["ac_line"]:
        notes.append(
            f"AC line changed during the run: {started['power']['ac_line']} -> "
            f"{ended['power']['ac_line']}"
        )

    meta = started
    meta["load"]["loadavg_1m_end"] = ended["load"]["loadavg_1m_start"]
    meta["load"]["cpu_busy_pct_end"] = ended["load"]["cpu_busy_pct_start"]
    meta.update(
        {
            "session_date": args.session_date,
            "tag": args.tag,
            "source": source,
            "library": {
                "event_loop": loop_used,
                "uvicorn_loop_flag": args.uvicorn_loop,
                "server": "uvicorn, one worker per server, 127.0.0.1, default "
                "settings, LOG_LEVEL=WARNING",
                "baseline": {"ref": args.baseline_ref, "sha": baseline_sha},
                "constraints": "bench/constraints-py311.txt",
                "constraints_match": not mismatches,
            },
            "instrumentation": "iqg-diagnostics-endpoint: current_diagnostics and "
            "current_diagnostics_hist64 call POST /quality/diagnostics "
            f"(build {DIAG_BUILD}); baseline_quality, current_quality, the "
            "pipeline A/B and the input-limits probe are uninstrumented",
            "timer": benchmeta.perf_counter_timer(),
            "validity": {
                "valid": not reasons,
                "reasons": reasons,
                "runs": runs,
                "tests": tests,
            },
            "power_end": ended["power"],
            "date_utc_end": ended["date_utc"],
            "notes": notes,
        }
    )
    meta["raw_files"] = benchmeta.raw_file_entries(env_dir)
    benchmeta.write_json(env_dir / "meta.json", meta)
    benchmeta.write_json(env_dir / "summary.json", build_summary(env_dir, meta))
    log("valid" if not reasons else f"INVALID: {'; '.join(reasons)}")


if __name__ == "__main__":
    sys.exit(main())
