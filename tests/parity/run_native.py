#!/usr/bin/env python3
"""
run_native.py - The only writer of the parity golden files.

Runs every case in cases.json through app.services.pipeline.outcome (the entry
both API routes and the portfolio's Pyodide demo use) with the case's settings
and the case list's histogram bin count, and checks each result against the
case's stated intent.

  python tests/parity/run_native.py --write   # write golden/<case>.json and meta.json
  python tests/parity/run_native.py --check   # regenerate in memory, compare bytes

Local only: the goldens record one exact environment (the bench venv built from
bench/constraints-py311.txt), while upstream CI installs unpinned requirements,
so no pytest test compares them.
"""

import argparse
import ast
import datetime
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

PARITY = Path(__file__).resolve().parent
REPO = PARITY.parents[1]
FIXTURES = PARITY / "fixtures"
GOLDEN = PARITY / "golden"
CASES = PARITY / "cases.json"
SCHEMA = "iqg-parity-golden/1"
# The pipeline's import closure: the files the portfolio vendors and runs.
SOURCES = (
    "app/__init__.py",
    "app/core/__init__.py",
    "app/core/config.py",
    "app/core/timing.py",
    "app/models/__init__.py",
    "app/models/schemas.py",
    "app/services/__init__.py",
    "app/services/preprocess.py",
    "app/services/quality.py",
    "app/services/diagnostics.py",
    "app/services/pipeline.py",
)
EXPECT_KEYS = {"status", "is_ok", "cause", "blur_margin_max"}
_isolation: list[tempfile.TemporaryDirectory] = []


def _settings_fields() -> set[str]:
    # Read from the source, because importing app already builds a Settings.
    tree = ast.parse((REPO / "app/core/config.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            return {
                n.target.id
                for n in node.body
                if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and not n.target.id.startswith("_")
            }
    raise SystemExit("run_native: no Settings class in app/core/config.py")


def isolate() -> None:
    """
    Settings exactly as the browser sees them: no settings environment variable
    and no .env (an empty temporary working directory). Call before importing app.
    """
    fields = _settings_fields()
    for key in list(os.environ):
        if key.lower() in fields:
            del os.environ[key]
    tmp = tempfile.TemporaryDirectory(prefix="iqg-parity-")
    _isolation.append(tmp)
    os.chdir(tmp.name)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from app.core.config import Settings

    if set(Settings.model_fields) != fields:
        raise SystemExit("run_native: Settings fields differ from the parsed source")


def sha256_lf(path: Path) -> str:
    """sha256 of a text file as git stores it (CRLF turned into LF)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")


def load_cases() -> dict:
    return json.loads(CASES.read_text(encoding="utf-8"))


def golden_for(case: dict, bins: int) -> dict:
    from app.core.config import Settings
    from app.services import pipeline

    data = (PARITY / case["file"]).read_bytes()
    s = Settings(**case["settings"])
    out = pipeline.outcome(data, case["content_type"], s, bins)
    if out["status"] != 200:
        return {
            "error": {
                "status": out["status"],
                "detail": out["detail"],
                "cause": out["cause"],
            }
        }
    golden = json.loads(str(out["response_json"]))
    del golden["diagnostics"]["timings"]
    del golden["diagnostics"]["versions"]
    return golden


def contradiction(case: dict, golden: dict) -> str | None:
    """Why the golden contradicts the case's stated intent, or None."""
    expect = case["expect"]
    unknown = set(expect) - EXPECT_KEYS
    if unknown:
        return f"unknown expect keys {sorted(unknown)}"
    error = golden.get("error")
    status = error["status"] if error else 200
    if status != expect["status"]:
        return f"status {status}, expected {expect['status']}"
    if "is_ok" in expect and (error or golden["is_ok"] != expect["is_ok"]):
        return f"is_ok {None if error else golden['is_ok']}, expected {expect['is_ok']}"
    if "cause" in expect and (error or {}).get("cause") != expect["cause"]:
        return f"cause {(error or {}).get('cause')}, expected {expect['cause']}"
    if "blur_margin_max" in expect:
        # The blur check passes, by no more than the stated margin.
        checks = [] if error else golden["diagnostics"]["decision"]["checks"]
        margins = [c["margin"] for c in checks if c["name"] == "blur_min"]
        if len(margins) != 1 or not 0 <= margins[0] <= expect["blur_margin_max"]:
            return f"blur margin {margins}, expected 0 to {expect['blur_margin_max']}"
    return None


def meta(cases: dict) -> dict:
    import pydantic
    import pydantic_settings

    from app.core.config import Settings
    from app.services import diagnostics

    versions = diagnostics.versions(Settings()).model_dump()
    del versions["app"]
    versions["pydantic"] = pydantic.VERSION
    versions["pydantic_settings"] = pydantic_settings.__version__
    return {
        "schema": SCHEMA,
        "environment": "ubuntu-wsl2",
        "python": platform.python_version(),
        "versions": versions,
        "bins": cases["bins"],
        "command": "python tests/parity/run_native.py --write",
        "date": datetime.date.today().isoformat(),
        "sources": {path: sha256_lf(REPO / path) for path in SOURCES},
        "fixtures": {
            f"fixtures/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(FIXTURES.iterdir())
            if p.is_file()
        },
        "cases_sha256": sha256_lf(CASES),
    }


def build() -> tuple[dict[str, bytes], dict]:
    """Every case's golden bytes (in case order) and the meta object."""
    cases = load_cases()
    names = [case["name"] for case in cases["cases"]]
    if len(set(names)) != len(names) or "meta" in names:
        raise SystemExit("run_native: case names must be unique and not 'meta'")
    used = {case["file"] for case in cases["cases"]}
    on_disk = {f"fixtures/{p.name}" for p in FIXTURES.iterdir() if p.is_file()}
    if used != on_disk:
        raise SystemExit(
            f"run_native: fixtures without a case {sorted(on_disk - used)}, "
            f"cases without a fixture {sorted(used - on_disk)}"
        )
    goldens: dict[str, bytes] = {}
    for case in cases["cases"]:
        golden = golden_for(case, cases["bins"])
        why = contradiction(case, golden)
        if why:
            print(f"PARITY_INTENT_MISMATCH {case['name']}: {why}")
            raise SystemExit(1)
        goldens[case["name"]] = dump(golden)
    return goldens, meta(cases)


def write(goldens: dict[str, bytes], info: dict) -> None:
    GOLDEN.mkdir(exist_ok=True)
    keep = {f"{name}.json" for name in goldens} | {"meta.json"}
    for stale in GOLDEN.glob("*.json"):
        if stale.name not in keep:
            stale.unlink()
    for name, data in goldens.items():
        (GOLDEN / f"{name}.json").write_bytes(data)
    (GOLDEN / "meta.json").write_bytes(dump(info))
    print(f"PARITY_GOLDEN_WRITTEN {len(goldens)} cases")


def check(goldens: dict[str, bytes], info: dict) -> int:
    expected = {f"{name}.json" for name in goldens} | {"meta.json"}
    present = {p.name for p in GOLDEN.glob("*.json")}
    if present != expected:
        print(
            f"PARITY_GOLDEN_DIFF files: missing {sorted(expected - present)}, "
            f"extra {sorted(present - expected)}"
        )
        return 1
    for name, data in goldens.items():
        if (GOLDEN / f"{name}.json").read_bytes() != data:
            print(f"PARITY_GOLDEN_DIFF {name}.json differs from the native result")
            return 1
    stored = json.loads((GOLDEN / "meta.json").read_text(encoding="utf-8"))
    for key in ("schema", "versions", "bins", "sources", "fixtures", "cases_sha256"):
        if stored.get(key) != info[key]:
            print(
                f"PARITY_GOLDEN_DIFF meta.json {key}: {stored.get(key)} != {info[key]}"
            )
            return 1
    print(f"PARITY_GOLDEN_OK {len(goldens)} cases")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the golden files")
    mode.add_argument("--check", action="store_true", help="compare with them")
    args = ap.parse_args()
    isolate()
    goldens, info = build()
    if args.write:
        write(goldens, info)
        return 0
    return check(goldens, info)


if __name__ == "__main__":
    sys.exit(main())
