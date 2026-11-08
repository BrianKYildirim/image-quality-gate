"""Re-measures every row of the README input-limits table, before and after.

  python bench/limits_probe.py --baseline-ref 8e47d74 --runs 5 --out limits.json
  python bench/limits_probe.py --baseline <checkout> --runs 5 --out limits.json

"before" is a checkout of the code before the input limits were added (--baseline,
or --baseline-ref extracted with git archive); "after" is this tree. For every run,
case and variant it starts a fresh single-worker uvicorn server on 127.0.0.1 with
default settings, reads the server's peak resident memory, sends one request,
reads the peak again and stops the server. It records the status code, the
client-side elapsed time (first byte sent to response body read) and the growth of
the server's peak resident memory: PeakWorkingSetSize on Windows (through
K32GetProcessMemoryInfo on the server PID), VmHWM from /proc/<pid>/status on Linux.

Cases (the README rows):
  upload_100mib_content_length  100 MiB multipart upload with Content-Length
  upload_100mib_chunked         the same upload with Transfer-Encoding: chunked
  png_bomb_12000x12000          small solid-gray PNG that decodes to 144 MP
  truncated_sharp_small         the first half of tests/assets/sharp_small.jpg
  jpeg_5000x5000                valid photo-like JPEG of exactly 25 MP

The client never sends Expect: 100-continue. While uploading it checks for an early
response before every 64 KiB and stops sending when one arrives, as curl does.
Inputs are generated deterministically (Pillow, scripts/bench_quality.py
make_photo). Standard library plus the app's own Pillow and NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_quality  # noqa: E402
import benchmeta  # noqa: E402

MIB = 2**20
UPLOAD_BYTES = 100 * MIB
SEND_CHUNK = 64 * 1024
TIMEOUT_S = 180.0
BOUNDARY = bench_quality.BOUNDARY
# Settings fields of app/core/config.py; removed from the server environment so
# every server runs with the defaults.
SETTINGS_ENV = set(bench_quality.SETTINGS_ENV) | {"max_image_pixels"}
VARIANTS = ("before", "after")
CASES = (
    "upload_100mib_content_length",
    "upload_100mib_chunked",
    "png_bomb_12000x12000",
    "truncated_sharp_small",
    "jpeg_5000x5000",
)
# Runs in the server interpreter itself, so the PID is the process that serves
# (on Windows a venv's python.exe is a launcher that starts the real interpreter
# as a child process).
SERVER_CODE = (
    "import os, sys\n"
    "with open(os.environ['IQG_PROBE_PID_FILE'], 'w') as f:\n"
    "    f.write(str(os.getpid()))\n"
    "from uvicorn.main import main\n"
    "sys.argv[0] = 'uvicorn'\n"
    "main()\n"
)


# ---------- Inputs ----------


def part_head(filename: str, content_type: str) -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()


PART_TAIL = f"\r\n--{BOUNDARY}--\r\n".encode()


def make_inputs() -> dict[str, dict[str, Any]]:
    """Every case: the file bytes (None for the generated zero upload) and facts."""
    from PIL import Image

    buf = BytesIO()
    Image.new("L", (12000, 12000), 128).save(buf, format="PNG")
    bomb = buf.getvalue()
    sharp = (REPO / "tests" / "assets" / "sharp_small.jpg").read_bytes()
    truncated = sharp[: len(sharp) // 2]
    photo = bench_quality.make_photo(5000, 5000)
    upload = {
        "filename": "zeros.jpg",
        "content_type": "image/jpeg",
        "data": None,
        "input_bytes": UPLOAD_BYTES,
        "sha256": None,
    }
    cases: dict[str, dict[str, Any]] = {
        "upload_100mib_content_length": {
            **upload,
            "transfer": "content-length",
            "description": "100 MiB multipart upload (zero bytes) with "
            "Content-Length",
        },
        "upload_100mib_chunked": {
            **upload,
            "transfer": "chunked",
            "description": "100 MiB multipart upload (zero bytes), "
            "Transfer-Encoding: chunked, 64 KiB chunks, no Content-Length",
        },
        "png_bomb_12000x12000": {
            "filename": "bomb.png",
            "content_type": "image/png",
            "data": bomb,
            "transfer": "content-length",
            "description": "solid gray (L, 128) PNG of 12000 x 12000 = 144 MP, "
            "Pillow default compression",
        },
        "truncated_sharp_small": {
            "filename": "sharp_small.jpg",
            "content_type": "image/jpeg",
            "data": truncated,
            "transfer": "content-length",
            "description": "first half of tests/assets/sharp_small.jpg "
            f"({len(truncated)} of {len(sharp)} bytes)",
        },
        "jpeg_5000x5000": {
            "filename": "photo_5000x5000.jpg",
            "content_type": "image/jpeg",
            "data": photo,
            "transfer": "content-length",
            "description": "valid 5000 x 5000 (exactly 25 MP) photo-like JPEG, "
            "bench_quality.make_photo",
        },
    }
    for info in cases.values():
        if info["data"] is not None:
            info["input_bytes"] = len(info["data"])
            info["sha256"] = hashlib.sha256(info["data"]).hexdigest()
    return cases


def body_pieces(info: dict[str, Any]) -> tuple[list[tuple[str, str]], Iterator[bytes]]:
    """(extra headers, body pieces already framed for the transfer encoding)."""
    head = part_head(info["filename"], info["content_type"])

    def file_pieces() -> Iterator[bytes]:
        if info["data"] is not None:
            data = info["data"]
            for i in range(0, len(data), SEND_CHUNK):
                yield data[i : i + SEND_CHUNK]
            return
        zeros = bytes(SEND_CHUNK)
        for _ in range(UPLOAD_BYTES // SEND_CHUNK):
            yield zeros

    def multipart() -> Iterator[bytes]:
        yield head
        yield from file_pieces()
        yield PART_TAIL

    if info["transfer"] == "chunked":

        def framed() -> Iterator[bytes]:
            for piece in multipart():
                yield f"{len(piece):x}\r\n".encode() + piece + b"\r\n"
            yield b"0\r\n\r\n"

        return [("Transfer-Encoding", "chunked")], framed()
    length = len(head) + info["input_bytes"] + len(PART_TAIL)
    return [("Content-Length", str(length))], multipart()


# ---------- Server ----------


def peak_resident_bytes(pid: int) -> int | None:
    """Peak resident memory of a process: PeakWorkingSetSize or VmHWM."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
        handle = k32.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ok = k32.K32GetProcessMemoryInfo(
                wintypes.HANDLE(handle), ctypes.byref(counters), counters.cb
            )
            return int(counters.PeakWorkingSetSize) if ok else None
        finally:
            k32.CloseHandle(wintypes.HANDLE(handle))
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


MEMORY_SOURCE = (
    "PeakWorkingSetSize (K32GetProcessMemoryInfo on the server PID)"
    if sys.platform == "win32"
    else "VmHWM from /proc/<pid>/status"
)


class Server:
    """A fresh single-worker uvicorn server for one request."""

    def __init__(self, app_dir: Path, workdir: Path, loop: str) -> None:
        self.port = bench_quality.free_port()
        pidfile = workdir / f"server-{self.port}.pid"
        env = {k: v for k, v in os.environ.items() if k.lower() not in SETTINGS_ENV}
        env.update(
            LOG_LEVEL="WARNING",
            PYTHONWARNINGS="ignore",
            PYTHONDONTWRITEBYTECODE="1",
            IQG_PROBE_PID_FILE=str(pidfile),
        )
        cmd = [sys.executable, "-c", SERVER_CODE, "app.main:app"]
        cmd += ["--app-dir", str(app_dir), "--host", "127.0.0.1"]
        cmd += ["--port", str(self.port), "--log-level", "warning", "--loop", loop]
        self.log = open(workdir / f"server-{self.port}.err", "wb")
        self.proc = subprocess.Popen(
            cmd, cwd=workdir, env=env, stdout=subprocess.DEVNULL, stderr=self.log
        )
        self.pid = self._wait_ready(pidfile)

    def _wait_ready(self, pidfile: Path) -> int:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.request("GET", "/health", headers={"Connection": "close"})
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status == 200 and pidfile.is_file():
                    return int(pidfile.read_text())
            except OSError:
                time.sleep(0.2)
        self.stop()
        sys.exit(f"server on port {self.port} did not start")

    def stop(self) -> None:
        pid = getattr(self, "pid", None)
        if pid is not None and pid != self.proc.pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass  # already gone
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


# ---------- Client ----------


def response_complete(data: bytes) -> bool:
    end = data.find(b"\r\n\r\n")
    if end < 0:
        return False
    for line in data[:end].split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            return len(data) - end - 4 >= int(value)
    return False  # read to EOF


def exchange(port: int, info: dict[str, Any]) -> dict[str, Any]:
    """One request: send (stopping at an early response), then read the response."""
    headers, pieces = body_pieces(info)
    head = f"POST /quality HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
    head += f"Content-Type: multipart/form-data; boundary={BOUNDARY}\r\n"
    head += "".join(f"{k}: {v}\r\n" for k, v in headers) + "Connection: close\r\n\r\n"
    sock = socket.create_connection(("127.0.0.1", port), timeout=TIMEOUT_S)
    sock.setblocking(False)
    deadline = time.monotonic() + TIMEOUT_S
    sent, stopped = 0, None
    pending = memoryview(head.encode())
    t0 = time.perf_counter_ns()
    try:
        while stopped is None:
            if not pending:
                nxt = next(pieces, None)
                if nxt is None:
                    break
                pending = memoryview(nxt)
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("upload timed out")
            readable, writable, _ = select.select([sock], [sock], [], left)
            if readable:
                stopped = "response arrived before the upload finished"
            elif writable:
                try:
                    n = sock.send(pending[:SEND_CHUNK])
                except BlockingIOError:
                    continue
                except OSError as e:  # reset or broken pipe: the server closed
                    stopped = f"send stopped: {type(e).__name__}"
                    break
                sent += n
                pending = pending[n:]
        data = b""
        while not response_complete(data):
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("response timed out")
            readable, _, _ = select.select([sock], [], [], left)
            if not readable:
                continue
            try:
                chunk = sock.recv(65536)
            except BlockingIOError:
                continue
            except OSError:
                break  # reset after the response (or before any)
            if not chunk:
                break
            data += chunk
        elapsed = time.perf_counter_ns() - t0
    finally:
        sock.close()
    status = None
    if data.startswith(b"HTTP/"):
        status = int(data.split(b"\r\n", 1)[0].split()[1])
    body = data.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in data else b""
    try:
        parsed = json.loads(body) if body and len(body) <= 4096 else None
    except ValueError:
        parsed = None
    return {
        "status": status,
        "elapsed_ns": elapsed,
        "sent_bytes": sent,
        "stopped_early": stopped,
        "response_bytes": len(body),
        "response_sha256": hashlib.sha256(body).hexdigest() if body else None,
        "response": parsed,
    }


# ---------- Probe ----------


def probe(
    baseline: Path,
    runs: int,
    cases: list[str],
    loop: str,
    inputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    apps = {"before": baseline.resolve(), "after": REPO}
    out = []
    with tempfile.TemporaryDirectory(prefix="iqg-limits-") as tmp:
        for run in range(1, runs + 1):
            for index, case in enumerate(cases):
                order = VARIANTS if (run + index) % 2 else VARIANTS[::-1]
                for variant in order:
                    server = Server(apps[variant], Path(tmp), loop)
                    try:
                        before = peak_resident_bytes(server.pid)
                        result = exchange(server.port, inputs[case])
                        after = peak_resident_bytes(server.pid)
                    finally:
                        server.stop()
                    growth = None
                    if before is not None and after is not None:
                        growth = after - before
                    out.append(
                        {
                            "run": run,
                            "case": case,
                            "variant": variant,
                            "peak_before_bytes": before,
                            "peak_after_bytes": after,
                            "peak_growth_bytes": growth,
                            **result,
                        }
                    )
                    ms = result["elapsed_ns"] / 1e6
                    mib = "?" if growth is None else f"{growth / MIB:+.1f} MiB"
                    print(
                        f"run {run} {case:30s} {variant:6s} {result['status']} "
                        f"{ms:9.1f} ms {mib}",
                        flush=True,
                    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", type=Path, help="checkout of the 'before' code")
    ap.add_argument("--baseline-ref", help="or: commit to extract with git archive")
    ap.add_argument("--runs", type=int, default=5, help="fresh servers per case (5)")
    ap.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    ap.add_argument("--uvicorn-loop", choices=("auto", "asyncio"), default="auto")
    ap.add_argument("--out", type=Path, help="write every run as JSON")
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    with tempfile.TemporaryDirectory(prefix="iqg-limits-base-") as tmp:
        baseline = args.baseline
        if baseline is None:
            if not args.baseline_ref:
                ap.error("give --baseline or --baseline-ref")
            baseline = Path(tmp) / "baseline"
            benchmeta.extract_ref(REPO, args.baseline_ref, baseline)
        if not (baseline / "app" / "main.py").is_file():
            sys.exit(f"--baseline has no app/main.py: {baseline}")
        inputs = make_inputs()
        measurements = probe(baseline, args.runs, args.cases, args.uvicorn_loop, inputs)
    report = {
        "schema": "iqg-limits/1",
        "runs": args.runs,
        "variants": {
            "before": "the baseline checkout (--baseline / --baseline-ref)",
            "after": "this tree",
        },
        "server": "fresh single-worker uvicorn per run, case and variant; "
        "127.0.0.1; default settings; one GET /health before the request",
        "uvicorn_loop": args.uvicorn_loop,
        "event_loop": benchmeta.event_loop(args.uvicorn_loop),
        "memory_source": MEMORY_SOURCE,
        "elapsed": "client side, time.perf_counter_ns from the first byte sent to "
        "the response body read; no Expect: 100-continue",
        "cases": {
            case: {k: v for k, v in inputs[case].items() if k != "data"}
            for case in args.cases
        },
        "measurements": measurements,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        benchmeta.write_json(args.out, report)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
