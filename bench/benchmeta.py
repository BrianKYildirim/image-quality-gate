"""Environment metadata for benchmark results (schema bench-meta/1).

Used by bench/run_bench.py. Collects the facts a reader needs to reproduce or judge a
run: processor name, cores and memory, OS and WSL details, AC power and power scheme,
CPU busy and load average, the Python interpreter and installed package versions, the
event loop uvicorn uses, and the git state of the measured source. Standard library
only (Python 3.11). Adapted from ratelimit-cpp's bench/benchmeta.py; the two repos
stay independent.

  python bench/benchmeta.py
      print what collect() records on this machine, as JSON

Windows values come from the registry and kernel32 (winreg, ctypes), not
platform.processor(), which reports "Intel64 Family 6 Model ..." instead of the
processor name. Under WSL the CPU busy and load figures describe the WSL VM only; AC
power and the power scheme are read from the Windows host through interop when it is
available, and are recorded as not observed otherwise.
"""

from __future__ import annotations

import configparser
import datetime
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "bench-meta/1"
RESULTS_PREFIX = "bench/results/"

IS_WINDOWS = sys.platform == "win32"


def _run(args: list[str], timeout: float = 30, cwd: str | None = None) -> str | None:
    """stdout of a helper command as text, or None if it cannot run."""
    try:
        p = subprocess.run(
            args, capture_output=True, timeout=timeout, cwd=cwd, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    out = p.stdout
    if b"\x00" in out:  # wsl.exe writes UTF-16LE
        return out.decode("utf-16-le", errors="replace").replace("\x00", "")
    return out.decode("utf-8", errors="replace")


def _read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def is_wsl() -> bool:
    version = (_read("/proc/version") or "").lower()
    return sys.platform.startswith("linux") and "microsoft" in version


def utc_now() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- Hardware ----------


def _win_reg(path: str, name: str) -> Any:
    if not IS_WINDOWS:
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k:
            return winreg.QueryValueEx(k, name)[0]
    except OSError:
        return None


def cpu_model() -> str | None:
    """Processor name: ProcessorNameString on Windows, /proc/cpuinfo on Linux."""
    if IS_WINDOWS:
        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        name = _win_reg(key, "ProcessorNameString")
        return str(name).strip() if name else None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def physical_cores() -> int | None:
    if IS_WINDOWS:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        size = wintypes.DWORD(0)
        # 0 = RelationProcessorCore
        k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
            return None
        count, off = 0, 0
        while off < size.value:
            relation, entry = struct.unpack_from("<II", buf.raw, off)
            count += relation == 0
            off += entry
        return count or None
    pairs: set[tuple[str | None, str]] = set()
    cores_field: int | None = None
    phys: str | None = None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        key, _, value = (s.strip() for s in line.partition(":"))
        if key == "physical id":
            phys = value
        elif key == "core id":
            pairs.add((phys, value))
        elif key == "cpu cores" and cores_field is None:
            cores_field = int(value)
    return len(pairs) or cores_field


def memory_bytes() -> int | None:
    if IS_WINDOWS:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
            ] + [
                (n, ctypes.c_ulonglong)
                for n in (
                    "ullTotalPhys",
                    "ullAvailPhys",
                    "ullTotalPageFile",
                    "ullAvailPageFile",
                    "ullTotalVirtual",
                    "ullAvailVirtual",
                    "ullAvailExtendedVirtual",
                )
            ]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullTotalPhys) if ok else None
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def hardware() -> dict[str, Any]:
    return {
        "cpu_model": cpu_model(),
        "physical_cores": physical_cores(),
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory_bytes(),
    }


# ---------- Load and power ----------


def _system_times() -> tuple[int, int] | None:
    """(busy, total) CPU time counters since boot, in arbitrary units."""
    if IS_WINDOWS:
        import ctypes

        idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        total = kernel.value + user.value  # kernel time includes idle time
        return total - idle.value, total
    for line in (_read("/proc/stat") or "").splitlines():
        if line.startswith("cpu "):
            f = [int(x) for x in line.split()[1:]]
            idle_time = f[3] + (f[4] if len(f) > 4 else 0)  # idle + iowait
            total_time = sum(f[:8])  # guest time is already counted in user
            return total_time - idle_time, total_time
    return None


def cpu_busy_pct(sample_s: float = 1.0) -> float | None:
    """Percent of all CPUs busy over sample_s seconds (GetSystemTimes, /proc/stat)."""
    a = _system_times()
    time.sleep(sample_s)
    b = _system_times()
    if not a or not b or b[1] == a[1]:
        return None
    return round(100.0 * (b[0] - a[0]) / (b[1] - a[1]), 2)


def loadavg() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):  # not available on Windows
        return None


def _scheme_name(text: str | None) -> str | None:
    for line in (text or "").splitlines():
        m = re.search(r"\(([^()]+)\)\s*\*?\s*$", line.strip())
        if m:
            return m.group(1).strip()
    return None


def power_status() -> dict[str, Any]:
    """{ ac_line, scheme[, note] }. ac_line is None (with a note) if not observed."""
    out: dict[str, Any]
    if IS_WINDOWS:
        import ctypes
        from ctypes import wintypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", wintypes.DWORD),
                ("BatteryFullLifeTime", wintypes.DWORD),
            ]

        s = SYSTEM_POWER_STATUS()
        ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
        ac = {0: False, 1: True}.get(s.ACLineStatus) if ok else None
        scheme = _scheme_name(_run(["powercfg", "/getactivescheme"]))
        out = {
            "ac_line": ac,
            "scheme": scheme or "unknown (powercfg failed)",
            "source": "GetSystemPowerStatus; powercfg /getactivescheme",
        }
        if ac is None:
            out["note"] = "GetSystemPowerStatus reported the AC line status as unknown"
        return out
    if is_wsl():  # Windows host through interop (RESEARCH A3)
        line = _run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; "
                "[System.Windows.Forms.SystemInformation]::PowerStatus"
                ".PowerLineStatus",
            ],
            cwd="/mnt/c",
        )
        ac = {"online": True, "offline": False}.get((line or "").strip().lower())
        scheme = _scheme_name(_run(["powercfg.exe", "/getactivescheme"], cwd="/mnt/c"))
        out = {
            "ac_line": ac,
            "scheme": scheme
            or "unknown (powercfg.exe not reachable through WSL interop)",
            "source": "Windows host through WSL interop: "
            "PowerStatus.PowerLineStatus; powercfg.exe /getactivescheme",
        }
        if ac is None:
            out["note"] = (
                "AC line status not observed: powershell.exe was not reachable "
                "through WSL interop"
            )
        return out
    ac = None
    for p in sorted(Path("/sys/class/power_supply").glob("*/online")):
        ac = (_read(p) or "").strip() == "1"
    out = {
        "ac_line": ac,
        "scheme": "unknown (no power scheme on this OS)",
        "source": "/sys/class/power_supply",
    }
    if ac is None:
        out["note"] = "no AC adapter is visible in /sys/class/power_supply"
    return out


# ---------- OS ----------


def wsl_info() -> dict[str, Any] | None:
    """WSL version, .wslconfig (parsed, or None when absent), distro and CPUs."""
    if not is_wsl():
        return None
    version = None
    text = _run(["wsl.exe", "--version"], cwd="/mnt/c")
    m = re.search(r"WSL[^:\n]*:\s*([0-9][0-9.]*)", text or "")
    if m:
        version = m.group(1)
    if not version:
        proc_version = (_read("/proc/version") or "").lower()
        version = "2" if "wsl2" in proc_version else "unknown"
    wslconfig = None
    echo = _run(["cmd.exe", "/c", "echo %USERPROFILE%"], cwd="/mnt/c")
    profile = (echo or "").strip()
    linux_profile = (_run(["wslpath", "-u", profile]) or "").strip() if profile else ""
    cfg_path = Path(linux_profile) / ".wslconfig" if linux_profile else None
    if cfg_path and cfg_path.is_file():
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(cfg_path.read_text(encoding="utf-8", errors="replace"))
        wslconfig = {s: dict(parser.items(s)) for s in parser.sections()}
    if cfg_path is None:
        note: str | None = "Windows user profile not reachable through interop"
    else:
        note = "absent: WSL defaults apply" if wslconfig is None else None
    return {
        "version": version,
        "wslconfig": wslconfig,
        "distro": os.environ.get("WSL_DISTRO_NAME"),
        "nproc": os.cpu_count(),
        "wslconfig_note": note,
    }


def os_info(environment_id: str) -> dict[str, Any]:
    if IS_WINDOWS:
        nt = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        build = str(
            _win_reg(nt, "CurrentBuildNumber") or platform.version().split(".")[-1]
        )
        ubr = _win_reg(nt, "UBR")
        name = str(_win_reg(nt, "ProductName") or "Windows")
        # ProductName still says "Windows 10" on Windows 11
        if build.isdigit() and int(build) >= 22000:
            name = name.replace("Windows 10", "Windows 11")
        return {
            "family": "Windows",
            "name": name,
            "build": f"{build}.{ubr}" if ubr is not None else build,
            "version": platform.version(),
            "display_version": _win_reg(nt, "DisplayVersion"),
            "kernel": None,
            "wsl": None,
        }
    rel: dict[str, str] = {}
    for line in (_read("/etc/os-release") or "").splitlines():
        k, _, v = line.partition("=")
        rel[k] = v.strip().strip('"')
    wsl = wsl_info()
    if environment_id == "ubuntu-wsl2" and wsl is None:
        wsl = {"version": "not detected", "wslconfig": None}
    return {
        "family": platform.system(),
        "name": rel.get("PRETTY_NAME") or platform.system(),
        "build": rel.get("VERSION_ID") or platform.version(),
        "kernel": platform.release(),
        "libc": " ".join(platform.libc_ver()).strip() or None,
        "wsl": wsl,
    }


# ---------- Python ----------


def packages() -> dict[str, str]:
    """name -> version of every distribution installed for this interpreter."""
    out: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            out[re.sub(r"[-_.]+", "-", name).lower()] = dist.version
    return dict(sorted(out.items()))


def python_toolchain() -> dict[str, Any]:
    """Python shape of meta.toolchain: version, build string, bare executable name."""
    return {
        "version": platform.python_version(),
        "implementation": sys.implementation.name,
        "build": sys.version,
        "executable_name": Path(sys.executable).name,
        "packages": packages(),
    }


def event_loop(uvicorn_loop: str = "auto") -> str:
    """The loop uvicorn runs: uvloop when --loop auto finds it installed, else asyncio."""
    if uvicorn_loop == "auto" and importlib.util.find_spec("uvloop") is not None:
        return "uvloop"
    return "asyncio"


def perf_counter_timer() -> dict[str, Any]:
    info = time.get_clock_info("perf_counter")
    return {
        "name": f"time.perf_counter_ns ({info.implementation})",
        "resolution_ns": info.resolution * 1e9,
        "overhead_ns": None,
        "overhead_note": "not subtracted; a request takes milliseconds",
        "monotonic": info.monotonic,
    }


# ---------- Source ----------


def _git(repo_dir: Path, *args: str) -> str:
    p = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return p.stdout


def git_source(repo_dir: Path, repo_slug: str) -> dict[str, Any]:
    """sha, dirty (tracked changes outside bench/results/), paths and git describe."""

    def paths(porcelain: str) -> list[tuple[str, str]]:
        out = []
        for line in porcelain.splitlines():
            if len(line) > 3:
                path = line[3:].split(" -> ")[-1].strip('"')
                if not path.startswith(RESULTS_PREFIX):
                    out.append((line[:2], path))
        return out

    tracked = paths(_git(repo_dir, "status", "--porcelain", "--untracked-files=no"))
    everything = _git(repo_dir, "status", "--porcelain", "--untracked-files=all")
    untracked = [p for code, p in paths(everything) if code == "??"]
    return {
        "repo": repo_slug,
        "sha": _git(repo_dir, "rev-parse", "HEAD").strip(),
        "dirty": bool(tracked),
        "modified": sorted(p for _, p in tracked),
        "untracked": sorted(untracked),
        "describe": _git(repo_dir, "describe", "--always", "--dirty").strip(),
        "branch": _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD").strip(),
    }


def tree_hash(repo_dir: Path, rev: str, path: str) -> str | None:
    """git tree id of <rev>:<path>, or None when it cannot be resolved."""
    try:
        return _git(repo_dir, "rev-parse", f"{rev}:{path}").strip()
    except subprocess.CalledProcessError:
        return None


def extract_ref(repo_dir: Path, ref: str, dest: Path) -> str:
    """Extract `git archive <ref>` into dest with tarfile (portable: no git worktree,
    no tar binary). Returns the full commit sha of ref."""
    sha = _git(repo_dir, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    archive = subprocess.run(
        ["git", "-C", str(repo_dir), "archive", "--format=tar", sha],
        capture_output=True,
        check=True,
    ).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(dest, filter="data")
    return sha


# ---------- Files ----------


def raw_file_entries(env_dir: Path) -> list[dict[str, Any]]:
    """[{ path, bytes, sha256 }] for every file under <env_dir>/raw, sorted by path."""
    out = []
    for f in (env_dir / "raw").rglob("*"):
        if f.is_file():
            data = f.read_bytes()
            out.append(
                {
                    "path": f.relative_to(env_dir).as_posix(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return sorted(out, key=lambda e: str(e["path"]))


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path: Path, obj: Any) -> None:
    """Deterministic JSON: indent 1, sorted keys, UTF-8, LF, trailing newline."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(dumps(obj))


# ---------- Assembly ----------


def load_sample() -> tuple[float | None, float | None]:
    """(loadavg_1m, cpu_busy_pct) now. Windows has no load average; busy takes 1 s."""
    return loadavg(), cpu_busy_pct(1.0)


def collect(
    project: str,
    environment_id: str,
    command: list[str],
    workload: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A bench-meta/1 document. extra supplies the run-specific fields (source,
    toolchain, library, timer, validity, raw_files, notes, ...) and overrides
    anything collected here. load_*_end fields start equal to the start sample; the
    runner replaces them with a second collect() after the measured steps."""
    la, busy = load_sample()
    meta: dict[str, Any] = {
        "schema": SCHEMA,
        "project": project,
        "environment_id": environment_id,
        "date_utc": utc_now(),
        "command": list(command),
        "workload": workload,
        "hardware": hardware(),
        "os": os_info(environment_id),
        "power": power_status(),
        "load": {
            "loadavg_1m_start": la,
            "loadavg_1m_end": la,
            "cpu_busy_pct_start": busy,
            "cpu_busy_pct_end": busy,
        },
        "toolchain": python_toolchain(),
        "library": {},
        "instrumentation": "none",
        "notes": [],
    }
    meta.update(extra or {})
    return meta


if __name__ == "__main__":
    env_id = "ubuntu-wsl2" if is_wsl() else "windows-native" if IS_WINDOWS else "linux"
    probe = collect("probe", env_id, ["python", "bench/benchmeta.py"], {"probe": True})
    print(dumps(probe), end="")
