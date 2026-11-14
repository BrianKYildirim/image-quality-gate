#!/usr/bin/env python3
"""
make_fixtures.py - Writes the parity fixtures into tests/parity/fixtures/.

Every file is either a copy of a test asset or generated deterministically, so
running it again gives the same bytes. Byte-identical output holds for the
pinned bench venv (bench/constraints-py311.txt), whose Pillow and numpy versions
fix the encoders.

  python tests/parity/make_fixtures.py
  python tests/parity/make_fixtures.py --memory-probe DIR   # only the large photo

--memory-probe writes the 5000 x 5000 photo used to measure the browser demo's
memory to DIR, which must be outside this repository, and nothing else. It is
not a parity fixture.
"""

import argparse
import hashlib
import json

import struct
import sys
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

PARITY = Path(__file__).resolve().parent
REPO = PARITY.parents[1]
FIXTURES = PARITY / "fixtures"
ASSETS = REPO / "tests" / "assets"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(PARITY))

from bench_quality import make_photo  # noqa: E402
from run_native import isolate  # noqa: E402

# The portfolio's Dark UI sample (static/strivoimg1.jpg), committed here as a copy.
STRIVO = "strivoimg1.jpg"
STRIVO_SHA256 = "04a53e985769217fc6db199736e0e06a31bf4b9b9aeac29039351d20d4eef6ad"
NEAR_THRESHOLD_MARGIN = 0.1


def put(name: str, data: bytes) -> None:
    (FIXTURES / name).write_bytes(data)


def png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def photo_pixels() -> np.ndarray:
    return np.asarray(Image.open(BytesIO(make_photo(640, 480))).convert("RGB"))


def claim_dimensions(data: bytes, width: int, height: int) -> bytes:
    # tests/conftest.py _claim_dimensions: rewrite the PNG IHDR size and its CRC.
    ihdr = b"IHDR" + struct.pack(">II", width, height) + data[24:29]
    crc = struct.pack(">I", zlib.crc32(ihdr))
    return data[:12] + ihdr + crc + data[33:]


def blur_margin(data: bytes) -> float:
    from app.core.config import Settings
    from app.services import pipeline

    s = Settings()
    out = pipeline.outcome(data, "image/png", s)
    if out["status"] != 200:
        raise SystemExit(f"make_fixtures: near-threshold candidate rejected: {out}")
    return json.loads(str(out["response_json"]))["blur_score"] - s.blur_min


def near_threshold() -> tuple[bytes, float, float]:
    """A PNG whose blur score is at most NEAR_THRESHOLD_MARGIN above blur_min."""
    sharp = photo_pixels().astype(np.float64)
    blurred = np.asarray(
        Image.fromarray(photo_pixels()).filter(ImageFilter.GaussianBlur(2))
    ).astype(np.float64)
    lo, hi = 0.0, 1.0  # blend weight of the blurred copy
    for _ in range(40):
        t = (lo + hi) / 2
        mixed = np.floor((1 - t) * sharp + t * blurred + 0.5).astype(np.uint8)
        data = png(Image.fromarray(mixed))
        margin = blur_margin(data)
        if 0 <= margin <= NEAR_THRESHOLD_MARGIN:
            return data, t, margin
        if margin > NEAR_THRESHOLD_MARGIN:
            lo = t
        else:
            hi = t
    raise SystemExit("make_fixtures: no blend weight gives a near-threshold blur score")


def underexposed() -> bytes:
    # RGB * 0.15 rounded half up, in integers: floor(x * 0.15 + 0.5).
    rgb = np.asarray(Image.open(ASSETS / "sharp_small.jpg").convert("RGB"))
    dark = (rgb.astype(np.uint32) * 15 + 50) // 100
    return png(Image.fromarray(dark.astype(np.uint8)))


def generate() -> dict[str, bytes]:
    photo = Image.fromarray(photo_pixels())
    files: dict[str, bytes] = {}
    # The demo's Sharp and Blurry samples, byte for byte.
    for name in ("sharp_small.jpg", "blurry_small.jpg"):
        files[name] = (ASSETS / name).read_bytes()
    # The bench photos: one under the resize limit, one resized and rotated.
    files["photo_1600x1200.jpg"] = make_photo(1600, 1200)
    big = make_photo(4032, 3024, 6)
    if len(big) <= 1024 * 1024:  # the too_large case uploads it with a limit of 1
        raise SystemExit("make_fixtures: photo_4032x3024_exif6.jpg is not over 1 MiB")
    files["photo_4032x3024_exif6.jpg"] = big
    # EXIF orientations 2-8 (mirror, rotations, transpose, transverse).
    for n in range(2, 9):
        files[f"exif{n}.jpg"] = make_photo(640, 480, n)
    # PNG with alpha: a left-to-right ramp, fully transparent on the left.
    alpha = np.clip((np.arange(640) - 64) * 255 // 512, 0, 255).astype(np.uint8)
    rgba = np.dstack([np.asarray(photo), np.broadcast_to(alpha, (480, 640))])
    files["rgba.png"] = png(Image.fromarray(rgba, "RGBA"))
    # Grayscale (L) and palette (P, 64 colours) PNGs.
    files["gray_l.png"] = png(photo.convert("L"))
    files["palette_p.png"] = png(photo.quantize(colors=64))
    # A CMYK JPEG.
    buf = BytesIO()
    photo.convert("CMYK").save(buf, format="JPEG", quality=90)
    files["cmyk.jpg"] = buf.getvalue()
    # The Dark sample: an underexposed copy of sharp_small (fails brightness).
    files["underexposed.png"] = underexposed()
    # A single pixel: Laplacian variance 0, histogram over a constant array.
    files["one_pixel.png"] = png(Image.new("L", (1, 1), 128))
    # Rejections: a truncated JPEG, a header over the pixel limit, no bytes.
    files["truncated.jpg"] = files["photo_1600x1200.jpg"][
        : len(files["photo_1600x1200.jpg"]) // 2
    ]
    small = png(Image.new("L", (32, 32), 128))
    files["overlimit_header.png"] = claim_dimensions(small, 6000, 6000)
    files["empty.jpg"] = b""
    # Just above the blur threshold; a PNG, so it decodes identically everywhere.
    data, t, margin = near_threshold()
    files["near_threshold.png"] = data
    print(f"NEAR_THRESHOLD t={t!r} blur_margin={margin!r}")
    return files


def memory_probe(target: Path) -> int:
    target = target.resolve()
    if target == REPO or REPO in target.parents:
        raise SystemExit("make_fixtures: --memory-probe must be outside the repository")
    target.mkdir(parents=True, exist_ok=True)
    (target / "photo_5000x5000.jpg").write_bytes(make_photo(5000, 5000))
    print(f"MEMORY_PROBE_WRITTEN {target / 'photo_5000x5000.jpg'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--memory-probe", type=Path, metavar="DIR")
    args = ap.parse_args()
    if args.memory_probe is not None:
        return memory_probe(args.memory_probe)
    strivo = FIXTURES / STRIVO
    if (
        not strivo.is_file()
        or hashlib.sha256(strivo.read_bytes()).hexdigest() != STRIVO_SHA256
    ):
        raise SystemExit(
            f"make_fixtures: {strivo} must be the portfolio's static/{STRIVO} "
            f"(sha256 {STRIVO_SHA256})"
        )
    isolate()
    FIXTURES.mkdir(exist_ok=True)
    files = generate()
    for name, data in files.items():
        put(name, data)
    expected = set(files) | {STRIVO}
    present = {p.name for p in FIXTURES.iterdir()}
    if present != expected:
        raise SystemExit(
            f"make_fixtures: unexpected files {sorted(present - expected)}"
        )
    for name in sorted(expected):
        print(f"FIXTURE {name} {(FIXTURES / name).stat().st_size}")
    print(f"FIXTURES_WRITTEN {len(expected)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
