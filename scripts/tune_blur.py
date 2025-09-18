#!/usr/bin/env python3
"""
tune_blur.py — Compute blur (variance of Laplacian) + brightness for a folder of images,
summarize distribution, and suggest BLUR_MIN using an Otsu threshold over blur scores.
Optionally saves a CSV and histogram PNGs.

Usage (examples):
  python tune_blur.py --dir tests/assets
  python tune_blur.py --dir samples --csv out/blur_scores.csv --plot out

Notes:
- Requires Pillow, numpy, opencv-python[-headless].
- --plot requires matplotlib (optional).
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import cv2
from PIL import Image, ImageOps

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Record:
    path: Path
    width: int
    height: int
    blur_score: float
    brightness: float


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p


def load_and_prepare(p: Path, max_dim: int) -> Tuple[np.ndarray, int, int]:
    img = Image.open(p)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.asarray(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray, img.width, img.height


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


# replace in tune_blur.py (or add a flag)
def otsu_threshold(values: np.ndarray, bins: int = 128, log=False) -> float:
    v = values[values > 0]
    if v.size < 3:
        return float(v.mean()) if v.size else 0.0
    if log:
        v = np.log10(v)
    hist, edges = np.histogram(v, bins=bins)
    prob = hist.astype(float) / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / np.clip(omega * (1 - omega), 1e-12, None)
    k = int(np.nanargmax(sigma_b2))
    thr = (edges[k] + edges[k + 1]) / 2.0
    return float(10**thr) if log else float(thr)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summarize(records: List[Record], bright_min: int, bright_max: int) -> None:
    if not records:
        print("No images found.", file=sys.stderr)
        return
    blur = np.array([r.blur_score for r in records], dtype=np.float64)
    bright = np.array([r.brightness for r in records], dtype=np.float64)

    print(f"\nAnalyzed {len(records)} images.")
    print(
        "Blur score (variance of Laplacian):\n"
        f"  min={blur.min():.2f}  p10={percentile(blur,10):.2f}  "
        f"p25={percentile(blur,25):.2f}  median={percentile(blur,50):.2f}  "
        f"p75={percentile(blur,75):.2f}  p90={percentile(blur,90):.2f}  max={blur.max():.2f}"
    )
    print(
        "Brightness (0..255):\n"
        f"  min={bright.min():.1f}  p10={percentile(bright,10):.1f}  "
        f"p25={percentile(bright,25):.1f}  median={percentile(bright,50):.1f}  "
        f"p75={percentile(bright,75):.1f}  p90={percentile(bright,90):.1f}  max={bright.max():.1f}"
    )

    # Suggest brightness bounds from central 95%
    br_lo = max(0, round(percentile(bright, 2.5)))
    br_hi = min(255, round(percentile(bright, 97.5)))
    # Suggest blur threshold via Otsu
    blur_suggest = otsu_threshold(blur, bins=128, log=True)

    print("\nSuggested thresholds:")
    print(f"  BLUR_MIN (log-Otsu) ≈ {blur_suggest:.1f}")
    print(
        f"  BRIGHT_MIN..BRIGHT_MAX ≈ {br_lo}..{br_hi}  (current: {bright_min}..{bright_max})"
    )
    print("\nAction: Put these in your .env and re-run service to validate.")


def write_csv(records: List[Record], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "width", "height", "blur_score", "brightness"])
        for r in records:
            w.writerow(
                [
                    str(r.path),
                    r.width,
                    r.height,
                    f"{r.blur_score:.4f}",
                    f"{r.brightness:.2f}",
                ]
            )
    print(f"Wrote CSV: {csv_path}")


def maybe_plot(records: List[Record], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # optional
    except Exception:
        print(
            "Plotting skipped (matplotlib not installed). Run: pip install matplotlib",
            file=sys.stderr,
        )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    blur = np.array([r.blur_score for r in records], dtype=np.float64)
    bright = np.array([r.brightness for r in records], dtype=np.float64)

    # Blur histogram
    plt.figure()
    plt.hist(blur, bins=40)
    plt.title("Blur score (variance of Laplacian)")
    plt.xlabel("blur_score")
    plt.ylabel("count")
    p = out_dir / "hist_blur.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Wrote plot: {p}")

    # Brightness histogram
    plt.figure()
    plt.hist(bright, bins=40)
    plt.title("Brightness (0..255)")
    plt.xlabel("brightness")
    plt.ylabel("count")
    p2 = out_dir / "hist_brightness.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print(f"Wrote plot: {p2}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute blur + brightness stats for images in a folder."
    )
    ap.add_argument(
        "--dir", required=True, type=Path, help="Folder containing images (recurses)."
    )
    ap.add_argument(
        "--max-dim",
        type=int,
        default=1600,
        help="Cap on larger image side before scoring.",
    )
    ap.add_argument(
        "--csv", type=Path, default=None, help="Optional path to write CSV results."
    )
    ap.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional folder to write histogram PNGs (requires matplotlib).",
    )
    ap.add_argument("--bright-min", type=int, default=20)
    ap.add_argument("--bright-max", type=int, default=235)
    args = ap.parse_args()

    if not args.dir.exists():
        print(f"--dir not found: {args.dir}", file=sys.stderr)
        sys.exit(2)

    records: List[Record] = []
    for p in iter_images(args.dir):
        try:
            gray, w, h = load_and_prepare(p, args.max_dim)
            bs = variance_of_laplacian(gray)
            br = brightness(gray)
            records.append(
                Record(path=p, width=w, height=h, blur_score=bs, brightness=br)
            )
        except Exception as e:
            print(f"[warn] failed to process {p}: {e}", file=sys.stderr)

    summarize(records, args.bright_min, args.bright_max)
    if args.csv:
        write_csv(records, args.csv)
    if args.plot:
        maybe_plot(records, args.plot)


if __name__ == "__main__":
    main()
