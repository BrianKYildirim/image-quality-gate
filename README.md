# Image Quality Gate

A small web service that checks whether an uploaded photo is **sharp enough** and **reasonably exposed** before an application accepts it. It returns the measurements behind its answer, so callers can see why a photo passed or failed.

Built with Python, FastAPI, OpenCV, Pillow and NumPy. It runs on a CPU; no GPU or machine-learning model is involved.

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [API](#api)
- [Diagnostics](#diagnostics)
- [Configuration](#configuration)
- [Input limits and security](#input-limits-and-security)
- [Design decisions](#design-decisions)
- [Testing](#testing)
- [Performance](#performance)
- [Challenges and lessons learned](#challenges-and-lessons-learned)
- [Limitations and future work](#limitations-and-future-work)
- [Project structure](#project-structure)
- [Development](#development)

## What it does

Apps that collect photos from people, such as receipts, damage reports or profile pictures, often receive blurry or very dark shots that are useless later. This service is a quick automated first check. You send it an image and it answers with:

- a **blur score**: higher means more fine detail and sharp edges;
- a **brightness** value from 0 (black) to 255 (white);
- **`is_ok`**: whether both values are inside the configured limits.

It does not judge what is *in* the picture. A sharp, well-lit photo of the wrong thing still passes. See [Limitations](#limitations-and-future-work).

## How it works

```text
upload ─► request size check ─► content type check ─► file size check
       ─► read image header ─► pixel count check ─► decode pixels (incomplete files rejected)
       ─► fix EXIF orientation ─► convert to RGB ─► shrink so the long side ≤ 1600 px
       ─► grayscale ─► blur score + brightness ─► compare with thresholds ─► response
```

**In plain terms:** the service puts the photo the right way up and shrinks it to a standard size. It then measures how crisp the edges are and how bright the image is on average, and compares both numbers with limits you configure.

**Technically:**

1. **Normalize.** Pillow opens the image, applies the EXIF orientation (`ImageOps.exif_transpose`) and converts to RGB. If the longest side is above `RESIZE_MAX_DIM` (default 1600), the image is downscaled with Lanczos resampling. OpenCV then converts it to 8-bit grayscale (BGR → GRAY, i.e. 0.299 R + 0.587 G + 0.114 B).
2. **Blur score: variance of the Laplacian.** The Laplacian (`cv2.Laplacian` with OpenCV's default 3×3 kernel) is a second-derivative filter. It produces large positive and negative values at edges and values near zero in smooth areas. A sharp image has many strong edges, so the filtered values are spread widely and their variance is high. Blur softens edges, the values cluster around zero, and the variance drops.
3. **Brightness:** the mean of the grayscale image.
4. **Decision:**

   ```text
   is_ok = blur_score >= BLUR_MIN  AND  BRIGHT_MIN <= brightness <= BRIGHT_MAX
   ```

   All three comparisons are inclusive. A value exactly on a threshold passes.

The two test images in `tests/assets` show the difference. Both are 256×256 JPEGs with almost the same brightness, but very different blur scores:

| Image | `blur_score` | `brightness` | `is_ok` (default thresholds) |
|---|---:|---:|---|
| `sharp_small.jpg` | 1033.12 | 64.75 | `true` |
| `blurry_small.jpg` | 1.16 | 64.74 | `false` (blur below 140) |

## Quick start

Tested with Python 3.11, the version CI uses.

```bash
git clone https://github.com/BrianKYildirim/image-quality-gate.git
cd image-quality-gate
python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

Then, in another terminal:

```bash
curl -F "file=@tests/assets/sharp_small.jpg;type=image/jpeg" http://localhost:8080/quality
```

```json
{"blur_score":1033.1184387207031,"brightness":64.75344848632812,"width":256,"height":256,"is_ok":true,"thresholds":{"blur_min":140.0,"bright_min":20,"bright_max":235}}
```

Interactive API docs are served at `http://localhost:8080/docs` (Swagger UI) and `/redoc`.

No `.env` file is needed: without one, the defaults in [Configuration](#configuration) apply. Note that `.env.example` sets lower thresholds (`BLUR_MIN=70`, `BRIGHT_MIN=10`), so copying it to `.env` changes the decisions.

### Docker

```bash
docker build -t image-quality-gate:local .
docker run --rm -p 8080:8080 image-quality-gate:local
```

The Dockerfile copies `.env.example` into the image as `.env`, so a container started this way uses `BLUR_MIN=70`, `BRIGHT_MIN=10` and `APP_VERSION=0.1.0` unless you override them with `-e` or `--env-file`. The container runs uvicorn with two worker processes on port 8080. `docker compose up --build` does the same; `docker-compose.yml` also loads a `.env` file from the project folder, so create one first, for example by copying `.env.example`.

## API

| Method and path | Purpose |
|---|---|
| `POST /quality` | Score an image (multipart form field `file`). |
| `POST /quality/diagnostics` | The same, plus a full breakdown of what was measured and decided. See [Diagnostics](#diagnostics). |
| `GET /quality/config` | Thresholds, limits and library versions in effect, without uploading anything. |
| `GET /health` | `{"status": "ok"}` |
| `GET /version` | `{"app": APP_NAME, "version": APP_VERSION}` |
| `GET /metrics` | Prometheus metrics, when `PROMETHEUS_ENABLED=true`. |

Every response carries an `X-Request-ID` header: the one the client sent, or a generated `rid-<uuid>`.

### `POST /quality` response

| Field | Type | Meaning |
|---|---|---|
| `blur_score` | float | Variance of the Laplacian of the processed grayscale image. |
| `brightness` | float | Mean of the processed grayscale image, 0–255. |
| `width`, `height` | int | Size of the processed image: after orientation, before grayscale, at most `RESIZE_MAX_DIM` on the long side. |
| `is_ok` | bool | The decision described above. |
| `thresholds` | object | `blur_min`, `bright_min` and `bright_max` used for this request. |

### Errors

| Status | `detail` | When |
|---:|---|---|
| 400 | `Invalid image data` | The file cannot be decoded, is incomplete (truncated), or has more pixels than `MAX_IMAGE_PIXELS`. |
| 413 | `File too large` | The request body is larger than `MAX_UPLOAD_MB` MiB plus 64 KiB, or the file itself is larger than `MAX_UPLOAD_MB` MiB. |
| 415 | `Unsupported media type` | The content type declared for the file does not start with `image/`. |
| 422 | FastAPI validation error | The `file` field is missing, or a query parameter is invalid. |

## Diagnostics

`POST /quality/diagnostics` runs exactly the same pipeline as `/quality` and returns the same fields, plus a `diagnostics` object. It is meant for debugging, threshold tuning and demos: it shows what the service measured and decided, which settings it used, what it observed about the image, and how long each step took.

- **Every value comes from the run that produced the response.** Nothing is re-computed separately or estimated. The extra statistics and the optional histogram are calculated after the decision, from the same arrays.
- **`/quality` is untouched by this.** Its response, its errors and its OpenAPI schema are byte-for-byte the same as before diagnostics existed; the tests check this. The only exceptions are deliberate: incomplete files and images over the pixel limit are now rejected (see [Input limits](#input-limits-and-security)).

```bash
curl -F "file=@tests/assets/sharp_small.jpg;type=image/jpeg" \
     -H "X-Request-ID: demo-1" \
     "http://localhost:8080/quality/diagnostics?laplacian_histogram_bins=8"
```

`laplacian_histogram_bins` (2–256) is optional; without it, `laplacian_histogram` is `null`.

Real output for that request with default settings (timings depend on the machine and change from run to run):

```json
{
  "blur_score": 1033.1184387207031,
  "brightness": 64.75344848632812,
  "width": 256,
  "height": 256,
  "is_ok": true,
  "thresholds": { "blur_min": 140.0, "bright_min": 20, "bright_max": 235 },
  "diagnostics": {
    "schema_version": 1,
    "request_id": "demo-1",
    "measurements": {
      "blur_score": 1033.1184387207031,
      "brightness": 64.75344848632812,
      "brightness_std": 110.9875994256436,
      "gray_min": 0,
      "gray_max": 255,
      "laplacian_mean": 0.0,
      "laplacian_min": -510.0,
      "laplacian_max": 259.0,
      "pixel_count": 65536
    },
    "decision": {
      "rule": "is_ok = (blur_score >= blur_min) AND (bright_min <= brightness) AND (brightness <= bright_max)",
      "combine": "all",
      "checks": [
        { "name": "blur_min", "measurement": "blur_score", "comparator": ">=", "threshold": 140.0, "value": 1033.1184387207031, "passed": true, "margin": 893.1184387207031 },
        { "name": "bright_min", "measurement": "brightness", "comparator": ">=", "threshold": 20, "value": 64.75344848632812, "passed": true, "margin": 44.753448486328125 },
        { "name": "bright_max", "measurement": "brightness", "comparator": "<=", "threshold": 235, "value": 64.75344848632812, "passed": true, "margin": 170.24655151367188 }
      ],
      "is_ok": true
    },
    "settings": {
      "blur_min": { "value": 140.0, "source": "default" },
      "bright_min": { "value": 20, "source": "default" },
      "bright_max": { "value": 235, "source": "default" },
      "resize_max_dim": { "value": 1600, "source": "default" },
      "max_upload_mb": { "value": 6, "source": "default" },
      "max_upload_bytes": { "value": 6291456, "source": "derived" },
      "max_request_bytes": { "value": 6356992, "source": "derived" },
      "max_image_pixels": { "value": 25000000, "source": "default" },
      "pillow_max_image_pixels": { "value": 89478485, "source": "pillow" }
    },
    "image": {
      "upload_bytes": 2294,
      "content_type": "image/jpeg",
      "format": "JPEG",
      "mode": "RGB",
      "original_width": 256,
      "original_height": 256,
      "exif_orientation": null,
      "orientation_applied": false,
      "oriented_width": 256,
      "oriented_height": 256,
      "resized": false,
      "scale": 1.0,
      "processed_width": 256,
      "processed_height": 256
    },
    "timings": {
      "clock": "perf_counter_ns",
      "stages": [
        { "name": "read_upload", "offset_ns": 15852, "duration_ns": 7621 },
        { "name": "open", "offset_ns": 26978, "duration_ns": 127329 },
        { "name": "decode", "offset_ns": 157628, "duration_ns": 465284 },
        { "name": "exif_transpose", "offset_ns": 625176, "duration_ns": 157067 },
        { "name": "convert_rgb", "offset_ns": 783299, "duration_ns": 169536 },
        { "name": "to_array", "offset_ns": 960761, "duration_ns": 281968 },
        { "name": "rgb_to_bgr", "offset_ns": 1244533, "duration_ns": 51998 },
        { "name": "grayscale", "offset_ns": 1322197, "duration_ns": 46225 },
        { "name": "laplacian", "offset_ns": 1376165, "duration_ns": 167757 },
        { "name": "statistics", "offset_ns": 1545534, "duration_ns": 929954 },
        { "name": "decision", "offset_ns": 2479576, "duration_ns": 25977 },
        { "name": "extra_statistics", "offset_ns": 2533322, "duration_ns": 564427 },
        { "name": "laplacian_histogram", "offset_ns": 3102098, "duration_ns": 5284501 }
      ],
      "stages_sum_ns": 8279644,
      "handler_total_ns": 8390306
    },
    "versions": {
      "app": "0.0.0",
      "python": "3.11.17",
      "numpy": "2.4.6",
      "opencv": "5.0.0",
      "pillow": "13.0.0",
      "libjpeg": "6.2",
      "libjpeg_turbo": "3.2.0"
    },
    "laplacian_histogram": {
      "bins": 8,
      "bin_edges": [-510.0, -413.875, -317.75, -221.625, -125.5, -29.375, 66.75, 162.875, 259.0],
      "counts": [4, 0, 508, 0, 2, 64506, 0, 516]
    }
  }
}
```

The response also carries a standard `Server-Timing` header with the same stage durations in milliseconds; for the response above it read `read_upload;dur=0.008, open;dur=0.127, decode;dur=0.465, ..., handler_total;dur=8.390`. Browser developer tools display it.

<details>
<summary><strong>Field reference</strong></summary>

Categories:

- **computed**: calculated by the pipeline from the image.
- **derived**: a fixed function of other reported values.
- **measured**: a duration on the monotonic clock `time.perf_counter_ns`.
- **metadata**: observed from the upload, the decoder, the settings or the runtime.

| Field | Unit | Category | Meaning |
|---|---|---|---|
| `schema_version` | — | metadata | Format version of `diagnostics` (currently 1). |
| `request_id` | — | metadata | Same as the `X-Request-ID` response header. |
| `measurements.blur_score`, `brightness` | gray level², gray level | computed | The same floats as the top-level fields. |
| `measurements.brightness_std` | gray level | computed | Standard deviation of the processed grayscale image. |
| `measurements.gray_min`, `gray_max` | gray level | computed | Darkest and brightest pixel. |
| `measurements.laplacian_mean`, `_min`, `_max` | gray level | computed | Of the Laplacian array whose variance is `blur_score`. |
| `measurements.pixel_count` | pixels | computed | Pixels scored (processed width × height). |
| `decision.rule`, `combine` | — | metadata | The rule in words; `all` means every check must pass. |
| `decision.checks[].name`, `measurement`, `comparator`, `threshold` | — | metadata | What is compared with what, and how. |
| `decision.checks[].value`, `passed` | — | computed | The compared value and the result of the comparison. |
| `decision.checks[].margin` | as the value | derived | `value − threshold` for `>=`, `threshold − value` for `<=`. It is ≥ 0 exactly when the check passed. |
| `decision.is_ok` | — | computed | The same as the top-level `is_ok`. |
| `settings.<name>.value`, `source` | — | metadata | Value in effect, and where it came from: `default`, `env` (environment variable), `dotenv` (`.env` file), `init` (passed in code), `derived` or `pillow`. Sources are recorded when settings load, following pydantic-settings' precedence. |
| `image.upload_bytes`, `content_type` | bytes, — | metadata | Size of the uploaded file and the type the client declared for it. |
| `image.format`, `mode` | — | metadata | As decoded by Pillow, e.g. `JPEG`/`RGB`, `PNG`/`RGBA`, `GIF`/`P`. |
| `image.original_width`, `original_height` | px | metadata | Decoded size, before orientation. |
| `image.exif_orientation` | — | metadata | Orientation tag as read by Pillow (EXIF 0x0112, or the XMP fallback Pillow uses); `null` if absent. |
| `image.orientation_applied` | — | derived | `true` for orientations 2–8, the values `exif_transpose` flips or rotates. |
| `image.oriented_width`, `oriented_height` | px | metadata | Size after orientation. Width and height swap for orientations 5–8. |
| `image.resized`, `scale` | —, ratio | derived, computed | `scale = min(1, RESIZE_MAX_DIM / longest side)`; `resized` is `scale < 1`. |
| `image.processed_width`, `processed_height` | px | metadata | Size that was scored (`int(size × scale)`, truncated). Equals the top-level `width`/`height`. |
| `timings.stages[].name`, `offset_ns`, `duration_ns` | ns | metadata, measured | Each step, in the order it ran; the offset is relative to handler entry. |
| `timings.stages_sum_ns` | ns | derived | Sum of the stage durations. |
| `timings.handler_total_ns` | ns | measured | Handler entry until the diagnostics are assembled. |
| `versions.*` | — | metadata | App version setting, and the Python, NumPy, OpenCV and Pillow versions loaded in the process, plus Pillow's JPEG codec (libjpeg API level and libjpeg-turbo version). |
| `laplacian_histogram.bin_edges`, `counts` | gray level, pixels | computed | `numpy.histogram` of the exact Laplacian array used for `blur_score`: signed values, equal-width bins from its minimum to its maximum, no clipping or scaling. Counts sum to `pixel_count`. |

**What the timings cover:** the work inside the request handler, from its first line until the diagnostics are assembled. Stages: `read_upload`, `open` (header only), `decode`, `exif_transpose`, `convert_rgb`, `resize` (only when it happens), `to_array`, `rgb_to_bgr`, `grayscale`, `laplacian`, `statistics`, `decision`, `extra_statistics` and `laplacian_histogram` (only when requested).

**What they do not cover:**

- network transfer;
- receiving and parsing the multipart upload, which happens before the handler starts;
- time spent waiting for a free worker;
- middleware;
- building and sending the response.

The handler runs its CPU-bound work directly on the event loop, so while a worker processes one image, other requests to that worker wait. That waiting shows up only as client-side latency.

</details>

## Configuration

Settings come from environment variables, then a `.env` file in the working directory, then the defaults below. Variable names are case-insensitive.

| Variable | Default | Meaning |
|---|---:|---|
| `BLUR_MIN` | `140.0` | Minimum blur score. |
| `BRIGHT_MIN` | `20` | Minimum brightness (0–255). |
| `BRIGHT_MAX` | `235` | Maximum brightness (0–255). |
| `RESIZE_MAX_DIM` | `1600` | Longest side before scoring (minimum 256). |
| `MAX_UPLOAD_MB` | `6` | Maximum file size in MiB (1 MiB = 1,048,576 bytes; allowed range 1–25). The whole request may be up to 64 KiB larger, to leave room for the multipart framing. |
| `MAX_IMAGE_PIXELS` | `25000000` | Maximum width × height (25 megapixels). |
| `LOG_LEVEL` | `INFO` | `DEBUG` adds one structured timing record per request, with no image data or file names. |
| `LOG_JSON` | `true` | JSON log lines. |
| `PROMETHEUS_ENABLED` | `true` | Serve `/metrics`. |
| `APP_NAME` | `image-quality-gate` | Returned by `/version`. |
| `APP_VERSION` | `0.0.0` | Returned by `/version` and in diagnostics. |

The blur score depends on the image source and on `RESIZE_MAX_DIM`, so tune thresholds on your own images; see [Threshold tuning](#threshold-tuning).

## Input limits and security

Uploaded files are untrusted input. The checks run from cheapest to most expensive, so a bad request is turned away before it costs much. Each rejection reuses the service's existing status codes and messages, so clients need no changes.

| Check | What it stops | How |
|---|---|---|
| Request size | Huge uploads tying up memory or disk | A small ASGI middleware (`app/core/body_limit.py`) answers 413 straight away if `Content-Length` is above the limit, without reading the body. For chunked uploads with no length, it counts bytes as they arrive and stops at the first chunk past the limit. |
| Content type | Non-image uploads | 415 unless the declared type starts with `image/`. |
| File size | Files just over `MAX_UPLOAD_MB` | The handler reads at most the limit plus one byte. |
| Pixel count | "Decompression bombs": small files that expand into huge images | After Pillow reads the header (width and height) and before any pixels are decoded, images over `MAX_IMAGE_PIXELS` get a 400. |
| Incomplete files | Truncated uploads scored as if they were complete | Pillow's default behavior is kept: decoding raises an error when the data ends before the image is complete. |

Measured on the laptop described under [Performance](#performance), comparing the code before these checks with the current code. Each case used a fresh single-worker uvicorn server and `curl`; times are single runs on an idle machine, and memory is the growth of the server's peak resident memory:

| Request | Before | After |
|---|---|---|
| 100 MiB upload (with `Content-Length`) | 413 after the whole upload was received (423 ms); +101 MiB | 413 in 3 ms; nothing read; +1 MiB |
| 100 MiB chunked upload | 413 after the whole upload (452 ms); +101 MiB | 413 in 19 ms; +2 MiB |
| 165 KiB PNG that decodes to 12000 × 12000 (144 MP) | **200** after 2.9 s; +824 MiB | 400 in 5 ms; no measurable growth |
| First half of `sharp_small.jpg` (truncated) | **200**, scored as sharp: the missing part was filled with flat gray | 400 |
| Valid 5000 × 5000 JPEG (exactly 25 MP) | 200 in 551 ms; +287 MiB | 200 in 566 ms with identical scores; +287 MiB |

Some details worth knowing:

- **A JPEG missing only its final end-of-image marker is rejected**, even if its pixels are complete. The decoder cannot tell it apart from a file cut off at the end of its data, and the service relies on the decoder rather than guessing. Extra bytes after the end of a JPEG, or a PNG without its final `IEND` chunk, are still accepted and score exactly the same as the intact file.
- **Corruption that doesn't end the data early is not detected.** For example, zeroed bytes in the middle of a JPEG decode without an error and are scored as decoded. Pillow does not report these cases, so the service does not claim to.
- **The pixel limit is a policy choice.** 25 MP covers common 12 MP phone photos, but rejects images from cameras that shoot at higher resolutions (for example 48 MP modes). Raise `MAX_IMAGE_PIXELS` if you need them, and plan for the memory: a 25 MP image used about 290 MiB of server memory in the test above.

### Still needed in a real deployment

The service on its own is not hardened for direct exposure to the internet. The deployment files were not changed as part of this work. Before public use:

- **Run it behind a reverse proxy** that limits body size and upload time. Uvicorn has no request-body timeout, so a client can send an upload very slowly and hold a connection open. Example for nginx (not tested here):

  ```nginx
  client_max_body_size 7m;    # slightly above MAX_UPLOAD_MB
  client_body_timeout 15s;    # drop very slow uploads
  ```

- **Add authentication and rate limiting.** There are none, and CORS allows every origin.
- **Limit concurrency.** Each image is processed on the worker's event loop, and the worker can't start another request until it finishes. On the idle test laptop, a 12 MP photo took about 0.26 s and a 25 MP photo about 0.55 s, and a 25 MP image used about 290 MiB. Several large images at once can exhaust a small container: `docker-compose.yml` sets a 512 MB limit and the Dockerfile starts two workers.
- **Restrict `/quality/diagnostics`, `/quality/config` and `/metrics`** if you don't want to reveal thresholds, library versions or traffic data.
- **Keep Pillow and OpenCV up to date.** They parse untrusted files.

## Design decisions

- **Classic image measurements instead of a machine-learning model.** The variance of the Laplacian and the mean brightness are fast on a CPU, easy to explain, and need no training data or GPU. The trade-off is that they measure the image, not its meaning, and the thresholds need tuning for each image source.
- **Normalize before measuring.** Fixing the orientation and capping the longest side at 1600 px keeps the work bounded and makes scores from different cameras more comparable. The trade-off is that the blur score depends on `RESIZE_MAX_DIM`, so thresholds should be re-tuned if it changes.
- **A transparent rule.** Three inclusive comparisons combined with AND are easy to reason about, and the diagnostics endpoint shows each one with its margin.
- **Diagnostics as a separate endpoint over one shared pipeline.** A flag on `/quality` would have changed its response schema. A separate endpoint leaves existing clients completely unaffected, and because both routes call the same function, the two can't drift apart.
- **Report only what was actually measured.** Every diagnostic value is computed in the same run as the decision. Timings are taken where the work really happens: Pillow decodes lazily, so decoding is triggered explicitly to be timed on its own. Setting sources are recorded when settings load, not reconstructed later.
- **Fail early, reuse existing responses.** The new limits run before the expensive steps and return the service's existing 400 and 413 responses.
- **Let the decoder decide what is incomplete.** Rejecting truncated files relies on Pillow's own end-of-data detection instead of a homemade heuristic. That is stricter in one rare case (JPEGs missing their end marker) but never guesses.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

There are 130 tests (many are parameterized cases). They take about 6 seconds on the laptop described under [Performance](#performance) when it is otherwise idle.

| File | What it checks |
|---|---|
| `tests/integration/test_compat.py` | `/quality` responses (status, body, headers) and its OpenAPI schema are byte-identical to a verbatim copy of the original implementation, over 30+ images and error cases. The intentional behavior changes are tested separately. |
| `tests/integration/test_api_diagnostics.py` | Diagnostics match the decision exactly; boundary decisions over HTTP; setting sources; image metadata; timing order and bounds; `Server-Timing`; request IDs; no secrets or file paths in any output; logging; concurrent requests. |
| `tests/integration/test_input_limits.py` | Pixel limit, request size limit (including a real chunked upload that is cut off) and truncated files, through the API. |
| `tests/unit/test_decision.py` | A value exactly on each threshold passes and one floating-point step past it fails; `is_ok` equals the AND of the checks and the original rule; the Laplacian array is kept only when diagnostics ask for it. |
| `tests/unit/test_preprocess.py` | Identical pixels to the original code for EXIF orientations 1–9, resizing, and RGBA, palette, grayscale and CMYK inputs; reported metadata. |
| `tests/unit/test_input_safety.py` | Truncated files are rejected per format; the pixel limit is applied before any pixel decoding. |
| `tests/unit/test_body_limit.py`, `test_config.py`, `test_timing.py`, `test_quality_math.py` | The middleware on its own, settings provenance, the stage timer, and the basic scoring. |

CI (GitHub Actions, Python 3.11) runs `ruff check .`, `black --check .`, `mypy app` and `pytest -q`. The tests were also checked against deliberately broken copies of the code (for example a `<` instead of `<=`, or a re-enabled truncated-image flag), and each break made tests fail.

## Performance

`scripts/bench_quality.py` compares this code with an older checkout over real HTTP. It starts one single-worker uvicorn server per variant and sends each round of requests to every variant in a shuffled order. It reports the median and spread, plus the *paired* difference per round, which cancels out background load on a shared machine. It also times the image pipeline in-process, alternating the old and new code call by call.

```bash
git worktree add ../iqg-baseline <older-commit>
python scripts/bench_quality.py --baseline ../iqg-baseline --json out/bench.json
```

One run, against the code before diagnostics and the input limits were added:

- **Machine and software:** Intel i7-1185G7 laptop, Linux under WSL2, Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, Pillow 13.0.0, FastAPI 0.161.1, uvicorn 0.54.0.
- **Rounds:** 300 per small image, 60 per photo.
- **Load average:** 0.24 at the start, 1.13 at the end.

Values are in milliseconds: the baseline median, then the paired difference with its 25th and 75th percentiles.

| Image | Baseline `/quality` | `/quality` now | `/quality/diagnostics` | `+ 64-bin histogram` |
|---|---:|---:|---:|---:|
| `blurry_small.jpg` (256×256) | 3.19 | +0.05 [−0.37, +0.50] | +0.60 [+0.22, +1.13] | +2.06 [+1.57, +2.92] |
| `sharp_small.jpg` (256×256) | 2.93 | +0.12 [−0.15, +0.44] | +0.61 [+0.31, +1.10] | +1.98 [+1.58, +2.96] |
| 1600×1200 JPEG | 26.5 | +0.3 [−2.1, +3.3] | +14.0 [+10.8, +16.5] | +32.4 [+28.6, +37.3] |
| 4032×3024 JPEG, EXIF 6 | 262.6 | +0.9 [−12.4, +16.4] | +1.4 [−10.3, +9.2] | +17.6 [+6.9, +35.2] |

How to read this:

- **The default endpoint shows no measurable change.** Each paired difference is small next to its own spread. In-process, the extra bookkeeping (stage timer, per-check results) cost a median of about 20 µs on a ~0.55 ms small-image pipeline.
- **Diagnostics cost extra:** mostly the additional statistics over the full arrays, plus the histogram when requested. They are meant for debugging and demos.
- **Keep diagnostics traffic separate.** In an earlier run where one server handled both endpoints, `/quality` on the 1600×1200 image was about 4 ms slower after diagnostics requests on the same worker, because of how memory was reused. That is why the benchmark gives each variant its own server.
- **These are single-run numbers from one laptop, not a load test.**

## Challenges and lessons learned

- **Timing a library that works lazily.** `Image.open` in Pillow only reads the header; the pixels are decoded the first time they are needed, which turned out to be inside the EXIF-orientation step. Timing the code naively would have blamed orientation for the cost of decoding. Lesson: read the library's source before instrumenting it.
- **Proving that nothing changed.** "Opt-in" diagnostics are only safe if the default response is unchanged. The tests keep a verbatim copy of the original endpoint and compare raw response bytes over a corpus of images and error cases. They also compare the OpenAPI schema.
- **Benchmarking on a busy laptop.** Comparing one block of requests with another gave contradictory answers: one run showed a 14% slowdown, another a 14% speed-up. Switching to interleaved requests with paired differences, plus A/A control runs (old code against itself), made the results stable enough to trust.
- **That method found real regressions, twice.** Both times the refactored code freed a large array later than the original did, so later steps had to map fresh memory instead of reusing the freed memory:
  - The rotated copy of the image, held a little too long, cost about 200 µs per small image in an in-process comparison.
  - The 15 MB Laplacian array, held until the end of the request so diagnostics could use it, added about 5–6 ms to a 1600×1200 image (roughly 20%). Counting minor page faults per request (about 3,300 more than the original) confirmed the cause.

  In both cases, freeing the array at the same point as the original code made the fault counts identical again. Diagnostics now ask to keep the Laplacian explicitly. Lesson: in Python code that handles large buffers, *when* an object is released can matter for speed. A related lesson about the benchmark itself: requests to one endpoint can change the speed of another endpoint in the same process, so each variant now gets its own server.
- **Instrumentation has a cost.** The first stage timer used `@contextlib.contextmanager` and cost 7–17 µs per stage on this laptop. A small class-based timer brought eleven stages down to about 6 µs in total (best case).
- **A security review turned up real problems.**
  - A 165 KiB PNG could make the service decode a 144-megapixel image (+824 MiB of memory, and seconds of CPU time).
  - A 100 MiB upload was fully received and read into memory before being rejected.
  - A truncated JPEG was scored as a sharp photo.

  The root cause of the last one was a single line, `ImageFile.LOAD_TRUNCATED_IMAGES = True`. It looked like a harmless safety setting, but it told Pillow to silently fill in missing data. Each problem now has a fix and tests, with before/after measurements above.
- **Reproducing CI locally without internet access.** The development machine's Linux environment had no network access. To run the same Python version as CI, Python 3.11 was built from source with a verified checksum, and the dependencies were downloaded on the host and installed offline. Lesson: checking downloads and pinning down the environment matters as much as the code.
- **Unpinned dependencies move under you.** A fresh install picked up Starlette 1.9, whose test client now asks for a package called `httpx2` (it still works with `httpx`, with a warning), and blur scores can differ slightly between OpenCV and Pillow versions. The diagnostics report the library versions that produced each result; pinning the dependencies is listed as future work.

## Limitations and future work

- **The measurements don't understand content.** A sharp photo of a blank wall has few edges and gets a low blur score; a noisy photo can score high. Brightness is a global average, so a strongly backlit subject can still pass.
- **Thresholds need tuning per image source.** The defaults are a starting point, not a validated setting.
- **Formats:** whatever Pillow can decode (JPEG, PNG, GIF, WebP, TIFF, BMP and others). HEIC, the default format on many iPhones, is not supported without an extra plugin.
- **Future work:**
  - pin dependency versions;
  - move image processing off the event loop (a process pool);
  - add authentication and rate limiting;
  - tune thresholds on a labeled set of real photos;
  - make `scripts/tune_blur.py` import the service's pipeline instead of re-implementing it.

## Project structure

```text
app/
  main.py                    # App setup: middleware, routes, /version
  api/routes_quality.py      # /quality, /quality/diagnostics, /quality/config, /health
  core/body_limit.py         # Request body size limit (ASGI middleware)
  core/config.py             # Settings and where each value came from
  core/logging.py            # JSON logging
  core/metrics.py            # Prometheus instrumentation
  core/timing.py             # Per-request stage timer
  models/schemas.py          # Response models
  services/preprocess.py     # Decode, orientation, pixel limit, resize
  services/quality.py        # Blur score, brightness, decision
  services/diagnostics.py    # Builds the diagnostics response
scripts/
  bench_quality.py           # Latency benchmark
  tune_blur.py               # Threshold tuning helper
tests/
  assets/                    # sharp_small.jpg, blurry_small.jpg
  unit/, integration/
.github/workflows/ci.yml     # Lint, type-check and test on Python 3.11
Dockerfile, docker-compose.yml, Makerfile, requirements.txt, requirements-dev.txt, .env.example
```

## Development

The common commands are in a file named `Makerfile`, so pass it to `make` explicitly:

```bash
make -f Makerfile run     # uvicorn with auto-reload on port 8080
make -f Makerfile test    # pytest -q
make -f Makerfile lint    # ruff and mypy (CI also runs black --check)
make -f Makerfile fmt     # black, then ruff --fix
make -f Makerfile build   # docker build
```

### Threshold tuning

`scripts/tune_blur.py` scores every image in a folder and suggests thresholds. `BLUR_MIN` comes from Otsu's method on the log of the blur scores; the brightness range covers the middle 95% of images.

```bash
python scripts/tune_blur.py --dir path/to/your/images --csv out/scores.csv --plot out/
```

It re-implements the same preprocessing steps rather than importing the service code. For the two test images its scores match the service's exactly, but it does not apply the service's upload and pixel limits. `--plot` needs matplotlib, which is in `requirements.txt`. Put representative sharp, blurry, dark and bright images from your real source in the folder, then confirm the suggested values through the API.
