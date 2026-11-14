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

1. **Normalize.** Pillow opens the image, applies the EXIF orientation (`ImageOps.exif_transpose`) and converts to RGB. If the longest side is above `RESIZE_MAX_DIM` (default <!-- claim:iqg.config.resize_max_dim -->1600), the image is downscaled with Lanczos resampling. OpenCV then converts it to 8-bit grayscale (BGR → GRAY, i.e. 0.299 R + 0.587 G + 0.114 B).
2. **Blur score: variance of the Laplacian.** The Laplacian (`cv2.Laplacian` with OpenCV's default 3×3 kernel) is a second-derivative filter. It produces large positive and negative values at edges and values near zero in smooth areas. A sharp image has many strong edges, so the filtered values are spread widely and their variance is high. Blur softens edges, the values cluster around zero, and the variance drops.
3. **Brightness:** the mean of the grayscale image.
4. **Decision:**

   ```text
   is_ok = blur_score >= BLUR_MIN  AND  BRIGHT_MIN <= brightness <= BRIGHT_MAX
   ```

   All three comparisons are inclusive. A value exactly on a threshold passes.

The two test images in `tests/assets` show the difference. Both are 256×256 JPEGs with almost the same brightness, but very different blur scores:

<!-- claim-waiver-begin: computed outputs of the scoring pipeline (blur score and brightness) for the two test images, not performance figures; the full-precision response for sharp_small.jpg is shown under Quick start -->
| Image | `blur_score` | `brightness` | `is_ok` (default thresholds) |
|---|---:|---:|---|
| `sharp_small.jpg` | 1033.12 | 64.75 | `true` |
| `blurry_small.jpg` | 1.16 | 64.74 | `false` (blur below 140) |
<!-- claim-waiver-end -->

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
| <!-- claim-waiver: HTTP status code, not a measurement -->400 | `Invalid image data` | The file cannot be decoded, is incomplete (truncated), or has more pixels than `MAX_IMAGE_PIXELS`. |
| <!-- claim-waiver: HTTP status code, not a measurement -->413 | `File too large` | The request body is larger than `MAX_UPLOAD_MB` MiB plus <!-- claim:iqg.config.multipart_allowance_kib -->64 KiB, or the file itself is larger than `MAX_UPLOAD_MB` MiB. |
| <!-- claim-waiver: HTTP status code, not a measurement -->415 | `Unsupported media type` | The content type declared for the file does not start with `image/`. |
| <!-- claim-waiver: HTTP status code, not a measurement -->422 | FastAPI validation error | The `file` field is missing, or a query parameter is invalid. |

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
| `BLUR_MIN` | <!-- claim:iqg.config.blur_min -->`140.0` | Minimum blur score. |
| `BRIGHT_MIN` | <!-- claim:iqg.config.bright_min -->`20` | Minimum brightness (0–255). |
| `BRIGHT_MAX` | <!-- claim:iqg.config.bright_max -->`235` | Maximum brightness (0–255). |
| `RESIZE_MAX_DIM` | <!-- claim:iqg.config.resize_max_dim -->`1600` | Longest side before scoring (minimum 256). |
| `MAX_UPLOAD_MB` | <!-- claim:iqg.config.max_upload_mb -->`6` | Maximum file size in MiB (<!-- claim-waiver: unit definition, not a measurement -->1 MiB = <!-- claim-waiver: unit definition, not a measurement -->1,048,576 bytes; allowed range 1–25). The whole request may be up to <!-- claim:iqg.config.multipart_allowance_kib -->64 KiB larger, to leave room for the multipart framing. |
| `MAX_IMAGE_PIXELS` | <!-- claim:iqg.config.max_image_pixels -->`25000000` | Maximum width × height (25 megapixels). |
| `LOG_LEVEL` | `INFO` | `DEBUG` adds one structured timing record per request, with no image data or file names. |
| `LOG_JSON` | `true` | JSON log lines. |
| `PROMETHEUS_ENABLED` | `true` | Serve `/metrics`. |
| `APP_NAME` | `image-quality-gate` | Returned by `/version`. |
| `APP_VERSION` | <!-- claim-waiver: version string, not a number -->`0.0.0` | Returned by `/version` and in diagnostics. |

The blur score depends on the image source and on `RESIZE_MAX_DIM`, so tune thresholds on your own images; see [Threshold tuning](#threshold-tuning).

## Input limits and security

Uploaded files are untrusted input. The checks run from cheapest to most expensive, so a bad request is turned away before it costs much. Each rejection reuses the service's existing status codes and messages, so clients need no changes.

| Check | What it stops | How |
|---|---|---|
| Request size | Huge uploads tying up memory or disk | A small ASGI middleware (`app/core/body_limit.py`) answers 413 straight away if `Content-Length` is above the limit, without reading the body. For chunked uploads with no length, it counts bytes as they arrive and stops at the first chunk past the limit. |
| Content type | Non-image uploads | <!-- claim-waiver: HTTP status code, not a measurement -->415 unless the declared type starts with `image/`. |
| File size | Files just over `MAX_UPLOAD_MB` | The handler reads at most the limit plus one byte. |
| Pixel count | "Decompression bombs": small files that expand into huge images | After Pillow reads the header (width and height) and before any pixels are decoded, images over `MAX_IMAGE_PIXELS` get a 400. |
| Incomplete files | Truncated uploads scored as if they were complete | Pillow's default behavior is kept: decoding raises an error when the data ends before the image is complete. |

Measured with `bench/limits_probe.py` in the benchmark session described under [Performance](#performance), comparing the code before these checks (`8e47d74`) with the current code. Each run sent one request to a fresh single-worker uvicorn server, and each case ran 5 times per version. Times are medians, with the lowest and highest run in parentheses. Memory is the median growth of the server's peak resident memory over the request.

**Ubuntu on WSL2**

<!-- claims-table:begin id=iqg.limits.ubuntu-wsl2 -->
| Request | Version | Response | Time (ms) | Peak memory growth (MiB) |
|---|---|---|---:|---:|
| 100 MiB upload (with `Content-Length`) | before | HTTP 413 after the whole upload was received | 302 (250–315) | 101 |
|  | now | HTTP 413 before the upload finished; body not read | 1.1 (0.7–2.3) | 0.3 |
| 100 MiB chunked upload | before | HTTP 413 after the whole upload was received | 286 (264–413) | 101 |
|  | now | HTTP 413 before the upload finished | 23 (16–56) | 2.5 |
| 165 KiB PNG that decodes to 12000 × 12000 (144 MP) | before | **HTTP 200**: the whole image was decoded | 1,939 (1,887–2,529) | 825 |
|  | now | HTTP 400 before any pixels were decoded | 6.4 (5.3–16) | 0.9 |
| First half of `sharp_small.jpg` (truncated) | before | **HTTP 200**, scored as sharp: the missing part was filled with flat gray | 8.2 (6.0–18) | 5.7 |
|  | now | HTTP 400 | 6.3 (5.0–15) | 1.3 |
| Valid 5000 × 5000 JPEG (exactly 25 MP) | before | HTTP 200 | 467 (458–515) | 292 |
|  | now | HTTP 200, byte-identical response | 492 (469–515) | 292 |
<!-- claims-table:end -->

**Native Windows 11**

<!-- claims-table:begin id=iqg.limits.windows-native -->
| Request | Version | Response | Time (ms) | Peak memory growth (MiB) |
|---|---|---|---:|---:|
| 100 MiB upload (with `Content-Length`) | before | HTTP 413 after the whole upload was received | 522 (372–572) | 100 |
|  | now | HTTP 413 before the upload finished; body not read | 1.7 (1.1–3.7) | 0.3 |
| 100 MiB chunked upload | before | HTTP 413 after the whole upload was received | 456 (442–504) | 102 |
|  | now | HTTP 413 before the upload finished | 32 (28–72) | 2.5 |
| 165 KiB PNG that decodes to 12000 × 12000 (144 MP) | before | **HTTP 200**: the whole image was decoded | 1,326 (1,319–1,356) | 825 |
|  | now | HTTP 400 before any pixels were decoded | 27 (17–28) | 0.9 |
| First half of `sharp_small.jpg` (truncated) | before | **HTTP 200**, scored as sharp: the missing part was filled with flat gray | 26 (19–42) | 3.0 |
|  | now | HTTP 400 | 16 (13–26) | 0.8 |
| Valid 5000 × 5000 JPEG (exactly 25 MP) | before | HTTP 200 | 424 (405–457) | 290 |
|  | now | HTTP 200, byte-identical response | 426 (408–437) | 290 |
<!-- claims-table:end -->

Some details worth knowing:

- **A JPEG missing only its final end-of-image marker is rejected**, even if its pixels are complete. The decoder cannot tell it apart from a file cut off at the end of its data, and the service relies on the decoder rather than guessing. Extra bytes after the end of a JPEG, or a PNG without its final `IEND` chunk, are still accepted and score exactly the same as the intact file.
- **Corruption that doesn't end the data early is not detected.** For example, zeroed bytes in the middle of a JPEG decode without an error and are scored as decoded. Pillow does not report these cases, so the service does not claim to.
- **The pixel limit is a policy choice.** 25 MP covers common 12 MP phone photos, but rejects images from cameras that shoot at higher resolutions (for example 48 MP modes). Raise `MAX_IMAGE_PIXELS` if you need them, and plan for the memory: a 25 MP image grew the server's peak memory by <!-- claim:iqg.limits.jpeg_5000x5000.after.rss_growth_mib.ubuntu-wsl2 -->292 MiB on Ubuntu and <!-- claim:iqg.limits.jpeg_5000x5000.after.rss_growth_mib.windows-native -->290 MiB on Windows in the probe above.

### Still needed in a real deployment

The service on its own is not hardened for direct exposure to the internet. The deployment files were not changed as part of this work. Before public use:

- **Run it behind a reverse proxy** that limits body size and upload time. Uvicorn has no request-body timeout, so a client can send an upload very slowly and hold a connection open. Example for nginx (not tested here):

  <!-- claim-waiver-begin: example nginx configuration (not tested here), not a measurement -->
  ```nginx
  client_max_body_size 7m;    # slightly above MAX_UPLOAD_MB
  client_body_timeout 15s;    # drop very slow uploads
  ```
  <!-- claim-waiver-end -->

- **Add authentication and rate limiting.** There are none, and CORS allows every origin.
- **Limit concurrency.** Each image is processed on the worker's event loop, and the worker can't start another request until it finishes. In the benchmark session, `/quality` took a median of <!-- claim:iqg.photo_4032x3024.current_quality.p50.ubuntu-wsl2 -->237.0 ms on Ubuntu and <!-- claim:iqg.photo_4032x3024.current_quality.p50.windows-native -->229.6 ms on Windows for a 12 MP photo. The 25 MP image in the input-limits probe took <!-- claim:iqg.limits.jpeg_5000x5000.after.elapsed_ms.ubuntu-wsl2 -->492 ms and <!-- claim:iqg.limits.jpeg_5000x5000.after.elapsed_ms.windows-native -->426 ms and grew the server's peak memory by about <!-- claim:iqg.limits.jpeg_5000x5000.after.rss_growth_mib.ubuntu-wsl2 -->292 MiB. Several large images at once can exhaust a small container: `docker-compose.yml` sets a <!-- claim:iqg.config.docker_memory_mib -->512 MiB memory limit and the Dockerfile starts <!-- claim:iqg.config.docker_workers -->2 worker processes.
- **Restrict `/quality/diagnostics`, `/quality/config` and `/metrics`** if you don't want to reveal thresholds, library versions or traffic data.
- **Keep Pillow and OpenCV up to date.** They parse untrusted files.

## Design decisions

- **Classic image measurements instead of a machine-learning model.** The variance of the Laplacian and the mean brightness are fast on a CPU, easy to explain, and need no training data or GPU. The trade-off is that they measure the image, not its meaning, and the thresholds need tuning for each image source.
- **Normalize before measuring.** Fixing the orientation and capping the longest side at <!-- claim:iqg.config.resize_max_dim -->1600 px keeps the work bounded and makes scores from different cameras more comparable. The trade-off is that the blur score depends on `RESIZE_MAX_DIM`, so thresholds should be re-tuned if it changes.
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

There are <!-- claim:iqg.tests.ubuntu-wsl2 -->130 tests (many are parameterized cases). The benchmark session described under [Performance](#performance) ran them in both environments, and all of them passed on native Windows too (<!-- claim:iqg.tests.windows-native -->130 tests).

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

## Browser parity

`app/services/pipeline.py` is the FastAPI-free entry both routes run through, and the same code the [portfolio](https://github.com/BrianKYildirim/portfolio) runs in the browser through Pyodide. `tests/parity/` holds the fixture images, the intended outcome of each case and the native golden answers the portfolio compares its Pyodide results against; only `tests/parity/run_native.py` writes the goldens. See [tests/parity/README.md](tests/parity/README.md).

## Performance

`scripts/bench_quality.py` compares this code with an older version over real HTTP. It starts one single-worker uvicorn server per variant and sends each round of requests to every variant in a shuffled order. It reports per-request percentiles, plus the *paired* difference per round, which cancels out background load on a shared machine. It also times the image pipeline in-process, alternating the old and new code call by call.

`bench/run_bench.py` runs the whole measurement for one environment: five `bench_quality.py` sessions on fresh servers, the input-limits probe (`bench/limits_probe.py`) and the test suite. It writes the raw per-request timings, `meta.json` (machine, OS, Python and package versions, load, power) and `summary.json` to `bench/results/<date>-<commit>/<environment>/`. The baseline is commit `8e47d74`, the code before diagnostics and the input limits were added, which the runner extracts with `git archive`.

```bash
python bench/run_bench.py --env ubuntu-wsl2      # or: --env windows-native
```

<!-- bench-methodology:begin -->
**How these numbers were measured.** One session on 2026-11-06, at commit `a0834d4`, results in `bench/results/2026-11-06-a0834d4/`. Both environments ran on the same 4-core, 8-thread laptop (Intel i7-1185G7), one after the other, with CPython 3.11.18 and the same pinned packages (`bench/constraints-py311.txt`: NumPy 2.4.6, OpenCV 5.0.0, Pillow 13.0.0, FastAPI 0.161.1, uvicorn 0.54.0). On Ubuntu 26.04 under WSL2, uvicorn runs on uvloop; on native Windows 11 it runs on asyncio, because uvloop doesn't support Windows.

- **Warm-up:** each session starts fresh servers and sends 3 rounds per image before measuring; they aren't kept. The in-process comparison makes one warm-up call per side and image.
- **Timer:** `time.perf_counter_ns` on the client, from sending a request to reading the whole response: `CLOCK_MONOTONIC` with <!-- claim:iqg.timer.resolution_ns.ubuntu-wsl2 -->1 ns resolution on Ubuntu, and `QueryPerformanceCounter` with <!-- claim:iqg.timer.resolution_ns.windows-native -->100 ns resolution on Windows. A request takes milliseconds, so the timer's own cost isn't subtracted.
- **Repetitions:** 5 sessions per environment, each with 300 rounds per small image and 60 per photo, plus 400 in-process pairs per small image and 20 per photo. A latency is the median of the five per-session p50 values, with the lowest and highest session in parentheses. A difference is the median of all paired per-round differences pooled over the five sessions, with their 25th and 75th percentiles in parentheses.
- **Outliers:** no requests are dropped. A whole session is marked invalid, kept in `raw/` and left out of the statistics if the laptop is on battery, if the CPU stays busy after waiting, or if a step fails. No session in this run was invalid.
- **Noise control:** every round sends one request to each variant in a shuffled order, each variant has its own server (so one endpoint's memory use can't slow another), and differences are paired per round. AC power, Windows power mode Balanced, other applications closed, one environment at a time. Before each step the runner waited until CPU use was below <!-- claim-waiver: threshold setting of the runner's quiet gate (--max-busy-pct), not a measurement -->10%. On Windows, CPU use was <!-- claim:iqg.meta.cpu_busy_start.windows-native -->14.12% when the session started and <!-- claim:iqg.meta.cpu_busy_end.windows-native -->4.87% when it ended. The 1-minute load average of the WSL VM, which can't see activity on the Windows host, was <!-- claim:iqg.meta.loadavg_start.ubuntu-wsl2 -->0 at the start and <!-- claim:iqg.meta.loadavg_end.ubuntu-wsl2 -->1.12 at the end.
<!-- bench-methodology:end -->

Latencies and differences are in milliseconds. The two diagnostics rows measure the instrumented endpoint `/quality/diagnostics`, which computes extra statistics; they are not the cost of `/quality`.

**Ubuntu on WSL2**

<!-- claims-table:begin id=iqg.performance.ubuntu-wsl2 -->
| Measurement (ms) | `blurry_small.jpg` (256×256) | `sharp_small.jpg` (256×256) | 1600×1200 JPEG | 4032×3024 JPEG, EXIF 6 |
|---|---:|---:|---:|---:|
| Baseline `/quality`, p50 | 2.64 (2.59–2.85) | 2.58 (2.45–3.33) | 24.9 (24.3–25.3) | 237.5 (236.8–241.0) |
| `/quality` now, difference | +0.06 (−0.16 to +0.32) | +0.06 (−0.19 to +0.29) | 0.0 (−1.1 to +1.2) | −1.5 (−7.6 to +4.0) |
| `/quality/diagnostics` (instrumented), difference | +0.57 (+0.34 to +0.95) | +0.55 (+0.32 to +0.89) | +11.9 (+10.5 to +14.4) | +0.1 (−6.8 to +4.7) |
| `/quality/diagnostics` + 64-bin histogram (instrumented), difference | +1.73 (+1.44 to +2.54) | +1.72 (+1.40 to +2.48) | +29.6 (+27.9 to +32.5) | +17.6 (+10.7 to +23.3) |
<!-- claims-table:end -->

**Native Windows 11**

<!-- claims-table:begin id=iqg.performance.windows-native -->
| Measurement (ms) | `blurry_small.jpg` (256×256) | `sharp_small.jpg` (256×256) | 1600×1200 JPEG | 4032×3024 JPEG, EXIF 6 |
|---|---:|---:|---:|---:|
| Baseline `/quality`, p50 | 2.38 (2.20–2.48) | 2.32 (2.18–2.48) | 34.2 (33.8–34.9) | 228.9 (226.5–230.6) |
| `/quality` now, difference | 0.00 (−0.16 to +0.19) | 0.00 (−0.16 to +0.16) | −0.2 (−2.1 to +1.5) | −1.0 (−6.2 to +4.8) |
| `/quality/diagnostics` (instrumented), difference | +0.49 (+0.27 to +0.69) | +0.49 (+0.28 to +0.69) | +12.2 (+10.7 to +13.9) | +12.4 (+5.9 to +17.9) |
| `/quality/diagnostics` + 64-bin histogram (instrumented), difference | +1.68 (+1.42 to +2.07) | +1.64 (+1.40 to +1.96) | +29.6 (+27.8 to +32.0) | +28.1 (+22.8 to +34.0) |
<!-- claims-table:end -->

How to read this:

- **The default endpoint shows no measurable change.** In both environments, the 25th-to-75th-percentile range of every `/quality` difference includes zero. In-process, the extra bookkeeping (stage timer, per-check results) cost a median of <!-- claim:iqg.pipeline.sharp_small.delta.ubuntu-wsl2 -->0.018 ms per call on Ubuntu and <!-- claim:iqg.pipeline.sharp_small.delta.windows-native -->0.018 ms on Windows for `sharp_small.jpg`, whose whole pipeline took <!-- claim:iqg.pipeline.sharp_small.current.ubuntu-wsl2 -->0.54 ms and <!-- claim:iqg.pipeline.sharp_small.current.windows-native -->0.77 ms.
- **Diagnostics cost extra:** mostly the additional statistics over the full arrays, plus the histogram when requested. They are meant for debugging and demos. On the 4032×3024 photo, the diagnostics difference without a histogram is within the noise on Ubuntu but not on Windows.
- **Keep diagnostics traffic separate.** In an earlier run where one server handled both endpoints, `/quality` on the 1600×1200 image was measurably slower after diagnostics requests on the same worker, because of how memory was reused. That is why the benchmark gives each variant its own server.
- **These are measurements from one laptop, not a load test.**

**Windows and Ubuntu.** I state a difference only where the two environments' lowest-to-highest session ranges don't overlap. For baseline `/quality` (p50), `blurry_small.jpg` was faster on Windows 11, native by <!-- claim:iqg.cmp.blurry_small.baseline_quality.p50 -->~1.11×, medians of 5 sessions each; for `sharp_small.jpg` the ranges overlap, so there is <!-- claim:iqg.cmp.sharp_small.baseline_quality.p50 -->no measurable difference (medians of 5 sessions each). The 1600×1200 photo was faster on Ubuntu on WSL2 by <!-- claim:iqg.cmp.photo_1600x1200.baseline_quality.p50 -->~1.37×, medians of 5 sessions each, and the 4032×3024 photo was faster on Windows 11, native by <!-- claim:iqg.cmp.photo_4032x3024.baseline_quality.p50 -->~1.04×, medians of 5 sessions each. Without HTTP, the current image pipeline itself was faster on Ubuntu on WSL2 by <!-- claim:iqg.cmp.pipeline.sharp_small.current -->~1.43×, medians of 5 sessions each for `sharp_small.jpg`, and faster on Ubuntu on WSL2 by <!-- claim:iqg.cmp.pipeline.photo_1600x1200.current -->~1.83×, medians of 5 sessions each for the 1600×1200 photo. Why is a hypothesis I haven't verified: the image work itself runs faster on Ubuntu, while the per-request path around it (uvicorn on uvloop inside the WSL2 VM versus asyncio on Windows, plus loopback networking) costs more on Ubuntu, which outweighs the image work for small images. A run with `--uvicorn-loop asyncio` on Linux would separate the event loop from the rest.

## Challenges and lessons learned

- **Timing a library that works lazily.** `Image.open` in Pillow only reads the header; the pixels are decoded the first time they are needed, which turned out to be inside the EXIF-orientation step. Timing the code naively would have blamed orientation for the cost of decoding. Lesson: read the library's source before instrumenting it.
- **Proving that nothing changed.** "Opt-in" diagnostics are only safe if the default response is unchanged. The tests keep a verbatim copy of the original endpoint and compare raw response bytes over a corpus of images and error cases. They also compare the OpenAPI schema.
- **Benchmarking on a busy laptop.** Comparing one block of requests with another gave contradictory answers: one run showed a clear slowdown, another a speed-up of the same size. Switching to interleaved requests with paired differences, plus A/A control runs (old code against itself), made the results stable enough to trust.
- **That method found real regressions, twice.** Both times the refactored code freed a large array later than the original did, so later steps had to map fresh memory instead of reusing the freed memory:
  - The rotated copy of the image, held a little too long, made each small image measurably slower in an in-process comparison.
  - The <!-- claim-waiver: size arithmetic, 1600 x 1200 values x 8 bytes (float64), not a measurement -->15 MB Laplacian array of a 1600×1200 image, held until the end of the request so diagnostics could use it, added a few milliseconds per request. Counting minor page faults per request (thousands more than the original) confirmed the cause.

  In both cases, freeing the array at the same point as the original code made the fault counts identical again. Diagnostics now ask to keep the Laplacian explicitly. Lesson: in Python code that handles large buffers, *when* an object is released can matter for speed. A related lesson about the benchmark itself: requests to one endpoint can change the speed of another endpoint in the same process, so each variant now gets its own server.
- **Instrumentation has a cost.** The first stage timer used `@contextlib.contextmanager`, and its cost per stage was large enough to show up next to the stages it timed. A small class-based timer made it much cheaper.
- **A security review turned up real problems.**
  - A <!-- claim-waiver: size of the probe's input file (168,457 bytes), a workload parameter, not a measurement -->165 KiB PNG could make the service decode a 144-megapixel image (<!-- claim:iqg.limits.png_bomb_12000x12000.before.rss_growth_mib.ubuntu-wsl2 -->+825 MiB of peak memory, and more than a second per request, in the probe above).
  - A <!-- claim-waiver: size of the probe's test upload, a workload parameter, not a measurement -->100 MiB upload was fully received and read into memory before being rejected.
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

`scripts/tune_blur.py` scores every image in a folder and suggests thresholds. `BLUR_MIN` comes from Otsu's method on the log of the blur scores; the brightness range covers the middle <!-- claim:iqg.design.tune_brightness_central_pct -->95% of images.

```bash
python scripts/tune_blur.py --dir path/to/your/images --csv out/scores.csv --plot out/
```

It re-implements the same preprocessing steps rather than importing the service code. For the two test images its scores match the service's exactly, but it does not apply the service's upload and pixel limits. `--plot` needs matplotlib, which is in `requirements.txt`. Put representative sharp, blurry, dark and bright images from your real source in the folder, then confirm the suggested values through the API.
