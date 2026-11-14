# Parity: native service vs the browser (Pyodide)

The portfolio at [github.com/BrianKYildirim/portfolio](https://github.com/BrianKYildirim/portfolio)
runs this service's pipeline in the browser through Pyodide. These files fix the
native side of that comparison: a set of input images, what each one is meant to
show, and the exact answer the native pipeline gives for each of them.

```
tests/parity/
  make_fixtures.py        writes fixtures/ (copies of test assets plus generated images)
  cases.json              every case: fixture, content type, settings, intended outcome
  run_native.py           the only writer of golden/ (--write) and its check (--check)
  fixtures/               the input files, kept byte for byte (-text)
  golden/<case>.json      the native answer for each case
  golden/meta.json        native environment, versions and the hashes binding the goldens
  tolerances.json         native-vs-Pyodide tolerances, written by the portfolio's comparison tool
```

## How it works

`run_native.py` calls `app.services.pipeline.outcome(data, content_type, settings, bins)`
for every case, in `cases.json` order. That is the entry both API routes run through
and the one the browser calls, so both sides run the same code with the same
settings and the same histogram bin count (`bins` in `cases.json`).

- Before importing `app`, the runner removes every environment variable named after
  a `Settings` field and moves to an empty temporary directory, so no environment
  value or `.env` file reaches `Settings`. The browser has neither. Each case's
  `settings` are passed to `Settings(**settings)` and show up as source `init`.
- A successful case's golden is the `QualityDiagnosticsResponse` JSON without
  `diagnostics.timings` and `diagnostics.versions` (the versions are recorded once,
  in `meta.json`). A rejected case's golden is
  `{"error": {"status", "detail", "cause"}}`, where `cause` is the class name of the
  error behind a 400, or null.
- Every golden is written with sorted keys, a one-space indent, UTF-8 and LF, so
  equal answers give byte-identical files.
- Each case states its intent in `expect`: the status, and where it is the point of
  the case, `is_ok`, the error `cause`, or `blur_margin_max` (the blur check passes
  by no more than this). A golden that contradicts its intent stops the runner with
  `PARITY_INTENT_MISMATCH <case>` and exit code one, so a wrong answer cannot become
  a golden quietly.
- `meta.json` records the environment, the Python and library versions (numpy,
  OpenCV, Pillow, libjpeg, libjpeg-turbo, pydantic, pydantic-settings), the bin count,
  the sha256 of each `app/` source in the pipeline's import closure (in the LF form
  git stores), the sha256 of every fixture and of `cases.json`. The goldens are only
  valid for exactly those inputs.

### Golden files come only from the native runner

`run_native.py --write` is the only way golden files are written. The Pyodide side
(the portfolio's comparison tool) has no write mode, so the browser build can never
produce its own expected output. Never edit a golden by hand, and never drop or
rewrite a case to make parity pass.

`run_native.py --check` builds every golden in memory and compares it with the files
byte for byte, along with the versions, bin count and hashes in `meta.json` (not the
date). It prints `PARITY_GOLDEN_OK <n> cases`, or the first difference and exits
with code one.

### Local only

The goldens record one exact environment: the WSL bench venv built from
`bench/constraints-py311.txt` (CPython 3.11, OpenCV, Pillow and numpy at the
versions in `meta.json`). Upstream CI installs unpinned requirements, so another
OpenCV or Pillow release could change the last digits of a score. That is why no
pytest test compares the goldens; run the check by hand in the pinned venv.

`tolerances.json` is written by the portfolio's comparison tool. It holds the
tolerances allowed between the native goldens and the Pyodide results (Pyodide
ships its own builds of OpenCV, Pillow and libjpeg), the reason for each, and the
largest difference measured.

## Commands

From the repository root, in the pinned venv:

```bash
python tests/parity/make_fixtures.py          # write fixtures/
python tests/parity/run_native.py --write     # write golden/ (after a deliberate change only)
python tests/parity/run_native.py --check     # reproduce golden/ byte for byte
```

`make_fixtures.py` is deterministic: running it again gives identical bytes. It
copies `tests/assets/sharp_small.jpg` and `blurry_small.jpg`, generates the rest
(the photos come from `make_photo` in `scripts/bench_quality.py`), and checks that
`fixtures/strivoimg1.jpg` is the portfolio's Dark UI sample by its sha256. It
stops if the fixture folder holds anything else.

`make_fixtures.py --memory-probe DIR` writes only a 5000 x 5000 photo to `DIR`,
which must be outside the repository. It is used to measure the browser demo's
memory and is not a parity fixture.

## Cases

| Case | Fixture | What it covers | Intent |
|------|---------|----------------|--------|
| `blurry_small` | `blurry_small.jpg` | The demo's Blurry sample (a copy of `tests/assets/blurry_small.jpg`) | 200, fails |
| `blurry_small_overrides` | `blurry_small.jpg` | Settings passed in: `blur_min` 70.0 and `bright_min` 10, reported as source `init` | 200 |
| `cmyk` | `cmyk.jpg` | A CMYK JPEG, converted to RGB | 200 |
| `empty` | `empty.jpg` | An empty file | 400, `UnidentifiedImageError` |
| `exif2` to `exif8` | `exif2.jpg` to `exif8.jpg` | EXIF orientations two to eight: mirror, rotations, transpose and transverse | 200 |
| `gray_l` | `gray_l.png` | A grayscale (L) PNG | 200 |
| `near_threshold` | `near_threshold.png` | A blend of a photo and a blurred copy whose blur score is just above `blur_min`; a PNG, so it decodes identically everywhere | 200, passes, blur margin at most 0.1 |
| `no_content_type` | `sharp_small.jpg` | An upload without a content type | 415 |
| `not_image` | `sharp_small.jpg` | Image bytes sent as `text/plain` | 415, no cause |
| `one_pixel` | `one_pixel.png` | A single pixel: Laplacian variance zero, a histogram over a constant array | 200, fails |
| `overlimit_header` | `overlimit_header.png` | A small PNG whose header claims 6000 x 6000, rejected before any pixel is decoded | 400, `ImageTooLarge` |
| `palette_p` | `palette_p.png` | A palette (P) PNG with a 64-colour palette | 200 |
| `photo_1600x1200` | `photo_1600x1200.jpg` | The bench photo at the resize limit | 200 |
| `photo_4032x3024_exif6` | `photo_4032x3024_exif6.jpg` | The bench phone-sized photo: rotated by EXIF orientation six, then resized | 200 |
| `rgba` | `rgba.png` | A PNG with an alpha channel ramping from fully transparent to opaque | 200 |
| `sharp_small` | `sharp_small.jpg` | The demo's Sharp sample (a copy of `tests/assets/sharp_small.jpg`) | 200, passes |
| `strivoimg1` | `strivoimg1.jpg` | The demo's Dark UI sample (the portfolio's `static/strivoimg1.jpg`) | 200 |
| `too_large` | `photo_4032x3024_exif6.jpg` | The phone-sized photo with `max_upload_mb` set to one | 413 |
| `truncated` | `truncated.jpg` | The first half of `photo_1600x1200.jpg` | 400, `OSError` |
| `underexposed` | `underexposed.png` | The demo's Dark sample: `sharp_small.jpg` with RGB scaled by 0.15, rounded half up | 200, fails |

"passes" and "fails" are `is_ok`. A case without one is there for its image
metadata, measurements and histogram, which parity compares field by field.
