# Image Quality Gate (FastAPI + OpenCV)

A small, production-ready microservice that scores uploaded images for sharpness and brightness and returns a simple ``is_ok`` decision. It is designed to sit next to your upload pipeline (for example, a Spring Boot app that uses S3 presigned uploads) and reject unusable photos before they enter your dataset.

TL;DR
- Blur metric: Variance of Laplacian (higher means sharper)
- Brightness metric: Mean grayscale in the range 0 to 255
- Decision rule: is_ok equals true when ``blur_score`` is greater than or equal to ``BLUR_MIN`` and brightness is between ``BRIGHT_MIN`` and ``BRIGHT_MAX``
- Recommended thresholds from tuning: ``BLUR_MIN equals 160``, ``BRIGHT_MIN equals 40``, ``BRIGHT_MAX equals 180``
- Endpoints provided: ``POST /quality, GET /health, GET /metrics, GET /version``

-------------------------------------------------------------------------------

## Table of Contents

1. [Why this exists](#1-why-this-exists)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [API](#4-api)
   - POST /quality
   - GET /health
   - GET /version
   - GET /metrics
5. [Configuration](#5-configuration)
6. [Quickstart](#6-quickstart)
7. [Run in Docker](#7-run-in-docker)
8. [Directory structure](#8-directory-structure)
9. [Threshold tuning](#9-threshold-tuning)
10. [Performance expectations](#10-performance-expectations)
11. [Security and robustness](#11-security-and-robustness)
12. [Testing and CI](#12-testing-and-ci)
13. [Troubleshooting on Windows and Docker Desktop](#13-troubleshooting-on-windows-and-docker-desktop)
14. [Acknowledgments](#14-acknowledgments)

-------------------------------------------------------------------------------

## 1. Why this exists

Crowdsourced or reporting apps collect many photos. Some are blurry, dark, or overexposed, which hurts downstream use such as triage, machine learning, or public dashboards. This service provides a fast, deterministic gate to keep the image set clean without adding GPU or heavy ML complexity.

-------------------------------------------------------------------------------

## 2. Features

- Fast: OpenCV executes C or C++ under the hood and runs well on CPUs without GPUs.
- Deterministic: Variance of Laplacian for blur plus simple brightness thresholds.
- Configurable: All thresholds are set by environment variables; no redeploy required to tune.
- Operational: Prometheus metrics endpoint, JSON logs, health, and version endpoints.
- Secure by default: Multipart-only in this version, type and size checks, EXIF orientation fix, and CORS can be restricted.
- Portable: Single Docker image with minimal footprint.
- Tested: Unit and integration tests with fixture images.

-------------------------------------------------------------------------------

## 3. Architecture

Request flow for the minimum viable product

1. The client uploads the original photo to object storage using a presigned URL that your backend issues.
2. Your backend fetches the photo bytes or a thumbnail and sends them to this service using ``POST`` to the quality endpoint.
3. This service computes both ``blur_score`` using the variance of Laplacian on a grayscale version of the image and brightness using the mean grayscale intensity.
4. The response includes the metrics, image dimensions, ``is_ok`` decision, and the thresholds that were applied.
5. Your backend either rejects the photo and returns an appropriate error to the client, or accepts the photo and persists metadata, including the ``blur_score``.

-------------------------------------------------------------------------------

## 4. API

``POST /quality``
- Purpose: Accept a multipart form upload with field name file and return ``blur_score``, ``brightness``, ``width``, ``height``, ``is_ok``, and the thresholds used.
- Success: Returns a ``200 OK`` response with a JSON body describing the metrics and decision.
- Errors: Returns ``415`` for unsupported media type, ``413`` for payload that is too large, and ``400`` for invalid image data.

``GET /health``
- Purpose: Liveness probe.
- Success: Returns ``200 OK`` with a small JSON body indicating status ok.

``GET /version``
- Purpose: Exposes app name and version from the environment for debugging and rollouts.

``GET /metrics``
- Purpose: Exposes Prometheus metrics in text format, including request counters and latency histograms.

-------------------------------------------------------------------------------

## 5. Configuration

Create a ``.env`` file or provide environment variables at runtime. The following variables are supported.

- ``BLUR_MIN``: numeric. Minimum blur score required for acceptance. Example 160.
- ``BRIGHT_MIN``: integer. Minimum brightness allowed. Example 40.
- ``BRIGHT_MAX``: integer. Maximum brightness allowed. Example 180.
- ``RESIZE_MAX_DIM``: integer. Maximum side length before scoring to bound CPU use. Example 1600.
- ``MAX_UPLOAD_MB``: integer. Maximum file size accepted in megabytes. Example 6.
- ``LOG_LEVEL``: string. Example INFO.
- ``LOG_JSON``: boolean. Example true.
- ``PROMETHEUS_ENABLED``: boolean. Example true.
- ``APP_NAME``: string. Example image-quality-gate.
- ``APP_VERSION``: string. Example 0.1.0.

Note: If you change ``RESIZE_MAX_DIM``, you should re-run threshold tuning because scores change with resolution.

-------------------------------------------------------------------------------

## 6. Quickstart

1. Create a Python virtual environment for your platform.
2. Activate the environment.
3. Install the dependencies from requirements. Use binary wheels only on Windows to avoid building from source.
4. Copy .env example to .env and adjust thresholds if needed.
5. Start the service using Uvicorn on port 8080.
6. Visit slash docs for interactive API documentation or call the endpoints directly.
7. Test with the included fixture images to verify that sharp photos return ``is_ok`` equals true and blurry photos return ``is_ok`` equals false.

-------------------------------------------------------------------------------

## 7. Run in Docker

1. Build the Docker image from the Dockerfile located at the project root.
2. Run the container and publish port 8080 while loading environment variables from your .env file.
3. Alternatively, use Docker Compose. The compose file builds the image, sets the environment, and publishes port 8080.
4. If you are mixing architectures, set the platform to linux slash amd64 as appropriate.

-------------------------------------------------------------------------------

## 8. Directory structure

High-level layout

- app folder: application code
  - main module: FastAPI app factory, CORS, version, metrics setup
  - api routes: quality and health endpoints
  - core config: environment settings using pydantic settings
  - core logging: JSON logging helpers
  - core metrics: Prometheus instrumentator setup
  - services preprocess: EXIF orientation fix, resize, and color conversions
  - services quality: blur and brightness metrics and decision function
  - models schemas: response models
- scripts folder: contains the tune_blur helper for threshold tuning, as well as sample images you can test
- tests folder: assets and unit plus integration tests
- .env example, requirements files, Dockerfile, docker compose file, pytest configuration, optional Makefile, README, and changelog

-------------------------------------------------------------------------------

## 9. Threshold tuning

The scripts folder includes a tune_blur helper that scans a folder of images, computes blur and brightness, and suggests a BLUR_MIN value using a log space Otsu threshold which handles heavy tails. It also supports a labeled sweep mode that assumes a samples folder with subfolders named sharp and any folder with a name that begins with blur such as blur_motion or blur_defocus.

Typical usage
- Place a set of images in the samples folder.
- Run the tuner with or without labels to obtain suggested thresholds.
- Optionally save a CSV of results and histogram images for documentation.
- Update the dot env file with the chosen BLUR_MIN, BRIGHT_MIN, and BRIGHT_MAX values.
- Restart the service and spot check decisions on a random selection of images.

A good starting point from a labeled dataset of 1000 images with equal counts of sharp, motion blur, and defocus blur resulted in ``BLUR_MIN`` equal to 160, ``BRIGHT_MIN`` equal to 40, and ``BRIGHT_MAX`` equal to 180. Your numbers may vary depending on camera sources and lighting conditions.

-------------------------------------------------------------------------------

## 10. Performance expectations

On a typical CPU with no GPU and with ``RESIZE_MAX_DIM`` equal to 1600 on 1080p inputs:
- Median latency often falls between approximately 30 and 60 milliseconds.
- 95th percentile latency often falls between approximately 80 and 120 milliseconds.
- Throughput can exceed 100 requests per second locally using two Uvicorn workers.

If you change ``RESIZE_MAX_DIM``, expect both latency and blur scores to shift. Re-tune thresholds after such changes.

-------------------------------------------------------------------------------

## 11. Security and robustness

- Multipart only in this version; no remote URL fetches, which avoids SSRF risk.
- Reject non-images with status ``415`` and oversize requests with status ``413``.
- Correct EXIF orientation before processing.
- Restrict CORS to your domains for production deployments.
- Use structured JSON logs and do not log raw image bytes.
- Processing is idempotent and safe to retry.

-------------------------------------------------------------------------------

## 12. Testing and CI

- Unit tests cover the quality math and preprocessing edge cases.
- Integration tests exercise the API endpoint using fixture images.
- A GitHub Actions workflow installs dependencies, runs linting and type checks, and executes the test suite on every push and pull request.

-------------------------------------------------------------------------------

## 13. Troubleshooting on Windows and Docker Desktop

- If pip attempts to build from source, use the pinned versions in requirements and install with wheel only flags to avoid building components like OpenCV and NumPy from source.
- If Docker Desktop shows errors related to the named pipe or API version, ensure the active Docker context is desktop dash linux, restart Docker Desktop, and rebuild without cache. As a fallback, build and run with plain docker build and docker run without Compose.
- Avoid running Docker or pip commands from MSYS or Git Bash when on Windows; prefer PowerShell or Command Prompt to prevent path and SSL issues.

-------------------------------------------------------------------------------

## 14. Acknowledgments

Built with FastAPI, OpenCV, NumPy, and Pillow. Metrics are provided by the Prometheus FastAPI Instrumentator.
