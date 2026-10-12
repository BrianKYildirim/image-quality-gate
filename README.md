# Image Quality Gate

A FastAPI microservice that evaluates uploaded images for **sharpness** and **brightness**.

The service uses OpenCV and Pillow to analyze an image, calculate quality metrics, and return a simple `is_ok` decision that can be used by an upload pipeline, moderation system, or backend application.

## How It Works

When an image is uploaded, the service:

1. Validates that the file is an image.
2. Rejects files larger than the configured upload limit.
3. Corrects the image's EXIF orientation.
4. Resizes oversized images to reduce processing cost.
5. Converts the image to grayscale.
6. Calculates:
   - **Blur score** using the variance of Laplacian
   - **Brightness** using the mean grayscale intensity
7. Compares both metrics against configurable thresholds.
8. Returns the metrics and a final `is_ok` decision.

```text
Image upload
     │
     ▼
Validate file type and size
     │
     ▼
Normalize orientation and dimensions
     │
     ▼
Convert to grayscale
     │
     ├── Calculate blur score
     ├── Calculate brightness
     │
     ▼
Apply configured thresholds
     │
     ▼
Return quality decision
```

## Features

- FastAPI REST API
- Multipart image uploads
- Blur detection using variance of Laplacian
- Brightness detection using grayscale intensity
- Configurable quality thresholds
- EXIF orientation correction
- Automatic image resizing
- Upload size validation
- JSON logging
- Request IDs through the `X-Request-ID` header
- Health and version endpoints
- Optional Prometheus metrics
- Docker and Docker Compose support
- Unit and integration tests
- No GPU required

## API Endpoints

### `POST /quality`

Analyzes an uploaded image.

The request must use `multipart/form-data` with a field named `file`.

#### Example with cURL

```bash
curl -X POST "http://localhost:8080/quality" \
  -F "file=@tests/assets/sharp_small.jpg"
```

#### Successful response

```json
{
  "blur_score": 513.42,
  "brightness": 121.8,
  "width": 1280,
  "height": 960,
  "is_ok": true,
  "thresholds": {
    "blur_min": 140.0,
    "bright_min": 20,
    "bright_max": 235
  }
}
```

#### Response fields

| Field | Type | Description |
|---|---:|---|
| `blur_score` | `float` | Variance of Laplacian; higher values generally indicate a sharper image |
| `brightness` | `float` | Average grayscale intensity from `0` to `255` |
| `width` | `integer` | Processed image width |
| `height` | `integer` | Processed image height |
| `is_ok` | `boolean` | Whether the image satisfies the configured quality thresholds |
| `thresholds` | `object` | Threshold values used for the decision |

#### Decision rule

An image passes when:

```text
blur_score >= BLUR_MIN
AND
BRIGHT_MIN <= brightness <= BRIGHT_MAX
```

### `GET /health`

Returns the service health status.

```bash
curl http://localhost:8080/health
```

Response:

```json
{
  "status": "ok"
}
```

### `GET /version`

Returns the configured application name and version.

```bash
curl http://localhost:8080/version
```

Response:

```json
{
  "app": "image-quality-gate",
  "version": "0.1.0"
}
```

### `GET /metrics`

Returns Prometheus-compatible metrics when metrics are enabled.

```bash
curl http://localhost:8080/metrics
```

## HTTP Error Responses

| Status | Meaning |
|---:|---|
| `400` | The uploaded file is not valid image data |
| `413` | The uploaded file exceeds the configured size limit |
| `415` | The uploaded file is not an image |
| `500` | Unexpected server-side error |

## Technology Stack

- **Python**
- **FastAPI**
- **Uvicorn**
- **OpenCV**
- **Pillow**
- **NumPy**
- **Pydantic Settings**
- **Prometheus FastAPI Instrumentator**
- **Docker**

## Project Structure

```text
image-quality-gate/
├── app/
│   ├── api/
│   │   └── routes_quality.py       # API routes
│   ├── core/
│   │   ├── config.py               # Environment-based settings
│   │   ├── logging.py              # Logging configuration
│   │   └── metrics.py              # Prometheus metrics setup
│   ├── models/
│   │   └── schemas.py              # Response schemas
│   ├── services/
│   │   ├── preprocess.py           # Image loading and normalization
│   │   └── quality.py              # Blur, brightness, and decision logic
│   └── main.py                     # FastAPI application entry point
├── scripts/
│   ├── samples/                    # Sample images for analysis
│   └── tune_blur.py                # Blur threshold tuning utility
├── tests/
│   ├── assets/                     # Test images
│   ├── integration/                # API integration tests
│   └── unit/                       # Unit tests
├── .env.example                    # Example configuration
├── Dockerfile                      # Production container definition
├── docker-compose.yml              # Docker Compose configuration
├── Makerfile                       # Common development commands
├── requirements.txt                # Runtime dependencies
└── README.md
```

## Requirements

- Python 3.11 or newer
- `pip`
- Optional: Docker and Docker Compose

## Local Development

### 1. Clone the repository

```bash
git clone https://github.com/BrianKYildirim/image-quality-gate.git
cd image-quality-gate
```

### 2. Create a virtual environment

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

### 5. Start the development server

```bash
uvicorn app.main:app --reload --port 8080
```

The API will be available at:

```text
http://localhost:8080
```

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## Configuration

Configuration is loaded from environment variables or a local `.env` file.

| Variable | Default | Description |
|---|---:|---|
| `BLUR_MIN` | `140.0` | Minimum accepted variance of Laplacian score |
| `BRIGHT_MIN` | `20` | Minimum accepted brightness |
| `BRIGHT_MAX` | `235` | Maximum accepted brightness |
| `RESIZE_MAX_DIM` | `1600` | Maximum image width or height before processing |
| `MAX_UPLOAD_MB` | `6` | Maximum accepted upload size in megabytes |
| `LOG_LEVEL` | `INFO` | Application log level |
| `LOG_JSON` | `true` | Enables JSON-formatted logs |
| `PROMETHEUS_ENABLED` | `true` | Enables the `/metrics` endpoint |
| `APP_NAME` | `image-quality-gate` | Application name returned by `/version` |
| `APP_VERSION` | `0.0.0` | Application version returned by `/version` |

Example configuration:

```dotenv
BLUR_MIN=140
BRIGHT_MIN=20
BRIGHT_MAX=235
RESIZE_MAX_DIM=1600
MAX_UPLOAD_MB=6
LOG_LEVEL=INFO
LOG_JSON=true
PROMETHEUS_ENABLED=true
APP_NAME=image-quality-gate
APP_VERSION=0.1.0
```

### Threshold guidance

Blur scores depend on image content, resolution, camera quality, and preprocessing settings. If `RESIZE_MAX_DIM` changes, blur thresholds should be evaluated again.

The included tuning script can help estimate a useful blur threshold for a collection of sample images.

## Docker

### Build the image

```bash
docker build -t image-quality-gate:local .
```

### Run the container

```bash
docker run --rm \
  --env-file .env \
  -p 8080:8080 \
  image-quality-gate:local
```

The service will be available at:

```text
http://localhost:8080
```

### Run with Docker Compose

```bash
docker compose up --build
```

Docker Compose publishes the service on port `8080`.

To stop it:

```bash
docker compose down
```

## Testing

Install development dependencies if needed, then run:

```bash
pytest -q
```

The test suite includes:

- Unit tests for blur calculations
- Unit tests for brightness calculations
- Image preprocessing tests
- API integration tests
- Validation tests for uploaded files
- Response schema checks

## Development Commands

The repository includes a `Makerfile` with common commands.

```bash
make run
```

Starts the development server with reload enabled.

```bash
make test
```

Runs the test suite.

```bash
make lint
```

Runs Ruff and mypy.

```bash
make fmt
```

Formats the code with Black and applies Ruff fixes.

```bash
make build
```

Builds the local Docker image.

## Threshold Tuning

The `scripts/tune_blur.py` utility can be used to inspect a folder of images and help select quality thresholds.

A typical tuning workflow is:

1. Add representative sharp, blurry, dark, and overexposed images to `scripts/samples/`.
2. Run the tuning script.
3. Review the generated scores.
4. Select suitable `BLUR_MIN`, `BRIGHT_MIN`, and `BRIGHT_MAX` values.
5. Update `.env`.
6. Test the API against a representative sample of images.

Thresholds should be selected for the image sources used by your application rather than treated as universal values.

## Production Considerations

Before exposing this service publicly:

- Restrict CORS origins instead of allowing every origin.
- Place the service behind a reverse proxy or API gateway.
- Add authentication if the endpoint is not internal.
- Configure request timeouts and rate limits.
- Monitor `/health` and `/metrics`.
- Avoid logging image contents or sensitive metadata.
- Keep upload limits aligned with available memory.
- Use multiple Uvicorn workers when appropriate.
- Re-tune thresholds when image sources or preprocessing settings change.

## Use Cases

This service can be integrated into:

- User-upload validation workflows
- Crowdsourced reporting platforms
- Document or receipt ingestion pipelines
- Image moderation systems
- Computer vision preprocessing pipelines
- Content management systems
- Machine learning data preparation workflows

## Limitations

The quality decision is based only on blur and brightness. An image may pass the quality gate while still containing:

- The wrong subject
- Incorrect framing
- Compression artifacts
- Motion distortion not captured by the threshold
- Unwanted objects
- Poor composition
- Unreadable text

For more advanced validation, this service can be combined with OCR, object detection, face detection, or application-specific rules.

## License

No license is currently specified for this repository. Add a license file before distributing or reusing the project publicly.

---

Built with Python, FastAPI, OpenCV, Pillow, and NumPy.
