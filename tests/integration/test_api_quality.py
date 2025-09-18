from fastapi.testclient import TestClient
from app.main import app
from pathlib import Path

client = TestClient(app)


def test_quality_endpoint_accepts_image():
    p = Path(__file__).parent.parent / "assets" / "sharp_small.jpg"
    with p.open("rb") as f:
        resp = client.post(
            "/quality", files={"file": ("sharp_small.jpg", f, "image/jpeg")}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert {"blur_score", "brightness", "width", "height", "is_ok"}.issubset(
        body.keys()
    )
