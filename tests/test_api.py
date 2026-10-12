import sys
from pathlib import Path

# Add project directories to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from config import API_KEY


def test_health():
    client = app.test_client()

    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["success"] is True
    assert data["status"] == "DeepGuard API is running."


def test_predict_without_api_key():
    client = app.test_client()

    response = client.post("/api/predict")

    assert response.status_code == 401

    data = response.get_json()

    assert data["success"] is False
    assert data["error"] == "API key is required."


def test_predict_with_wrong_api_key():
    client = app.test_client()

    response = client.post(
        "/api/predict",
        headers={
            "X-API-Key": "Wrong-Key"
        }
    )

    assert response.status_code == 403

    data = response.get_json()

    assert data["success"] is False
    assert data["error"] == "Invalid API key."


def test_predict_with_correct_api_key():
    client = app.test_client()

    response = client.post(
        "/api/predict",
        headers={
            "X-API-Key": API_KEY
        }
    )

    # Authentication should succeed.
    # The request then fails because no image was provided.
    assert response.status_code == 400

    data = response.get_json()

    assert data["success"] is False
    assert data["error"] == "No image file provided."


if __name__ == "__main__":
    print("=" * 60)
    print("DeepGuard API Test")
    print("=" * 60)

    test_health()
    print("✓ Health endpoint test passed")

    test_predict_without_api_key()
    print("✓ Missing API key test passed")

    test_predict_with_wrong_api_key()
    print("✓ Invalid API key test passed")

    test_predict_with_correct_api_key()
    print("✓ Valid API key test passed")

    print("=" * 60)
    print("All API tests passed successfully")
    print("=" * 60)