import sys
from pathlib import Path
import json
import io

import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DEEPGUARD IMPORTS
# ============================================================

from app import app
from config import API_KEY
from training.config import TEST_JSON, DATASET_ROOT


# ============================================================
# GET TEST VIDEO
# ============================================================

def get_test_video():
    """
    Get the first video listed in test.json.
    """

    with open(
        TEST_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        test_data = json.load(file)

    if not test_data:
        raise RuntimeError(
            "test.json is empty."
        )

    first_item = test_data[0]

    video_path = (
        first_item.get("video_path")
        or first_item.get("path")
        or first_item.get("file_path")
        or first_item.get("image_path")
    )

    if video_path is None:
        raise RuntimeError(
            "Could not find a video path in the "
            "first test.json entry."
        )

    video_path = Path(video_path)

    # Dataset paths in JSON are relative to the
    # FaceForensics++ dataset directory.
    if not video_path.is_absolute():
        video_path = DATASET_ROOT / video_path

    if not video_path.exists():
        raise FileNotFoundError(
            f"Test video does not exist:\n"
            f"{video_path}"
        )

    return video_path


# ============================================================
# EXTRACT FRAME
# ============================================================

def extract_frame(video_path):
    """
    Extract a readable frame from the test video.
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open test video:\n"
            f"{video_path}"
        )

    frame = None

    # Try several frame positions.
    frame_positions = [
        0,
        10,
        30,
        60,
        100
    ]

    for position in frame_positions:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            position
        )

        success, current_frame = (
            capture.read()
        )

        if (
            success
            and current_frame is not None
            and current_frame.size > 0
        ):
            frame = current_frame
            break

    capture.release()

    if frame is None:
        raise RuntimeError(
            f"Could not extract a frame from:\n"
            f"{video_path}"
        )

    return frame


# ============================================================
# TEST PREDICTION API
# ============================================================

def test_prediction_api():

    client = app.test_client()

    # --------------------------------------------------------
    # Get test video
    # --------------------------------------------------------

    video_path = get_test_video()

    print(
        f"Using test video: {video_path}"
    )

    # --------------------------------------------------------
    # Extract frame
    # --------------------------------------------------------

    frame = extract_frame(
        video_path
    )

    height, width = frame.shape[:2]

    print(
        f"Extracted frame resolution: "
        f"{width} x {height}"
    )

    # --------------------------------------------------------
    # Encode frame as JPEG
    # --------------------------------------------------------

    success, encoded_image = cv2.imencode(
        ".jpg",
        frame
    )

    if not success:
        raise RuntimeError(
            "Could not encode video frame as JPEG."
        )

    image_bytes = encoded_image.tobytes()

    print(
        f"JPEG size: {len(image_bytes):,} bytes"
    )

    # --------------------------------------------------------
    # Send image to REST API
    # --------------------------------------------------------

    response = client.post(
        "/api/predict",

        headers={
            "X-API-Key": API_KEY
        },

        data={
            "file": (
                io.BytesIO(image_bytes),
                "test_frame.jpg"
            )
        },

        content_type="multipart/form-data"
    )

    # --------------------------------------------------------
    # Display response
    # --------------------------------------------------------

    print(
        f"HTTP status: {response.status_code}"
    )

    data = response.get_json()

    print(
        f"API response: {data}"
    )

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    assert response.status_code == 200, (
        f"Expected HTTP 200 but received "
        f"{response.status_code}: {data}"
    )

    assert data is not None, (
        "API returned an empty response."
    )

    assert data["success"] is True, (
        f"API prediction failed: {data}"
    )

    assert data["prediction"] in [
        "REAL",
        "FAKE"
    ], (
        f"Invalid prediction: "
        f"{data.get('prediction')}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DeepGuard Prediction API Test")
    print("=" * 60)

    test_prediction_api()

    print("=" * 60)
    print("Prediction API test passed successfully")
    print("=" * 60)