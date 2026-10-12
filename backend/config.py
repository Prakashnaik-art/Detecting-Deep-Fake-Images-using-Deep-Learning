import os
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "training"
    / "checkpoints"
    / "best_model.pth"
)

FACE_DETECTOR_DIR = (
    PROJECT_ROOT
    / "models"
)

PROTOTXT_PATH = (
    FACE_DETECTOR_DIR
    / "deploy.prototxt"
)

CAFFE_MODEL_PATH = (
    FACE_DETECTOR_DIR
    / "res10_300x300_ssd_iter_140000.caffemodel"
)

UPLOAD_DIR = (
    PROJECT_ROOT
    / "uploads"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
)


# --------------------------------------------------
# Create required directories
# --------------------------------------------------

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Environment variables
# --------------------------------------------------

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

API_KEY = os.getenv(
    "DEEPGUARD_API_KEY"
)


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

IMAGE_SIZE = 224

NUM_CLASSES = 2

CLASS_NAMES = {
    0: "REAL",
    1: "FAKE",
}


# --------------------------------------------------
# Face detector configuration
# --------------------------------------------------

FACE_CONFIDENCE = 0.60

MIN_FACE_SIZE = 50


# --------------------------------------------------
# Flask configuration
# --------------------------------------------------

MAX_CONTENT_LENGTH = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
}


# --------------------------------------------------
# Validation
# --------------------------------------------------

if API_KEY is None:
    raise RuntimeError(
        "DEEPGUARD_API_KEY is not configured in .env"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"DeepGuard model not found:\n{MODEL_PATH}"
    )

if not PROTOTXT_PATH.exists():
    raise FileNotFoundError(
        f"Face detector prototxt not found:\n"
        f"{PROTOTXT_PATH}"
    )

if not CAFFE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face detector model not found:\n"
        f"{CAFFE_MODEL_PATH}"
    )