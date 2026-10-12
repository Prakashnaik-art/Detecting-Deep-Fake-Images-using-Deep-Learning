from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_JSON = DATASET_DIR / "train.json"
VAL_JSON = DATASET_DIR / "val.json"
TEST_JSON = DATASET_DIR / "test.json"

CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "training" / "results"


# Create output directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ORIGINAL DATASET
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\praka\Downloads\FaceForensics++_C23"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 2

CLASS_NAMES = {
    0: "REAL",
    1: "FAKE",
}


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 16

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

RANDOM_HORIZONTAL_FLIP = 0.5

RANDOM_ROTATION = 10


# ============================================================
# MODEL CHECKPOINTS
# ============================================================

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "best_model.pth"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR / "last_model.pth"
)


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]