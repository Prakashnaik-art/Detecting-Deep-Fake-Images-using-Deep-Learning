import json
import random
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    DATASET_ROOT,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ============================================================
# FACE DETECTOR
# ============================================================

# Face detector files
FACE_DETECTOR_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
)

PROTOTXT_PATH = FACE_DETECTOR_DIR / "deploy.prototxt"

CAFFE_MODEL_PATH = (
    FACE_DETECTOR_DIR
    / "res10_300x300_ssd_iter_140000.caffemodel"
)


# ------------------------------------------------------------
# Load OpenCV face detector
# ------------------------------------------------------------

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


FACE_NET = cv2.dnn.readNetFromCaffe(
    str(PROTOTXT_PATH),
    str(CAFFE_MODEL_PATH),
)


FACE_CONFIDENCE = 0.60

MIN_FACE_SIZE = 50


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


val_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(json_path):
    """
    Load dataset JSON file.
    """

    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"Dataset JSON not found:\n{json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"{json_path} must contain a JSON list."
        )

    return data


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(frame):
    """
    Detect all faces in a frame.

    Returns:
        list of tuples:
        (x1, y1, x2, y2, confidence)
    """

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )

    FACE_NET.setInput(blob)

    detections = FACE_NET.forward()

    faces = []

    for i in range(
        detections.shape[2]
    ):

        confidence = float(
            detections[0, 0, i, 2]
        )

        if confidence < FACE_CONFIDENCE:
            continue

        box = detections[
            0, 0, i, 3:7
        ] * torch.tensor(
            [width, height, width, height]
        ).numpy()

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        face_width = x2 - x1
        face_height = y2 - y1

        if (
            face_width < MIN_FACE_SIZE
            or face_height < MIN_FACE_SIZE
        ):
            continue

        faces.append(
            (
                x1,
                y1,
                x2,
                y2,
                confidence,
            )
        )

    return faces


# ============================================================
# FACE CROP
# ============================================================

def crop_faces(frame):
    """
    Detect faces and return cropped face images.
    """

    faces = detect_faces(frame)

    crops = []

    for (
        x1,
        y1,
        x2,
        y2,
        confidence,
    ) in faces:

        face = frame[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            continue

        # OpenCV BGR -> RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(face)

        crops.append(image)

    return crops


# ============================================================
# VIDEO FRAME SAMPLING
# ============================================================

def get_frame_indices(
    total_frames,
    num_frames,
):
    """
    Select evenly distributed frame indices.
    """

    if total_frames <= 0:
        return []

    if total_frames <= num_frames:

        return list(
            range(total_frames)
        )

    step = total_frames / num_frames

    indices = [
        int(i * step)
        for i in range(num_frames)
    ]

    return indices


# ============================================================
# VIDEO DATASET
# ============================================================

class DeepGuardVideoDataset(Dataset):

    def __init__(
        self,
        json_path,
        transform=None,
        frames_per_video=3,
    ):

        self.samples = load_json(
            json_path
        )

        self.transform = transform

        self.frames_per_video = (
            frames_per_video
        )

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in {json_path}"
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        relative_path = sample["path"]

        label = int(
            sample["label"]
        )

        video_path = (
            DATASET_ROOT
            / relative_path
        )

        # ----------------------------------------------------
        # Check video
        # ----------------------------------------------------

        if not video_path.exists():

            raise FileNotFoundError(
                f"Video not found:\n"
                f"{video_path}"
            )

        # ----------------------------------------------------
        # Open video
        # ----------------------------------------------------

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():

            raise RuntimeError(
                f"Could not open video:\n"
                f"{video_path}"
            )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:

            capture.release()

            raise RuntimeError(
                f"Video contains no frames:\n"
                f"{video_path}"
            )

        # ----------------------------------------------------
        # Select frames
        # ----------------------------------------------------

        indices = get_frame_indices(
            total_frames,
            self.frames_per_video,
        )

        face_images = []

        # ----------------------------------------------------
        # Extract frames
        # ----------------------------------------------------

        for frame_index in indices:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            success, frame = (
                capture.read()
            )

            if not success:
                continue

            crops = crop_faces(
                frame
            )

            face_images.extend(
                crops
            )

        capture.release()

        # ----------------------------------------------------
        # If no face was found
        # ----------------------------------------------------

        if len(face_images) == 0:

            # Try a random frame as fallback
            capture = cv2.VideoCapture(
                str(video_path)
            )

            random_index = random.randint(
                0,
                max(0, total_frames - 1)
            )

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                random_index,
            )

            success, frame = (
                capture.read()
            )

            capture.release()

            if success:

                height, width = (
                    frame.shape[:2]
                )

                # Center crop fallback
                side = min(
                    height,
                    width
                )

                x1 = (
                    width - side
                ) // 2

                y1 = (
                    height - side
                ) // 2

                face = frame[
                    y1:y1 + side,
                    x1:x1 + side
                ]

                face = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB,
                )

                face_images.append(
                    Image.fromarray(face)
                )

        # ----------------------------------------------------
        # Still no image
        # ----------------------------------------------------

        if len(face_images) == 0:

            raise RuntimeError(
                "Could not extract a usable "
                f"face/frame from:\n"
                f"{video_path}"
            )

        # ----------------------------------------------------
        # Select one face
        # ----------------------------------------------------

        image = random.choice(
            face_images
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label