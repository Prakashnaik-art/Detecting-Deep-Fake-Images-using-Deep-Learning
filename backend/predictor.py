import sys
from pathlib import Path

import cv2

from PIL import Image
from torchvision import transforms


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# BACKEND IMPORTS
# ============================================================

from face_detector import load_face_detector
from model import load_model


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

from training.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CLASS_NAMES,
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose(
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
# DEEPGUARD PREDICTOR
# ============================================================

class DeepGuardPredictor:

    def __init__(self):

        print(
            "Loading DeepGuard predictor..."
        )

        self.detector = load_face_detector()

        self.model = load_model()

        print(
            "DeepGuard predictor loaded successfully"
        )

    # ========================================================
    # PREPARE FACE
    # ========================================================

    def prepare_face(
        self,
        image,
        box,
    ):

        x1, y1, x2, y2 = box

        face = image[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            return None

        # BGR → RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Convert to PIL
        face = Image.fromarray(
            face
        )

        # Apply model preprocessing
        face = image_transform(
            face
        )

        return face

    # ========================================================
    # PREDICT IMAGE
    # ========================================================

    def predict(self, image):

        if image is None:
            raise ValueError(
                "Input image is empty."
            )

        # Detect all faces
        faces = self.detector.detect_faces(
            image
        )

        if len(faces) == 0:
            raise ValueError(
                "No face detected in the image."
            )

        predictions = []

        # Predict every detected face
        for face_info in faces:

            face_tensor = self.prepare_face(
                image,
                face_info["box"]
            )

            if face_tensor is None:
                continue

            prediction = self.model.predict(
                face_tensor
            )

            predictions.append(
                {
                    "prediction": prediction,
                    "label": CLASS_NAMES[
                        prediction
                    ],
                }
            )

        if len(predictions) == 0:
            raise ValueError(
                "Could not process any "
                "detected face."
            )

        # ====================================================
        # OVERALL RESULT
        # ====================================================
        #
        # If at least one face is FAKE,
        # the complete image is classified
        # as FAKE.
        #
        # Otherwise it is classified as REAL.
        # ====================================================

        fake_faces = sum(
            1
            for result in predictions
            if result["prediction"] == 1
        )

        if fake_faces > 0:
            overall_prediction = 1
        else:
            overall_prediction = 0

        overall_label = CLASS_NAMES[
            overall_prediction
        ]

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {
            "prediction": overall_prediction,
            "label": overall_label,
            "face_count": len(predictions),
            "face_predictions": predictions,
        }


# ============================================================
# LOAD PREDICTOR
# ============================================================

def load_predictor():

    return DeepGuardPredictor()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DeepGuard Predictor Test")
    print("=" * 60)

    predictor = load_predictor()

    print(
        "Predictor loaded successfully"
    )

    print("=" * 60)