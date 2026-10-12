import cv2

from config import (
    PROTOTXT_PATH,
    CAFFE_MODEL_PATH,
    FACE_CONFIDENCE,
    MIN_FACE_SIZE,
)


class FaceDetector:

    def __init__(self):

        self.net = cv2.dnn.readNetFromCaffe(
            str(PROTOTXT_PATH),
            str(CAFFE_MODEL_PATH),
        )

    def detect_faces(self, image):

        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(
                image,
                (300, 300),
            ),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )

        self.net.setInput(blob)

        detections = self.net.forward()

        faces = []

        for i in range(
            detections.shape[2]
        ):

            confidence = float(
                detections[0, 0, i, 2]
            )

            if confidence < FACE_CONFIDENCE:
                continue

            box = (
                detections[
                    0, 0, i, 3:7
                ]
                * [width, height, width, height]
            )

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
                {
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "confidence": confidence,
                }
            )

        return faces


def load_face_detector():

    return FaceDetector()


if __name__ == "__main__":

    print("=" * 60)
    print("DeepGuard Face Detector Test")
    print("=" * 60)

    detector = load_face_detector()

    print(
        "Face detector loaded successfully"
    )

    print(
        f"Confidence threshold: "
        f"{FACE_CONFIDENCE}"
    )

    print(
        f"Minimum face size: "
        f"{MIN_FACE_SIZE}"
    )

    print("=" * 60)