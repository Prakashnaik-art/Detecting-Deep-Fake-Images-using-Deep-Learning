
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from PIL import Image
from torchvision import transforms

from backend.model import load_model
from backend.config import UPLOAD_DIR


def main():
    image_path = input(
        "Enter the full path to a JPG or PNG image: "
    ).strip().strip('"')

    image_path = Path(image_path)

    if not image_path.is_file():
        print("ERROR: Image file not found.")
        return

    try:
        # Load image in RGB format
        image = Image.open(image_path).convert("RGB")

        # Preprocess image for EfficientNet-B0
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image_tensor = transform(image)

        # Load trained DeepGuard model
        model = load_model()

        # Get local model prediction
        local_prediction = model.predict(image_tensor)

        class_names = {
            0: "REAL",
            1: "FAKE"
        }

        # Save the heatmap in the uploads directory
        output_path = (
            Path(UPLOAD_DIR)
            / f"{image_path.stem}_heatmap.jpg"
        )

        saved_path = model.generate_heatmap(
            image_tensor=image_tensor,
            original_image_path=image_path,
            output_path=output_path,
            target_class=local_prediction
        )

        # Verify that the image was saved correctly
        heatmap = cv2.imread(str(saved_path))

        if heatmap is None:
            raise RuntimeError(
                "Saved heatmap could not be reopened."
            )

        print("\nHeatmap generated successfully.")
        print(
            "Local model prediction:",
            class_names.get(local_prediction, "UNKNOWN")
        )
        print("Saved path:", saved_path)
        print(
            "Image dimensions:",
            f"{heatmap.shape[1]} x {heatmap.shape[0]}"
        )

    except Exception as error:
        print(f"\nERROR: {error}")
        raise


if __name__ == "__main__":
    main()
