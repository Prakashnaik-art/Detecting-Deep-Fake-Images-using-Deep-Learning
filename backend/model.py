
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from backend.config import MODEL_PATH, NUM_CLASSES
from training.config import DEVICE
from training.model import create_model


# ============================================================
# DEEPGUARD MODEL
# ============================================================

class DeepGuardModel:

    def __init__(self):
        self.device = DEVICE

        # Create EfficientNet-B0 architecture
        self.model = create_model(num_classes=NUM_CLASSES)

        # Load trained checkpoint
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.checkpoint_epoch = checkpoint.get("epoch", None)

        # Final convolutional layer used for Grad-CAM
        self.target_layer = self.model.model.features[-1]

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            prediction = outputs.argmax(dim=1).item()

        return prediction

    # ========================================================
    # GRAD-CAM HEATMAP
    # ========================================================

    def generate_heatmap(
        self,
        image_tensor,
        original_image_path,
        output_path,
        target_class=None,
        overlay_weight=0.40
    ):
        """
        Generate a Grad-CAM overlay from the local EfficientNet-B0.

        Classes:
            0 = REAL
            1 = FAKE

        This visualization explains the local model, not Sightengine.
        """

        image_tensor = image_tensor.to(self.device)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        activations = []
        gradients = []

        def save_activations(_module, _inputs, output):
            activations.append(output)

        def save_gradients(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        activation_handle = self.target_layer.register_forward_hook(
            save_activations
        )

        gradient_handle = self.target_layer.register_full_backward_hook(
            save_gradients
        )

        try:
            self.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                outputs = self.model(image_tensor)

                if target_class is None:
                    target_class = int(outputs.argmax(dim=1).item())

                if target_class not in (0, 1):
                    raise ValueError(
                        "target_class must be 0 (REAL) or 1 (FAKE)."
                    )

                outputs[:, target_class].sum().backward()

            feature_maps = activations[0].detach()
            feature_gradients = gradients[0].detach()

            # Average gradients across spatial dimensions
            weights = feature_gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            # Weighted combination of feature maps
            cam = (weights * feature_maps).sum(
                dim=1,
                keepdim=True
            )

            cam = F.relu(cam)

            # Resize to model input resolution
            cam = F.interpolate(
                cam,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )

            cam = cam[0, 0].cpu().numpy()

            # Normalize activation values to [0, 1]
            cam -= cam.min()
            maximum = cam.max()

            if maximum > 1e-8:
                cam /= maximum
            else:
                cam.fill(0)

        finally:
            activation_handle.remove()
            gradient_handle.remove()
            self.model.zero_grad(set_to_none=True)

        # Read original image
        original = cv2.imread(str(original_image_path))

        if original is None:
            raise ValueError(
                "Could not read the original image for the heatmap."
            )

        height, width = original.shape[:2]

        # Match heatmap dimensions to original image
        cam = cv2.resize(cam, (width, height))

        # Convert activation map to a blue-to-red color map
        heatmap = cv2.applyColorMap(
            (np.clip(cam, 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Blend original image and heatmap
        overlay = cv2.addWeighted(
            original,
            1.0 - overlay_weight,
            heatmap,
            overlay_weight,
            0
        )

        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), overlay):
            raise RuntimeError("Could not save the generated heatmap.")

        return output_path


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    return DeepGuardModel()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DeepGuard Backend Model Test")
    print("=" * 60)

    model = load_model()

    print("Model loaded successfully")
    print(f"Device: {model.device}")
    print(f"Checkpoint epoch: {model.checkpoint_epoch}")
    print(f"Grad-CAM target layer: {model.target_layer}")

    print("=" * 60)
