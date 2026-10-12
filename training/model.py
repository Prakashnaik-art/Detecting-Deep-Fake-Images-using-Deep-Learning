import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class DeepGuardModel(nn.Module):
    """
    DeepGuard EfficientNet-B0 binary classifier.

    Classes:
        0 = REAL
        1 = FAKE
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # Build the architecture without downloading ImageNet weights.
        # DeepGuard's trained checkpoint is loaded separately.
        self.model = efficientnet_b0(weights=None)

        # Replace the original ImageNet classifier.
        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)


def create_model(num_classes=2):
    """Create the DeepGuard model architecture."""
    return DeepGuardModel(num_classes=num_classes)


if __name__ == "__main__":
    from training.config import NUM_CLASSES, DEVICE

    model = create_model(NUM_CLASSES).to(DEVICE)

    print("DeepGuard model created successfully")
    print(f"Device: {DEVICE}")
    print(f"Number of classes: {NUM_CLASSES}")

    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
        device=DEVICE,
    )

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
