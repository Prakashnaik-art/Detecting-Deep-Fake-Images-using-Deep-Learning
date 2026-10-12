import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from config import (
    TEST_JSON,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    BEST_MODEL_PATH,
    NUM_CLASSES,
)

from dataset import (
    DeepGuardVideoDataset,
    val_transform,
)

from model import create_model


def main():

    print("=" * 60)
    print("DeepGuard Model Evaluation")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Test dataset: {TEST_JSON}")
    print(f"Model: {BEST_MODEL_PATH}")
    print()

    # --------------------------------------------------
    # Check model
    # --------------------------------------------------

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model not found:\n{BEST_MODEL_PATH}"
        )

    # --------------------------------------------------
    # Load test dataset
    # --------------------------------------------------

    print("Loading test dataset...")

    test_dataset = DeepGuardVideoDataset(
        TEST_JSON,
        transform=val_transform,
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )
    print()

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    print("Loading model...")

    model = create_model(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print()

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    all_labels = []
    all_predictions = []

    print("Running evaluation...")

    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="Testing",
        )

        for images, labels in progress:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print()

    print("Confusion Matrix:")
    print(
        "                 Predicted"
    )
    print(
        "              REAL    FAKE"
    )

    print(
        f"Actual REAL   "
        f"{matrix[0, 0]:5d}   "
        f"{matrix[0, 1]:5d}"
    )

    print(
        f"Actual FAKE   "
        f"{matrix[1, 0]:5d}   "
        f"{matrix[1, 1]:5d}"
    )

    print()

    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "REAL",
                "FAKE",
            ],
            zero_division=0,
        )
    )

    print("=" * 60)
    print("Evaluation completed")
    print("=" * 60)


if __name__ == "__main__":
    main()