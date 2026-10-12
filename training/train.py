import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    TRAIN_JSON,
    VAL_JSON,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    DEVICE,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    NUM_CLASSES,
)

from dataset import (
    DeepGuardVideoDataset,
    train_transform,
    val_transform,
)

from model import create_model


def load_samples(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_class_weights(samples):
    """
    Calculate inverse-frequency class weights.

    Class 0 = REAL
    Class 1 = FAKE
    """
    counts = [0] * NUM_CLASSES

    for sample in samples:
        label = int(sample["label"])
        counts[label] += 1

    total = sum(counts)

    weights = [
        total / (NUM_CLASSES * count)
        if count > 0
        else 0.0
        for count in counts
    ]

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    with torch.no_grad():
        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_accuracy,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    torch.save(checkpoint, path)


def main():
    print("=" * 60)
    print("DeepGuard EfficientNet-B0 Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    print("Loading datasets...")

    train_dataset = DeepGuardVideoDataset(
        TRAIN_JSON,
        transform=train_transform,
    )

    val_dataset = DeepGuardVideoDataset(
        VAL_JSON,
        transform=val_transform,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # --------------------------------------------------
    # Class weights
    # --------------------------------------------------

    train_samples = load_samples(TRAIN_JSON)

    class_weights = calculate_class_weights(
        train_samples
    )

    print("Class weights:")
    print(
        f"REAL: {class_weights[0].item():.4f}"
    )
    print(
        f"FAKE: {class_weights[1].item():.4f}"
    )
    print()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("Creating model...")

    model = create_model(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    best_val_loss = float("inf")

    print("Starting training...")
    print()

    for epoch in range(NUM_EPOCHS):

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # Save last checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_loss,
            val_accuracy,
            LAST_MODEL_PATH,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_accuracy,
                BEST_MODEL_PATH,
            )

            print(
                "✓ Best model saved"
            )

        print()

    print("=" * 60)
    print("Training completed")
    print("=" * 60)

    print(
        f"Best model: {BEST_MODEL_PATH}"
    )

    print(
        f"Last model: {LAST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()