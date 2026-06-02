from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

TRAIN_CSV = RESULTS_DIR / "train_metadata.csv"
VAL_CSV = RESULTS_DIR / "val_metadata.csv"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CropDataset(Dataset):
    def __init__(self, csv_path, label_to_idx):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = label_to_idx

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = self.transform(
            Image.open(row["image_path"]).convert("RGB")
        )

        y = self.label_to_idx[row["label"]]

        return x, torch.tensor(y, dtype=torch.long)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x).argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.numel()

    return correct / max(total, 1)


def main():
    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        raise FileNotFoundError(
            "train_metadata.csv or val_metadata.csv not found. "
            "Run: python3 src/02_data_pipeline.py first."
        )

    labels = sorted(
        pd.read_csv(TRAIN_CSV)["label"].unique().tolist()
    )

    label_to_idx = {
        label: idx for idx, label in enumerate(labels)
    }

    train_loader = DataLoader(
        CropDataset(TRAIN_CSV, label_to_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        CropDataset(VAL_CSV, label_to_idx),
        batch_size=BATCH_SIZE
    )

    model = ImprovedCNN(len(labels)).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    rows = []

    print(f"Training on device: {DEVICE}")
    print(f"Classes: {labels}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            loss = criterion(model(x), y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader)

        rows.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch}/{EPOCHS}: "
            f"loss={avg_loss:.4f}, "
            f"val_accuracy={val_accuracy:.3f}"
        )

    metrics = pd.DataFrame(rows)

    metrics.to_csv(
        RESULTS_DIR / "training_metrics.csv",
        index=False
    )

    plt.figure()
    plt.plot(
        metrics["epoch"],
        metrics["train_loss"],
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("CNN Training Loss")
    plt.savefig(
        RESULTS_DIR / "training_loss.png",
        bbox_inches="tight"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_to_idx": label_to_idx,
        "image_size": IMAGE_SIZE,
        "model_name": "ImprovedCNN"
    }

    torch.save(
        checkpoint,
        RESULTS_DIR / "initial_model.pt"
    )

    torch.save(
        checkpoint,
        MODELS_DIR / "cnn_crop_disease_model.pt"
    )

    print("Model saved to:")
    print(RESULTS_DIR / "initial_model.pt")
    print(MODELS_DIR / "cnn_crop_disease_model.pt")


if __name__ == "__main__":
    main()
