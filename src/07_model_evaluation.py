from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATHS = [
    RESULTS_DIR / "initial_model.pt",
    MODELS_DIR / "cnn_crop_disease_model.pt",
]

CLASSIFICATION_REPORT_TXT = RESULTS_DIR / "classification_report.txt"
CLASSIFICATION_REPORT_JSON = RESULTS_DIR / "classification_report.json"
CONFUSION_MATRIX_PNG = RESULTS_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_CSV = RESULTS_DIR / "confusion_matrix.csv"

IMAGE_SIZE = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ImageFolderFromCSV(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        possible_path_cols = ["image_path", "filepath", "file_path", "path", "image"]
        possible_label_cols = ["label", "class", "target", "disease"]

        self.path_col = None
        self.label_col = None

        for col in possible_path_cols:
            if col in self.df.columns:
                self.path_col = col
                break

        for col in possible_label_cols:
            if col in self.df.columns:
                self.label_col = col
                break

        if self.path_col is None or self.label_col is None:
            raise ValueError(
                f"CSV must contain image path and label columns. Found columns: {list(self.df.columns)}"
            )

        self.labels = sorted(self.df[self.label_col].astype(str).unique().tolist())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(str(row[self.path_col]))

        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path

        label = str(row[self.label_col])
        label_idx = self.label_to_idx[label]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_idx


def find_model_path():
    for path in MODEL_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No trained model found. Expected one of:\n"
        + "\n".join(str(path) for path in MODEL_PATHS)
        + "\nRun training first: python3 src/03_train_initial_model.py"
    )


def find_test_csv():
    possible = [
        DATA_DIR / "test.csv",
        DATA_DIR / "processed" / "test.csv",
        DATA_DIR / "processed_dataset" / "test.csv",
        DATA_DIR / "test_metadata.csv",
        RESULTS_DIR / "test.csv",
    ]

    for path in possible:
        if path.exists():
            return path

    csv_files = list(DATA_DIR.rglob("*test*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        "Could not find a test CSV file. Expected something like:\n"
        "data/test.csv or data/processed/test.csv\n"
        "Run your data pipeline first: python3 src/02_data_pipeline.py"
    )


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "label_to_idx" in checkpoint:
        label_to_idx = checkpoint["label_to_idx"]
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        num_classes = len(label_to_idx)
    else:
        raise ValueError("Checkpoint must contain label_to_idx.")

    model = ImprovedCNN(num_classes=num_classes).to(DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError("Checkpoint must contain model_state_dict or state_dict.")

    model.eval()

    image_size = checkpoint.get("image_size", IMAGE_SIZE)

    return model, idx_to_label, image_size


def evaluate_model(model, dataloader):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return y_true, y_pred


def save_confusion_matrix(cm, class_names):
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(CONFUSION_MATRIX_CSV)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.colorbar()

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=220)
    plt.close()


def main():
    model_path = find_model_path()
    test_csv = find_test_csv()

    print(f"Using model: {model_path}")
    print(f"Using test CSV: {test_csv}")
    print(f"Device: {DEVICE}")

    model, idx_to_label, image_size = load_model(model_path)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    test_dataset = ImageFolderFromCSV(test_csv, transform=transform)

    class_names = [
        idx_to_label[i] for i in range(len(idx_to_label))
    ]

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    y_true, y_pred = evaluate_model(model, test_loader)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    print(report_text)

    CLASSIFICATION_REPORT_TXT.write_text(report_text, encoding="utf-8")
    CLASSIFICATION_REPORT_JSON.write_text(
        json.dumps(report_dict, indent=2),
        encoding="utf-8",
    )

    save_confusion_matrix(cm, class_names)

    print("\nSaved output files:")
    print(f"- {CLASSIFICATION_REPORT_TXT}")
    print(f"- {CLASSIFICATION_REPORT_JSON}")
    print(f"- {CONFUSION_MATRIX_PNG}")
    print(f"- {CONFUSION_MATRIX_CSV}")


if __name__ == "__main__":
    main()
