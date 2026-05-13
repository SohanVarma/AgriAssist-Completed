from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

RESULTS_DIR = Path("results")
MODEL_PATH = RESULTS_DIR / "initial_model.pt"

IMAGE_SIZE = 128
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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Train first using: python3 src/03_train_initial_model.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model = ImprovedCNN(len(label_to_idx)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, idx_to_label


def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model, idx_to_label = load_model()

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_label = idx_to_label[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    return predicted_label, confidence_score


def main():
    image_path = input("\nEnter image path: ").strip()

    if not image_path:
        raise ValueError("Image path cannot be empty.")

    prediction, confidence = predict_image(image_path)

    print("\nPrediction Result")
    print("=================")
    print(f"Predicted Disease: {prediction}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()