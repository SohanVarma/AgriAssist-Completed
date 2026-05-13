from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
MODEL_PATH = RESULTS_DIR / "initial_model.pt"
VAL_CSV = RESULTS_DIR / "val_metadata.csv"
OUTPUT_PATH = RESULTS_DIR / "gradcam_example.png"

IMAGE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImprovedCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 8 * 8, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_validation_image():
    if not VAL_CSV.exists():
        raise FileNotFoundError(
            "val_metadata.csv not found. Run: python3 src/02_data_pipeline.py"
        )

    df = pd.read_csv(VAL_CSV)

    if df.empty:
        raise ValueError("Validation metadata is empty.")

    row = df.iloc[0]
    image_path = Path(row["image_path"])

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    return image_path, image


def make_gradcam(model, image_tensor, class_index):
    gradients = []
    activations = []

    target_layer = model.features[12]  # final Conv2d layer

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()

    output = model(image_tensor)
    score = output[0, class_index]

    score.backward()

    forward_handle.remove()
    backward_handle.remove()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Train first using: python3 src/03_train_initial_model.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model = ImprovedCNN(len(label_to_idx)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_path, image = load_validation_image()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    logits = model(image_tensor)
    pred_idx = logits.argmax(dim=1).item()
    pred_label = idx_to_label[pred_idx]

    cam = make_gradcam(model, image_tensor, pred_idx)

    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_resized)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_resized)
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.title(f"Grad-CAM: {pred_label}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)

    print(f"Grad-CAM saved to: {OUTPUT_PATH}")
    print(f"Image used: {image_path}")
    print(f"Predicted class: {pred_label}")


if __name__ == "__main__":
    main()