from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
MODEL_PATH = RESULTS_DIR / "initial_model.pt"
VAL_CSV = RESULTS_DIR / "val_metadata.csv"
OUTPUT_PATH = RESULTS_DIR / "gradcam_example.png"


class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 16 * 16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_first_validation_image():
    import pandas as pd

    df = pd.read_csv(VAL_CSV)
    image_path = Path(df.iloc[0]["image_path"])
    return image_path, Image.open(image_path).convert("RGB")


def make_gradcam(model, image_tensor, class_index):
    gradients = []
    activations = []

    target_layer = model.features[3]

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
        raise FileNotFoundError("Train the model first using: python3 src/03_train_initial_model.py")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model = SmallCNN(len(label_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    image_path, image = load_first_validation_image()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    x = transform(image).unsqueeze(0)
    logits = model(x)
    pred_idx = logits.argmax(dim=1).item()
    pred_label = idx_to_label[pred_idx]

    cam = make_gradcam(model, x, pred_idx)

    image_resized = image.resize((64, 64))

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_resized)
    plt.imshow(cam, alpha=0.45, cmap="jet")
    plt.title(f"Grad-CAM: {pred_label}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"Grad-CAM saved to {OUTPUT_PATH}")
    print(f"Image used: {image_path}")
    print(f"Predicted class: {pred_label}")


if __name__ == "__main__":
    main()
