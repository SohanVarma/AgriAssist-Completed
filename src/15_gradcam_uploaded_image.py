from pathlib import Path
from importlib import import_module

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image


predict_module = import_module("13_predict_uploaded_image")

load_model = predict_module.load_model
DEVICE = predict_module.DEVICE


def make_gradcam(model, image_tensor, class_index):
    gradients = []
    activations = []

    # Last Conv2d layer in your ImprovedCNN feature extractor:
    # features = [
    # 0 Conv2d, 1 BatchNorm, 2 ReLU, 3 MaxPool,
    # 4 Conv2d, 5 BatchNorm, 6 ReLU, 7 MaxPool,
    # 8 Conv2d, 9 BatchNorm, 10 ReLU, 11 MaxPool,
    # 12 Conv2d, 13 BatchNorm, 14 ReLU, 15 MaxPool
    # ]
    target_layer = model.features[12]

    def forward_hook(module, inputs, output):
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

    if not gradients or not activations:
        raise RuntimeError("Grad-CAM hooks did not capture gradients or activations.")

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min < 1e-8:
        cam = cam * 0
    else:
        cam = (cam - cam_min) / (cam_max - cam_min)

    return cam


def generate_gradcam_for_image(image_path, output_path):
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, idx_to_label, image_size = load_model()
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    original_image = Image.open(image_path).convert("RGB")
    image_resized = original_image.resize((image_size, image_size))
    image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)

    class_index = predicted_idx.item()
    predicted_label = idx_to_label[class_index]
    confidence_score = confidence.item() * 100

    cam = make_gradcam(model, image_tensor, class_index)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_resized)
    plt.title("Uploaded Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_resized)
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.title(f"Grad-CAM: {predicted_label}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close("all")

    return {
        "label": predicted_label,
        "confidence": confidence_score,
        "gradcam_path": str(output_path),
    }


if __name__ == "__main__":
    test_image_path = input("Enter image path: ").strip()

    if not test_image_path:
        raise ValueError("Image path cannot be empty.")

    result = generate_gradcam_for_image(
        image_path=test_image_path,
        output_path="web/static/outputs/gradcam_test.png",
    )

    print("Grad-CAM generated successfully")
    print(result)
