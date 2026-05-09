from pathlib import Path
import json
import math
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

SOURCE_DIR = Path('data/sample_crop_dataset/leaf_spot')
TARGET_DIR = Path('data/sample_crop_dataset_synthetic/leaf_spot')
RESULTS_DIR = Path('results')
MODEL_PATH = RESULTS_DIR / 'mini_ddpm_leaf_spot.pt'
SUMMARY_PATH = RESULTS_DIR / 'synthetic_generation_summary.json'

TARGET_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 32
TIMESTEPS = 50
EPOCHS = 3
BATCH_SIZE = 8
NUM_SYNTHETIC_IMAGES = 8


class LeafImageDataset(Dataset):
    def __init__(self, image_dir: Path):
        self.paths = sorted(image_dir.glob('*.png'))
        if not self.paths:
            raise FileNotFoundError(
                'No source images found. Run: python3 src/01_create_sample_dataset.py'
            )

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(image)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn.Conv2d(3 + 1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, x, t):
        batch = x.shape[0]
        t_img = t.view(batch, 1, 1, 1).expand(batch, 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, t_img], dim=1))


def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)


def add_noise(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x0)
    sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    noisy = sqrt_alpha * x0 + sqrt_one_minus * noise
    return noisy, noise


def save_tensor_image(tensor, path):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    image = transforms.ToPILImage()(tensor.cpu())
    image.save(path)


def train_ddpm():
    dataset = LeafImageDataset(SOURCE_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TinyDenoiser()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    betas = linear_beta_schedule(TIMESTEPS)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    losses = []

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for x0 in loader:
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), dtype=torch.long)
            noisy, target_noise = add_noise(
                x0,
                t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
            )

            t_normalized = t.float() / TIMESTEPS
            predicted_noise = model(noisy, t_normalized)

            loss = criterion(predicted_noise, target_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        losses.append(avg_loss)
        print(f'Epoch {epoch}/{EPOCHS} - DDPM noise prediction loss: {avg_loss:.4f}')

    torch.save(model.state_dict(), MODEL_PATH)
    return model, betas, alphas, alphas_cumprod, losses


@torch.no_grad()
def sample_images(model, betas, alphas, alphas_cumprod):
    model.eval()

    generated = []

    for img_idx in range(NUM_SYNTHETIC_IMAGES):
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

        for step in reversed(range(TIMESTEPS)):
            t = torch.tensor([step], dtype=torch.float32) / TIMESTEPS
            predicted_noise = model(x, t)

            beta = betas[step]
            alpha = alphas[step]
            alpha_cumprod = alphas_cumprod[step]

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            )

            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

        output_path = TARGET_DIR / f'ddpm_synthetic_leaf_spot_{img_idx:03d}.png'
        save_tensor_image(x.squeeze(0), output_path)
        generated.append(str(output_path))

    return generated


def main():
    model, betas, alphas, alphas_cumprod, losses = train_ddpm()
    generated_paths = sample_images(model, betas, alphas, alphas_cumprod)

    summary = {
        'component': 'working lightweight DDPM synthetic image generator',
        'target_class': 'leaf_spot',
        'source_dir': str(SOURCE_DIR),
        'output_dir': str(TARGET_DIR),
        'model_path': str(MODEL_PATH),
        'timesteps': TIMESTEPS,
        'epochs': EPOCHS,
        'image_size': IMG_SIZE,
        'synthetic_images_created': len(generated_paths),
        'generated_images': generated_paths,
        'training_losses': losses,
        'note': 'This is a compact DDPM implementation for runnable academic demonstration, not a production-scale diffusion model.'
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
