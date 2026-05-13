from pathlib import Path
import json
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = Path("data/PlantVillage")
OUTPUT_DIR = Path("results/ddpm_outputs")
MODEL_DIR = Path("results/models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASS_KEYWORD = "leaf_spot"

IMG_SIZE = 64
BATCH_SIZE = 4
TIMESTEPS = 300
EPOCHS = 50
LEARNING_RATE = 2e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# DATASET
# =========================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_target_class_folder(root: Path, keyword: str):
    keyword = keyword.lower()

    matching = [
        p for p in root.rglob("*")
        if p.is_dir() and keyword in p.name.lower().replace(" ", "_")
    ]

    if not matching:
        raise FileNotFoundError(
            f"No class folder containing '{keyword}' found."
        )

    return matching[0]


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = []

        for p in Path(root_dir).rglob("*"):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                self.paths.append(p)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img


# =========================================================
# POSITIONAL EMBEDDING
# =========================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device

        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)

        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )

        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()),
            dim=-1
        )

        return embeddings


# =========================================================
# SIMPLE UNET
# =========================================================

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.relu(self.bnorm1(self.conv1(x)))

        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]

        h = h + time_emb

        h = self.relu(self.bnorm2(self.conv2(h)))

        return h


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        time_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(3, 64, 3, padding=1)

        self.down1 = Block(64, 128, time_dim)
        self.down2 = Block(128, 256, time_dim)

        self.up1 = Block(256, 128, time_dim)
        self.up2 = Block(128, 64, time_dim)

        self.output = nn.Conv2d(64, 3, 1)

        self.pool = nn.MaxPool2d(2)

        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        x = self.conv0(x)

        d1 = self.down1(x, t)
        p1 = self.pool(d1)

        d2 = self.down2(p1, t)

        u1 = self.upscale(d2)
        u1 = self.up1(u1, t)

        u2 = self.upscale(u1)
        u2 = self.up2(u2, t)

        return self.output(u2)


# =========================================================
# DIFFUSION SCHEDULER
# =========================================================

betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)

alphas = 1. - betas

alphas_cumprod = torch.cumprod(alphas, axis=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)


def get_index(vals, t, x_shape):
    batch_size = t.shape[0]

    out = vals.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(DEVICE)


def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = get_index(
        sqrt_alphas_cumprod,
        t,
        x_0.shape
    )

    sqrt_one_minus_alphas_cumprod_t = get_index(
        sqrt_one_minus_alphas_cumprod,
        t,
        x_0.shape
    )

    noisy_image = (
        sqrt_alphas_cumprod_t * x_0
        + sqrt_one_minus_alphas_cumprod_t * noise
    )

    return noisy_image, noise


# =========================================================
# TRAINING
# =========================================================

def train():
    target_dir = find_target_class_folder(
        DATA_ROOT,
        TARGET_CLASS_KEYWORD
    )

    dataset = PlantDiseaseDataset(target_dir)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = SimpleUNet().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_fn = nn.MSELoss()

    print(f"Training on device: {DEVICE}")
    print(f"Dataset size: {len(dataset)}")

    for epoch in range(EPOCHS):
        loop = tqdm(loader)

        epoch_loss = 0

        for batch in loop:
            optimizer.zero_grad()

            batch = batch.to(DEVICE)

            t = torch.randint(
                0,
                TIMESTEPS,
                (batch.shape[0],),
                device=DEVICE
            ).long()

            x_noisy, noise = forward_diffusion_sample(
                batch,
                t
            )

            noise_pred = model(x_noisy, t)

            loss = loss_fn(noise_pred, noise)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(loss=loss.item())

        print(
            f"Epoch {epoch+1} Average Loss: "
            f"{epoch_loss / len(loader):.4f}"
        )

    model_path = MODEL_DIR / "unet_ddpm_leaf_spot.pt"

    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")

    generate_images(model)

    summary = {
        "model": "UNet DDPM",
        "epochs": EPOCHS,
        "timesteps": TIMESTEPS,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "dataset": str(target_dir),
        "device": DEVICE
    }

    with open(
        OUTPUT_DIR / "ddpm_training_summary.json",
        "w"
    ) as f:
        json.dump(summary, f, indent=2)


# =========================================================
# SAMPLING
# =========================================================

@torch.no_grad()
def sample_timestep(x, t, model):
    betas_t = get_index(betas, t, x.shape)

    sqrt_one_minus_alphas_cumprod_t = get_index(
        sqrt_one_minus_alphas_cumprod,
        t,
        x.shape
    )

    sqrt_recip_alphas_t = torch.sqrt(
        1.0 / get_index(alphas, t, x.shape)
    )

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t)
        / sqrt_one_minus_alphas_cumprod_t
    )

    posterior_variance_t = betas_t

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(
            posterior_variance_t
        ) * noise


@torch.no_grad()
def generate_images(model):
    model.eval()

    img = torch.randn(
        (16, 3, IMG_SIZE, IMG_SIZE),
        device=DEVICE
    )

    for i in tqdm(
        reversed(range(TIMESTEPS)),
        total=TIMESTEPS
    ):
        t = torch.full(
            (16,),
            i,
            device=DEVICE,
            dtype=torch.long
        )

        img = sample_timestep(img, t, model)

    img = (img.clamp(-1, 1) + 1) / 2

    output_path = OUTPUT_DIR / "generated_leaf_spot_samples.png"

    save_image(img, output_path, nrow=4)

    print(f"Generated images saved to: {output_path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    train()