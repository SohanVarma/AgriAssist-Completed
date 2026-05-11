from pathlib import Path
import json
import math
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

SOURCE_DIR = Path('data/sample_crop_dataset/leaf_spot')
TARGET_DIR = Path('data/sample_crop_dataset_synthetic/leaf_spot')
RESULTS_DIR = Path('results')
MODEL_PATH = RESULTS_DIR / 'unet_ddpm_leaf_spot.pt'
SUMMARY_PATH = RESULTS_DIR / 'synthetic_generation_summary.json'
GRID_PATH = RESULTS_DIR / 'ddpm_generated_grid.png'

TARGET_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 64
TIMESTEPS = 300
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
NUM_SYNTHETIC_IMAGES = 16
BASE_CHANNELS = 64
TIME_EMBED_DIM = 256

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class LeafImageDataset(Dataset):
    def __init__(self, image_dir: Path):
        self.paths = sorted(image_dir.glob('*.png'))
        if not self.paths:
            raise FileNotFoundError(
                'No source images found. Run: python3 src/01_create_sample_dataset.py'
            )

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(image)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(b, c, h * w).transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return x + residual


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNetDDPM(nn.Module):
    def __init__(self, image_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.downsample1 = Downsample(base_channels)

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.downsample2 = Downsample(base_channels * 2)

        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.downsample3 = Downsample(base_channels * 4)

        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        self.mid_attn = AttentionBlock(base_channels * 8)
        self.mid2 = ResidualBlock(base_channels * 8, base_channels * 4, time_emb_dim)

        self.upsample3 = Upsample(base_channels * 4)
        self.up3 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)

        self.upsample2 = Upsample(base_channels * 2)
        self.up2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)

        self.upsample1 = Upsample(base_channels)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 3, padding=1),
        )

    def forward(self, x, time):
        t = self.time_mlp(time)

        x0 = self.init_conv(x)

        x1 = self.down1(x0, t)
        x = self.downsample1(x1)

        x2 = self.down2(x, t)
        x = self.downsample2(x2)

        x3 = self.down3(x, t)
        x = self.downsample3(x3)

        x = self.mid1(x, t)
        x = self.mid_attn(x)
        x = self.mid2(x, t)

        x = self.upsample3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x, t)

        x = self.upsample2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)

        x = self.upsample1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, t)

        return self.out(x)


class DiffusionScheduler:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=DEVICE)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=DEVICE), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noisy = sqrt_alpha * x0 + sqrt_one_minus * noise
        return noisy, noise

    @torch.no_grad()
    def sample_step(self, model, x, step):
        batch_size = x.shape[0]
        t = torch.full((batch_size,), step, device=DEVICE, dtype=torch.long)
        t_float = t.float()

        predicted_noise = model(x, t_float)

        beta = self.betas[step]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[step]
        sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas[step])

        model_mean = sqrt_recip_alpha * (
            x - beta * predicted_noise / sqrt_one_minus_alpha_cumprod
        )

        if step == 0:
            return model_mean

        posterior_var = self.posterior_variance[step]
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var) * noise


def train_ddpm():
    dataset = LeafImageDataset(SOURCE_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = UNetDDPM(base_channels=BASE_CHANNELS, time_emb_dim=TIME_EMBED_DIM).to(DEVICE)
    scheduler = DiffusionScheduler(TIMESTEPS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    losses = []

    print(f'Training UNet DDPM on device: {DEVICE}')
    print(f'Images: {len(dataset)}, image_size: {IMG_SIZE}, timesteps: {TIMESTEPS}, epochs: {EPOCHS}')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for x0 in loader:
            x0 = x0.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=DEVICE).long()
            noisy, target_noise = scheduler.add_noise(x0, t)

            predicted_noise = model(noisy, t.float())
            loss = criterion(predicted_noise, target_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        losses.append(avg_loss)
        print(f'Epoch {epoch}/{EPOCHS} - UNet DDPM loss: {avg_loss:.4f}')

    torch.save({
        'model_state_dict': model.state_dict(),
        'image_size': IMG_SIZE,
        'timesteps': TIMESTEPS,
        'base_channels': BASE_CHANNELS,
        'time_embed_dim': TIME_EMBED_DIM,
        'losses': losses,
    }, MODEL_PATH)

    return model, scheduler, losses


@torch.no_grad()
def sample_images(model, scheduler):
    model.eval()
    x = torch.randn(NUM_SYNTHETIC_IMAGES, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    for step in reversed(range(TIMESTEPS)):
        x = scheduler.sample_step(model, x, step)

    images = ((x.clamp(-1, 1) + 1) / 2).cpu()
    generated_paths = []

    for idx, image_tensor in enumerate(images):
        path = TARGET_DIR / f'unet_ddpm_leaf_spot_{idx:03d}.png'
        transforms.ToPILImage()(image_tensor).save(path)
        generated_paths.append(str(path))

    utils.save_image(images, GRID_PATH, nrow=4)
    return generated_paths


def main():
    model, scheduler, losses = train_ddpm()
    generated_paths = sample_images(model, scheduler)

    summary = {
        'component': 'UNet-based DDPM synthetic crop disease image generator',
        'target_class': 'leaf_spot',
        'source_dir': str(SOURCE_DIR),
        'output_dir': str(TARGET_DIR),
        'model_path': str(MODEL_PATH),
        'generated_grid': str(GRID_PATH),
        'device': DEVICE,
        'architecture': 'UNet with residual blocks, sinusoidal timestep embeddings, attention bottleneck, downsampling and upsampling paths',
        'timesteps': TIMESTEPS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'image_size': IMG_SIZE,
        'synthetic_images_created': len(generated_paths),
        'generated_images': generated_paths,
        'training_losses': losses,
        'note': 'This is a substantially heavier academic DDPM implementation. For higher quality, train longer on a larger real crop disease dataset using GPU.'
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
