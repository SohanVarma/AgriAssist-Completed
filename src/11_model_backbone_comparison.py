from pathlib import Path
import json
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DATASET_ROOT = Path('data/sample_crop_dataset')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 2


class CropDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.classes = sorted([p.name for p in root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            for img_path in (root_dir / cls).glob('*.png'):
                self.samples.append((img_path, self.class_to_idx[cls]))

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), label


def train_model(model, train_loader, val_loader, device):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / max(total, 1)
    return accuracy


def build_resnet(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vit(num_classes):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def main():
    dataset = CropDataset(DATASET_ROOT)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}

    print('Training ResNet18...')
    resnet = build_resnet(len(dataset.classes))
    resnet_acc = train_model(resnet, train_loader, val_loader, device)
    results['ResNet18'] = {
        'validation_accuracy': round(resnet_acc, 4)
    }

    print('Training Vision Transformer...')
    vit = build_vit(len(dataset.classes))
    vit_acc = train_model(vit, train_loader, val_loader, device)
    results['VisionTransformer_B16'] = {
        'validation_accuracy': round(vit_acc, 4)
    }

    summary = {
        'component': 'working backbone comparison experiment',
        'dataset_root': str(DATASET_ROOT),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'results': results,
    }

    output_path = RESULTS_DIR / 'backbone_comparison_results.json'
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps(summary, indent=2))
    print(f'Saved results to: {output_path}')


if __name__ == '__main__':
    main()
