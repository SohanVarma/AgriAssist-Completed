from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

RESULTS_DIR = Path('results')
MODEL_PATH = RESULTS_DIR / 'initial_model.pt'
VAL_CSV = RESULTS_DIR / 'val_metadata.csv'


class CropDataset(Dataset):
    def __init__(self, csv_path, label_to_idx):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = label_to_idx
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        image = self.transform(image)
        label = self.label_to_idx[row['label']]
        return image, label


class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 16 * 16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model = SmallCNN(len(label_to_idx))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = CropDataset(VAL_CSV, label_to_idx)
    loader = DataLoader(dataset, batch_size=16)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(label_to_idx.keys())
    )

    cm = confusion_matrix(y_true, y_pred)

    (RESULTS_DIR / 'classification_report.txt').write_text(report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=list(label_to_idx.keys()),
        yticklabels=list(label_to_idx.keys())
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png')

    print(report)


if __name__ == '__main__':
    main()
