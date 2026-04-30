from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

REAL_DATA_ROOT = Path("data/real_crop_dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_training_folder(root: Path) -> Path:
    """
    Finds the Kaggle dataset training folder.
    Expected dataset after download:
    data/real_crop_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train
    or a similar nested train directory.
    """
    if not root.exists():
        raise FileNotFoundError(
            "Real dataset not found. Run: python3 src/06_download_real_dataset.py"
        )

    train_dirs = [p for p in root.rglob("train") if p.is_dir()]
    if not train_dirs:
        raise FileNotFoundError(
            "Could not find a 'train' folder inside data/real_crop_dataset."
        )

    # Choose the train folder with the most class folders/images
    train_dirs = sorted(train_dirs, key=lambda p: len(list(p.rglob("*"))), reverse=True)
    return train_dirs[0]


def build_metadata(train_dir: Path, max_images_per_class: int = 80) -> pd.DataFrame:
    rows = []

    class_dirs = [p for p in train_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found inside {train_dir}")

    for class_dir in class_dirs:
        label = class_dir.name
        images = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        # Limit per class so training is fast for demo
        images = sorted(images)[:max_images_per_class]

        for img_path in images:
            rows.append({
                "image_path": str(img_path),
                "label": label,
                "source": "kaggle_new_plant_diseases_dataset"
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No images found in the Kaggle training folder.")

    return df


def main():
    train_dir = find_training_folder(REAL_DATA_ROOT)
    print(f"Using Kaggle train folder: {train_dir}")

    df = build_metadata(train_dir)

    df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    df = df[df["exists"]].drop(columns=["exists"])

    min_class_count = df["label"].value_counts().min()
    stratify_col = df["label"] if min_class_count >= 2 else None

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify_col
    )

    train_df.to_csv(RESULTS_DIR / "train_metadata.csv", index=False)
    val_df.to_csv(RESULTS_DIR / "val_metadata.csv", index=False)

    summary = {
        "seed": SEED,
        "dataset": "Kaggle New Plant Diseases Dataset",
        "dataset_root": str(REAL_DATA_ROOT),
        "train_folder_used": str(train_dir),
        "total_images_loaded": int(len(df)),
        "train_images": int(len(train_df)),
        "validation_images": int(len(val_df)),
        "number_of_classes": int(df["label"].nunique()),
        "classes": sorted(df["label"].unique().tolist()),
        "class_distribution": df["label"].value_counts().to_dict(),
        "status": "Data pipeline working and Kaggle dataset loaded"
    }

    (RESULTS_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
