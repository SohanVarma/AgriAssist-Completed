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
SAMPLE_DATA_ROOT = Path("data/sample_crop_dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_training_folder(root: Path) -> Path | None:
    """Return the largest nested train folder if a real Kaggle dataset exists."""
    if not root.exists():
        return None

    train_dirs = [p for p in root.rglob("train") if p.is_dir()]
    if not train_dirs:
        return None

    return sorted(train_dirs, key=lambda p: len(list(p.rglob("*"))), reverse=True)[0]


def build_metadata_from_class_folders(dataset_dir: Path, source: str, max_images_per_class: int | None = None) -> pd.DataFrame:
    rows = []
    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]

    if not class_dirs:
        raise ValueError(f"No class folders found inside {dataset_dir}")

    for class_dir in sorted(class_dirs):
        label = class_dir.name
        images = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        images = sorted(images)

        if max_images_per_class is not None:
            images = images[:max_images_per_class]

        for img_path in images:
            rows.append({
                "image_path": str(img_path),
                "label": label,
                "source": source,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found inside {dataset_dir}")
    return df


def load_dataset() -> tuple[pd.DataFrame, str, str]:
    """
    Prefer the real Kaggle dataset when available.
    Fall back to the generated sample dataset so the project remains runnable for evaluation.
    """
    real_train_dir = find_training_folder(REAL_DATA_ROOT)

    if real_train_dir is not None:
        print(f"Using Kaggle train folder: {real_train_dir}")
        df = build_metadata_from_class_folders(
            real_train_dir,
            source="kaggle_new_plant_diseases_dataset",
            max_images_per_class=80,
        )
        return df, "Kaggle New Plant Diseases Dataset", str(real_train_dir)

    metadata_csv = SAMPLE_DATA_ROOT / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            "No dataset found. Run either:\n"
            "  python3 src/01_create_sample_dataset.py\n"
            "or\n"
            "  python3 src/06_download_real_dataset.py"
        )

    print("Real dataset not found. Using generated sample dataset for demo run.")
    df = pd.read_csv(metadata_csv)
    return df, "Generated sample crop disease dataset", str(SAMPLE_DATA_ROOT)


def main() -> None:
    df, dataset_name, dataset_location = load_dataset()

    df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    df = df[df["exists"]].drop(columns=["exists"])

    if df.empty:
        raise ValueError("Metadata loaded, but none of the image paths exist.")

    min_class_count = df["label"].value_counts().min()
    stratify_col = df["label"] if min_class_count >= 2 else None

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify_col,
    )

    train_df.to_csv(RESULTS_DIR / "train_metadata.csv", index=False)
    val_df.to_csv(RESULTS_DIR / "val_metadata.csv", index=False)

    summary = {
        "seed": SEED,
        "dataset": dataset_name,
        "dataset_location": dataset_location,
        "total_images_loaded": int(len(df)),
        "train_images": int(len(train_df)),
        "validation_images": int(len(val_df)),
        "number_of_classes": int(df["label"].nunique()),
        "classes": sorted(df["label"].unique().tolist()),
        "class_distribution": df["label"].value_counts().to_dict(),
        "status": "Data pipeline completed successfully",
    }

    (RESULTS_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
