from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

REAL_DATA_ROOT = Path("data/PlantVillage")

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_metadata_from_class_folders(
    dataset_dir: Path,
    source: str,
    max_images_per_class: int | None = 300
) -> pd.DataFrame:

    rows = []

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}"
        )

    class_dirs = [
        p for p in dataset_dir.iterdir()
        if p.is_dir()
    ]

    if not class_dirs:
        raise ValueError(
            f"No class folders found inside {dataset_dir}"
        )

    for class_dir in sorted(class_dirs):
        label = class_dir.name

        images = [
            p for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        images = sorted(images)

        if max_images_per_class is not None:
            images = images[:max_images_per_class]

        for img_path in images:
            rows.append({
                "image_path": str(img_path),
                "label": label,
                "source": source
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No images found inside {dataset_dir}"
        )

    return df


def load_dataset() -> tuple[pd.DataFrame, str, str]:

    df = build_metadata_from_class_folders(
        REAL_DATA_ROOT,
        source="PlantVillage external crop disease dataset",
        max_images_per_class=300
    )

    return (
        df,
        "PlantVillage external crop disease dataset",
        str(REAL_DATA_ROOT)
    )


def main() -> None:
    df, dataset_name, dataset_location = load_dataset()

    df["exists"] = df["image_path"].apply(
        lambda p: Path(p).exists()
    )

    df = df[df["exists"]].drop(columns=["exists"])

    if df.empty:
        raise ValueError(
            "Metadata loaded, but none of the image paths exist."
        )

    min_class_count = df["label"].value_counts().min()

    stratify_col = df["label"] if min_class_count >= 2 else None

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify_col
    )

    train_df.to_csv(
        RESULTS_DIR / "train_metadata.csv",
        index=False
    )

    val_df.to_csv(
        RESULTS_DIR / "val_metadata.csv",
        index=False
    )

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
        "status": "Data pipeline completed successfully"
    }

    (RESULTS_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()