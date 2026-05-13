from pathlib import Path
import json
import random

from PIL import Image, ImageEnhance, ImageOps

REAL_DATA_ROOT = Path("data/PlantVillage")
TARGET_CLASS_KEYWORD = "leaf_spot"

TARGET_DIR = Path("data/sample_crop_dataset_synthetic/leaf_spot")
RESULTS_DIR = Path("results")

TARGET_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_target_class_folder(dataset_root: Path, class_keyword: str) -> Path:
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}. Make sure PlantVillage is inside data/PlantVillage"
        )

    keyword = class_keyword.lower()

    matching_dirs = [
        p for p in dataset_root.rglob("*")
        if p.is_dir() and keyword in p.name.lower().replace(" ", "_")
    ]

    if not matching_dirs:
        raise FileNotFoundError(
            f"No class folder matching '{class_keyword}' found inside {dataset_root}"
        )

    return sorted(
        matching_dirs,
        key=lambda p: len(list(p.rglob("*"))),
        reverse=True
    )[0]


def augment_image(img: Image.Image) -> Image.Image:
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    if random.random() > 0.5:
        img = ImageOps.flip(img)

    img = ImageEnhance.Color(img).enhance(random.uniform(1.1, 1.5))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(1.1, 1.5))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.2))

    return img


def main():
    source_dir = find_target_class_folder(
        REAL_DATA_ROOT,
        TARGET_CLASS_KEYWORD
    )

    image_paths = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_paths = image_paths[:100]

    generated = 0
    generated_files = []

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        aug = augment_image(img)

        output_path = TARGET_DIR / f"synthetic_{generated:04d}_{img_path.name}"
        aug.save(output_path)

        generated += 1
        generated_files.append(str(output_path))

    summary = {
        "component": "synthetic crop disease data generation from PlantVillage dataset",
        "dataset_source": str(REAL_DATA_ROOT),
        "source_class_folder": str(source_dir),
        "target_class_keyword": TARGET_CLASS_KEYWORD,
        "synthetic_images_created": generated,
        "output_dir": str(TARGET_DIR),
        "generated_files_sample": generated_files[:10],
        "note": "This version uses the external PlantVillage dataset as the image source."
    }

    (RESULTS_DIR / "synthetic_generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()