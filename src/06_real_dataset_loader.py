from pathlib import Path
from datasets import load_dataset
from PIL import Image

SAVE_DIR = Path("data/plantvillage_subset")
MAX_IMAGES = 500

def get_image(sample):
    for key in ["image", "img", "jpg"]:
        if key in sample:
            return sample[key]
    raise KeyError(f"No image column found. Available columns: {list(sample.keys())}")

def get_label(sample):
    for key in ["label", "labels", "disease"]:
        if key in sample:
            return sample[key]
    return "unknown"

def main():
    print("Loading PlantVillage from HuggingFace...")
    dataset = load_dataset("mohanty/PlantVillage", split="train", trust_remote_code=True)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    subset = dataset.select(range(min(MAX_IMAGES, len(dataset))))

    for i, sample in enumerate(subset):
        img = get_image(sample)
        label = get_label(sample)

        class_dir = SAVE_DIR / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(img, Image.Image):
            img = Image.open(img)

        img.convert("RGB").save(class_dir / f"{i}.png")

    print(f"Saved {len(subset)} PlantVillage images to {SAVE_DIR}")

if __name__ == "__main__":
    main()
