import os
import requests
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# ---------- CONFIG ----------
PLANTVILLAGE_DIR = "data/plantvillage"
INATURALIST_DIR = "data/inaturalist_sample"
MAX_IMAGES = 500

# ---------- PLANTVILLAGE LOADER ----------
def load_plantvillage():
    print("Downloading PlantVillage dataset...")
    dataset = load_dataset("mohanty/PlantVillage", split="train")

    os.makedirs(PLANTVILLAGE_DIR, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset)):
        if i >= MAX_IMAGES:
            break

        img = sample["image"]
        label = str(sample["label"])

        class_dir = os.path.join(PLANTVILLAGE_DIR, label)
        os.makedirs(class_dir, exist_ok=True)

        img.save(os.path.join(class_dir, f"{i}.png"))

    print(f"Saved {MAX_IMAGES} PlantVillage images.")

# ---------- INATURALIST SAMPLE LOADER ----------
def load_inaturalist_sample():
    print("Fetching sample images from iNaturalist...")

    url = "https://api.inaturalist.org/v1/observations?photos=true&per_page=50"
    response = requests.get(url)
    data = response.json()

    os.makedirs(INATURALIST_DIR, exist_ok=True)

    count = 0
    for obs in data["results"]:
        if "photos" not in obs or len(obs["photos"]) == 0:
            continue

        img_url = obs["photos"][0]["url"].replace("square", "large")

        try:
            img_data = requests.get(img_url).content
            img = Image.open(BytesIO(img_data)).convert("RGB")

            img.save(os.path.join(INATURALIST_DIR, f"inat_{count}.jpg"))
            count += 1
        except:
            continue

    print(f"Saved {count} iNaturalist sample images.")

# ---------- MAIN ----------
if __name__ == "__main__":
    load_plantvillage()
    load_inaturalist_sample()
