import json
import shutil
from pathlib import Path

CONFIG_PATH = Path("data/dataset_config.json")
TARGET_DATASET_NAME = "New Plant Diseases Dataset"


def load_dataset_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    for dataset in config.get("dataset_sources", []):
        if dataset.get("name") == TARGET_DATASET_NAME:
            return dataset

    raise ValueError(f"Could not find '{TARGET_DATASET_NAME}' in dataset_config.json")


def main() -> None:
    config = load_dataset_config()

    local_dir = Path(config["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    if any(local_dir.iterdir()):
        print(f"Dataset already exists at {local_dir}")
        return

    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "Install kagglehub first: python3 -m pip install kagglehub"
        )

    print(f"Downloading {config['name']} from Kaggle...")
    downloaded_path = Path(kagglehub.dataset_download(config["dataset_slug"]))

    print(f"Downloaded to: {downloaded_path}")
    print(f"Copying dataset to: {local_dir}")

    for item in downloaded_path.iterdir():
        target = local_dir / item.name

        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    print("Dataset ready.")


if __name__ == "__main__":
    main()
