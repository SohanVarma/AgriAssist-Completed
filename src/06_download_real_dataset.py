import json
import shutil
from pathlib import Path

CONFIG_PATH = Path("data/dataset_config.json")


def load_active_dataset() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    active_dataset = config.get("active_image_dataset")

    for dataset in config.get("dataset_sources", []):
        if dataset.get("name") == active_dataset:
            return dataset

    raise ValueError(
        f"Active dataset '{active_dataset}' not found in dataset_config.json"
    )


def download_kaggle_dataset(config: dict, local_dir: Path) -> None:
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "Install kagglehub first: python3 -m pip install kagglehub"
        )

    print(f"Downloading dataset from Kaggle link: {config['url']}")

    downloaded_path = Path(
        kagglehub.dataset_download(config["dataset_slug"])
    )

    print(f"Downloaded to temporary location: {downloaded_path}")
    print(f"Copying dataset to: {local_dir}")

    for item in downloaded_path.iterdir():
        target = local_dir / item.name

        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main() -> None:
    dataset_config = load_active_dataset()

    local_dir = Path(dataset_config["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    if any(local_dir.iterdir()):
        print(f"Dataset already exists at {local_dir}")
        return

    source_type = dataset_config.get("source", "").lower()

    if source_type == "kaggle":
        download_kaggle_dataset(dataset_config, local_dir)
    else:
        raise NotImplementedError(
            f"Automatic download not implemented yet for source type: {source_type}"
        )

    print(f"Dataset ready at: {local_dir}")


if __name__ == "__main__":
    main()
