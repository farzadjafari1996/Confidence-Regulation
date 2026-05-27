import os
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download


def setup_cache() -> Path:
    """Configure Hugging Face to store dataset files inside the project cache directory."""

    project_root = Path(__file__).resolve().parent
    cache_dir = project_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return cache_dir


def download_dataset(dataset_id: str, cache_dir: Path) -> str:
    """Download a Hugging Face dataset snapshot into the project cache."""

    print(f"downloading the dataset : {dataset_id}")
    local_dataset_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        resume_download=True,
    )
    print(f"dataset downloaded : {dataset_id}")
    return local_dataset_path


def download_all_datasets(dataset_ids: List[str], cache_dir: Path) -> List[str]:
    """Download all requested datasets and return the ones that completed successfully."""

    downloaded_datasets: List[str] = []

    for dataset_id in dataset_ids:
        try:
            download_dataset(dataset_id, cache_dir)
            downloaded_datasets.append(dataset_id)
        except Exception:
            print(f"[skip] Could not download dataset {dataset_id}")

    return downloaded_datasets