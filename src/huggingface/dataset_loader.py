from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download

def download_dataset(dataset_id: str, cache_dir: Path) -> str:
    dataset_cache_dir = Path(cache_dir) / f"datasets--{dataset_id.replace('/', '--')}"
    snapshots_dir = dataset_cache_dir / "snapshots"

    if snapshots_dir.is_dir():
        snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if snapshot_dirs:
            latest_snapshot = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)
            print(f"dataset loaded from cache : {dataset_id}")
            return str(latest_snapshot)

    print(f"downloading the dataset : {dataset_id}")
    local_dataset_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    print(f"dataset downloaded : {dataset_id}")
    return local_dataset_path

def download_all_datasets(dataset_ids: List[str], cache_dir: Path) -> List[str]:
    downloaded_datasets: List[str] = []

    for dataset_id in dataset_ids:
        try:
            download_dataset(dataset_id, cache_dir)
            downloaded_datasets.append(dataset_id)
        except Exception:
            print(f"[skip] Could not download dataset {dataset_id}")

    return downloaded_datasets