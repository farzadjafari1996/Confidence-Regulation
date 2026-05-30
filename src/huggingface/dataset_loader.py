import json
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

LAMA_DATA_URL = "https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz"
LAMA_CONFIG_FILES = {
    "trex": ["TREx"],
    "squad": ["Squad/test.jsonl"],
    "google_re": [
        "Google_RE/date_of_birth_test.jsonl",
        "Google_RE/place_of_birth_test.jsonl",
        "Google_RE/place_of_death_test.jsonl",
    ],
    "conceptnet": ["ConceptNet/test.jsonl"],
}
DATASET_CONFIGS = {
    "mandarjoshi/trivia_qa": "rc",
}

def download_dataset(dataset_id: str, cache_dir: Path) -> str:
    if dataset_id == "facebook/lama":
        lama_root = Path(cache_dir) / "facebook-lama"
        archive_path = lama_root / "negated_data.tar.gz"
        extract_dir = lama_root / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        if not any(extract_dir.iterdir()):
            lama_root.mkdir(parents=True, exist_ok=True)
            if not archive_path.exists():
                urllib.request.urlretrieve(LAMA_DATA_URL, archive_path)

            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(extract_dir)

        print(f"dataset loaded from cache : {dataset_id}")
        return str(extract_dir)

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

def load_dataset_entries(dataset_id: str, cache_dir: Path, n: Optional[int] = None):
    if dataset_id == "facebook/lama":
        return _load_lama_entries(cache_dir, n)

    dataset_config = DATASET_CONFIGS.get(dataset_id)
    if dataset_config is None:
        dataset = load_dataset(dataset_id, split="train", cache_dir=str(cache_dir))
    else:
        dataset = load_dataset(
            dataset_id,
            dataset_config,
            split="train",
            cache_dir=str(cache_dir),
        )
    limit = len(dataset) if n is None else min(n, len(dataset))
    return [dataset[index] for index in range(limit)]


def _load_lama_entries(cache_dir: Path, n: Optional[int], config_name: str = "squad"):
    lama_root = Path(cache_dir) / "facebook-lama"
    archive_path = lama_root / "negated_data.tar.gz"
    extract_dir = lama_root / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not any(extract_dir.iterdir()):
        lama_root.mkdir(parents=True, exist_ok=True)
        if not archive_path.exists():
            urllib.request.urlretrieve(LAMA_DATA_URL, archive_path)

        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extract_dir)

    config_files = LAMA_CONFIG_FILES.get(config_name)
    if config_files is None:
        raise ValueError(f"Unknown facebook/lama config: {config_name}")

    entries = []
    for relative_path in config_files:
        file_path = extract_dir / relative_path
        if relative_path == "TREx":
            jsonl_files = sorted(file_path.rglob("*.jsonl"))
        else:
            jsonl_files = [file_path]

        for jsonl_file in jsonl_files:
            with jsonl_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    entries.append(json.loads(line))
                    if n is not None and len(entries) >= n:
                        return entries

    return entries