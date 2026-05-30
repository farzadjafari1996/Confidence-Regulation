from huggingface_hub import snapshot_download
from pathlib import Path

def download_resources(model_ids, dataset_ids, cache_dir):
    for model_id in model_ids:
        if not _is_model_downloaded(model_id, cache_dir):
            print(f"model {model_id} was downloaded over net")
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
            )
        else:
            print(f"model {model_id} was cached")
        

    for dataset_id in dataset_ids:
        if not _is_dataset_downloaded(dataset_id, cache_dir):
            print(f"dataset {dataset_id} was downloaded over net")
            snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
        else:
            print(f"dataset {dataset_id} was cached")


def _is_model_downloaded(model_id, cache_dir):
    model_cache_dir = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = model_cache_dir / "snapshots"
    return snapshots_dir.is_dir() and any(p.is_dir() for p in snapshots_dir.iterdir())


def _is_dataset_downloaded(dataset_id, cache_dir):
    dataset_cache_dir = Path(cache_dir) / f"datasets--{dataset_id.replace('/', '--')}"
    snapshots_dir = dataset_cache_dir / "snapshots"
    return snapshots_dir.is_dir() and any(p.is_dir() for p in snapshots_dir.iterdir())