import os
from pathlib import Path

def setup_cache() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return cache_dir
