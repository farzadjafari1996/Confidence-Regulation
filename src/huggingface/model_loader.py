from pathlib import Path
from typing import List, Tuple, Optional

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_all_models(model_ids: List[str], cache_dir: Path) -> List[str]:
    """Download missing model snapshots into `cache_dir` and return available models.

    Models that already have snapshots under `cache/models--<repo>` are skipped.
    """
    downloaded_models: List[str] = []
    available_models: List[str] = []
    missing_models: List[str] = []

    for model_id in model_ids:
        model_cache_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
        snapshots_dir = model_cache_dir / "snapshots"
        if snapshots_dir.is_dir() and any(p.is_dir() for p in snapshots_dir.iterdir()):
            available_models.append(model_id)
        else:
            missing_models.append(model_id)

    if not missing_models:
        return available_models

    for model_id in missing_models:
        try:
            print(f"downloading the model : {model_id}")
            snapshot_download(repo_id=model_id, cache_dir=str(cache_dir))
            downloaded_models.append(model_id)
            available_models.append(model_id)
            print(f"model downloaded : {model_id}")
        except Exception as exc:
            print(f"[skip] Could not download {model_id}")
    return available_models


def load_model(model_id: str, cache_dir: Path) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load a tokenizer and causal LM for `model_id` from `cache_dir` (or download it).

    Returns a tuple `(tokenizer, model)`.
    """
    model_cache_dir = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = model_cache_dir / "snapshots"
    local_repo_path: Optional[str] = None

    if snapshots_dir.is_dir() and any(p.is_dir() for p in snapshots_dir.iterdir()):
        latest_snapshot = max(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            default=None,
        )
        if latest_snapshot is not None:
            local_repo_path = str(latest_snapshot)
            print(f"model loaded from cache : {model_id}")

    if local_repo_path is None:
        print(f"downloading the model : {model_id}")
        try:
            local_repo_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                resume_download=True,
            )
            print(f"model downloaded : {model_id}")
        except Exception as exc:
            print(f"[skip] Could not download {model_id}")
            raise

    tokenizer_kwargs = {
        "cache_dir": str(cache_dir),
        "trust_remote_code": True,
        "use_fast": True,
    }

    tokenizer = AutoTokenizer.from_pretrained(local_repo_path, **tokenizer_kwargs)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            local_repo_path,
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        msg = str(exc)
        if "Unrecognized configuration class" in msg or "for this kind of AutoModel" in msg:
            raise RuntimeError(
                f"Could not load {model_id}: transformers does not recognize the model config. "
                "Try upgrading `transformers` to a newer version that includes support for this model, "
                "or allow remote code with `trust_remote_code=True`. Original error: "
                f"{msg}"
            ) from exc
        raise

    print(f"model loaded : {model_id}")
    return tokenizer, model
