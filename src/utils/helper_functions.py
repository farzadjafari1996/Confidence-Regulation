import torch
from pathlib import Path

from src.huggingface.dataset_loader import download_dataset
from src.huggingface.model_loader import load_model

def download_resources(model_ids, dataset_ids, cache_dir):
    for model_id in model_ids:
        try:
            tokenizer, model = load_model(model_id, cache_dir)
            output = run_model_smoke_test(model, tokenizer, model_id)
            print(f"[ok] model working properly: {model_id} -> {output}")
        except Exception:
            print(f"[skip] model not working properly: {model_id}")

    for dataset_id in dataset_ids:
        try:
            dataset_path = download_dataset(dataset_id, cache_dir)
            run_dataset_smoke_test(dataset_path, dataset_id)
            print(f"[ok] dataset working properly: {dataset_id}")
        except Exception:
            print(f"[skip] dataset not working properly: {dataset_id}")


def run_model_smoke_test(model, tokenizer, name):
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        out[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded


def run_dataset_smoke_test(dataset_path, name):
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset cache path does not exist: {dataset_path}")

    has_content = any(dataset_dir.rglob("*"))
    if not has_content:
        raise RuntimeError(f"Dataset cache is empty: {dataset_path}")

    return True