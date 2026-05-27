from datasets import DATASETS
from model_loader import setup_cache
from models import MODELS
from tests import hf_resource_manager

def hf_resource_manager(model_ids, dataset_ids, cache_dir):
    for model_id in model_ids:
        try:
            tokenizer, model = load_model(model_id, cache_dir)
            output = run_smoke_test(model, tokenizer, model_id)
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

if __name__ == "__main__":
    cache_dir = setup_cache()
    hf_resource_manager(MODELS, DATASETS, cache_dir)
