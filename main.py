from src.huggingface.model_loader import load_model
from src.huggingface.dataset_loader import load_dataset_entries
from src.huggingface.datasets import DATASETS
from src.huggingface.models import MODELS
from src.utils.cache import setup_cache
from src.utils.helper_functions import download_resources
from src.tests.accuracy import _evaluate_model_on_dataset

if __name__ == "__main__":
    cache_dir = setup_cache()
    #download_resources(MODELS, DATASETS, cache_dir)

    for model_id in MODELS:
        tokenizer, model = load_model(model_id, cache_dir)
        dataset_entries = load_dataset_entries("facebook/lama", cache_dir)
        counter, dataset_length, accuracy = _evaluate_model_on_dataset(tokenizer, model, dataset_entries)

        print(f"Model: {model_id}")
        print(f"Dataset: facebook/lama")
        print(f"Counter: {counter}")
        print(f"Dataset length: {dataset_length}")
        print(f"Accuracy: {accuracy:.4f}")
        print("---")
