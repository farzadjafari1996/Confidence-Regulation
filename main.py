from src.huggingface.datasets import DATASETS
from src.huggingface.models import MODELS
from src.utils.cache import setup_cache
from src.utils.helper_functions import download_resources


if __name__ == "__main__":
    cache_dir = setup_cache()
    download_resources(MODELS, DATASETS, cache_dir)
