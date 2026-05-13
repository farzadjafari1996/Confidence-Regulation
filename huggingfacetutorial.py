import os
from pathlib import Path
from huggingface_hub import model_info

# ============ SETUP: HuggingFace Cache Directory ============
# Configure HuggingFace to store models locally in ./cache folder
project_root = Path(__file__).parent
cache_dir = project_root / "cache"
cache_dir.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(cache_dir)
print(f"✓ HuggingFace cache directory set to: {cache_dir}")
print(f"  Models and datasets will be stored locally in this project\n")
# ============================================================


# Your code starts here
if __name__ == "__main__":
    for model_id in [
        "meta-llama/Llama-3.1-8B",
        "mistralai/Mistral-7B-v0.3",
        "Qwen/Qwen2.5-7B",
    ]:
        try:
            info = model_info(model_id)
            print(f"✓ {model_id}  —  {info.modelId}")
        except Exception as e:
            print(f"✗ {model_id}  —  {e}")
