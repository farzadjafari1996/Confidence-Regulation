# Neuron-Entropy
Research implementation used for the "Trends in AI and LLMs" project.

## Overview
This repository evaluates causal language models on knowledge-recall datasets (e.g. TriviaQA, LAMA) by generating short completions and checking if gold answers appear in model outputs.

## Requirements
- Python 3.10+ recommended
- A CUDA-compatible GPU for fast evaluation (optional but recommended)
- Install dependencies from `requirements.txt` (the file is configured to use the PyTorch CUDA index present in this repo)

## Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd Neuron-Entropy
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv-cuda
# On Windows (PowerShell)
.venv-cuda\Scripts\Activate.ps1
# On macOS/Linux
source .venv-cuda/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Notes:
- The `requirements.txt` uses the PyTorch CUDA wheel index (`cu128`) shown at the top of the file. If you do not have CUDA or prefer CPU-only PyTorch, adjust the `--index-url` and `torch` entry accordingly.

## Usage
Run the main evaluation script:
```bash
python main.py
```

By default `main.py` reads `MODELS` from `constants.py` and evaluates each model on the configured datasets. Results are printed to stdout.

## Project structure
- `main.py`: Entry point. Loads tokenizers/models and runs evaluations.
- `accuracy.py`: Evaluation utilities and dataset prompt builders for TriviaQA and LAMA.
- `constants.py`: Lists `MODELS` and `DATASETS` used by the scripts.
- `cache.py`: Sets up a local HuggingFace cache directory and environment variables used by `datasets` and `transformers`.
- `requirements.txt`: Pinned Python dependencies and PyTorch CUDA wheel index.

## Example
Run a single model run (prints accuracy):
```bash
python main.py
```

## Notes & Recommendations
- The code uses `transformers.AutoModelForCausalLM.from_pretrained(..., device_map="auto")` which works best with a recent `transformers` and a GPU.
- `cache.setup_cache()` configures `HF_HOME` to a `./cache` directory inside the project so models and dataset artifacts are stored locally.
- If you plan to evaluate large models or many examples, consider running on a machine with sufficient GPU RAM or using model offloading tools.

If you'd like, I can also:
- add a small CLI flag to `main.py` to control `max_examples` for quick tests
- provide a discrete CPU-only requirements variant
