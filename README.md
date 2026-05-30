# Neuron-Entropy
Research Implementation of Neuron Entropy for Trends in AI and LLMs

## Setup Instructions

### Prerequisites
- Python 3.12 recommended
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Neuron-Entropy
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv-cuda
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
       .venv-cuda\Scripts\Activate.ps1
     ```
   - On macOS/Linux:
     ```bash
       source .venv-cuda/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` is configured for CUDA (`cu128`) PyTorch.

5. **Configure HuggingFace cache (optional)**
   To store HuggingFace models locally in this project instead of your home directory:
   ```bash
   .\setup_env.ps1
   ```
   This sets `HF_HOME` to `./cache` folder and creates it if needed.

## Requirements
All project dependencies are listed in `requirements.txt` and include:
- PyTorch (deep learning framework)
- Transformers (HuggingFace models)
- Accelerate (distributed training)
- Datasets (dataset loading)
- And other data science/ML libraries

## Usage
Run the main script:
```bash
python main.py
```
