"""
Llama 3.2 1B Instruct — Ext 1: QA ablation on LAMA T-REx + TriviaQA.

Usage:
  python src/llama/ext1_qa_ablation_instruct.py --device cuda
  python src/llama/ext1_qa_ablation_instruct.py --split sanity
  python src/llama/ext1_qa_ablation_instruct.py --check-functions

Requires Phase 1: outputs/meta-llama_Llama-3.2-1B-Instruct/phase1/
Outputs: outputs/meta-llama_Llama-3.2-1B-Instruct/ext1/, outputs/meta-llama_Llama-3.2-1B-Instruct/ext1_sanity/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from hyperparameters import ROOT  # noqa: E402
from qa.cli import add_ext1_cli, run_ext1_main  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "llama_instruct.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama 3.2 1B Instruct — Ext 1 QA ablation")
    add_ext1_cli(p, DEFAULT_CONFIG)
    p.set_defaults(device="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    run_ext1_main(parse_args())


if __name__ == "__main__":
    main()
