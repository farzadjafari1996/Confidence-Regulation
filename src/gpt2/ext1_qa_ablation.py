"""
GPT-2 — Ext 1: QA ablation on LAMA T-REx + TriviaQA.

Usage:
  .venv\\Scripts\\Activate.ps1   # Windows — use venv for CUDA
  python src/gpt2/ext1_qa_ablation.py --device cuda
  python src/gpt2/ext1_qa_ablation.py --split sanity
  python src/gpt2/ext1_qa_ablation.py --check-functions
  python src/gpt2/ext1_qa_ablation.py --plots-only --dataset lama --split sanity

Requires Phase 1: outputs/gpt2-small/phase1/
Outputs: outputs/gpt2-small/ext1/, outputs/gpt2-small/ext1_sanity/
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

DEFAULT_CONFIG = ROOT / "config" / "gpt2.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 — Ext 1 QA ablation")
    add_ext1_cli(p, DEFAULT_CONFIG)
    p.set_defaults(device="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    run_ext1_main(parse_args())


if __name__ == "__main__":
    main()
