"""
Llama 3.2 1B Instruct — Ext 2: inter-layer MLP→MLP null-space weight projection.

Usage:
  python src/llama/ext2_weight_projection_instruct.py --device cuda
  python src/llama/ext2_weight_projection_instruct.py --check-functions
  python src/llama/ext2_weight_projection_instruct.py --plots-only

Outputs: outputs/meta-llama_Llama-3.2-1B-Instruct/ext2/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from hyperparameters import ROOT  # noqa: E402
from llama.load import load_llama  # noqa: E402
from weight_proj.cli import add_ext2_cli, run_ext2_main  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "llama_instruct.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama 3.2 1B Instruct — Ext 2 weight projection")
    add_ext2_cli(p, DEFAULT_CONFIG)
    p.set_defaults(device="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    run_ext2_main(parse_args(), load_model_fn=load_llama)


if __name__ == "__main__":
    main()
