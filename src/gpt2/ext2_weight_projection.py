"""
GPT-2 — Ext 2: inter-layer MLP→MLP null-space weight projection.

For each consecutive MLP pair (ℓ → ℓ+1), project W_out[ℓ] onto the dark
subspace of W_in[ℓ+1] (bottom-k left singular vectors; TL forward is resid @ W_in)
and compute ρ. Also runs last-MLP → W_U as a Phase-1 sanity check.

Usage:
  .venv\\Scripts\\Activate.ps1
  python src/gpt2/ext2_weight_projection.py --device cuda
  python src/gpt2/ext2_weight_projection.py --check-functions
  python src/gpt2/ext2_weight_projection.py --plots-only
  python src/gpt2/ext2_weight_projection.py --png

Outputs: outputs/gpt2-small/ext2/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from gpt2.load import load_gpt2  # noqa: E402
from hyperparameters import ROOT  # noqa: E402
from weight_proj.cli import add_ext2_cli, run_ext2_main  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "gpt2.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 — Ext 2 weight projection")
    add_ext2_cli(p, DEFAULT_CONFIG)
    p.set_defaults(device="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    run_ext2_main(parse_args(), load_model_fn=load_gpt2)


if __name__ == "__main__":
    main()
