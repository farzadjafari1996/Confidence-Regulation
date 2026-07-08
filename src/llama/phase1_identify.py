"""
Llama 3.2 — Phase 1: static entropy-neuron identification.

SwiGLU last-layer neurons (silu(gate)*up, w_out = down_proj rows), top rho, k=20 null space.

Usage:
  python src/llama/phase1_identify.py
  python src/llama/phase1_identify.py --config config/llama.yaml
  python src/llama/phase1_identify.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda
  python src/llama/phase1_identify.py --png

Outputs: outputs/<model-slug>/phase1/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from llama.core import run_phase1_identify  # noqa: E402
from llama.load import load_llama  # noqa: E402
from run_config import add_config_argument, add_phase1_cli, build_model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama 3.2 — Phase 1 identification")
    add_config_argument(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_phase1_cli(p, "llama")
    return p.parse_args()


def main() -> None:
    import gpt2.core as core

    args = parse_args()
    core._SAVE_PNG = args.png

    cfg, out_base = build_model_config(args, "llama")
    out_dir = cfg.output_dir(out_base) / "phase1"

    run_phase1_identify(
        cfg,
        out_dir,
        device=args.device,
        load_model_fn=load_llama,
    )


if __name__ == "__main__":
    main()
