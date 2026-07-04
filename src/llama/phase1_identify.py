"""
Llama 3.2 — Phase 1: static entropy-neuron identification.

SwiGLU last-layer neurons (silu(gate)*up, w_out = down_proj rows), top 1% by rho, k=20 null space.

Usage:
  python src/llama/phase1_identify.py
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
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from gpt2.core import ModelRunConfig  # noqa: E402
from llama.core import LLAMA_NULLSPACE_K, run_phase1_identify  # noqa: E402
from llama.load import LLAMA_32_1B_BASE, load_llama  # noqa: E402

DEFAULT_MODEL = LLAMA_32_1B_BASE
DEFAULT_OUT = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama 3.2 — Phase 1 identification")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace Llama 3.2 checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    p.add_argument("--entropy-top-pct", type=float, default=0.01, help="Top fraction by rho")
    p.add_argument(
        "--dtype",
        default=None,
        help="float16 | bfloat16 | float32 (default: float16 on cuda, float32 on cpu)",
    )
    p.add_argument("--png", action="store_true")
    return p.parse_args()


def main() -> None:
    import gpt2.core as core

    args = parse_args()
    core._SAVE_PNG = args.png

    cfg = ModelRunConfig(
        name=args.model,
        entropy_top_pct=args.entropy_top_pct,
        udark_k=LLAMA_NULLSPACE_K,
        dtype=args.dtype,
    )
    out_dir = cfg.output_dir(args.out_dir) / "phase1"

    run_phase1_identify(
        cfg,
        out_dir,
        device=args.device,
        load_model_fn=load_llama,
    )


if __name__ == "__main__":
    main()
