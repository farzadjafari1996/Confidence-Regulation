"""
GPT-2 family — Phase 1: static entropy-neuron identification.

Selects top 1% neurons by rho (paper §3.3), writes plots and candidate list.
Selection is purely metric-based; paper neuron IDs are used for verification only.

Usage:
  python src/gpt2/phase1_identify.py
  python src/gpt2/phase1_identify.py --model gpt2-medium --device cuda
  python src/gpt2/phase1_identify.py --png

Outputs: outputs/<model-slug>/phase1/
  static_neuron_metrics.csv
  top1pct_rho_neurons.json   ← input for phase2_ablation.py
  fig2d_singular_values.html
  fig2d_rho_vs_norm.html
  fig2a_norm_vs_logitvar.html
  summary.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from gpt2.load import load_gpt2  # noqa: E402
from gpt2.core import (  # noqa: E402
    GPT2_PAPER_ENTROPY_NEURONS,
    ModelRunConfig,
    run_phase1_identify,
)

DEFAULT_MODEL = "gpt2-small"
DEFAULT_OUT = ROOT / "outputs"


# Paper null-space dimension for GPT-2 Small (§3.3, k=12)
GPT2_NULLSPACE_K = 12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 family — Phase 1 identification")
    p.add_argument("--model", default=DEFAULT_MODEL, help="TransformerLens GPT-2 checkpoint name")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    p.add_argument("--entropy-top-pct", type=float, default=0.01, help="Top fraction by rho")
    p.add_argument("--png", action="store_true")
    return p.parse_args()


def main() -> None:
    import gpt2.core as core

    args = parse_args()
    core._SAVE_PNG = args.png

    cfg = ModelRunConfig(
        name=args.model,
        entropy_top_pct=args.entropy_top_pct,
        udark_k=GPT2_NULLSPACE_K if args.model == "gpt2-small" else None,
    )
    out_dir = cfg.output_dir(args.out_dir) / "phase1"

    paper_neurons = GPT2_PAPER_ENTROPY_NEURONS if cfg.name == "gpt2-small" else None

    run_phase1_identify(
        cfg,
        out_dir,
        device=args.device,
        load_model_fn=load_gpt2,
        paper_neurons=paper_neurons,
    )


if __name__ == "__main__":
    main()
