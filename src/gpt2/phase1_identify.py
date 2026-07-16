"""
GPT-2 family — Phase 1: static entropy-neuron identification.

Selects top neurons by rho (paper §3.3), writes plots and candidate list.
Selection is purely metric-based; paper neuron IDs are used for verification only.

Usage:
  python src/gpt2/phase1_identify.py
  python src/gpt2/phase1_identify.py --config config/gpt2.yaml
  python src/gpt2/phase1_identify.py --model gpt2-medium --device cuda
  python src/gpt2/phase1_identify.py --png

Outputs: outputs/gpt2-small/phase1/
  static_neuron_metrics.csv
  top1pct_rho_neurons.json   ← input for phase2_ablation.py
  fig2a_norm_vs_logitvar.html
  fig2d_singular_values.html
  entropy_vs_random_unembed_alignment.html
  entropy_vs_random_unembed_alignment_means.html  (when write_align_mode is both or means_only)
  summary.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from gpt2.load import load_gpt2  # noqa: E402
from gpt2.core import GPT2_PAPER_ENTROPY_NEURONS, run_phase1_identify  # noqa: E402
from run_config import add_config_argument, add_phase1_cli, build_model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 family — Phase 1 identification")
    add_config_argument(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_phase1_cli(p, "gpt2")
    return p.parse_args()


def main() -> None:
    import gpt2.core as core

    args = parse_args()
    core._SAVE_PNG = args.png

    cfg, out_base = build_model_config(args, "gpt2")
    out_dir = cfg.output_dir(out_base) / "phase1"
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
