"""
Llama 3.2 — Phase 2b: logit distribution comparison after mean ablation.

Requires Phase 1 outputs in outputs/<model-slug>/phase1/.
Default ablation set: 12 entropy neurons + 90 random controls.

Usage:
  python src/llama/phase2b_distribution.py --device cuda
  python src/llama/phase2b_distribution.py --config config/llama.yaml
  python src/llama/phase2b_distribution.py --branch both
  python src/llama/phase2b_distribution.py --plots-only

Outputs: outputs/<model-slug>/phase2b/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd
import torch

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from gpt2.distribution_ablation import (  # noqa: E402
    plot_all_distribution_ablation,
    run_phase2b_distribution_ablation,
    write_phase2b_summary,
)
from gpt2.core import load_phase1_outputs  # noqa: E402
from hyperparameters import ModelRunConfig  # noqa: E402
from llama.load import load_llama  # noqa: E402
from run_config import add_config_argument, add_phase2b_cli, build_model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama — Phase 2b distribution ablation")
    add_config_argument(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_phase2b_cli(p, "llama")
    return p.parse_args()


def _plots_only(
    cfg: ModelRunConfig,
    phase1_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    branch: str,
) -> None:
    feather = out_dir / "distribution_metrics.feather"
    if not feather.exists():
        raise FileNotFoundError(f"No distribution metrics at {feather}")

    entropy_labels, static_df = load_phase1_outputs(phase1_dir)
    dist_df = pd.read_feather(feather)
    plot_all_distribution_ablation(
        dist_df,
        static_df,
        entropy_labels,
        out_dir,
        cfg.name,
        branch=branch,  # type: ignore[arg-type]
    )
    write_phase2b_summary(
        dist_df,
        cfg,
        out_dir,
        entropy_labels,
        dist_df["component_name"].unique().tolist(),
        branch=branch,  # type: ignore[arg-type]
        elapsed_sec=0.0,
    )


def main() -> None:
    import gpt2.core as core

    args = parse_args()
    core._SAVE_PNG = args.png

    cfg, out_base = build_model_config(args, "llama")
    phase1_dir = args.phase1_dir or cfg.output_dir(out_base) / "phase1"
    out_dir = cfg.output_dir(out_base) / "phase2b"

    if args.plots_only:
        _plots_only(cfg, phase1_dir, out_dir, branch=args.branch)
        return

    run_phase2b_distribution_ablation(
        cfg,
        phase1_dir,
        out_dir,
        device=args.device,
        load_model_fn=load_llama,
        branch=args.branch,
        softmax_top_k=args.softmax_top_k,
        n_example_tokens=args.n_example_tokens,
    )


if __name__ == "__main__":
    main()
