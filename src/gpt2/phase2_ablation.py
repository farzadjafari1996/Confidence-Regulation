"""
GPT-2 family — Phase 2: mean ablation (TE vs DE_LN).

Requires Phase 1 outputs in outputs/<model-slug>/phase1/.
Ablation set: all top-1% rho neurons + 30 random controls.

Usage:
  python src/gpt2/phase2_ablation.py --device cuda
  python src/gpt2/phase2_ablation.py --k 8 --data-end 600
  python src/gpt2/phase2_ablation.py --plots-only   # reuse ablation.feather

Outputs: outputs/<model-slug>/phase2/
  ablation.feather
  fig2c_te_vs_de_ln.html
  fig2e_rho_vs_ln_fraction.html
  fig1_entropy_vs_argmax.html
  rho_vs_ln_fraction.csv
  summary.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

SRC = pathlib.Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from gpt2.load import load_gpt2  # noqa: E402
from gpt2.phase1_identify import GPT2_NULLSPACE_K  # noqa: E402
from gpt2.core import (  # noqa: E402
    GPT2_PAPER_ENTROPY_NEURONS,
    ModelRunConfig,
    load_phase1_outputs,
    plot_b2_rho_vs_ln_fraction,
    plot_fig1_entropy_vs_argmax,
    plot_fig2c_te_de_box,
    run_phase2_ablation,
    write_phase2_summary,
)

DEFAULT_MODEL = "gpt2-small"
DEFAULT_OUT = ROOT / "outputs"

# ~3 min on RTX 4070 with 60 neurons (30 entropy + 30 random); increase if faster
DEFAULT_K = 6
DEFAULT_DATA_END = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 family — Phase 2 ablation")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    p.add_argument("--phase1-dir", type=pathlib.Path, default=None)
    p.add_argument("--k", type=int, default=DEFAULT_K, help="Ablation batches sampled")
    p.add_argument("--data-end", type=int, default=DEFAULT_DATA_END, help="C4 sequences for activations")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--png", action="store_true")
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from existing ablation.feather",
    )
    return p.parse_args()


def _plots_only(
    cfg: ModelRunConfig,
    phase1_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    paper_neurons: list[str] | None,
) -> None:
    feather = out_dir / "ablation.feather"
    if not feather.exists():
        raise FileNotFoundError(f"No ablation at {feather}")

    entropy_labels, static_df = load_phase1_outputs(phase1_dir)
    ablation_df = pd.read_feather(feather)

    plot_fig2c_te_de_box(ablation_df, out_dir, entropy_labels, cfg.name)
    agg = (
        ablation_df.assign(
            te=lambda d: (d["loss_post_ablation"] - d["loss"]).abs(),
            de_ln=lambda d: (d["loss_post_ablation_with_frozen_ln"] - d["loss"]).abs(),
        )
        .groupby("component_name", as_index=False)
        .agg({"te": "mean", "de_ln": "mean"})
    )
    agg = agg.merge(static_df[["component_name", "rho"]], on="component_name")
    agg["is_entropy"] = agg["component_name"].isin(entropy_labels)
    agg["ln_mediated_fraction"] = 1 - agg["de_ln"] / agg["te"].replace(0, np.nan)
    agg.to_csv(out_dir / "rho_vs_ln_fraction.csv", index=False)
    plot_b2_rho_vs_ln_fraction(agg, out_dir, cfg.name, highlight_col="is_entropy")
    plot_fig1_entropy_vs_argmax(ablation_df, out_dir, entropy_labels, cfg.name)
    write_phase2_summary(
        ablation_df,
        cfg,
        out_dir,
        entropy_labels,
        ablation_df["component_name"].unique().tolist(),
        elapsed_sec=0.0,
        paper_neurons=paper_neurons,
    )


def main() -> None:
    import gpt2.core as core

    args = parse_args()
    core._SAVE_PNG = args.png

    cfg = ModelRunConfig(
        name=args.model,
        k=args.k,
        data_end=args.data_end,
        batch_size=args.batch_size,
        udark_k=GPT2_NULLSPACE_K if args.model == "gpt2-small" else None,
    )
    phase1_dir = args.phase1_dir or cfg.output_dir(args.out_dir) / "phase1"
    out_dir = cfg.output_dir(args.out_dir) / "phase2"
    paper_neurons = GPT2_PAPER_ENTROPY_NEURONS if cfg.name == "gpt2-small" else None

    if args.plots_only:
        _plots_only(cfg, phase1_dir, out_dir, paper_neurons=paper_neurons)
        return

    run_phase2_ablation(
        cfg,
        phase1_dir,
        out_dir,
        device=args.device,
        load_model_fn=load_gpt2,
        n_random=args.n_random,
        seed=args.seed,
        paper_neurons=paper_neurons,
    )


if __name__ == "__main__":
    main()
