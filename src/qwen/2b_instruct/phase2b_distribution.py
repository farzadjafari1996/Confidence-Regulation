"""
Qwen 3.5 2B (instruct) — Phase 2b: logit distribution comparison after mean ablation.

Requires Phase 1 outputs in outputs/Qwen_Qwen3.5-2B/phase1/.

Usage:
  python src/qwen/2b_instruct/phase2b_distribution.py --device cuda
  python src/qwen/2b_instruct/phase2b_distribution.py --branch both
  python src/qwen/2b_instruct/phase2b_distribution.py --plots-only

Outputs: outputs/Qwen_Qwen3.5-2B/phase2b/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd
import torch

SRC = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from gpt2.distribution_ablation import (  # noqa: E402
    plot_all_distribution_ablation,
    run_phase2b_distribution_ablation,
    write_phase2b_summary,
)
from hyperparameters import ROOT, ModelRunConfig  # noqa: E402
from qwen.core import (  # noqa: E402
    filter_entropy_activation_df,
    get_entropy_activation_df,
    load_phase1_outputs,
    load_tokenized_data,
)
from qwen.load import QWEN_35_2B_INSTRUCT, load_qwen  # noqa: E402
from run_config import add_config_argument, add_phase2b_cli, build_model_config  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "qwen3_2b.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen 3.5 2B instruct — Phase 2b distribution ablation")
    add_config_argument(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_phase2b_cli(p, "qwen")
    p.set_defaults(config=DEFAULT_CONFIG, model=QWEN_35_2B_INSTRUCT)
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

    cfg, out_base = build_model_config(args, "qwen")
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
        load_model_fn=load_qwen,
        branch=args.branch,
        softmax_top_k=args.softmax_top_k,
        n_example_tokens=args.n_example_tokens,
        load_tokenized_fn=load_tokenized_data,
        get_entropy_df_fn=get_entropy_activation_df,
        filter_df_fn=filter_entropy_activation_df,
    )


if __name__ == "__main__":
    main()
