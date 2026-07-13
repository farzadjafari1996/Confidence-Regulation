"""
Qwen 3.5 2B (base) — Phase 1: static entropy-neuron identification.

Usage:
  python src/qwen/2b_base/phase1_identify.py --device cuda

Outputs: outputs/Qwen_Qwen3.5-2B-Base/phase1/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from hyperparameters import ROOT  # noqa: E402
from qwen.core import run_phase1_identify  # noqa: E402
from qwen.load import QWEN_35_2B_BASE, load_qwen  # noqa: E402
from run_config import add_config_argument, add_phase1_cli, build_model_config  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "qwen3_2b.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen 3.5 2B base — Phase 1 identification")
    add_config_argument(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_phase1_cli(p, "qwen")
    p.set_defaults(config=DEFAULT_CONFIG, model=QWEN_35_2B_BASE)
    return p.parse_args()


def main() -> None:
    import gpt2.core as core

    args = parse_args()
    core._SAVE_PNG = args.png

    cfg, out_base = build_model_config(args, "qwen")
    out_dir = cfg.output_dir(out_base) / "phase1"

    run_phase1_identify(
        cfg,
        out_dir,
        device=args.device,
        load_model_fn=load_qwen,
    )


if __name__ == "__main__":
    main()
