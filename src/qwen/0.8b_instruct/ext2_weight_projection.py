"""
Qwen 3.5 0.8B (instruct) — Ext 2: inter-layer MLP→MLP null-space weight projection.

Usage:
  python src/qwen/0.8b_instruct/ext2_weight_projection.py --device cuda

Outputs: outputs/Qwen_Qwen3.5-0.8B/ext2/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from hyperparameters import ROOT  # noqa: E402
from qwen.load import QWEN_35_08B_INSTRUCT, load_qwen  # noqa: E402
from weight_proj.cli import add_ext2_cli, run_ext2_main  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "qwen3_0.8b.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen 3.5 0.8B instruct — Ext 2 weight projection")
    add_ext2_cli(p, DEFAULT_CONFIG)
    p.set_defaults(
        config=DEFAULT_CONFIG,
        model=QWEN_35_08B_INSTRUCT,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main() -> None:
    run_ext2_main(parse_args(), load_model_fn=load_qwen)


if __name__ == "__main__":
    main()
