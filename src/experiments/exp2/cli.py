"""Shared CLI helpers for Exp 2 weight-projection entry points."""

from __future__ import annotations

import argparse
import pathlib

import torch

from exp2.runner import run_exp2
from exp2.sanity import run_unit_checks, write_sanity_report
from exp2.settings import Exp2Config, load_exp2_config


def add_exp2_cli(parser: argparse.ArgumentParser, default_config: pathlib.Path) -> None:
    parser.add_argument("--config", type=pathlib.Path, default=default_config)
    parser.add_argument("--model", default=None, help="Override model id from YAML (Qwen base/instruct)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument(
        "--check-functions",
        action="store_true",
        help="Run Ext2 unit checks (no model load)",
    )
    parser.add_argument("--png", action="store_true")


def run_exp2_main(args: argparse.Namespace, load_model_fn) -> None:
    cfg: Exp2Config = load_exp2_config(args.config)
    if getattr(args, "model", None):
        cfg.model = args.model

    if args.check_functions:
        print("Ext2 unit checks")
        results = run_unit_checks()
        out = cfg.study_dir()
        write_sanity_report(out, results)
        failed = [r for r in results if r.get("passed") is False]
        if failed:
            raise SystemExit(f"{len(failed)} Ext2 unit check(s) failed")
        print("All Ext2 unit checks passed")
        return

    device = args.device or cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print(
            "CUDA unavailable; using CPU. "
            "Activate the project venv first (e.g. .venv\\Scripts\\Activate.ps1 on Windows)."
        )

    run_exp2(
        cfg,
        device=device,
        load_model_fn=load_model_fn,
        save_png=args.png,
        plots_only=args.plots_only,
    )
