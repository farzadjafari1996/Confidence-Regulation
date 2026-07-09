"""Shared CLI helpers for Ext 1 QA ablation entry points."""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import torch

from gpt2.load import load_gpt2
from hyperparameters import GPT2_HPARAMS, LLAMA_HPARAMS, ModelRunConfig
from llama.load import load_llama
from qa.plots import write_ext1_plots
from qa.runner import run_model_dataset
from qa.sanity_checks import run_all_sanity_checks
from qa.settings import (
    DatasetName,
    Ext1Config,
    SplitName,
    ext1_dataset_dir,
    load_ext1_config,
)


def add_ext1_cli(parser: argparse.ArgumentParser, default_config: pathlib.Path) -> None:
    parser.add_argument("--config", type=pathlib.Path, default=default_config)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset", choices=["lama", "triviaqa"], default=None)
    parser.add_argument("--split", choices=["sanity", "main"], default=None)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--check-functions", action="store_true", help="Run QA module sanity checks")
    parser.add_argument("--png", action="store_true")


def _model_run_config(ext1_cfg: Ext1Config) -> ModelRunConfig:
    spec = ext1_cfg.model_spec()
    base = GPT2_HPARAMS if spec.family == "gpt2" else LLAMA_HPARAMS
    return ModelRunConfig.from_hparams(spec.model, base, dtype=ext1_cfg.dtype)


def _load_model(ext1_cfg: Ext1Config, run_cfg: ModelRunConfig, device: str):
    spec = ext1_cfg.model_spec()
    if spec.family == "gpt2":
        model, tokenizer = load_gpt2(run_cfg, device)
        model.set_use_attn_result(False)
        return model, tokenizer
    return load_llama(run_cfg, device)


def _plots_only(
    ext1_cfg: Ext1Config,
    dataset: DatasetName,
    split_name: SplitName,
) -> None:
    model_name = ext1_cfg.model_spec().model
    out_dir = ext1_dataset_dir(ext1_cfg.out_dir, split_name, dataset)
    csv_path = out_dir / "detailed.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    main_df = df[df["ablation_mode"] == "mean_all_entropy"]
    write_ext1_plots(
        main_df,
        out_dir,
        dataset=dataset,
        model_title=model_name,
        split_name=split_name,
        full_df=df,
    )


def run_ext1_main(args: argparse.Namespace) -> None:
    import gpt2.core as core

    core._SAVE_PNG = args.png
    ext1_cfg = load_ext1_config(args.config)
    model_name = ext1_cfg.model_spec().model

    if args.check_functions:
        run_all_sanity_checks(args.config, model_name=model_name)
        return

    device = args.device or ext1_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print(
            "CUDA unavailable; using CPU. "
            "Activate the project venv first (e.g. .venv\\Scripts\\Activate.ps1 on Windows)."
        )

    datasets: list[DatasetName] = ["lama", "triviaqa"]
    if args.dataset:
        datasets = [args.dataset]  # type: ignore[assignment]

    splits: list[SplitName] = ["sanity", "main"]
    if args.split:
        splits = [args.split]  # type: ignore[assignment]

    if args.plots_only:
        for dataset in datasets:
            for split_name in splits:
                _plots_only(ext1_cfg, dataset, split_name)
        return

    run_cfg = _model_run_config(ext1_cfg)
    phase1_dir = ext1_cfg.out_dir / "phase1"
    if not phase1_dir.exists():
        raise FileNotFoundError(f"Run Phase 1 first — missing {phase1_dir}")

    torch.manual_seed(ext1_cfg.seed)
    print(f"\nLoading {run_cfg.name} ({ext1_cfg.model_spec().family})...")
    model, tokenizer = _load_model(ext1_cfg, run_cfg, device)

    for dataset in datasets:
        run_model_dataset(
            model,
            tokenizer,
            run_cfg,
            ext1_cfg,
            ext1_cfg.model_spec(),
            dataset=dataset,
            phase1_dir=phase1_dir,
            out_base=ext1_cfg.out_dir,
            device=device,
            splits=splits,
        )

    if device.startswith("cuda"):
        del model
        torch.cuda.empty_cache()


# Backward-compatible aliases (deprecated)
add_phase3b_cli = add_ext1_cli
run_phase3b_main = run_ext1_main
