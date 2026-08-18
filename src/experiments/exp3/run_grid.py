"""Run the Exp 3 five-step grid: prepare / gather / train / eval / ablate.

Usage (from repo root, venv active):
  python src/experiments/exp3/run_grid.py --step prepare --split sanity --fast --device cuda
  python src/experiments/exp3/run_grid.py --step gather --split sanity --fast --device cuda
  python src/experiments/exp3/run_grid.py --step train --split sanity --fast --device cuda
  python src/experiments/exp3/run_grid.py --step eval --split sanity --fast
  python src/experiments/exp3/run_grid.py --step ablate --split sanity --fast --device cuda
  python src/experiments/exp3/run_grid.py --step all --split sanity --fast --device cuda
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_src = pathlib.Path(__file__).resolve().parent
while _src != _src.parent and not (
    (_src / "common").is_dir() and (_src / "experiments").is_dir() and (_src / "models").is_dir()
):
    _src = _src.parent
for _p in (_src, _src / "common", _src / "experiments", _src / "models"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from exp3.ablate_entropy_features import run_entropy_feature_ablation  # noqa: E402
from exp3.cli import add_arm_select_args, arm_select_kwargs  # noqa: E402
from exp3.evaluate import run_evaluate_arms  # noqa: E402
from exp3.gather_activations import run_gather  # noqa: E402
from exp3.model_utils import load_exp3_model, load_tokenizer  # noqa: E402
from exp3.prepare_datasets import run_prepare_datasets  # noqa: E402
from exp3.settings import DATASETS, apply_runtime_overrides, load_exp3_config, resolved_datasets  # noqa: E402
from exp3.train_arms import run_train_arms  # noqa: E402
from hyperparameters import ROOT  # noqa: E402

GRID = [
    ("gpt2", ROOT / "config" / "gpt2.yaml", None),
    ("llama_instruct", ROOT / "config" / "llama_instruct.yaml", None),
    ("qwen_2b_instruct", ROOT / "config" / "qwen3_2b_instruct.yaml", None),
    ("smollm_instruct", ROOT / "config" / "smollm2_1.7b_instruct.yaml", None),
    ("pythia_1.4b", ROOT / "config" / "pythia_1.4b.yaml", None),
    ("gemma_2b_instruct", ROOT / "config" / "gemma2_2b_it.yaml", None),
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 3 multi-model / multi-dataset grid")
    p.add_argument(
        "--step",
        choices=["prepare", "gather", "collect", "train", "eval", "ablate", "all"],
        required=True,
        help="'collect' is an alias for prepare then gather",
    )
    p.add_argument("--split", choices=["sanity", "main"], default="sanity")
    p.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    p.add_argument("--model", default="all", help="GRID key or 'all'")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=None, help="Override sanity and main sample sizes")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Tiny N (64 unless --n), fp16 on CUDA, fewer logistic/MLP/random draws",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    add_arm_select_args(p)
    p.add_argument("--n-controls", type=int, default=None, help="Ablation random-control count")
    return p.parse_args()


def _jobs(args: argparse.Namespace):
    datasets = resolved_datasets(args.dataset)
    models = [row for row in GRID if args.model == "all" or row[0] == args.model]
    if not models:
        raise SystemExit(f"Unknown --model {args.model}. Choose one of {[r[0] for r in GRID]}")
    for key, config, _ in models:
        for dataset in datasets:
            cfg = load_exp3_config(config)
            cfg.dataset = dataset  # type: ignore[assignment]
            apply_runtime_overrides(
                cfg,
                n=args.n,
                fast=args.fast,
                device=args.device,
            )
            yield key, dataset, cfg


def _prepare(cfg, args, tokenizer) -> None:
    run_prepare_datasets(cfg, args.split, tokenizer=tokenizer, force=args.force)


def _gather(cfg, args, *, model, tokenizer) -> None:
    run_gather(
        cfg,
        args.split,
        device=args.device,
        resume=not args.no_resume,
        force=args.force,
        model=model,
        tokenizer=tokenizer,
    )


def _train(cfg, args) -> None:
    run_train_arms(
        cfg,
        args.split,
        device=args.device,
        skip_phase1=False,
        force=args.force,
        **arm_select_kwargs(args),
    )


def _eval(cfg, args) -> None:
    run_evaluate_arms(
        cfg,
        args.split,
        skip_importance=True,
        device=args.device,
        skip_phase1=False,
        **arm_select_kwargs(args),
    )


def _ablate(cfg, args) -> None:
    kwargs = {}
    if args.n_controls is not None:
        kwargs["n_controls"] = args.n_controls
    elif args.fast:
        kwargs["n_controls"] = 3
    run_entropy_feature_ablation(
        cfg,
        args.split,
        "logistic",
        device=args.device,
        skip_phase1=False,
        **kwargs,
    )


def _wanted(step: str, args_step: str) -> bool:
    if args_step == "all":
        return True
    if args_step == "collect":
        return step in {"prepare", "gather"}
    return step == args_step


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            args.device = "cpu"
            print("CUDA unavailable; using CPU")

    tokenizers: dict[str, object] = {}
    models: dict[str, tuple[object, object]] = {}

    for key, dataset, cfg in _jobs(args):
        print(
            f"\n=== {args.step}  {key}  {dataset}  {args.split}  "
            f"n={cfg.sample_size(args.split)} dtype={cfg.dtype} ==="
        )
        if _wanted("prepare", args.step):
            if key not in tokenizers:
                print(f"Loading tokenizer for {cfg.model}...")
                tokenizers[key] = load_tokenizer(cfg)
            _prepare(cfg, args, tokenizers[key])
        if _wanted("gather", args.step):
            from exp3.dataset_io import collection_ready

            if args.force or not collection_ready(cfg.data_out_dir(args.split)):
                if key not in models:
                    models[key] = load_exp3_model(cfg, args.device)
                model, tokenizer = models[key]
                _gather(cfg, args, model=model, tokenizer=tokenizer)
            else:
                _gather(cfg, args, model=None, tokenizer=None)
        if _wanted("train", args.step):
            _train(cfg, args)
        if _wanted("eval", args.step):
            _eval(cfg, args)
        if _wanted("ablate", args.step):
            _ablate(cfg, args)


if __name__ == "__main__":
    main()
