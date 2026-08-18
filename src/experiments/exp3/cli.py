"""Shared CLI helpers for Exp 3 entrypoints."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Iterator

from exp3.settings import (
    DATASETS,
    LABELS,
    DatasetName,
    Exp3Config,
    load_exp3_config,
    resolved_datasets,
)


def bootstrap_src(start: pathlib.Path) -> pathlib.Path:
    src = start.resolve()
    while src != src.parent and not (
        (src / "common").is_dir() and (src / "experiments").is_dir() and (src / "models").is_dir()
    ):
        src = src.parent
    for extra in (src, src / "common", src / "experiments", src / "models"):
        s = str(extra)
        if s not in sys.path:
            sys.path.insert(0, s)
    return src


def add_step_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=pathlib.Path, required=True, help="Model YAML under config/")
    p.add_argument(
        "--dataset",
        choices=["all", *DATASETS],
        default="all",
        help="One cloze set, or all (qa_wikidata then lama_trex). Default: all.",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--split", choices=["sanity", "main"], default="sanity")
    p.add_argument("--force", action="store_true")


def add_arm_select_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--skip-mlp", action="store_true", help="Skip extra last-MLP MLP probes (already off by default)")
    p.add_argument("--skip-random", action="store_true")
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Neuron subset size for entropy / bottom-k / random last-MLP arms. "
        "Default: full phase-1 entropy set.",
    )
    p.add_argument(
        "--n-random",
        type=int,
        default=None,
        help="How many random last-MLP subsets to train. Default: 10.",
    )
    p.add_argument(
        "--headline-only",
        action="store_true",
        help="Only last-MLP all vs entropy. Drops bottom-k, random, extra MLP probes.",
    )
    p.add_argument(
        "--last-mlp-only",
        action="store_true",
        help="Default arm set. Last-MLP logistics: all, entropy, bottom-k, and n-random subsets.",
    )
    p.add_argument(
        "--include-resid",
        action="store_true",
        help="Also train resid_all and random residual subsets (arm 1).",
    )
    p.add_argument(
        "--include-mlp",
        action="store_true",
        help="Also train extra last-MLP MLP (non-linear) probes.",
    )
    p.add_argument(
        "--skip-resid",
        action="store_true",
        help="Skip residual arms (default; resid_all is off unless --include-resid).",
    )
    p.add_argument("--skip-bottom", action="store_true", help="Skip lowest-ρ bottom-k arms")


def add_train_args(p: argparse.ArgumentParser) -> None:
    add_step_args(p)
    p.add_argument("--label", choices=LABELS, default=None)
    p.add_argument("--skip-phase1", action="store_true")
    add_arm_select_args(p)


def arm_select_kwargs(args: argparse.Namespace) -> dict:
    headline = bool(getattr(args, "headline_only", False))
    include_resid = bool(getattr(args, "include_resid", False))
    last_mlp = bool(getattr(args, "last_mlp_only", False))
    k = getattr(args, "k", None)
    n_random = getattr(args, "n_random", None)
    skip_bottom = bool(getattr(args, "skip_bottom", False))
    if headline:
        include_bottom = False
    elif skip_bottom:
        include_bottom = False
    else:
        include_bottom = True
    skip_resid = True
    if include_resid and not headline and not last_mlp:
        skip_resid = False
    if getattr(args, "skip_resid", False):
        skip_resid = True
    include_mlp = bool(getattr(args, "include_mlp", False)) and not headline and not last_mlp
    if getattr(args, "skip_mlp", False):
        include_mlp = False
    return {
        "k": k,
        "n_random": n_random,
        "skip_random": bool(getattr(args, "skip_random", False)) or headline,
        "include_mlp": include_mlp,
        "skip_resid": skip_resid,
        "include_bottom": include_bottom,
        "headline_only": headline,
        "entropy_as_topk": k is not None,
    }


def apply_overrides(cfg: Exp3Config, args: argparse.Namespace) -> Exp3Config:
    if getattr(args, "dataset", None) and args.dataset != "all":
        cfg.dataset = resolved_datasets(args.dataset)[0]
    if getattr(args, "device", None):
        cfg.device = args.device
    return cfg


def load_cfg(args: argparse.Namespace, *, dataset: DatasetName | None = None) -> Exp3Config:
    cfg = load_exp3_config(args.config)
    if dataset is not None:
        cfg.dataset = dataset
    return apply_overrides(cfg, args)


def iter_cfgs(args: argparse.Namespace) -> Iterator[tuple[DatasetName, Exp3Config]]:
    for dataset in resolved_datasets(getattr(args, "dataset", "all")):
        yield dataset, load_cfg(args, dataset=dataset)


def resolve_device(device: str | None, fallback: str = "cuda") -> str:
    import torch

    chosen = device or fallback
    if chosen == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU")
        return "cpu"
    return chosen
