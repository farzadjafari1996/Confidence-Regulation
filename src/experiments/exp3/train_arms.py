"""Step 3: train logistic (and optional MLP) probes on all feature arms.

Usage (from repo root, venv active). Omitting --dataset runs Wikidata then T-REx:
  python src/experiments/exp3/train_arms.py --config config/llama_instruct.yaml --split main --device cuda
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

from exp3.arms import list_arms, probe_ready, resolve_matched_k  # noqa: E402
from exp3.cli import add_train_args, arm_select_kwargs, iter_cfgs, resolve_device  # noqa: E402
from exp3.settings import Exp3Config, LabelName, SplitName  # noqa: E402
from exp3.train_logistic import run_train_logistic  # noqa: E402
from exp3.train_mlp import run_train_mlp  # noqa: E402


def run_train_arms(
    cfg: Exp3Config,
    split_name: SplitName = "main",
    *,
    label: LabelName | None = None,
    device: str | None = None,
    phase1_dir: pathlib.Path | None = None,
    skip_phase1: bool = False,
    k: int | None = None,
    n_random: int | None = None,
    force: bool = False,
    skip_random: bool = False,
    include_mlp: bool | None = None,
    skip_resid: bool = True,
    include_bottom: bool | None = None,
    headline_only: bool = False,
    entropy_as_topk: bool = False,
) -> list[pathlib.Path]:
    label = cfg.resolved_label(label)
    device = device or cfg.device
    k, n_random, resolved_phase1 = resolve_matched_k(
        cfg,
        split_name,
        device=device,
        phase1_dir=phase1_dir,
        skip_phase1=skip_phase1,
        k=k,
        n_random=n_random,
    )
    arms = list_arms(
        cfg,
        k,
        n_random=n_random,
        skip_random=skip_random,
        include_mlp=include_mlp,
        skip_resid=skip_resid,
        include_bottom=include_bottom,
        headline_only=headline_only,
        entropy_as_topk=entropy_as_topk,
    )
    print(
        f"Arms k={k} n_random={n_random}: "
        + ", ".join(f"{a.probe}/{a.features}({a.name})" for a in arms)
    )
    saved: list[pathlib.Path] = []
    for arm in arms:
        if not force and probe_ready(cfg, split_name, arm.probe, arm.features):
            print(f"Skip {arm.probe}/{arm.features} ({arm.name} already trained)")
            continue
        print(f"Train {arm.probe}/{arm.features} ({arm.name})")
        if arm.probe == "logistic":
            path = run_train_logistic(
                cfg,
                split_name,
                label=label,
                features=arm.features,
                device=device,
                phase1_dir=resolved_phase1,
                skip_phase1=skip_phase1,
            )
        else:
            path = run_train_mlp(
                cfg,
                split_name,
                label=label,
                features=arm.features,
                device=device,
                phase1_dir=resolved_phase1,
                skip_phase1=skip_phase1,
            )
        saved.append(path)
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 3 step 3: train feature-arm probes")
    add_train_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for dataset, cfg in iter_cfgs(args):
        device = resolve_device(args.device or cfg.device)
        print(f"\n=== train  {dataset}  {args.split}  n={cfg.sample_size(args.split)} ===")
        run_train_arms(
            cfg,
            args.split,
            label=args.label,
            device=device,
            skip_phase1=args.skip_phase1,
            force=args.force,
            **arm_select_kwargs(args),
        )


if __name__ == "__main__":
    main()
