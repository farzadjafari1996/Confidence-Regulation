"""Step 1: download, filter, and cache Wikidata / LAMA T-REx items.

Usage (from repo root, venv active). Omitting --dataset runs both cloze sets:
  python src/experiments/exp3/prepare_datasets.py --config config/llama_instruct.yaml --split main
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

from exp3.cli import add_step_args, iter_cfgs  # noqa: E402
from exp3.data_loaders import prepare_splits  # noqa: E402
from exp3.model_utils import load_tokenizer  # noqa: E402
from exp3.settings import Exp3Config, SplitName  # noqa: E402
from hyperparameters import model_slug  # noqa: E402


def run_prepare_datasets(
    cfg: Exp3Config,
    split_name: SplitName,
    *,
    tokenizer,
    force: bool = False,
) -> dict[str, int]:
    slug = model_slug(cfg.model)
    splits = prepare_splits(cfg, slug, tokenizer, force=force)
    counts = {name: len(items) for name, items in splits.items()}
    print(
        f"Prepared {cfg.dataset} for {slug}: "
        f"sanity={counts.get('sanity', 0)} main={counts.get('main', 0)} "
        f"(requested {split_name}={cfg.sample_size(split_name)})"
    )
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 3 step 1: cache dataset items")
    add_step_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = None
    for dataset, cfg in iter_cfgs(args):
        print(f"\n=== prepare  {dataset}  {args.split}  n={cfg.sample_size(args.split)} ===")
        if tokenizer is None:
            print(f"Loading tokenizer for {cfg.model}...")
            tokenizer = load_tokenizer(cfg)
        run_prepare_datasets(cfg, args.split, tokenizer=tokenizer, force=args.force)


if __name__ == "__main__":
    main()
