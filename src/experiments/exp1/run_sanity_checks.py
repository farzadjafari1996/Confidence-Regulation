"""
Run Phase 3 sanity checks for all models.

Usage:
  python src/experiments/exp1/run_sanity_checks.py
  python src/experiments/exp1/run_sanity_checks.py --model gpt2-small
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_p = pathlib.Path(__file__).resolve().parent
while _p != _p.parent and not (_p / 'bootstrap.py').is_file():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path(__file__)

from hyperparameters import ROOT  # noqa: E402
from exp1.sanity_checks import run_all_sanity_checks  # noqa: E402

MODEL_CONFIGS = {
    "gpt2-small": ROOT / "config" / "gpt2.yaml",
    "meta-llama/Llama-3.2-1B": ROOT / "config" / "llama.yaml",
    "meta-llama/Llama-3.2-1B-Instruct": ROOT / "config" / "llama_instruct.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all Phase 3 QA sanity checks")
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    targets = (
        {args.model: MODEL_CONFIGS[args.model]}
        if args.model
        else MODEL_CONFIGS
    )
    results = {}
    for model, cfg_path in targets.items():
        results[model] = run_all_sanity_checks(cfg_path, model_name=model)

    failed = [m for m, r in results.items() if not r.get("all_passed")]
    if failed:
        raise SystemExit(f"Sanity checks failed for: {', '.join(failed)}")
    print("\nAll model sanity checks passed.")


if __name__ == "__main__":
    main()
