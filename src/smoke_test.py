"""
Model smoke tests (roadmap Model Registry checklist).

Loads each model, runs one forward pass (seq_len=64 by default), and prints
architecture metadata plus pass/fail checks before Phase 1/2 runs.

Usage:
  python src/smoke_test.py
  python src/smoke_test.py --model gpt2-small
  python src/smoke_test.py --model meta-llama/Llama-3.2-1B --device cuda
  python src/smoke_test.py --seq-len 64

Llama models require HF_TOKEN or reference_code/ablations/hf_token.txt.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
from transformer_lens import utils as tl_utils

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from gpt2.load import load_gpt2  # noqa: E402
from hyperparameters import (  # noqa: E402
    GPT2_HPARAMS,
    LLAMA_HPARAMS,
    ModelRunConfig,
    load_run_settings,
)
from llama.load import load_llama  # noqa: E402

Family = Literal["gpt2", "llama"]

MODEL_REGISTRY: dict[str, tuple[Family, pathlib.Path | None]] = {
    "gpt2-small": ("gpt2", SRC.parent / "config" / "gpt2.yaml"),
    "meta-llama/Llama-3.2-1B": ("llama", SRC.parent / "config" / "llama.yaml"),
    "meta-llama/Llama-3.2-1B-Instruct": ("llama", SRC.parent / "config" / "llama_instruct.yaml"),
}

EXPECTED_NORM: dict[Family, set[str]] = {
    "gpt2": {"LN", "LNPre"},
    "llama": {"RMSPre"},
}

REQUIRED_LLama_HOOK_SUFFIXES = ("hook_resid_post", "hook_post", "ln_final.hook_scale")


@dataclass
class SmokeResult:
    model: str
    family: Family
    device: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _model_config(model_name: str) -> ModelRunConfig:
    family, config_path = MODEL_REGISTRY[model_name]
    settings = load_run_settings(config_path, family)
    return ModelRunConfig.from_hparams(model_name, settings["hparams"], dtype=settings.get("dtype"))


def _check(name: str, ok: bool, checks: dict[str, bool], errors: list[str], detail: str = "") -> None:
    checks[name] = ok
    if not ok:
        msg = name if not detail else f"{name}: {detail}"
        errors.append(msg)


def _forward_gpt2(cfg: ModelRunConfig, device: str, seq_len: int) -> SmokeResult:
    result = SmokeResult(model=cfg.name, family="gpt2", device=device, passed=False)
    try:
        model, tokenizer = load_gpt2(cfg, device)
        ids = tokenizer("Hello world", return_tensors="pt")["input_ids"][:, :seq_len].to(device)
        logits, cache = model.run_with_cache(ids)
        d_model = model.cfg.d_model

        result.metrics = {
            "logits_shape": tuple(logits.shape),
            "n_layers": model.cfg.n_layers,
            "d_mlp": model.cfg.d_mlp,
            "d_model": d_model,
            "d_vocab": model.cfg.d_vocab,
            "normalization": model.cfg.normalization_type,
            "udark_start": cfg.udark_start(d_model),
            "nullspace_k": cfg.nullspace_k(d_model),
            "cache_keys_sample": sorted(cache.keys())[:8],
        }

        _check("forward_ok", logits.ndim == 3, result.checks, result.errors)
        _check(
            "logits_shape",
            logits.shape[0] == 1 and logits.shape[1] == ids.shape[1],
            result.checks,
            result.errors,
            f"got {tuple(logits.shape)}",
        )
        _check(
            "normalization",
            model.cfg.normalization_type in EXPECTED_NORM["gpt2"],
            result.checks,
            result.errors,
            str(model.cfg.normalization_type),
        )
        _check(
            "udark_start_paper",
            cfg.udark_start(d_model) == -12,
            result.checks,
            result.errors,
            f"got {cfg.udark_start(d_model)} (paper: -12 for gpt2-small)",
        )
        if device.startswith("cuda"):
            _check(
                "vram_under_12gb",
                torch.cuda.max_memory_allocated() / (1024**3) < 12.0,
                result.checks,
                result.errors,
                f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB peak",
            )
    except Exception as exc:
        result.errors.append(f"exception: {exc}")
        result.checks["forward_ok"] = False

    result.passed = bool(result.checks) and all(result.checks.values()) and not result.errors
    return result


def _forward_llama(cfg: ModelRunConfig, device: str, seq_len: int) -> SmokeResult:
    result = SmokeResult(model=cfg.name, family="llama", device=device, passed=False)
    try:
        model, tokenizer = load_llama(cfg, device)
        ids = tokenizer("Hello world", return_tensors="pt")["input_ids"][:, :seq_len].to(device)
        logits, cache = model.run_with_cache(ids)
        d_model = model.cfg.d_model
        layer = model.cfg.n_layers - 1

        result.metrics = {
            "logits_shape": tuple(logits.shape),
            "n_layers": model.cfg.n_layers,
            "d_mlp": model.cfg.d_mlp,
            "d_model": d_model,
            "d_vocab": model.cfg.d_vocab,
            "normalization": model.cfg.normalization_type,
            "udark_start": cfg.udark_start(d_model),
            "nullspace_k": cfg.nullspace_k(d_model),
            "cache_keys": sorted(cache.keys()),
            "resid_post_key": tl_utils.get_act_name("resid_post", layer),
            "post_key": tl_utils.get_act_name("post", layer),
        }

        _check("forward_ok", logits.ndim == 3, result.checks, result.errors)
        _check(
            "logits_shape",
            logits.shape[0] == 1 and logits.shape[1] == ids.shape[1],
            result.checks,
            result.errors,
            f"got {tuple(logits.shape)}",
        )
        _check(
            "normalization",
            model.cfg.normalization_type in EXPECTED_NORM["llama"],
            result.checks,
            result.errors,
            str(model.cfg.normalization_type),
        )
        _check(
            "udark_start_1pct",
            cfg.udark_start(d_model) == -20,
            result.checks,
            result.errors,
            f"got {cfg.udark_start(d_model)} (expected -20 for d_model=2048, k=20)",
        )
        for suffix in REQUIRED_LLama_HOOK_SUFFIXES:
            _check(
                f"hook_{suffix}",
                any(k.endswith(suffix) or k == suffix for k in cache),
                result.checks,
                result.errors,
                f"missing in {sorted(cache.keys())}",
            )
        if device.startswith("cuda"):
            _check(
                "vram_under_12gb",
                torch.cuda.max_memory_allocated() / (1024**3) < 12.0,
                result.checks,
                result.errors,
                f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB peak",
            )
    except Exception as exc:
        result.errors.append(f"exception: {exc}")
        result.checks["forward_ok"] = False

    result.passed = bool(result.checks) and all(result.checks.values()) and not result.errors
    return result


_RUNNERS: dict[Family, Callable[[ModelRunConfig, str, int], SmokeResult]] = {
    "gpt2": _forward_gpt2,
    "llama": _forward_llama,
}


def run_smoke_test(model_name: str, device: str, seq_len: int) -> SmokeResult:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}; choose from {list(MODEL_REGISTRY)}")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        torch.cuda.reset_peak_memory_stats()

    family, _ = MODEL_REGISTRY[model_name]
    cfg = _model_config(model_name)
    return _RUNNERS[family](cfg, device, seq_len)


def _print_result(result: SmokeResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"{result.model}  [{status}]  device={result.device}")
    print(f"{'=' * 60}")
    for key, val in result.metrics.items():
        print(f"  {key}: {val}")
    print("  checks:")
    for name, ok in result.checks.items():
        print(f"    {'✓' if ok else '✗'} {name}")
    if result.errors:
        print("  errors:")
        for err in result.errors:
            print(f"    - {err}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model load + forward smoke tests")
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY),
        default=None,
        help="Run one model (default: all registered models)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    targets = [args.model] if args.model else list(MODEL_REGISTRY)

    print(f"Smoke tests on {args.device}  seq_len={args.seq_len}")
    results = [run_smoke_test(name, args.device, args.seq_len) for name in targets]
    for result in results:
        _print_result(result)

    failed = [r.model for r in results if not r.passed]
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        raise SystemExit(1)
    print(f"\nAll {len(results)} smoke test(s) passed.")


if __name__ == "__main__":
    main()
