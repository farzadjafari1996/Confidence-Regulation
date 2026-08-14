"""Load YAML run settings and merge with CLI overrides."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Literal

from hyperparameters import (
    DEFAULT_OUTPUT_DIR,
    GEMMA_DEFAULT_MODEL,
    GEMMA_HPARAMS,
    GPT2_DEFAULT_MODEL,
    GPT2_HPARAMS,
    LLAMA_DEFAULT_MODEL,
    LLAMA_HPARAMS,
    PYTHIA_DEFAULT_MODEL,
    PYTHIA_HPARAMS,
    QWEN_DEFAULT_MODEL,
    QWEN_HPARAMS,
    SMOLLM_DEFAULT_MODEL,
    SMOLLM_HPARAMS,
    Hyperparameters,
    ModelRunConfig,
    load_run_settings,
)

Family = Literal["gpt2", "llama", "qwen", "pythia", "smollm", "gemma"]

_FAMILY_DEFAULTS: dict[Family, tuple[Hyperparameters, str]] = {
    "gpt2": (GPT2_HPARAMS, GPT2_DEFAULT_MODEL),
    "llama": (LLAMA_HPARAMS, LLAMA_DEFAULT_MODEL),
    "qwen": (QWEN_HPARAMS, QWEN_DEFAULT_MODEL),
    "pythia": (PYTHIA_HPARAMS, PYTHIA_DEFAULT_MODEL),
    "smollm": (SMOLLM_HPARAMS, SMOLLM_DEFAULT_MODEL),
    "gemma": (GEMMA_HPARAMS, GEMMA_DEFAULT_MODEL),
}

_HF_DTYPE_FAMILIES = frozenset({"llama", "qwen", "smollm", "gemma"})


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help="YAML config (default: config/<family>.yaml when present)",
    )


def add_phase1_cli(parser: argparse.ArgumentParser, family: Family) -> None:
    _, default_model = _FAMILY_DEFAULTS[family]
    parser.add_argument("--model", default=None, help=f"Default from config or {default_model}")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--entropy-top-pct", type=float, default=None)
    parser.add_argument(
        "--entropy-top-n",
        type=int,
        default=None,
        help="Fixed neuron count; effective count is min(pct, n)",
    )
    parser.add_argument(
        "--nullspace-top-pct",
        type=float,
        default=None,
        help="Fraction of d_model for null-space k; effective k is min(pct*d_model, nullspace_k)",
    )
    parser.add_argument(
        "--nullspace-k",
        type=int,
        default=None,
        help="Fixed null-space k; effective k is min(nullspace_top_pct*d_model, k)",
    )
    parser.add_argument("--seed", type=int, default=None)
    if family in _HF_DTYPE_FAMILIES:
        parser.add_argument(
            "--dtype",
            default=None,
            help="float16 | bfloat16 | float32 (default: float16 on cuda, float32 on cpu)",
        )
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--write-align-n-random", type=int, default=None)
    parser.add_argument(
        "--write-align-mode",
        choices=("full", "means_only", "both"),
        default=None,
        help="Unembed alignment plot: all neurons, means only, or both HTML files",
    )


def add_phase2_cli(parser: argparse.ArgumentParser, family: Family) -> None:
    _, default_model = _FAMILY_DEFAULTS[family]
    parser.add_argument("--model", default=None, help=f"Default from config or {default_model}")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--phase1-dir", type=pathlib.Path, default=None)
    parser.add_argument("--k", type=int, default=None, help="Ablation batches sampled")
    parser.add_argument("--data-end", type=int, default=None, help="C4 sequences for activations")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-random", type=int, default=None)
    parser.add_argument("--n-random-pct", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--nullspace-top-pct",
        type=float,
        default=None,
        help="Fraction of d_model for null-space k (must match phase 1)",
    )
    if family in ("gpt2", "pythia"):
        parser.add_argument(
            "--nullspace-k",
            type=int,
            default=None,
            help="Must match phase 1 if re-running static metrics",
        )
    if family in _HF_DTYPE_FAMILIES:
        parser.add_argument(
            "--dtype",
            default=None,
            help="float16 | bfloat16 | float32 (default: float16 on cuda, float32 on cpu)",
        )
    parser.add_argument("--png", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from existing ablation.feather",
    )


def _cli_hparam_overrides(args: argparse.Namespace) -> dict[str, Any]:
    arg_to_cfg = {
        "entropy_top_pct": "entropy_top_pct",
        "entropy_top_n": "entropy_top_n",
        "nullspace_top_pct": "nullspace_top_pct",
        "nullspace_k": "udark_k",
        "dataset": "dataset",
        "data_end": "data_end",
        "max_seq_length": "max_seq_length",
        "seed": "seed",
        "k": "k",
        "batch_size": "batch_size",
        "n_random": "n_random",
        "n_random_pct": "n_random_pct",
        "chunk_size": "chunk_size",
        "filter_start_pos": "filter_start_pos",
        "filter_end_pos": "filter_end_pos",
        "n_trace_neurons": "n_trace_neurons",
        "box_plot_n_random": "box_plot_n_random",
        "write_align_n_random": "write_align_n_random",
        "write_align_mode": "write_align_mode",
        "use_log2_entropy": "use_log2_entropy",
    }
    overrides: dict[str, Any] = {}
    for arg_name, cfg_key in arg_to_cfg.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[cfg_key] = value
    return overrides


def build_model_config(args: argparse.Namespace, family: Family) -> tuple[ModelRunConfig, pathlib.Path]:
    """Merge YAML config with CLI; CLI wins when a flag is explicitly set."""
    settings = load_run_settings(args.config, family)
    _, default_model = _FAMILY_DEFAULTS[family]

    model = args.model or settings.get("model") or default_model
    out_dir = args.out_dir or settings.get("out_dir") or DEFAULT_OUTPUT_DIR
    dtype = getattr(args, "dtype", None)
    if dtype is None:
        dtype = settings.get("dtype")

    cfg = ModelRunConfig.from_hparams(
        model,
        settings["hparams"],
        dtype=dtype,
        **_cli_hparam_overrides(args),
    )
    return cfg, out_dir
