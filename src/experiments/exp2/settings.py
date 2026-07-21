"""Exp 2 weight-projection configuration."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

from hyperparameters import (
    GPT2_HPARAMS,
    ROOT,
    load_config_yaml,
    model_output_dir,
    nullspace_k_from_pct_and_fixed,
)


@dataclass
class Exp2Config:
    """Minimal Exp 2 settings (static MLP→MLP ρ analysis)."""

    model: str = "gpt2-small"
    family: str = "gpt2"
    out_dir: pathlib.Path = ROOT / "outputs"
    seed: int = 42
    nullspace_top_pct: float = GPT2_HPARAMS.nullspace_top_pct
    nullspace_k: int | None = GPT2_HPARAMS.nullspace_k
    n_trace_neurons: int = 6
    n_align_random: int = 15
    dtype: str | None = None
    device: str = "cuda"
    config_path: pathlib.Path | None = None

    def study_dir(self) -> pathlib.Path:
        return model_output_dir(self.model, self.out_dir) / "exp2"

    def effective_k(self, d_model: int) -> int:
        return nullspace_k_from_pct_and_fixed(d_model, self.nullspace_top_pct, self.nullspace_k)

    def udark_start(self, d_model: int) -> int:
        return -self.effective_k(d_model)


def _resolve_path(value: Any, default: pathlib.Path) -> pathlib.Path:
    if value is None:
        return default
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_exp2_config(config_path: pathlib.Path | str) -> Exp2Config:
    """Load Exp 2 block from a family YAML; inherit nullspace knobs from hyperparameters."""
    path = pathlib.Path(config_path)
    raw = load_config_yaml(path)
    hp = raw.get("hyperparameters") if isinstance(raw.get("hyperparameters"), dict) else {}
    block = raw.get("exp2") if isinstance(raw.get("exp2"), dict) else {}

    nullspace_top_pct = block.get("nullspace_top_pct", hp.get("nullspace_top_pct", GPT2_HPARAMS.nullspace_top_pct))
    nullspace_k = block.get("nullspace_k", hp.get("nullspace_k", GPT2_HPARAMS.nullspace_k))
    if isinstance(nullspace_k, str) and nullspace_k.lower() in {"null", "none", ""}:
        nullspace_k = None

    dtype = block.get("dtype", raw.get("dtype"))
    if isinstance(dtype, str) and dtype.lower() in {"null", "none", ""}:
        dtype = None

    return Exp2Config(
        model=str(raw.get("model") or "gpt2-small"),
        family=str(raw.get("family") or "gpt2"),
        out_dir=_resolve_path(raw.get("out_dir"), ROOT / "outputs"),
        seed=int(block.get("seed", hp.get("seed", 42))),
        nullspace_top_pct=float(nullspace_top_pct),
        nullspace_k=int(nullspace_k) if nullspace_k is not None else None,
        n_trace_neurons=int(block.get("n_trace_neurons", 6)),
        n_align_random=int(block.get("n_align_random", 15)),
        dtype=dtype,
        device=str(block.get("device", "cuda")),
        config_path=path.resolve(),
    )
