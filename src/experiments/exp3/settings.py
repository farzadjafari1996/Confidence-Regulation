"""Ext 3 hallucination-probe configuration."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Literal

from hyperparameters import ROOT, load_config_yaml, model_output_dir

SplitName = Literal["sanity", "main"]
LabelName = Literal["top1", "top5", "confident_error", "multiclass", "gold_logprob"]
ProbeName = Literal["logistic", "mlp"]


@dataclass
class LamaSettings:
    hf_path: str = "facebook/lama"
    config: str = "trex"
    filter_single_token_gold: bool = True


@dataclass
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


@dataclass
class LogisticSettings:
    n_iter: int = 40
    cv: int = 3
    max_iter: int = 2000


@dataclass
class MlpSettings:
    n_iter: int = 20
    max_epochs: int = 80
    batch_size: int = 64
    patience: int = 10


@dataclass
class ImportanceSettings:
    n_repeats: int = 3
    top_k: int = 40


@dataclass
class Exp3Config:
    model: str = "gpt2-small"
    family: str = "gpt2"
    out_dir: pathlib.Path = ROOT / "outputs"
    data_dir: pathlib.Path = ROOT / "data" / "exp3"
    seed: int = 42
    device: str = "cuda"
    dtype: str | None = None
    sample_sizes: dict[str, int] = field(
        default_factory=lambda: {"sanity": 50, "main": 3000}
    )
    split_ratios: SplitRatios = field(default_factory=SplitRatios)
    checkpoint_every: int = 50
    label_default: LabelName = "top1"
    confident_error_top1_prob: float = 0.5
    lama: LamaSettings = field(default_factory=LamaSettings)
    logistic: LogisticSettings = field(default_factory=LogisticSettings)
    mlp: MlpSettings = field(default_factory=MlpSettings)
    importance: ImportanceSettings = field(default_factory=ImportanceSettings)
    config_path: pathlib.Path | None = None

    def sample_size(self, split_name: SplitName) -> int:
        return int(self.sample_sizes[split_name])

    def study_dir(self, split_name: SplitName) -> pathlib.Path:
        base = model_output_dir(self.model, self.out_dir)
        name = "exp3_sanity" if split_name == "sanity" else "exp3"
        return base / name

    def data_out_dir(self, split_name: SplitName) -> pathlib.Path:
        return self.study_dir(split_name) / "data"

    def probe_out_dir(self, split_name: SplitName, probe: ProbeName) -> pathlib.Path:
        return self.study_dir(split_name) / probe


def _resolve_path(value: Any, default: pathlib.Path) -> pathlib.Path:
    if value is None:
        return default
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_exp3_config(config_path: pathlib.Path | str) -> Exp3Config:
    path = pathlib.Path(config_path)
    raw = load_config_yaml(path)
    block = raw.get("exp3") if isinstance(raw.get("exp3"), dict) else {}

    dtype = block.get("dtype", raw.get("dtype"))
    if isinstance(dtype, str) and dtype.lower() in {"null", "none", ""}:
        dtype = None

    sizes = block.get("sample_sizes") or {}
    ratios_raw = block.get("split_ratios") or {}
    lama_raw = block.get("lama") or {}
    log_raw = block.get("logistic") or {}
    mlp_raw = block.get("mlp") or {}
    imp_raw = block.get("importance") or {}

    return Exp3Config(
        model=str(raw.get("model") or "gpt2-small"),
        family=str(raw.get("family") or "gpt2"),
        out_dir=_resolve_path(raw.get("out_dir"), ROOT / "outputs"),
        data_dir=_resolve_path(block.get("data_dir"), ROOT / "data" / "exp3"),
        seed=int(block.get("seed", 42)),
        device=str(block.get("device", "cuda")),
        dtype=dtype,
        sample_sizes={
            "sanity": int(sizes.get("sanity", 50)),
            "main": int(sizes.get("main", 3000)),
        },
        split_ratios=SplitRatios(
            train=float(ratios_raw.get("train", 0.70)),
            val=float(ratios_raw.get("val", 0.15)),
            test=float(ratios_raw.get("test", 0.15)),
        ),
        checkpoint_every=int(block.get("checkpoint_every", 50)),
        label_default=str(block.get("label_default", "top1")),  # type: ignore[arg-type]
        confident_error_top1_prob=float(block.get("confident_error_top1_prob", 0.5)),
        lama=LamaSettings(
            hf_path=str(lama_raw.get("hf_path", "facebook/lama")),
            config=str(lama_raw.get("config", "trex")),
            filter_single_token_gold=bool(lama_raw.get("filter_single_token_gold", True)),
        ),
        logistic=LogisticSettings(
            n_iter=int(log_raw.get("n_iter", 40)),
            cv=int(log_raw.get("cv", 3)),
            max_iter=int(log_raw.get("max_iter", 2000)),
        ),
        mlp=MlpSettings(
            n_iter=int(mlp_raw.get("n_iter", 20)),
            max_epochs=int(mlp_raw.get("max_epochs", 80)),
            batch_size=int(mlp_raw.get("batch_size", 64)),
            patience=int(mlp_raw.get("patience", 10)),
        ),
        importance=ImportanceSettings(
            n_repeats=int(imp_raw.get("n_repeats", 3)),
            top_k=int(imp_raw.get("top_k", 40)),
        ),
        config_path=path.resolve(),
    )
