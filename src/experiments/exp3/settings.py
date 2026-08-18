"""Exp 3 hallucination-probe configuration."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Literal

from hyperparameters import ROOT, load_config_yaml, model_output_dir, model_slug

SplitName = Literal["sanity", "main"]
LabelName = Literal[
    "top1",
    "top5",
    "confident_error",
    "multiclass",
    "gold_logprob",
    "option_logprob",
]
ProbeName = Literal["logistic", "mlp"]
# Built-ins: all | mlp_last | resid_all | entropy | rest
# Matched-k tags: topk_{k} | bottom_k{k} | random_k{k}_d{draw} | random_resid_k{k}_d{draw}
FeatureSet = str
DatasetName = Literal["qa_wikidata", "lama_trex"]

DATASETS: tuple[DatasetName, ...] = ("qa_wikidata", "lama_trex")


def resolved_datasets(dataset: str | None) -> list[DatasetName]:
    if dataset is None or dataset == "all":
        return list(DATASETS)
    name = str(dataset)
    if name not in DATASETS:
        raise ValueError(f"Unsupported Exp3 dataset: {dataset} (one of {DATASETS} or 'all')")
    return [name]  # type: ignore[list-item]


LABELS: tuple[LabelName, ...] = (
    "top1",
    "top5",
    "confident_error",
    "multiclass",
    "gold_logprob",
    "option_logprob",
)


@dataclass
class BigBenchSettings:
    hf_path: str = "tasksource/bigbench"
    config: str = "qa_wikidata"
    filter_single_token_gold: bool = True


@dataclass
class TrexSettings:
    hf_path: str = "facebook/lama"
    config: str = "trex"
    filter_single_token_gold: bool = True
    require_mask_at_end: bool = True


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
class MatchedKSettings:
    n_random: int = 10
    include_bottom: bool = True
    include_mlp: bool = False


@dataclass
class Exp3Config:
    model: str = "gpt2-small"
    family: str = "gpt2"
    use_chat_template: bool = False
    dataset: DatasetName = "qa_wikidata"
    out_dir: pathlib.Path = ROOT / "outputs"
    data_dir: pathlib.Path = ROOT / "data" / "exp3"
    seed: int = 42
    device: str = "cuda"
    dtype: str | None = None
    sample_sizes: dict[str, int] = field(
        default_factory=lambda: {"sanity": 200, "main": 10000}
    )
    split_ratios: SplitRatios = field(default_factory=SplitRatios)
    checkpoint_every: int = 250
    label_default: LabelName = "top1"
    confident_error_top1_prob: float = 0.5
    bigbench: BigBenchSettings = field(default_factory=BigBenchSettings)
    trex: TrexSettings = field(default_factory=TrexSettings)
    logistic: LogisticSettings = field(default_factory=LogisticSettings)
    mlp: MlpSettings = field(default_factory=MlpSettings)
    importance: ImportanceSettings = field(default_factory=ImportanceSettings)
    matched_k: MatchedKSettings = field(default_factory=MatchedKSettings)
    config_path: pathlib.Path | None = None

    def sample_size(self, split_name: SplitName) -> int:
        return int(self.sample_sizes[split_name])

    def study_dir(self, split_name: SplitName) -> pathlib.Path:
        """Probe / ablation artifacts under outputs/<slug>/exp3[_sanity]/<dataset>/."""
        base = model_output_dir(self.model, self.out_dir)
        name = "exp3_sanity" if split_name == "sanity" else "exp3"
        return base / name / self.dataset

    def data_out_dir(self, split_name: SplitName) -> pathlib.Path:
        """Collected activations under data/exp3/<slug>/<dataset>/<split>/."""
        return self.data_dir / model_slug(self.model) / self.dataset / split_name

    def probe_out_dir(
        self,
        split_name: SplitName,
        probe: ProbeName,
        *,
        features: FeatureSet = "all",
    ) -> pathlib.Path:
        from exp3.features import probe_dir_name

        return self.study_dir(split_name) / probe_dir_name(probe, features)

    def resolved_label(self, label: LabelName | None = None) -> LabelName:
        if label is not None:
            return label
        return self.label_default

    def effective_chat_template(self) -> bool:
        """Cloze probes use raw completion; chat wraps break next-token scoring."""
        return False


FAST_N = 64


def apply_runtime_overrides(
    cfg: Exp3Config,
    *,
    n: int | None = None,
    fast: bool = False,
    device: str | None = None,
) -> Exp3Config:
    """Cap sample sizes and shrink search/ablation knobs for smoke runs."""
    if device:
        cfg.device = device
    if (device or cfg.device) == "cuda" and not cfg.dtype:
        cfg.dtype = "float16"
    n_items = n if n is not None else (FAST_N if fast else None)
    if n_items is not None:
        cfg.sample_sizes["sanity"] = int(n_items)
        cfg.sample_sizes["main"] = int(n_items)
        cfg.checkpoint_every = min(cfg.checkpoint_every, max(8, int(n_items) // 4))
    if fast:
        cfg.logistic.n_iter = min(cfg.logistic.n_iter, 4)
        cfg.logistic.cv = min(cfg.logistic.cv, 2)
        cfg.logistic.max_iter = min(cfg.logistic.max_iter, 400)
        cfg.mlp.n_iter = min(cfg.mlp.n_iter, 2)
        cfg.mlp.max_epochs = min(cfg.mlp.max_epochs, 8)
        cfg.matched_k.n_random = min(cfg.matched_k.n_random, 2)
        cfg.importance.n_repeats = 1
    return cfg


def is_instruct_checkpoint(model: str) -> bool:
    name = model.lower()
    if name.endswith("-base") or name.endswith("_base"):
        return False
    if name in {"gpt2", "gpt2-small"} or "pythia" in name:
        return False
    if "instruct" in name or name.endswith("-it") or name.endswith("_it"):
        return True
    if "qwen" in name and not name.endswith("-base"):
        return True
    return False


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
    bb_raw = block.get("bigbench") or block.get("qa_wikidata") or {}
    trex_raw = block.get("trex") or block.get("lama_trex") or {}
    log_raw = block.get("logistic") or {}
    mlp_raw = block.get("mlp") or {}
    imp_raw = block.get("importance") or {}
    mk_raw = block.get("matched_k") or {}
    use_chat = bool(block.get("use_chat_template", False))

    dataset = str(block.get("dataset") or "qa_wikidata")
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported Exp3 dataset: {dataset} (one of {DATASETS})")

    return Exp3Config(
        model=str(raw.get("model") or "gpt2-small"),
        family=str(raw.get("family") or "gpt2"),
        use_chat_template=bool(use_chat),
        dataset=dataset,  # type: ignore[arg-type]
        out_dir=_resolve_path(raw.get("out_dir"), ROOT / "outputs"),
        data_dir=_resolve_path(block.get("data_dir"), ROOT / "data" / "exp3"),
        seed=int(block.get("seed", 42)),
        device=str(block.get("device", "cuda")),
        dtype=dtype,
        sample_sizes={
            "sanity": int(sizes.get("sanity", 200)),
            "main": int(sizes.get("main", 10000)),
        },
        split_ratios=SplitRatios(
            train=float(ratios_raw.get("train", 0.70)),
            val=float(ratios_raw.get("val", 0.15)),
            test=float(ratios_raw.get("test", 0.15)),
        ),
        checkpoint_every=int(block.get("checkpoint_every", 250)),
        label_default=str(block.get("label_default", "top1")),  # type: ignore[arg-type]
        confident_error_top1_prob=float(block.get("confident_error_top1_prob", 0.5)),
        bigbench=BigBenchSettings(
            hf_path=str(bb_raw.get("hf_path", "tasksource/bigbench")),
            config=str(bb_raw.get("config", "qa_wikidata")),
            filter_single_token_gold=bool(bb_raw.get("filter_single_token_gold", True)),
        ),
        trex=TrexSettings(
            hf_path=str(trex_raw.get("hf_path", "facebook/lama")),
            config=str(trex_raw.get("config", "trex")),
            filter_single_token_gold=bool(trex_raw.get("filter_single_token_gold", True)),
            require_mask_at_end=bool(trex_raw.get("require_mask_at_end", True)),
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
        matched_k=MatchedKSettings(
            n_random=int(mk_raw.get("n_random", 10)),
            include_bottom=bool(mk_raw.get("include_bottom", True)),
            include_mlp=bool(mk_raw.get("include_mlp", False)),
        ),
        config_path=path.resolve(),
    )
