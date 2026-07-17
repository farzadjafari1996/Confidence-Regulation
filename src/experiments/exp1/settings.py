"""Ext 1 QA ablation configuration."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Literal

from hyperparameters import ROOT, load_config_yaml, model_output_dir

AblationMode = Literal["mean_all_entropy", "per_neuron_avg"]
AblationBranch = Literal["te", "de_ln"]
DecodeMode = Literal["greedy", "sampling"]
DatasetName = Literal["lama", "triviaqa"]
SplitName = Literal["sanity", "main"]
Family = Literal["gpt2", "llama", "qwen"]


@dataclass
class ModelSpec:
    family: Family
    model: str
    use_chat_template: bool = False


@dataclass
class CalibrationSettings:
    dataset: str = "stas/c4-en-10k"
    data_end: int = 200
    max_seq_length: int = 256


@dataclass
class AblationSettings:
    include_random_control: bool = True
    sanity_modes: list[AblationMode] = field(
        default_factory=lambda: ["mean_all_entropy", "per_neuron_avg"]
    )
    main_modes: list[AblationMode] = field(default_factory=lambda: ["mean_all_entropy"])
    branches: list[AblationBranch] = field(default_factory=lambda: ["te", "de_ln"])
    chunk_size: int = 20


@dataclass
class LamaDatasetSettings:
    hf_path: str = "facebook/lama"
    config: str = "trex"
    filter_single_token_gold: bool = True
    max_new_tokens: int = 32


@dataclass
class TriviaQADatasetSettings:
    hf_path: str = "mandarjoshi/trivia_qa"
    config: str = "rc.nocontext"
    split: str = "validation"
    max_new_tokens: int = 64
    prompt_template: str = "Question: {question}\nAnswer:"


@dataclass
class DecodeGreedySettings:
    do_sample: bool = False


@dataclass
class DecodeSamplingSettings:
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class DecodingSettings:
    modes: list[DecodeMode] = field(default_factory=lambda: ["greedy", "sampling"])
    greedy: DecodeGreedySettings = field(default_factory=DecodeGreedySettings)
    sampling: DecodeSamplingSettings = field(default_factory=DecodeSamplingSettings)


@dataclass
class Exp1Config:
    out_dir: pathlib.Path = ROOT / "outputs"
    data_dir: pathlib.Path = ROOT / "data" / "exp1"
    seed: int = 42
    models: list[ModelSpec] = field(default_factory=list)
    sample_sizes: dict[str, int] = field(default_factory=lambda: {"sanity": 20, "main": 500})
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    ablation: AblationSettings = field(default_factory=AblationSettings)
    lama: LamaDatasetSettings = field(default_factory=LamaDatasetSettings)
    triviaqa: TriviaQADatasetSettings = field(default_factory=TriviaQADatasetSettings)
    decoding: DecodingSettings = field(default_factory=DecodingSettings)
    device: str = "cuda"
    dtype: str | None = None
    config_path: pathlib.Path | None = None

    def sample_size(self, split_name: SplitName) -> int:
        return int(self.sample_sizes[split_name])

    def ablation_modes(self, split_name: SplitName) -> list[AblationMode]:
        if split_name == "sanity":
            return list(self.ablation.sanity_modes)
        return list(self.ablation.main_modes)

    def model_spec(self) -> ModelSpec:
        if not self.models:
            raise ValueError("Exp1Config has no model defined")
        return self.models[0]


def _resolve_path(value: Any, default: pathlib.Path) -> pathlib.Path:
    if value is None:
        return default
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def exp1_study_dir(out_dir: pathlib.Path, split_name: SplitName) -> str:
    return "exp1_sanity" if split_name == "sanity" else "exp1"


def exp1_dataset_dir(out_dir: pathlib.Path, split_name: SplitName, dataset: DatasetName) -> pathlib.Path:
    return out_dir / exp1_study_dir(out_dir, split_name) / dataset


def _parse_exp1_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Read ext1 QA section (legacy: phase3 block with datasets/calibration)."""
    if "exp1" in raw:
        return raw["exp1"]
    p3 = raw.get("phase3") or {}
    if "datasets" in p3 or "calibration" in p3:
        return p3
    return {}


def _parse_models(raw: dict[str, Any], exp1_raw: dict[str, Any]) -> list[ModelSpec]:
    models_raw = raw.get("models") or []
    if models_raw:
        return [
            ModelSpec(
                family=m["family"],
                model=m["model"],
                use_chat_template=bool(m.get("use_chat_template", False)),
            )
            for m in models_raw
        ]
    if raw.get("model"):
        return [
            ModelSpec(
                family=raw["family"],
                model=raw["model"],
                use_chat_template=bool(exp1_raw.get("use_chat_template", False)),
            )
        ]
    return []


def load_exp1_config(path: pathlib.Path | str | None = None) -> Exp1Config:
    cfg_path = pathlib.Path(path) if path is not None else ROOT / "config" / "gpt2.yaml"
    raw = load_config_yaml(cfg_path) if cfg_path.exists() else {}
    exp1_raw = _parse_exp1_raw(raw)
    models = _parse_models(raw, exp1_raw)

    cal_raw = exp1_raw.get("calibration") or {}
    calibration = CalibrationSettings(
        dataset=cal_raw.get("dataset", "stas/c4-en-10k"),
        data_end=int(cal_raw.get("data_end", 200)),
        max_seq_length=int(cal_raw.get("max_seq_length", 256)),
    )

    abl_raw = exp1_raw.get("ablation") or {}
    ablation = AblationSettings(
        include_random_control=bool(abl_raw.get("include_random_control", True)),
        sanity_modes=list(abl_raw.get("sanity_modes", ["mean_all_entropy", "per_neuron_avg"])),
        main_modes=list(abl_raw.get("main_modes", ["mean_all_entropy"])),
        chunk_size=int(abl_raw.get("chunk_size", 20)),
        branches=list(abl_raw.get("branches", ["te", "de_ln"])),
    )

    lama_raw = (exp1_raw.get("datasets") or {}).get("lama") or {}
    lama = LamaDatasetSettings(
        hf_path=lama_raw.get("hf_path", "facebook/lama"),
        config=lama_raw.get("config", "trex"),
        filter_single_token_gold=bool(lama_raw.get("filter_single_token_gold", True)),
        max_new_tokens=int(lama_raw.get("max_new_tokens", 32)),
    )

    trivia_raw = (exp1_raw.get("datasets") or {}).get("triviaqa") or {}
    triviaqa = TriviaQADatasetSettings(
        hf_path=trivia_raw.get("hf_path", "mandarjoshi/trivia_qa"),
        config=trivia_raw.get("config", "rc.nocontext"),
        split=trivia_raw.get("split", "validation"),
        max_new_tokens=int(trivia_raw.get("max_new_tokens", 64)),
        prompt_template=trivia_raw.get("prompt_template", "Question: {question}\nAnswer:"),
    )

    dec_raw = exp1_raw.get("decoding") or {}
    greedy_raw = dec_raw.get("greedy") or {}
    sampling_raw = dec_raw.get("sampling") or {}
    decoding = DecodingSettings(
        modes=list(dec_raw.get("modes", ["greedy", "sampling"])),
        greedy=DecodeGreedySettings(do_sample=bool(greedy_raw.get("do_sample", False))),
        sampling=DecodeSamplingSettings(
            do_sample=bool(sampling_raw.get("do_sample", True)),
            temperature=float(sampling_raw.get("temperature", 1.0)),
            top_p=float(sampling_raw.get("top_p", 1.0)),
        ),
    )

    runtime = exp1_raw.get("runtime") or {}
    dtype = runtime.get("dtype")
    if isinstance(dtype, str) and dtype.lower() in {"null", "none", ""}:
        dtype = None

    seed = int(exp1_raw.get("seed", raw.get("seed", 42)))
    spec = models[0] if models else None
    out_base = _resolve_path(raw.get("out_dir"), ROOT / "outputs")
    out_dir = model_output_dir(spec.model, out_base) if spec else out_base

    return Exp1Config(
        out_dir=out_dir,
        data_dir=_resolve_path(exp1_raw.get("data_dir"), ROOT / "data" / "exp1"),
        seed=seed,
        models=models,
        sample_sizes={
            "sanity": int((exp1_raw.get("sample_sizes") or {}).get("sanity", 20)),
            "main": int((exp1_raw.get("sample_sizes") or {}).get("main", 500)),
        },
        calibration=calibration,
        ablation=ablation,
        lama=lama,
        triviaqa=triviaqa,
        decoding=decoding,
        device=str(runtime.get("device", "cuda")),
        dtype=dtype,
        config_path=cfg_path.resolve() if cfg_path.exists() else None,
    )


# Backward-compatible aliases (deprecated)
Phase3bConfig = Exp1Config
load_phase3b_config = load_exp1_config
phase3_study_dir = exp1_study_dir
phase3_dataset_dir = exp1_dataset_dir
