"""Phase 3 (paper Fig 4–5) configuration."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

from hyperparameters import ROOT, GPT2_PAPER_ENTROPY_NEURONS, load_config_yaml


@dataclass
class NeuronActivitySettings:
    target_neuron: str = "11.2378"
    k_batches: int = 10


@dataclass
class InductionSettings:
    num_sequences: int = 100
    seq_length: int = 100
    ablation_k: int = 50
    bos_seq_length: int = 50
    bos_num_examples: int = 100
    induction_heads: list[str] = field(
        default_factory=lambda: ["L5H1", "L5H5", "L6H9"]
    )
    joint_induction_heads: list[str] = field(
        default_factory=lambda: ["L5H1-L5H5", "L5H1-L5H5-L6H9"]
    )
    bos_baselines: list[str] = field(
        default_factory=lambda: ["L6H1", "L5H7", "L5H6", "L5H8", "L6H10", "L6H3"]
    )
    bos_target_neuron: str = "11.2378"
    trace_neurons: list[str] = field(default_factory=lambda: list(GPT2_PAPER_ENTROPY_NEURONS))


@dataclass
class Phase3Config:
    neuron_activity: NeuronActivitySettings = field(default_factory=NeuronActivitySettings)
    induction: InductionSettings = field(default_factory=InductionSettings)
    seed: int = 42
    config_path: pathlib.Path | None = None


def load_phase3_config(path: pathlib.Path | str | None = None) -> Phase3Config:
    cfg_path = pathlib.Path(path) if path is not None else ROOT / "config" / "gpt2.yaml"
    raw = load_config_yaml(cfg_path) if cfg_path.exists() else {}
    p3_raw: dict[str, Any] = raw.get("phase3") or {}

    na_raw = p3_raw.get("neuron_activity") or {}
    ind_raw = p3_raw.get("induction") or {}

    neuron_activity = NeuronActivitySettings(
        target_neuron=str(na_raw.get("target_neuron", "11.2378")),
        k_batches=int(na_raw.get("k_batches", 10)),
    )
    induction = InductionSettings(
        num_sequences=int(ind_raw.get("num_sequences", 100)),
        seq_length=int(ind_raw.get("seq_length", 100)),
        ablation_k=int(ind_raw.get("ablation_k", 50)),
        bos_seq_length=int(ind_raw.get("bos_seq_length", 50)),
        bos_num_examples=int(ind_raw.get("bos_num_examples", 100)),
        induction_heads=list(ind_raw.get("induction_heads", InductionSettings().induction_heads)),
        joint_induction_heads=list(
            ind_raw.get("joint_induction_heads", InductionSettings().joint_induction_heads)
        ),
        bos_baselines=list(ind_raw.get("bos_baselines", InductionSettings().bos_baselines)),
        bos_target_neuron=str(ind_raw.get("bos_target_neuron", "11.2378")),
        trace_neurons=list(ind_raw.get("trace_neurons", GPT2_PAPER_ENTROPY_NEURONS)),
    )
    seed = int(p3_raw.get("seed", raw.get("seed", 42)))
    return Phase3Config(
        neuron_activity=neuron_activity,
        induction=induction,
        seed=seed,
        config_path=cfg_path.resolve() if cfg_path.exists() else None,
    )
