"""Feature-arm inventory shared by train and evaluate."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from exp3.dataset_io import collection_ready, load_arrays
from exp3.settings import Exp3Config, ProbeName, SplitName
from hyperparameters import phase_output_dir


@dataclass(frozen=True)
class ArmSpec:
    name: str
    features: str
    probe: ProbeName = "logistic"


def probe_model_name(probe: ProbeName) -> str:
    return "best_model.joblib" if probe == "logistic" else "best_model.pt"


def probe_ready(
    cfg: Exp3Config, split_name: SplitName, probe: ProbeName, features: str
) -> bool:
    out = cfg.probe_out_dir(split_name, probe, features=features)
    return (out / probe_model_name(probe)).exists()


def list_arms(
    cfg: Exp3Config,
    k: int,
    *,
    n_random: int,
    skip_random: bool = False,
    include_mlp: bool | None = None,
    skip_resid: bool = True,
    include_bottom: bool | None = None,
    headline_only: bool = False,
    entropy_as_topk: bool = False,
) -> list[ArmSpec]:
    if headline_only:
        skip_resid = True
        skip_random = True
        include_mlp = False
        include_bottom = False
    include_mlp = cfg.matched_k.include_mlp if include_mlp is None else include_mlp
    include_bottom = (
        cfg.matched_k.include_bottom if include_bottom is None else include_bottom
    )
    entropy_features = f"topk_{k}" if entropy_as_topk else "entropy"
    arms: list[ArmSpec] = []
    if not skip_resid:
        arms.append(ArmSpec("resid_all", "resid_all"))
    arms.append(ArmSpec("mlp_last", "all"))
    arms.append(ArmSpec("entropy", entropy_features))
    if include_bottom:
        arms.append(ArmSpec("bottom_k", f"bottom_k{k}"))
    if not skip_random:
        for draw in range(n_random):
            arms.append(ArmSpec(f"random_mlp_d{draw:02d}", f"random_k{k}_d{draw:02d}"))
            if not skip_resid:
                arms.append(
                    ArmSpec(
                        f"random_resid_d{draw:02d}", f"random_resid_k{k}_d{draw:02d}"
                    )
                )
    if include_mlp:
        arms.append(ArmSpec("mlp_mlp_last", "all", probe="mlp"))
        arms.append(ArmSpec("mlp_entropy", entropy_features, probe="mlp"))
        if include_bottom:
            arms.append(ArmSpec("mlp_bottom_k", f"bottom_k{k}", probe="mlp"))
        if not skip_random:
            for draw in range(n_random):
                arms.append(
                    ArmSpec(
                        f"mlp_random_mlp_d{draw:02d}",
                        f"random_k{k}_d{draw:02d}",
                        probe="mlp",
                    )
                )
    return arms


def resolve_matched_k(
    cfg: Exp3Config,
    split_name: SplitName,
    *,
    device: str | None = None,
    phase1_dir: pathlib.Path | None = None,
    skip_phase1: bool = False,
    k: int | None = None,
    n_random: int | None = None,
) -> tuple[int, int, pathlib.Path]:
    """Return (k, n_random, phase1_dir). Requires gathered activations."""
    from exp3.ablate_entropy_features import ensure_phase1, load_entropy_neuron_indices

    data_dir = cfg.data_out_dir(split_name)
    if not collection_ready(data_dir):
        raise FileNotFoundError(
            f"Missing {data_dir}. Run exp3/gather_activations.py first."
        )
    data = load_arrays(data_dir)
    d_mlp = int(np.asarray(data["X_mlp_last"]).shape[1])
    resolved_phase1 = (
        pathlib.Path(phase1_dir)
        if phase1_dir is not None
        else phase_output_dir(cfg.model, "phase1", cfg.out_dir)
    )
    resolved_phase1 = ensure_phase1(
        cfg, resolved_phase1, device=device or cfg.device, skip_phase1=skip_phase1
    )
    if k is None:
        _, entropy_idx, _ = load_entropy_neuron_indices(resolved_phase1, d_mlp=d_mlp)
        k = int(len(entropy_idx))
    n_random = int(n_random if n_random is not None else cfg.matched_k.n_random)
    return int(k), n_random, resolved_phase1
