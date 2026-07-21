"""Feature-subset helpers for Exp3 probes (full / entropy-only / rest-only)."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from exp3.settings import Exp3Config, FeatureSet
from hyperparameters import phase_output_dir


def probe_dir_name(probe: str, features: FeatureSet) -> str:
    if features == "all":
        return probe
    return f"{probe}_{features}"


def resolve_feature_indices(
    cfg: Exp3Config,
    features: FeatureSet,
    *,
    d_mlp: int,
    phase1_dir: pathlib.Path | None = None,
    device: str | None = None,
    skip_phase1: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return column indices into X_full and metadata for the requested feature set."""
    if features == "all":
        idx = np.arange(d_mlp, dtype=np.int64)
        return idx, {
            "feature_set": "all",
            "n_features": int(d_mlp),
            "d_mlp": int(d_mlp),
            "entropy_indices": None,
            "entropy_neurons": None,
            "writer_layer": None,
            "phase1_dir": None,
        }

    # Lazy import avoids circular deps with ablate/evaluate/train modules.
    from exp3.ablate_entropy_features import ensure_phase1, load_entropy_neuron_indices

    resolved_phase1 = pathlib.Path(phase1_dir) if phase1_dir is not None else phase_output_dir(
        cfg.model, "phase1", cfg.out_dir
    )
    resolved_phase1 = ensure_phase1(
        cfg,
        resolved_phase1,
        device=device or cfg.device,
        skip_phase1=skip_phase1,
    )
    entropy_labels, entropy_idx, writer_layer = load_entropy_neuron_indices(
        resolved_phase1, d_mlp=d_mlp
    )
    if features == "entropy":
        idx = np.asarray(entropy_idx, dtype=np.int64)
    elif features == "rest":
        idx = np.setdiff1d(
            np.arange(d_mlp, dtype=np.int64),
            entropy_idx,
            assume_unique=False,
        )
    else:
        raise ValueError(f"Unknown feature set {features!r}")

    meta: dict[str, Any] = {
        "feature_set": features,
        "n_features": int(idx.size),
        "d_mlp": int(d_mlp),
        "entropy_indices": entropy_idx.tolist(),
        "entropy_neurons": entropy_labels,
        "writer_layer": writer_layer,
        "phase1_dir": str(resolved_phase1),
        "feature_indices": idx.tolist(),
    }
    return idx, meta


def select_features(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(X[:, indices], dtype=np.float32)
