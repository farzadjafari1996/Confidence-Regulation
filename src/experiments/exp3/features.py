"""Feature-subset helpers for Exp3 probes (full / entropy / rest / matched-k)."""

from __future__ import annotations

import pathlib
import re
from typing import Any

import numpy as np
import pandas as pd

from exp3.settings import Exp3Config, FeatureSet
from hyperparameters import phase_output_dir

_TOPK_RE = re.compile(r"^topk_(\d+)$")
_BOTTOM_RE = re.compile(r"^bottom_k(\d+)$")
_RANDOM_RE = re.compile(r"^random_k(\d+)_d(\d+)$")


def probe_dir_name(probe: str, features: FeatureSet) -> str:
    if features == "all":
        return probe
    return f"{probe}_{features}"


def load_rho_table(phase1_dir: pathlib.Path) -> pd.DataFrame:
    path = pathlib.Path(phase1_dir) / "static_neuron_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    required = {"neuron_index", "rho"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    out = df.copy()
    out["neuron_index"] = out["neuron_index"].astype(np.int64)
    out["rho"] = out["rho"].astype(np.float64)
    return out.sort_values("rho", ascending=False).reset_index(drop=True)


def _validate_indices(idx: np.ndarray, *, d_mlp: int, label: str) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        raise ValueError(f"{label}: empty feature index set")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{label}: duplicate indices")
    if idx.min() < 0 or idx.max() >= d_mlp:
        raise ValueError(f"{label}: indices out of range for d_mlp={d_mlp}")
    return idx


def select_topk_by_rho(rho_df: pd.DataFrame, k: int, *, d_mlp: int) -> np.ndarray:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(rho_df):
        raise ValueError(f"k={k} exceeds n_neurons={len(rho_df)}")
    idx = rho_df.nlargest(k, "rho")["neuron_index"].to_numpy(dtype=np.int64)
    return _validate_indices(idx, d_mlp=d_mlp, label=f"topk_{k}")


def select_bottom_by_rho(rho_df: pd.DataFrame, k: int, *, d_mlp: int) -> np.ndarray:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(rho_df):
        raise ValueError(f"k={k} exceeds n_neurons={len(rho_df)}")
    idx = rho_df.nsmallest(k, "rho")["neuron_index"].to_numpy(dtype=np.int64)
    return _validate_indices(idx, d_mlp=d_mlp, label=f"bottom_k{k}")


def select_random_non_entropy(
    *,
    d_mlp: int,
    entropy_idx: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    pool = np.setdiff1d(
        np.arange(d_mlp, dtype=np.int64),
        np.asarray(entropy_idx, dtype=np.int64),
        assume_unique=False,
    )
    if k > pool.size:
        raise ValueError(f"k={k} exceeds non-entropy pool size={pool.size}")
    rng = np.random.RandomState(seed)
    idx = rng.choice(pool, size=k, replace=False)
    return _validate_indices(np.sort(idx), d_mlp=d_mlp, label=f"random_k{k}")


def parse_matched_k_features(features: str) -> dict[str, Any] | None:
    """Parse matched-k feature tags; return None if not a matched-k tag."""
    m = _TOPK_RE.match(features)
    if m:
        return {"kind": "topk", "k": int(m.group(1))}
    m = _BOTTOM_RE.match(features)
    if m:
        return {"kind": "bottom", "k": int(m.group(1))}
    m = _RANDOM_RE.match(features)
    if m:
        return {"kind": "random", "k": int(m.group(1)), "draw": int(m.group(2))}
    return None


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
    entropy_idx = np.asarray(entropy_idx, dtype=np.int64)

    matched = parse_matched_k_features(str(features))
    if features == "entropy":
        idx = entropy_idx
    elif features == "rest":
        idx = np.setdiff1d(
            np.arange(d_mlp, dtype=np.int64),
            entropy_idx,
            assume_unique=False,
        )
    elif matched is not None:
        rho_df = load_rho_table(resolved_phase1)
        kind = matched["kind"]
        k = int(matched["k"])
        if kind == "topk":
            idx = select_topk_by_rho(rho_df, k, d_mlp=d_mlp)
        elif kind == "bottom":
            idx = select_bottom_by_rho(rho_df, k, d_mlp=d_mlp)
        else:
            draw = int(matched["draw"])
            # Deterministic per-draw seed; keep draws disjoint from cfg.seed base.
            idx = select_random_non_entropy(
                d_mlp=d_mlp,
                entropy_idx=entropy_idx,
                k=k,
                seed=int(cfg.seed) + 10_000 + draw,
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
    if matched is not None:
        meta["matched_k"] = matched
        if matched["kind"] in {"topk", "bottom"}:
            rho_map = load_rho_table(resolved_phase1).set_index("neuron_index")["rho"]
            meta["rho_values"] = [float(rho_map.loc[int(i)]) for i in idx]
    return idx, meta


def select_features(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(X[:, indices], dtype=np.float32)
