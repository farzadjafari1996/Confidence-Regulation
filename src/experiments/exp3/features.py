"""Feature-subset helpers for Exp3 probes (resid / last-MLP / entropy / matched-k)."""

from __future__ import annotations

import pathlib
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from exp3.settings import Exp3Config, FeatureSet
from hyperparameters import phase_output_dir

FeatureSource = Literal["mlp_last", "resid"]

_TOPK_RE = re.compile(r"^topk_(\d+)$")
_BOTTOM_RE = re.compile(r"^bottom_k(\d+)$")
_RANDOM_RE = re.compile(r"^random_k(\d+)_d(\d+)$")
_RANDOM_RESID_RE = re.compile(r"^random_resid_k(\d+)_d(\d+)$")

MLP_ONLY_FEATURES = frozenset({"all", "mlp_last", "entropy", "rest"})


def probe_dir_name(probe: str, features: FeatureSet) -> str:
    if features in {"all", "mlp_last"}:
        return probe
    return f"{probe}_{features}"


def mlp_probe_allowed(features: FeatureSet) -> bool:
    if features in MLP_ONLY_FEATURES:
        return True
    if parse_matched_k_features(str(features)) is not None:
        return True
    return False


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


def _validate_indices(idx: np.ndarray, *, dim: int, label: str) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        raise ValueError(f"{label}: empty feature index set")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{label}: duplicate indices")
    if idx.min() < 0 or idx.max() >= dim:
        raise ValueError(f"{label}: indices out of range for dim={dim}")
    return idx


def select_topk_by_rho(rho_df: pd.DataFrame, k: int, *, d_mlp: int) -> np.ndarray:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(rho_df):
        raise ValueError(f"k={k} exceeds n_neurons={len(rho_df)}")
    idx = rho_df.nlargest(k, "rho")["neuron_index"].to_numpy(dtype=np.int64)
    return _validate_indices(idx, dim=d_mlp, label=f"topk_{k}")


def select_bottom_by_rho(rho_df: pd.DataFrame, k: int, *, d_mlp: int) -> np.ndarray:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(rho_df):
        raise ValueError(f"k={k} exceeds n_neurons={len(rho_df)}")
    idx = rho_df.nsmallest(k, "rho")["neuron_index"].to_numpy(dtype=np.int64)
    return _validate_indices(idx, dim=d_mlp, label=f"bottom_k{k}")


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
    return _validate_indices(np.sort(idx), dim=d_mlp, label=f"random_k{k}")


def parse_matched_k_features(features: str) -> dict[str, Any] | None:
    """Parse last-MLP matched-k tags; return None if not a last-MLP matched-k tag."""
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
    n_resid: int | None = None,
    phase1_dir: pathlib.Path | None = None,
    device: str | None = None,
    skip_phase1: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return column indices and metadata. `source` is mlp_last or resid."""
    if features in {"all", "mlp_last"}:
        idx = np.arange(d_mlp, dtype=np.int64)
        return idx, {
            "feature_set": features,
            "source": "mlp_last",
            "n_features": int(d_mlp),
            "d_mlp": int(d_mlp),
            "entropy_indices": None,
            "entropy_neurons": None,
            "writer_layer": None,
            "phase1_dir": None,
        }

    if features == "resid_all":
        if n_resid is None:
            raise ValueError("resid_all requires n_resid")
        idx = np.arange(n_resid, dtype=np.int64)
        return idx, {
            "feature_set": "resid_all",
            "source": "resid",
            "n_features": int(n_resid),
            "d_mlp": int(d_mlp),
            "entropy_indices": None,
            "entropy_neurons": None,
            "writer_layer": None,
            "phase1_dir": None,
        }

    resid_match = _RANDOM_RESID_RE.match(str(features))
    if resid_match is not None:
        if n_resid is None:
            raise ValueError("random_resid requires n_resid")
        k = int(resid_match.group(1))
        draw = int(resid_match.group(2))
        rng = np.random.RandomState(int(cfg.seed) + 20_000 + draw)
        idx = np.sort(rng.choice(n_resid, size=k, replace=False).astype(np.int64))
        idx = _validate_indices(idx, dim=n_resid, label=str(features))
        return idx, {
            "feature_set": features,
            "source": "resid",
            "n_features": int(idx.size),
            "d_mlp": int(d_mlp),
            "matched_k": {"kind": "random_resid", "k": k, "draw": draw},
            "feature_indices": idx.tolist(),
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
        "source": "mlp_last",
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


def feature_matrix(
    data: dict[str, Any],
    cfg: Exp3Config,
    features: FeatureSet,
    *,
    phase1_dir: pathlib.Path | None = None,
    device: str | None = None,
    skip_phase1: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (X_subset, indices, meta) for the requested arm."""
    mlp = np.asarray(data["X_mlp_last"], dtype=np.float32)
    resid_shape = np.asarray(data["X_resid"]).shape
    n_resid = int(np.prod(resid_shape[1:]))
    idx, meta = resolve_feature_indices(
        cfg,
        features,
        d_mlp=mlp.shape[1],
        n_resid=n_resid,
        phase1_dir=phase1_dir,
        device=device,
        skip_phase1=skip_phase1,
    )
    source = meta.get("source", "mlp_last")
    if source == "resid":
        X_src = np.asarray(data["X_resid"], dtype=np.float32).reshape(mlp.shape[0], -1)
    else:
        X_src = mlp
    return select_features(X_src, idx), idx, meta
