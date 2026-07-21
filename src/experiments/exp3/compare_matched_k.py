"""Matched-k MLP probe comparison: entropy vs random vs bottom vs top-k rho sweep.

Trains fresh MLPs on fixed-size neuron subsets (unlike frozen full-probe ablation)
and writes a summary with the entropy AUROC percentile vs random-k null draws.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from exp3.evaluate import run_evaluate
from exp3.features import load_rho_table, select_topk_by_rho
from exp3.settings import Exp3Config, LabelName, SplitName
from exp3.train_mlp import run_train_mlp
from hyperparameters import phase_output_dir


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _test_metrics_path(cfg: Exp3Config, split_name: SplitName, features: str) -> pathlib.Path:
    return cfg.probe_out_dir(split_name, "mlp", features=features) / "test_metrics.json"


def _arm_ready(cfg: Exp3Config, split_name: SplitName, features: str) -> bool:
    out = cfg.probe_out_dir(split_name, "mlp", features=features)
    return (out / "best_model.pt").exists() and (out / "test_metrics.json").exists()


def _train_and_eval(
    cfg: Exp3Config,
    split_name: SplitName,
    features: str,
    *,
    label: LabelName,
    device: str,
    skip_phase1: bool,
    phase1_dir: pathlib.Path | None,
    force: bool,
) -> dict[str, Any]:
    out = cfg.probe_out_dir(split_name, "mlp", features=features)
    if force or not (out / "best_model.pt").exists():
        run_train_mlp(
            cfg,
            split_name,
            label=label,
            features=features,
            device=device,
            phase1_dir=phase1_dir,
            skip_phase1=skip_phase1,
        )
    if force or not (out / "test_metrics.json").exists():
        run_evaluate(
            cfg,
            split_name,
            "mlp",
            label=label,
            features=features,
            skip_importance=True,
        )
    payload = _read_json(out / "test_metrics.json")
    test = payload.get("test") or {}
    return {
        "features": features,
        "d_in": payload.get("d_in"),
        "accuracy": test.get("accuracy"),
        "f1": test.get("f1"),
        "auroc": test.get("auroc"),
        "path": str(out),
    }


def _percentile_of_score(scores: list[float], value: float) -> float:
    """Percentile of `value` within `scores` (fraction of scores <= value) * 100."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(100.0 * np.mean(arr <= value))


def run_matched_k_comparison(
    cfg: Exp3Config,
    split_name: SplitName = "main",
    *,
    label: LabelName | None = None,
    device: str | None = None,
    phase1_dir: pathlib.Path | None = None,
    skip_phase1: bool = True,
    k: int = 12,
    k_list: list[int] | None = None,
    n_random: int = 20,
    force: bool = False,
    skip_random: bool = False,
    skip_topk: bool = False,
    mlp_n_iter: int | None = None,
) -> pathlib.Path:
    label = label or cfg.label_default
    device = device or cfg.device
    if mlp_n_iter is not None:
        cfg.mlp.n_iter = int(mlp_n_iter)
    k_list = list(k_list) if k_list is not None else [4, 8, 12, 16, 20]
    if k not in k_list:
        k_list = sorted(set(k_list) | {k})

    resolved_phase1 = pathlib.Path(phase1_dir) if phase1_dir is not None else phase_output_dir(
        cfg.model, "phase1", cfg.out_dir
    )
    # Confirm Phase-1 top-k matches entropy set when k equals entropy count.
    data_dir = cfg.data_out_dir(split_name)
    from exp3.dataset_io import dataset_path, load_arrays

    data_path = dataset_path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run collect first.")
    d_mlp = int(np.asarray(load_arrays(data_path)["X_full"]).shape[1])
    rho_df = load_rho_table(resolved_phase1)
    topk_idx = select_topk_by_rho(rho_df, k, d_mlp=d_mlp)

    rows: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}

    # --- entropy (reuse existing mlp_entropy when present) ---
    entropy_features = "entropy"
    if force or not _arm_ready(cfg, split_name, entropy_features):
        arms["entropy_k12"] = _train_and_eval(
            cfg,
            split_name,
            entropy_features,
            label=label,
            device=device,
            skip_phase1=skip_phase1,
            phase1_dir=resolved_phase1,
            force=force,
        )
    else:
        payload = _read_json(_test_metrics_path(cfg, split_name, entropy_features))
        test = payload.get("test") or {}
        arms["entropy_k12"] = {
            "features": entropy_features,
            "d_in": payload.get("d_in"),
            "accuracy": test.get("accuracy"),
            "f1": test.get("f1"),
            "auroc": test.get("auroc"),
            "path": str(cfg.probe_out_dir(split_name, "mlp", features=entropy_features)),
            "reused": True,
        }
    rows.append({"arm": "entropy_k12", **arms["entropy_k12"]})

    # --- bottom-k ---
    bottom_features = f"bottom_k{k}"
    arms["bottom_k12"] = _train_and_eval(
        cfg,
        split_name,
        bottom_features,
        label=label,
        device=device,
        skip_phase1=skip_phase1,
        phase1_dir=resolved_phase1,
        force=force,
    )
    rows.append({"arm": "bottom_k12", **arms["bottom_k12"]})

    # --- top-k sweep (topk_12 aliases entropy when indices match) ---
    topk_arms: dict[str, Any] = {}
    if not skip_topk:
        entropy_idx_path = cfg.probe_out_dir(split_name, "mlp", features="entropy") / "train_metrics.json"
        entropy_idx: list[int] | None = None
        if entropy_idx_path.exists():
            em = _read_json(entropy_idx_path)
            entropy_idx = list((em.get("feature_meta") or {}).get("feature_indices") or [])

        for kk in k_list:
            tag = f"topk_{kk}"
            # Alias to entropy when identical to Phase-1 / entropy index set.
            if (
                kk == k
                and entropy_idx is not None
                and set(entropy_idx) == set(int(x) for x in topk_idx.tolist())
                and _arm_ready(cfg, split_name, "entropy")
            ):
                topk_arms[tag] = {
                    **arms["entropy_k12"],
                    "features": "entropy",
                    "alias_of": "entropy",
                    "k": kk,
                }
            else:
                topk_arms[tag] = _train_and_eval(
                    cfg,
                    split_name,
                    tag,
                    label=label,
                    device=device,
                    skip_phase1=skip_phase1,
                    phase1_dir=resolved_phase1,
                    force=force,
                )
                topk_arms[tag]["k"] = kk
            rows.append({"arm": tag, **topk_arms[tag]})
    arms["topk"] = topk_arms

    # --- random-k null draws ---
    random_rows: list[dict[str, Any]] = []
    if not skip_random:
        for draw in range(n_random):
            tag = f"random_k{k}_d{draw:02d}"
            row = _train_and_eval(
                cfg,
                split_name,
                tag,
                label=label,
                device=device,
                skip_phase1=skip_phase1,
                phase1_dir=resolved_phase1,
                force=force,
            )
            row["draw"] = draw
            random_rows.append(row)
            rows.append({"arm": tag, **row})

    random_aurocs = [float(r["auroc"]) for r in random_rows if r.get("auroc") is not None]
    entropy_auroc = arms["entropy_k12"].get("auroc")
    bottom_auroc = arms["bottom_k12"].get("auroc")
    null_summary: dict[str, Any] = {
        "n_random": len(random_aurocs),
        "auroc_mean": float(np.mean(random_aurocs)) if random_aurocs else None,
        "auroc_std": float(np.std(random_aurocs)) if random_aurocs else None,
        "auroc_min": float(np.min(random_aurocs)) if random_aurocs else None,
        "auroc_max": float(np.max(random_aurocs)) if random_aurocs else None,
        "entropy_percentile": (
            _percentile_of_score(random_aurocs, float(entropy_auroc))
            if entropy_auroc is not None and random_aurocs
            else None
        ),
        "bottom_percentile": (
            _percentile_of_score(random_aurocs, float(bottom_auroc))
            if bottom_auroc is not None and random_aurocs
            else None
        ),
    }
    arms["random_k12"] = {"draws": random_rows, "null_summary": null_summary}

    summary = {
        "model": cfg.model,
        "family": cfg.family,
        "split": split_name,
        "label": label,
        "probe": "mlp",
        "k": k,
        "k_list": k_list,
        "n_random": n_random,
        "phase1_dir": str(resolved_phase1),
        "d_mlp": d_mlp,
        "note": (
            "Fresh MLP probes trained on matched-k feature subsets. "
            "Unlike frozen full-probe ablation, each arm only sees its k dims at train time. "
            "topk_12 aliases entropy when Phase-1 selection equals top-12 by rho."
        ),
        "arms": {
            "entropy_k12": arms["entropy_k12"],
            "bottom_k12": arms["bottom_k12"],
            "topk": topk_arms,
            "random_null": null_summary,
        },
        "random_draws": random_rows,
    }

    out_dir = cfg.study_dir(split_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "matched_k_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "matched_k_summary.csv", index=False)
    print(f"Matched-k summary -> {summary_path}")
    print(
        f"  entropy AUROC={entropy_auroc}  "
        f"bottom={bottom_auroc}  "
        f"random mean±std="
        f"{null_summary['auroc_mean']}±{null_summary['auroc_std']}  "
        f"entropy pct={null_summary['entropy_percentile']}"
    )
    return summary_path
