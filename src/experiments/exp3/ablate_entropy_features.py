"""Step 4: probe dependence on Phase-1 high-ρ entropy-neuron features.

Freezes trained Exp3 probes, mean-replaces (and optionally zeros) entropy dims
in stored X_full, and compares necessity / sufficiency effects to matched
random-neuron controls. Optionally auto-runs Phase 1 if selection artifacts
are missing.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, percentileofscore, spearmanr

from exp3.dataset_io import dataset_path, ensure_splits, load_arrays
from exp3.evaluate import _load_logistic_predictor, _load_mlp_predictor
from exp3.labels import build_targets
from exp3.metrics import classification_metrics, regression_metrics
from exp3.settings import Exp3Config, LabelName, ProbeName, SplitName
from hyperparameters import ModelRunConfig, load_run_settings, phase_output_dir

AblationMode = Literal["mean", "zero"]
Intervention = Literal["necessity", "sufficiency"]

PredictFn = Callable[[np.ndarray], np.ndarray]


def _phase1_ready(phase1_dir: pathlib.Path) -> bool:
    return (phase1_dir / "top1pct_rho_neurons.json").exists() and (
        phase1_dir / "static_neuron_metrics.csv"
    ).exists()


def ensure_phase1(
    cfg: Exp3Config,
    phase1_dir: pathlib.Path,
    *,
    device: str,
    skip_phase1: bool = False,
) -> pathlib.Path:
    """Load existing Phase-1 outputs or run Phase 1 when missing."""
    phase1_dir = pathlib.Path(phase1_dir)
    if _phase1_ready(phase1_dir):
        print(f"Phase 1 artifacts found at {phase1_dir}")
        return phase1_dir
    if skip_phase1:
        raise FileNotFoundError(
            f"Missing Phase 1 outputs at {phase1_dir} and --skip-phase1 was set. "
            "Run phase1_identify.py first."
        )

    if cfg.config_path is None:
        raise FileNotFoundError(
            f"Missing Phase 1 at {phase1_dir} and Exp3Config has no config_path "
            "to auto-run Phase 1."
        )

    family = str(cfg.family).lower()
    print(f"Phase 1 missing at {phase1_dir}; auto-running for family={family}...")
    settings = load_run_settings(cfg.config_path, family)  # type: ignore[arg-type]
    model_cfg = ModelRunConfig.from_hparams(
        cfg.model,
        settings["hparams"],
        dtype=settings.get("dtype"),
    )
    out_base = pathlib.Path(cfg.out_dir)
    out_dir = model_cfg.phase_dir(out_base, "phase1")
    out_dir.mkdir(parents=True, exist_ok=True)

    if family == "llama":
        from llama.core import run_phase1_identify
        from llama.load import load_llama

        run_phase1_identify(model_cfg, out_dir, device=device, load_model_fn=load_llama)
    elif family == "gpt2":
        from gpt2.core import run_phase1_identify

        run_phase1_identify(model_cfg, out_dir, device=device)
    elif family == "qwen":
        from qwen.core import run_phase1_identify
        from qwen.load import load_qwen

        run_phase1_identify(model_cfg, out_dir, device=device, load_model_fn=load_qwen)
    elif family == "pythia":
        from pythia.core import run_phase1_identify
        from pythia.load import load_pythia

        run_phase1_identify(model_cfg, out_dir, device=device, load_model_fn=load_pythia)
    elif family == "smollm":
        from smollm.core import run_phase1_identify
        from smollm.load import load_smollm

        run_phase1_identify(model_cfg, out_dir, device=device, load_model_fn=load_smollm)
    elif family == "gemma":
        from gemma.core import run_phase1_identify
        from gemma.load import load_gemma

        run_phase1_identify(model_cfg, out_dir, device=device, load_model_fn=load_gemma)
    else:
        raise ValueError(f"Unsupported family for auto Phase 1: {family!r}")

    if not _phase1_ready(out_dir):
        raise RuntimeError(f"Phase 1 finished but artifacts still missing under {out_dir}")
    return out_dir


def load_entropy_neuron_indices(
    phase1_dir: pathlib.Path,
    *,
    d_mlp: int,
) -> tuple[list[str], np.ndarray, int]:
    """Return (labels, column indices, writer_layer) from Phase-1 selection JSON."""
    top1_path = pathlib.Path(phase1_dir) / "top1pct_rho_neurons.json"
    if not top1_path.exists():
        raise FileNotFoundError(f"Missing {top1_path}")
    payload = json.loads(top1_path.read_text(encoding="utf-8"))
    labels = list(payload.get("neurons") or [])
    if not labels:
        raise ValueError(f"No entropy neurons listed in {top1_path}")

    # Prefer explicit neuron_index from stats when present (avoids float coercion).
    stats = payload.get("stats") or []
    idx_from_stats: dict[str, int] = {}
    for row in stats:
        if not isinstance(row, dict):
            continue
        name = row.get("component_name")
        if name is not None and "neuron_index" in row:
            idx_from_stats[str(name)] = int(row["neuron_index"])
        elif name is not None and "neuron_index" not in row:
            # stats often only have component_name; parse below
            pass

    layers: list[int] = []
    idxs: list[int] = []
    for name in labels:
        name_s = str(name)
        if name_s in idx_from_stats:
            # Recover layer from label prefix.
            layer = int(float(name_s.split(".", 1)[0]))
            idx = idx_from_stats[name_s]
        else:
            parts = name_s.split(".")
            if len(parts) != 2:
                raise ValueError(f"Unexpected neuron label {name!r}; expected 'layer.idx'")
            layer = int(float(parts[0]))
            idx = int(float(parts[1]))
        if idx < 0 or idx >= d_mlp:
            raise ValueError(f"Neuron {name} index {idx} out of range for d_mlp={d_mlp}")
        layers.append(layer)
        idxs.append(idx)

    if len(set(layers)) != 1:
        raise ValueError(f"Entropy neurons span multiple layers: {sorted(set(layers))}")
    return [str(x) for x in labels], np.asarray(idxs, dtype=np.int64), int(layers[0])


def _replace_dims(
    X: np.ndarray,
    indices: np.ndarray,
    *,
    mode: AblationMode,
    fill: np.ndarray,
) -> np.ndarray:
    out = np.array(X, dtype=np.float32, copy=True)
    if indices.size == 0:
        return out
    if mode == "mean":
        out[:, indices] = fill[indices]
    elif mode == "zero":
        out[:, indices] = 0.0
    else:
        raise ValueError(f"Unknown ablation mode {mode!r}")
    return out


def apply_intervention(
    X: np.ndarray,
    target_indices: np.ndarray,
    *,
    intervention: Intervention,
    mode: AblationMode,
    fill: np.ndarray,
) -> np.ndarray:
    d = X.shape[1]
    if intervention == "necessity":
        return _replace_dims(X, target_indices, mode=mode, fill=fill)
    if intervention == "sufficiency":
        mask = np.ones(d, dtype=bool)
        mask[target_indices] = False
        other = np.flatnonzero(mask).astype(np.int64)
        return _replace_dims(X, other, mode=mode, fill=fill)
    raise ValueError(f"Unknown intervention {intervention!r}")


def _score_binary(
    predict_value: PredictFn,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if task == "regression":
        pred = np.asarray(predict_value(X), dtype=np.float64)
        return regression_metrics(y, pred), pred, pred

    scores = np.asarray(predict_value(X), dtype=np.float64)
    if task == "binary":
        pred = (scores >= 0.5).astype(np.int64)
        metrics = classification_metrics(y, pred, scores, task=task)
        return metrics, pred, scores
    pred = scores.astype(np.int64)
    metrics = classification_metrics(y, pred, None, task=task)
    return metrics, pred, scores


def _dependence_stats(
    y: np.ndarray,
    pred_base: np.ndarray,
    prob_base: np.ndarray,
    pred_abl: np.ndarray,
    prob_abl: np.ndarray,
    *,
    task: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "flip_rate": float(np.mean(pred_base != pred_abl)),
        "mean_abs_delta_score": float(np.mean(np.abs(prob_base - prob_abl))),
    }
    if task == "binary" and len(prob_base) > 2:
        if np.std(prob_base) > 1e-12 and np.std(prob_abl) > 1e-12:
            out["pearson_r"] = float(pearsonr(prob_base, prob_abl).statistic)
            out["spearman_r"] = float(spearmanr(prob_base, prob_abl).statistic)
        else:
            out["pearson_r"] = None
            out["spearman_r"] = None
    return out


def _metric_delta(base: dict[str, Any], abl: dict[str, Any], key: str) -> float | None:
    b, a = base.get(key), abl.get(key)
    if b is None or a is None:
        return None
    return float(b) - float(a)


def _summarize_controls(
    control_rows: list[dict[str, Any]],
    entropy_delta_auroc: float | None,
    *,
    intervention: Intervention,
) -> dict[str, Any]:
    deltas = [r["delta_auroc"] for r in control_rows if r.get("delta_auroc") is not None]
    summary: dict[str, Any] = {
        "n_controls": len(control_rows),
        "delta_auroc_mean": float(np.mean(deltas)) if deltas else None,
        "delta_auroc_std": float(np.std(deltas)) if deltas else None,
        "delta_auroc_entropy": entropy_delta_auroc,
        "percentile_vs_controls": None,
        "percentile_note": (
            "necessity: higher percentile => larger AUROC drop than random "
            "(more uniquely necessary). "
            "sufficiency: higher percentile => smaller AUROC drop than random "
            "(more uniquely sufficient)."
        ),
    }
    if deltas and entropy_delta_auroc is not None:
        if intervention == "necessity":
            # Larger drop => stronger necessity.
            score = entropy_delta_auroc
            ref = deltas
        else:
            # Smaller drop => stronger sufficiency.
            score = -entropy_delta_auroc
            ref = [-d for d in deltas]
        summary["percentile_vs_controls"] = float(percentileofscore(ref, score, kind="rank"))
    return summary


def _logistic_coef_summary(
    bundle: dict[str, Any],
    entropy_idx: np.ndarray,
    rng: np.random.Generator,
    n_controls: int,
) -> dict[str, Any] | None:
    model = bundle["pipeline"].named_steps.get("model")
    if model is None or not hasattr(model, "coef_"):
        return None
    coef = np.asarray(model.coef_, dtype=np.float64)
    abs_coef = np.mean(np.abs(coef), axis=0)
    d = abs_coef.shape[0]
    ent_mean = float(np.mean(abs_coef[entropy_idx])) if entropy_idx.size else None
    pool = np.setdiff1d(np.arange(d, dtype=np.int64), entropy_idx, assume_unique=False)
    k = int(entropy_idx.size)
    control_means: list[float] = []
    if k > 0 and pool.size >= k:
        for _ in range(n_controls):
            pick = rng.choice(pool, size=k, replace=False)
            control_means.append(float(np.mean(abs_coef[pick])))
    summary: dict[str, Any] = {
        "mean_abs_coef_entropy": ent_mean,
        "mean_abs_coef_all": float(np.mean(abs_coef)),
        "control_mean_abs_coef_mean": float(np.mean(control_means)) if control_means else None,
        "control_mean_abs_coef_std": float(np.std(control_means)) if control_means else None,
        "percentile_vs_controls": None,
    }
    if control_means and ent_mean is not None:
        summary["percentile_vs_controls"] = float(
            percentileofscore(control_means, ent_mean, kind="rank")
        )
    return summary


def _importance_crosscheck(
    out_dir: pathlib.Path,
    entropy_idx: np.ndarray,
) -> dict[str, Any] | None:
    path = out_dir / "importance.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    imps = np.asarray(payload.get("importances") or [], dtype=np.float64)
    if imps.size == 0:
        return None
    ranks = np.argsort(np.argsort(-imps))  # 0 = most important
    ent_imps = imps[entropy_idx]
    ent_ranks = ranks[entropy_idx]
    return {
        "source": str(path),
        "mean_importance": float(np.mean(ent_imps)),
        "mean_rank": float(np.mean(ent_ranks)),
        "best_rank": int(np.min(ent_ranks)),
        "per_neuron": [
            {
                "neuron_index": int(i),
                "importance": float(imps[i]),
                "rank": int(ranks[i]),
            }
            for i in entropy_idx
        ],
    }


def _write_plot(
    out_dir: pathlib.Path,
    *,
    intervention: Intervention,
    mode: AblationMode,
    entropy_delta: float | None,
    control_deltas: list[float],
    probe: str,
    label: str,
) -> None:
    if not control_deltas and entropy_delta is None:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if control_deltas:
        ax.hist(control_deltas, bins=min(20, max(5, len(control_deltas))), color="#4C78A8", alpha=0.85)
    if entropy_delta is not None:
        ax.axvline(
            entropy_delta,
            color="#E45756",
            linewidth=2,
            label=f"entropy delta_auroc={entropy_delta:.4f}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("delta_auroc (baseline - ablated)")
    ax.set_ylabel("Control count")
    ax.set_title(f"{probe}/{label}: {intervention} ({mode})")
    fig.tight_layout()
    fig.savefig(plots_dir / f"entropy_ablation_{intervention}_{mode}.png", dpi=150)
    plt.close(fig)


def run_entropy_feature_ablation(
    cfg: Exp3Config,
    split_name: SplitName,
    probe: ProbeName,
    *,
    label: LabelName | None = None,
    mode: Literal["mean", "zero", "both"] = "both",
    n_controls: int = 20,
    phase1_dir: pathlib.Path | None = None,
    skip_phase1: bool = False,
    device: str | None = None,
) -> pathlib.Path:
    label = label or cfg.label_default
    device = device or cfg.device
    out_dir = cfg.probe_out_dir(split_name, probe)
    data_dir = cfg.data_out_dir(split_name)
    data_path = dataset_path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset {data_path}. Run exp3/collect.py first.")

    resolved_phase1 = pathlib.Path(phase1_dir) if phase1_dir is not None else phase_output_dir(
        cfg.model, "phase1", cfg.out_dir
    )
    resolved_phase1 = ensure_phase1(
        cfg, resolved_phase1, device=device, skip_phase1=skip_phase1
    )

    data = load_arrays(data_path)
    splits = ensure_splits(cfg, split_name, data)
    y, task = build_targets(
        data,
        label,
        confident_error_top1_prob=cfg.confident_error_top1_prob,
    )
    X = np.asarray(data["X_full"], dtype=np.float32)
    train_idx = np.asarray(splits["train"], dtype=np.int64)
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    X_train, X_test = X[train_idx], X[test_idx]
    y_test = y[test_idx]
    train_mean = X_train.mean(axis=0).astype(np.float32)

    entropy_labels, entropy_idx, writer_layer = load_entropy_neuron_indices(
        resolved_phase1, d_mlp=X.shape[1]
    )
    print(
        f"Entropy neurons: n={len(entropy_idx)} layer={writer_layer} "
        f"idxs={entropy_idx.tolist()}"
    )

    bundle: dict[str, Any] | None = None
    if probe == "logistic":
        model_path = out_dir / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}. Run exp3/train_logistic.py first.")
        bundle = joblib.load(model_path)
        if bundle.get("label") != label:
            print(
                f"Warning: trained label={bundle.get('label')} but evaluating with label={label}"
            )
        predict_value, _ = _load_logistic_predictor(bundle, task)
    else:
        model_path = out_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}. Run exp3/train_mlp.py first.")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if ckpt.get("label") != label:
            print(f"Warning: trained label={ckpt.get('label')} but evaluating with label={label}")
        predict_value, _ = _load_mlp_predictor(ckpt, task)

    base_metrics, pred_base, score_base = _score_binary(predict_value, X_test, y_test, task)
    modes: list[AblationMode] = ["mean", "zero"] if mode == "both" else [mode]  # type: ignore[list-item]
    interventions: list[Intervention] = ["necessity", "sufficiency"]
    rng = np.random.default_rng(cfg.seed)
    d = X.shape[1]
    pool = np.setdiff1d(np.arange(d, dtype=np.int64), entropy_idx, assume_unique=False)
    k = int(entropy_idx.size)

    results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for intervention in interventions:
        for abl_mode in modes:
            X_ent = apply_intervention(
                X_test,
                entropy_idx,
                intervention=intervention,
                mode=abl_mode,
                fill=train_mean,
            )
            ent_metrics, pred_ent, score_ent = _score_binary(predict_value, X_ent, y_test, task)
            dep = _dependence_stats(
                y_test, pred_base, score_base, pred_ent, score_ent, task=task
            )
            delta_auroc = _metric_delta(base_metrics, ent_metrics, "auroc")
            delta_acc = _metric_delta(base_metrics, ent_metrics, "accuracy")
            delta_f1 = _metric_delta(base_metrics, ent_metrics, "f1")

            control_rows: list[dict[str, Any]] = []
            control_deltas: list[float] = []
            if k > 0 and pool.size >= k:
                for c_i in range(n_controls):
                    pick = rng.choice(pool, size=k, replace=False)
                    X_c = apply_intervention(
                        X_test,
                        pick,
                        intervention=intervention,
                        mode=abl_mode,
                        fill=train_mean,
                    )
                    c_metrics, pred_c, score_c = _score_binary(predict_value, X_c, y_test, task)
                    c_delta = _metric_delta(base_metrics, c_metrics, "auroc")
                    row = {
                        "control_id": c_i,
                        "indices": pick.tolist(),
                        "metrics": c_metrics,
                        "delta_auroc": c_delta,
                        "delta_accuracy": _metric_delta(base_metrics, c_metrics, "accuracy"),
                        "flip_rate": float(np.mean(pred_base != pred_c)),
                        "mean_abs_delta_score": float(np.mean(np.abs(score_base - score_c))),
                    }
                    control_rows.append(row)
                    if c_delta is not None:
                        control_deltas.append(c_delta)

            control_summary = _summarize_controls(
                control_rows, delta_auroc, intervention=intervention
            )
            key = f"{intervention}_{abl_mode}"
            results[key] = {
                "intervention": intervention,
                "mode": abl_mode,
                "metrics": ent_metrics,
                "delta_auroc": delta_auroc,
                "delta_accuracy": delta_acc,
                "delta_f1": delta_f1,
                "dependence": dep,
                "controls": control_summary,
            }
            csv_rows.append(
                {
                    "set": "entropy",
                    "intervention": intervention,
                    "mode": abl_mode,
                    "auroc": ent_metrics.get("auroc"),
                    "accuracy": ent_metrics.get("accuracy"),
                    "f1": ent_metrics.get("f1"),
                    "delta_auroc": delta_auroc,
                    "delta_accuracy": delta_acc,
                    "flip_rate": dep["flip_rate"],
                    "percentile_vs_controls": control_summary["percentile_vs_controls"],
                }
            )
            for crow in control_rows:
                csv_rows.append(
                    {
                        "set": f"random_{crow['control_id']}",
                        "intervention": intervention,
                        "mode": abl_mode,
                        "auroc": crow["metrics"].get("auroc"),
                        "accuracy": crow["metrics"].get("accuracy"),
                        "f1": crow["metrics"].get("f1"),
                        "delta_auroc": crow["delta_auroc"],
                        "delta_accuracy": crow["delta_accuracy"],
                        "flip_rate": crow["flip_rate"],
                        "percentile_vs_controls": None,
                    }
                )

            _write_plot(
                out_dir,
                intervention=intervention,
                mode=abl_mode,
                entropy_delta=delta_auroc,
                control_deltas=control_deltas,
                probe=probe,
                label=label,
            )
            print(
                f"  {key}: auroc={ent_metrics.get('auroc')} "
                f"delta_auroc={delta_auroc} "
                f"pct_vs_ctrl={control_summary['percentile_vs_controls']}"
            )

    coef_summary = None
    if probe == "logistic" and bundle is not None and task != "multiclass":
        coef_summary = _logistic_coef_summary(bundle, entropy_idx, rng, n_controls)

    payload = {
        "probe": probe,
        "label": label,
        "task": task,
        "n_test": int(len(test_idx)),
        "phase1_dir": str(resolved_phase1),
        "entropy_neurons": entropy_labels,
        "entropy_indices": entropy_idx.tolist(),
        "writer_layer": writer_layer,
        "n_controls": int(n_controls),
        "baseline": base_metrics,
        "results": results,
        "logistic_coef_summary": coef_summary,
        "importance_crosscheck": _importance_crosscheck(out_dir, entropy_idx),
        "note": (
            "Probe feature ablation only (frozen probe, fixed labels). "
            "Not a causal LLM intervention. Primary condition: train-mean replacement."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "entropy_feature_ablation.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Lightweight CSV without importing pandas (keep deps minimal).
    csv_path = out_dir / "entropy_feature_ablation.csv"
    headers = [
        "set",
        "intervention",
        "mode",
        "auroc",
        "accuracy",
        "f1",
        "delta_auroc",
        "delta_accuracy",
        "flip_rate",
        "percentile_vs_controls",
    ]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in csv_rows:
            f.write(
                ",".join(
                    "" if row[h] is None else str(row[h])
                    for h in headers
                )
                + "\n"
            )

    print(f"Entropy feature ablation -> {json_path}")
    print(f"  CSV -> {csv_path}")
    return json_path
