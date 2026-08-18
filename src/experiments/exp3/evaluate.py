"""Step 4: held-out test evaluation + per-neuron permutation importance.

Usage (from repo root, venv active). Omitting --dataset runs both cloze sets:
  python src/experiments/exp3/evaluate.py --config config/llama_instruct.yaml --split main
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

_src = pathlib.Path(__file__).resolve().parent
while _src != _src.parent and not (
    (_src / "common").is_dir() and (_src / "experiments").is_dir() and (_src / "models").is_dir()
):
    _src = _src.parent
for _p in (_src, _src / "common", _src / "experiments", _src / "models"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import joblib
import numpy as np
import torch
from sklearn.inspection import permutation_importance as sk_permutation_importance

from exp3.dataset_io import collection_ready, ensure_splits, load_arrays
from exp3.labels import build_targets
from exp3.metrics import classification_metrics, regression_metrics
from exp3.plots import write_exp3_plots
from exp3.settings import Exp3Config, FeatureSet, LabelName, ProbeName, SplitName
from exp3.train_mlp import ProbeMLP, _predict


class _CallableEstimator:
    """Thin sklearn-like wrapper so permutation_importance works for MLP predictors."""

    def __init__(self, predict_fn, task: str):
        self.predict_fn = predict_fn
        self.task = task

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        out = self.predict_fn(np.asarray(X))
        if self.task == "binary":
            return (out >= 0.5).astype(np.int64)
        return out

    def predict_proba(self, X):
        out = self.predict_fn(np.asarray(X))
        if self.task != "binary":
            raise AttributeError("predict_proba only for binary")
        out = np.asarray(out, dtype=np.float64)
        return np.column_stack([1.0 - out, out])

    def score(self, X, y):
        pred = self.predict(X)
        y = np.asarray(y)
        if self.task == "regression":
            return float(regression_metrics(y, pred)["r2"])
        if self.task == "binary":
            prob = self.predict_proba(X)[:, 1]
            m = classification_metrics(y, pred, prob, task=self.task)
            return float(m["auroc"] if m["auroc"] is not None else m["f1"])
        return float(classification_metrics(y, pred, None, task=self.task)["f1_macro"])


def _load_logistic_predictor(bundle: dict[str, Any], task: str):
    pipe = bundle["pipeline"]

    def predict_value(X: np.ndarray) -> np.ndarray:
        if task == "regression":
            return pipe.predict(X)
        if task == "binary":
            return pipe.predict_proba(X)[:, 1]
        return pipe.predict(X)

    return predict_value, pipe


def _load_mlp_predictor(ckpt: dict[str, Any], task: str):
    d_in = int(ckpt.get("d_in", ckpt["d_mlp"]))
    model = ProbeMLP(
        d_in,
        hidden=int(ckpt["hidden"]),
        n_classes=int(ckpt.get("n_classes", 2)),
        dropout=float(ckpt["dropout"]),
        task=task,
    )
    model.load_state_dict(ckpt["state_dict"])
    mean = np.asarray(ckpt["mean"])
    std = np.asarray(ckpt["std"])

    def predict_value(X: np.ndarray) -> np.ndarray:
        pred, prob = _predict(model, X, mean, std, task)
        if task == "binary":
            assert prob is not None
            return prob
        return pred

    return predict_value, _CallableEstimator(predict_value, task)


def _feature_indices_from_bundle(bundle: dict[str, Any], d_mlp: int) -> np.ndarray:
    if "feature_indices" in bundle and bundle["feature_indices"] is not None:
        return np.asarray(bundle["feature_indices"], dtype=np.int64)
    feature_set = bundle.get("feature_set", "all")
    if feature_set in {"all", "mlp_last"}:
        return np.arange(d_mlp, dtype=np.int64)
    raise ValueError(
        f"Probe bundle is missing feature_indices for feature_set={feature_set!r}"
    )


def _x_from_saved_indices(data: dict[str, Any], bundle: dict[str, Any], d_mlp: int) -> tuple[np.ndarray, np.ndarray]:
    from exp3.features import select_features as _sel

    feat_idx = _feature_indices_from_bundle(bundle, d_mlp)
    source = (bundle.get("feature_meta") or {}).get("source") or (
        "resid" if str(bundle.get("feature_set", "")).startswith("resid") or str(bundle.get("feature_set", "")).startswith("random_resid") else "mlp_last"
    )
    if source == "resid":
        X_src = np.asarray(data["X_resid"], dtype=np.float32).reshape(len(data["y_top1"]), -1)
    else:
        X_src = np.asarray(data["X_mlp_last"], dtype=np.float32)
    return _sel(X_src, feat_idx), feat_idx


def run_evaluate(
    cfg: Exp3Config,
    split_name: SplitName,
    probe: ProbeName,
    *,
    label: LabelName | None = None,
    features: FeatureSet = "all",
    skip_importance: bool = False,
) -> pathlib.Path:
    label = cfg.resolved_label(label)
    out_dir = cfg.probe_out_dir(split_name, probe, features=features)
    data_dir = cfg.data_out_dir(split_name)
    if not collection_ready(data_dir):
        raise FileNotFoundError(
            f"Missing dataset {data_dir}. Run exp3/gather_activations.py first."
        )

    data = load_arrays(data_dir)
    splits = ensure_splits(cfg, split_name, data)
    y, task = build_targets(
        data,
        label,
        confident_error_top1_prob=cfg.confident_error_top1_prob,
    )
    d_mlp = int(np.asarray(data["X_mlp_last"]).shape[1])
    test_idx = np.asarray(splits["test"], dtype=np.int64)

    bundle: dict[str, Any] | None = None
    ckpt: dict[str, Any] | None = None
    if probe == "logistic":
        model_path = out_dir / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing {model_path}. Run exp3/train_arms.py first."
            )
        bundle = joblib.load(model_path)
        if bundle.get("label") != label:
            print(
                f"Warning: trained label={bundle.get('label')} but evaluating with label={label}"
            )
        feat_idx = _feature_indices_from_bundle(bundle, d_mlp)
        X, feat_idx = _x_from_saved_indices(data, bundle, d_mlp)
        X_test, y_test = X[test_idx], y[test_idx]
        predict_value, estimator = _load_logistic_predictor(bundle, task)
        pipe = bundle["pipeline"]
        if task == "regression":
            pred = pipe.predict(X_test)
            test_metrics = regression_metrics(y_test, pred)
        else:
            pred = pipe.predict(X_test)
            prob = pipe.predict_proba(X_test)[:, 1] if task == "binary" else None
            test_metrics = classification_metrics(y_test, pred, prob, task=task)
        feature_set = bundle.get("feature_set", features)
        d_in = int(bundle.get("d_in", X.shape[1]))
    else:
        model_path = out_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing {model_path}. Run exp3/train_arms.py first."
            )
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if ckpt.get("label") != label:
            print(f"Warning: trained label={ckpt.get('label')} but evaluating with label={label}")
        feat_idx = _feature_indices_from_bundle(ckpt, d_mlp)
        X, feat_idx = _x_from_saved_indices(data, ckpt, d_mlp)
        X_test, y_test = X[test_idx], y[test_idx]
        predict_value, estimator = _load_mlp_predictor(ckpt, task)
        pred_or_prob = predict_value(X_test)
        if task == "regression":
            test_metrics = regression_metrics(y_test, pred_or_prob)
        elif task == "binary":
            pred = (pred_or_prob >= 0.5).astype(np.int64)
            test_metrics = classification_metrics(y_test, pred, pred_or_prob, task=task)
        else:
            test_metrics = classification_metrics(y_test, pred_or_prob, None, task=task)
        feature_set = ckpt.get("feature_set", features)
        d_in = int(ckpt.get("d_in", X.shape[1]))

    payload = {
        "probe": probe,
        "label": label,
        "task": task,
        "feature_set": feature_set,
        "d_in": d_in,
        "d_mlp": d_mlp,
        "n_test": int(len(test_idx)),
        "test": test_metrics,
    }
    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Test metrics -> {metrics_path}")
    print(f"  {test_metrics}")

    if skip_importance or features in {"resid_all"} or str(features).startswith("random_resid"):
        return metrics_path

    print(
        f"Computing permutation importance "
        f"(d={X_test.shape[1]}, repeats={cfg.importance.n_repeats})..."
    )
    scoring = "r2" if task == "regression" else ("roc_auc" if task == "binary" else "f1_macro")
    try:
        result = sk_permutation_importance(
            estimator,
            X_test,
            y_test,
            n_repeats=cfg.importance.n_repeats,
            random_state=cfg.seed,
            scoring=scoring,
            n_jobs=-1,
        )
        importances = result.importances_mean
    except Exception as exc:
        print(f"sklearn permutation_importance failed ({exc}); falling back")
        if probe == "logistic" and bundle is not None and task != "multiclass":
            coef = bundle["pipeline"].named_steps["model"].coef_
            importances = np.mean(np.abs(coef), axis=0)
        else:
            raise

    order = np.argsort(-importances)
    top_k = int(cfg.importance.top_k)
    # Map subset-local ranks back to original MLP neuron indices when available.
    local_to_global = feat_idx
    importance_payload = {
        "probe": probe,
        "label": label,
        "feature_set": feature_set,
        "baseline_note": "sklearn permutation_importance mean decrease in score",
        "top_neurons": [
            {
                "neuron_index": int(local_to_global[i]),
                "subset_index": int(i),
                "importance": float(importances[i]),
            }
            for i in order[:top_k]
        ],
        "importances": [float(x) for x in importances],
        "feature_indices": local_to_global.tolist(),
    }
    imp_path = out_dir / "importance.json"
    imp_path.write_text(json.dumps(importance_payload, indent=2), encoding="utf-8")
    write_exp3_plots(out_dir, importance_payload, test_metrics, probe=probe, label=label)
    print(f"Importance -> {imp_path}")
    return metrics_path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile_of_score(scores: list[float], value: float) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(100.0 * np.mean(arr <= value))


def run_evaluate_arms(
    cfg: Exp3Config,
    split_name: SplitName = "main",
    *,
    label: LabelName | None = None,
    skip_importance: bool = True,
    skip_random: bool = False,
    include_mlp: bool | None = None,
    skip_resid: bool = True,
    include_bottom: bool | None = None,
    headline_only: bool = False,
    entropy_as_topk: bool = False,
    device: str | None = None,
    phase1_dir: pathlib.Path | None = None,
    skip_phase1: bool = False,
    k: int | None = None,
    n_random: int | None = None,
) -> pathlib.Path:
    """Score every trained arm and write matched_k_summary.{json,csv}."""
    import pandas as pd

    from exp3.arms import list_arms, probe_ready, resolve_matched_k
    from exp3.features import load_rho_table, select_topk_by_rho

    label = cfg.resolved_label(label)
    k, n_random, resolved_phase1 = resolve_matched_k(
        cfg,
        split_name,
        device=device,
        phase1_dir=phase1_dir,
        skip_phase1=skip_phase1,
        k=k,
        n_random=n_random,
    )
    arms_spec = list_arms(
        cfg,
        k,
        n_random=n_random,
        skip_random=skip_random,
        include_mlp=include_mlp,
        skip_resid=skip_resid,
        include_bottom=include_bottom,
        headline_only=headline_only,
        entropy_as_topk=entropy_as_topk,
    )
    data = load_arrays(cfg.data_out_dir(split_name))
    d_mlp = int(np.asarray(data["X_mlp_last"]).shape[1])
    rho_df = load_rho_table(resolved_phase1)
    topk_idx = select_topk_by_rho(rho_df, k, d_mlp=d_mlp)

    rows: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}
    random_mlp: list[dict[str, Any]] = []
    random_resid: list[dict[str, Any]] = []

    for spec in arms_spec:
        if not probe_ready(cfg, split_name, spec.probe, spec.features):
            raise FileNotFoundError(
                f"Missing trained {spec.probe}/{spec.features} ({spec.name}). "
                "Run exp3/train_arms.py first."
            )
        metrics_path = run_evaluate(
            cfg,
            split_name,
            spec.probe,
            label=label,
            features=spec.features,
            skip_importance=skip_importance,
        )
        payload = _read_json(metrics_path)
        test = payload.get("test") or {}
        train_meta = {}
        train_path = metrics_path.parent / "train_metrics.json"
        if train_path.exists():
            train_meta = (_read_json(train_path).get("feature_meta") or {})
        row = {
            "arm": spec.name,
            "features": spec.features,
            "probe": spec.probe,
            "d_in": payload.get("d_in"),
            "accuracy": test.get("accuracy"),
            "f1": test.get("f1"),
            "auroc": test.get("auroc"),
            "path": str(metrics_path.parent),
            "feature_indices": train_meta.get("feature_indices"),
        }
        arms[spec.name] = row
        rows.append(row)
        if spec.name.startswith("random_mlp_"):
            random_mlp.append(row)
        elif spec.name.startswith("random_resid_"):
            random_resid.append(row)

    def _null(rows_in: list[dict[str, Any]], entropy_auroc: float | None) -> dict[str, Any]:
        aurocs = [float(r["auroc"]) for r in rows_in if r.get("auroc") is not None]
        return {
            "n_random": len(aurocs),
            "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
            "auroc_std": float(np.std(aurocs)) if aurocs else None,
            "auroc_min": float(np.min(aurocs)) if aurocs else None,
            "auroc_max": float(np.max(aurocs)) if aurocs else None,
            "entropy_percentile": (
                _percentile_of_score(aurocs, float(entropy_auroc))
                if entropy_auroc is not None and aurocs
                else None
            ),
        }

    entropy_auroc = arms.get("entropy", {}).get("auroc")
    entropy_idx = arms.get("entropy", {}).get("feature_indices") or []
    summary = {
        "model": cfg.model,
        "family": cfg.family,
        "dataset": cfg.dataset,
        "split": split_name,
        "label": label,
        "k": k,
        "n_random": n_random,
        "phase1_dir": str(resolved_phase1),
        "d_mlp": d_mlp,
        "topk_equals_entropy": set(int(x) for x in topk_idx.tolist())
        == set(int(x) for x in entropy_idx),
        "note": (
            "Default probes are logistic last-MLP all, entropy, bottom-k, "
            "and n-random last-MLP subsets. resid_all is off unless --include-resid."
        ),
        "arms": {
            "resid_all": arms.get("resid_all"),
            "mlp_last": arms.get("mlp_last"),
            "entropy": arms.get("entropy"),
            "bottom_k": arms.get("bottom_k"),
            "random_mlp_null": _null(random_mlp, entropy_auroc),
            "random_resid_null": _null(random_resid, entropy_auroc),
        },
    }
    out_dir = cfg.study_dir(split_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "matched_k_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "matched_k_summary.csv", index=False)
    print(f"Matched-k summary -> {summary_path}")
    extras = []
    if arms.get("resid_all"):
        extras.append(f"resid_all={arms['resid_all'].get('auroc')}")
    if summary["arms"]["random_mlp_null"]["entropy_percentile"] is not None:
        extras.append(
            f"random-mlp pct={summary['arms']['random_mlp_null']['entropy_percentile']}"
        )
    extra = ("  " + "  ".join(extras)) if extras else ""
    print(
        f"  entropy AUROC={entropy_auroc}  "
        f"mlp_last={arms.get('mlp_last', {}).get('auroc')}"
        f"{extra}"
    )
    return summary_path


def parse_args():
    import argparse

    from exp3.cli import add_train_args

    p = argparse.ArgumentParser(description="Exp 3 step 4: evaluate trained arms")
    add_train_args(p)
    p.add_argument(
        "--importance",
        action="store_true",
        help="Compute permutation importance (slow; off by default)",
    )
    return p.parse_args()


def main() -> None:
    from exp3.cli import arm_select_kwargs, iter_cfgs, resolve_device

    args = parse_args()
    for dataset, cfg in iter_cfgs(args):
        print(f"\n=== eval  {dataset}  {args.split}  n={cfg.sample_size(args.split)} ===")
        run_evaluate_arms(
            cfg,
            args.split,
            label=args.label,
            skip_importance=not args.importance,
            device=resolve_device(args.device or cfg.device),
            skip_phase1=args.skip_phase1,
            **arm_select_kwargs(args),
        )


if __name__ == "__main__":
    main()

