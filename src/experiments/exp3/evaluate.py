"""Step 3: held-out test evaluation + per-neuron permutation importance."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.inspection import permutation_importance as sk_permutation_importance

from exp3.dataset_io import dataset_path, ensure_splits, load_arrays
from exp3.labels import build_targets
from exp3.metrics import classification_metrics, regression_metrics
from exp3.plots import write_exp3_plots
from exp3.settings import Exp3Config, LabelName, ProbeName, SplitName
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
    model = ProbeMLP(
        int(ckpt["d_mlp"]),
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


def run_evaluate(
    cfg: Exp3Config,
    split_name: SplitName,
    probe: ProbeName,
    *,
    label: LabelName | None = None,
    skip_importance: bool = False,
) -> pathlib.Path:
    label = label or cfg.label_default
    out_dir = cfg.probe_out_dir(split_name, probe)
    data_dir = cfg.data_out_dir(split_name)
    data_path = dataset_path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset {data_path}. Run exp3/collect.py first.")

    data = load_arrays(data_path)
    splits = ensure_splits(cfg, split_name, data)
    y, task = build_targets(
        data,
        label,
        confident_error_top1_prob=cfg.confident_error_top1_prob,
    )
    X = np.asarray(data["X_full"], dtype=np.float32)
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    X_test, y_test = X[test_idx], y[test_idx]

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
        predict_value, estimator = _load_logistic_predictor(bundle, task)
        pipe = bundle["pipeline"]
        if task == "regression":
            pred = pipe.predict(X_test)
            test_metrics = regression_metrics(y_test, pred)
        else:
            pred = pipe.predict(X_test)
            prob = pipe.predict_proba(X_test)[:, 1] if task == "binary" else None
            test_metrics = classification_metrics(y_test, pred, prob, task=task)
    else:
        model_path = out_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}. Run exp3/train_mlp.py first.")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if ckpt.get("label") != label:
            print(f"Warning: trained label={ckpt.get('label')} but evaluating with label={label}")
        predict_value, estimator = _load_mlp_predictor(ckpt, task)
        pred_or_prob = predict_value(X_test)
        if task == "regression":
            test_metrics = regression_metrics(y_test, pred_or_prob)
        elif task == "binary":
            pred = (pred_or_prob >= 0.5).astype(np.int64)
            test_metrics = classification_metrics(y_test, pred, pred_or_prob, task=task)
        else:
            test_metrics = classification_metrics(y_test, pred_or_prob, None, task=task)

    payload = {
        "probe": probe,
        "label": label,
        "task": task,
        "n_test": int(len(test_idx)),
        "test": test_metrics,
    }
    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Test metrics -> {metrics_path}")
    print(f"  {test_metrics}")

    if skip_importance:
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
    importance_payload = {
        "probe": probe,
        "label": label,
        "baseline_note": "sklearn permutation_importance mean decrease in score",
        "top_neurons": [
            {"neuron_index": int(i), "importance": float(importances[i])}
            for i in order[:top_k]
        ],
        "importances": [float(x) for x in importances],
    }
    imp_path = out_dir / "importance.json"
    imp_path.write_text(json.dumps(importance_payload, indent=2), encoding="utf-8")
    write_exp3_plots(out_dir, importance_payload, test_metrics, probe=probe, label=label)
    print(f"Importance -> {imp_path}")
    return metrics_path
