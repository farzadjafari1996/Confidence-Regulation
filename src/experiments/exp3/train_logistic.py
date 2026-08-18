"""Step 2a: train logistic regression probe with randomized hyperparameter search."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exp3.dataset_io import collection_ready, ensure_splits, load_arrays
from exp3.features import feature_matrix
from exp3.labels import build_targets
from exp3.metrics import classification_metrics, regression_metrics
from exp3.settings import Exp3Config, FeatureSet, LabelName, SplitName


def _load_xy(
    cfg: Exp3Config,
    split_name: SplitName,
    label: LabelName,
) -> tuple[dict[str, Any], np.ndarray, dict[str, list[int]], str]:
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
    return data, y, splits, task


def run_train_logistic(
    cfg: Exp3Config,
    split_name: SplitName,
    *,
    label: LabelName | None = None,
    features: FeatureSet = "all",
    phase1_dir: pathlib.Path | None = None,
    device: str | None = None,
    skip_phase1: bool = False,
) -> pathlib.Path:
    label = cfg.resolved_label(label)
    out_dir = cfg.probe_out_dir(split_name, "logistic", features=features)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, y, splits, task = _load_xy(cfg, split_name, label)
    X, feat_idx, feat_meta = feature_matrix(
        data,
        cfg,
        features,
        phase1_dir=phase1_dir,
        device=device,
        skip_phase1=skip_phase1,
    )
    d_mlp = int(np.asarray(data["X_mlp_last"]).shape[1])
    print(
        f"Logistic features={features}: d_in={X.shape[1]} "
        f"(source={feat_meta.get('source')} d_mlp={d_mlp})"
    )
    train_idx = np.asarray(splits["train"], dtype=np.int64)
    val_idx = np.asarray(splits["val"], dtype=np.int64)
    # Fit search on train; score on val via refit using train+val for final? Plan: tune on val.
    # RandomizedSearchCV uses CV on train only; then evaluate val separately.
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if task == "regression":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        )
        param_dist = {
            "model__alpha": np.logspace(-3, 3, 50),
        }
        scoring = "r2"
    else:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=cfg.logistic.max_iter,
                        random_state=cfg.seed,
                    ),
                ),
            ]
        )
        param_dist = {
            "model__C": np.logspace(-3, 3, 40),
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
            "model__class_weight": [None, "balanced"],
        }
        if task == "multiclass":
            param_dist["model__multi_class"] = ["multinomial"]
        scoring = "roc_auc" if task == "binary" else "f1_macro"

    n_iter = min(cfg.logistic.n_iter, 40)
    n_jobs = 1 if len(X_train) < 400 else -1
    if task == "regression":
        cv = min(cfg.logistic.cv, 3)
        min_c = len(X_train)
    else:
        counts = np.bincount(np.asarray(y_train, dtype=np.int64).ravel())
        min_c = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
        cv = min(cfg.logistic.cv, max(2, min_c))
    if task == "binary" and min_c < 2:
        raise RuntimeError("Train set has a single class; collect more data or change --label")

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=min(n_iter, max(2, len(X_train))),
        scoring=scoring,
        cv=cv,
        random_state=cfg.seed,
        n_jobs=n_jobs,
        refit=True,
        error_score=np.nan,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Val metrics
    if task == "regression":
        pred_val = best.predict(X_val)
        val_metrics = regression_metrics(y_val, pred_val)
    else:
        pred_val = best.predict(X_val)
        if task == "binary" and hasattr(best, "predict_proba"):
            prob_val = best.predict_proba(X_val)[:, 1]
        else:
            prob_val = None
        val_metrics = classification_metrics(y_val, pred_val, prob_val, task=task)

    # Refit on train+val for the saved model (test reserved for Step 3)
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)
    best.fit(X_fit, y_fit)

    model_path = out_dir / "best_model.joblib"
    joblib.dump(
        {
            "pipeline": best,
            "label": label,
            "task": task,
            "best_params": search.best_params_,
            "d_mlp": d_mlp,
            "d_in": int(X.shape[1]),
            "feature_set": features,
            "feature_indices": feat_idx.tolist(),
            "feature_meta": feat_meta,
            "seed": cfg.seed,
        },
        model_path,
    )

    results = pd.DataFrame(search.cv_results_)
    results.to_csv(out_dir / "search_results.csv", index=False)

    train_metrics = {
        "label": label,
        "task": task,
        "feature_set": features,
        "d_in": int(X.shape[1]),
        "d_mlp": d_mlp,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "best_cv_score": float(search.best_score_) if search.best_score_ == search.best_score_ else None,
        "val": val_metrics,
        "class_balance_train": _class_balance(y_train, task),
        "class_balance_val": _class_balance(y_val, task),
        "feature_meta": feat_meta,
    }
    (out_dir / "train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2), encoding="utf-8"
    )
    print(f"Logistic probe saved -> {model_path}")
    print(f"  val metrics: {val_metrics}")
    return model_path


def _class_balance(y: np.ndarray, task: str) -> dict[str, Any]:
    if task == "regression":
        return {"mean": float(np.mean(y)), "std": float(np.std(y))}
    vals, counts = np.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v
