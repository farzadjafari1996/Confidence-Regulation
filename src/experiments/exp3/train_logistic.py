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

from exp3.dataset_io import dataset_path, ensure_splits, load_arrays
from exp3.features import resolve_feature_indices, select_features
from exp3.labels import build_targets
from exp3.metrics import classification_metrics, regression_metrics
from exp3.settings import Exp3Config, FeatureSet, LabelName, SplitName


def _load_xy(
    cfg: Exp3Config,
    split_name: SplitName,
    label: LabelName,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], str, dict[str, Any]]:
    data_dir = cfg.data_out_dir(split_name)
    path = dataset_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset {path}. Run exp3/collect.py first.")
    data = load_arrays(path)
    splits = ensure_splits(cfg, split_name, data)
    y, task = build_targets(
        data,
        label,
        confident_error_top1_prob=cfg.confident_error_top1_prob,
    )
    X = np.asarray(data["X_full"], dtype=np.float32)
    return X, y, splits, task, data


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
    label = label or cfg.label_default
    out_dir = cfg.probe_out_dir(split_name, "logistic", features=features)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_full, y, splits, task, _data = _load_xy(cfg, split_name, label)
    feat_idx, feat_meta = resolve_feature_indices(
        cfg,
        features,
        d_mlp=X_full.shape[1],
        phase1_dir=phase1_dir,
        device=device,
        skip_phase1=skip_phase1,
    )
    X = select_features(X_full, feat_idx)
    print(
        f"Logistic features={features}: d_in={X.shape[1]} "
        f"(from d_mlp={X_full.shape[1]})"
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
    cv = min(cfg.logistic.cv, max(2, len(np.unique(y_train)) if task != "regression" else 3))
    if task == "binary" and len(np.unique(y_train)) < 2:
        raise RuntimeError("Train set has a single class; collect more data or change --label")

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=cfg.seed,
        n_jobs=-1,
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
            "d_mlp": int(X_full.shape[1]),
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
        "d_mlp": int(X_full.shape[1]),
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
