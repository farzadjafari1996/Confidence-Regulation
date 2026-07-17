"""Classification / regression metrics for Ext3 probes."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    *,
    task: str = "binary",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if task == "binary":
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out["positive_rate"] = float(np.mean(y_true))
        if y_prob is not None and len(np.unique(y_true)) > 1:
            out["auroc"] = float(roc_auc_score(y_true, y_prob))
        else:
            out["auroc"] = None
    else:
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        out["auroc"] = None
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(y_true)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
