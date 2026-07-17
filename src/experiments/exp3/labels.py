"""Build training targets from stored Ext3 label fields."""

from __future__ import annotations

from typing import Any

import numpy as np

from exp3.settings import LabelName


def build_targets(
    data: dict[str, Any],
    label: LabelName,
    *,
    confident_error_top1_prob: float = 0.5,
) -> tuple[np.ndarray, str]:
    """Return (y, task) where task is 'binary', 'multiclass', or 'regression'."""
    y_top1 = np.asarray(data["y_top1"], dtype=np.int64)
    gold_rank = np.asarray(data["gold_rank"], dtype=np.int64)
    top1_prob = np.asarray(data["top1_prob"], dtype=np.float64)
    gold_logprob = np.asarray(data["gold_logprob"], dtype=np.float64)

    if label == "top1":
        return y_top1.copy(), "binary"
    if label == "top5":
        return (gold_rank < 5).astype(np.int64), "binary"
    if label == "confident_error":
        wrong = y_top1 == 0
        confident = top1_prob >= float(confident_error_top1_prob)
        return (wrong & confident).astype(np.int64), "binary"
    if label == "multiclass":
        # 0=correct, 1=wrong_confident, 2=wrong_uncertain
        y = np.zeros_like(y_top1)
        wrong = y_top1 == 0
        confident = top1_prob >= float(confident_error_top1_prob)
        y[wrong & confident] = 1
        y[wrong & ~confident] = 2
        return y, "multiclass"
    if label == "gold_logprob":
        return gold_logprob.copy(), "regression"
    raise ValueError(f"Unknown label: {label}")
