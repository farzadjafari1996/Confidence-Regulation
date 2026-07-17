"""Plots for Ext3 probe evaluation."""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def write_exp3_plots(
    out_dir: pathlib.Path,
    importance: dict[str, Any],
    test_metrics: dict[str, Any],
    *,
    probe: str,
    label: str,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    top = importance.get("top_neurons") or []
    if top:
        idxs = [t["neuron_index"] for t in top]
        vals = [t["importance"] for t in top]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(idxs)), vals, color="#4C78A8")
        ax.set_xticks(range(len(idxs)))
        ax.set_xticklabels([str(i) for i in idxs], rotation=90, fontsize=7)
        ax.set_ylabel("Permutation importance (Δscore)")
        ax.set_title(f"Ext3 {probe} / {label}: top-{len(idxs)} neurons")
        fig.tight_layout()
        fig.savefig(plots_dir / "top_neuron_importance.png", dpi=150)
        plt.close(fig)

    keys = [
        k
        for k in ("auroc", "f1", "f1_macro", "accuracy", "r2")
        if k in test_metrics and test_metrics[k] is not None
    ]
    if keys:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(keys, [float(test_metrics[k]) for k in keys], color="#F58518")
        ax.set_ylim(0, 1.05 if all(float(test_metrics[k]) <= 1.05 for k in keys) else None)
        ax.set_title(f"Ext3 {probe} / {label}: test metrics")
        fig.tight_layout()
        fig.savefig(plots_dir / "test_metrics.png", dpi=150)
        plt.close(fig)

    imps = np.asarray(importance.get("importances") or [], dtype=np.float64)
    if imps.size:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(imps, bins=50, color="#54A24B")
        ax.set_xlabel("Permutation importance")
        ax.set_ylabel("Neuron count")
        ax.set_title(f"Ext3 {probe} / {label}: importance distribution")
        fig.tight_layout()
        fig.savefig(plots_dir / "importance_hist.png", dpi=150)
        plt.close(fig)
