"""Synthetic data for Phase 3b sanity checks."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from exp1.settings import DatasetName


def make_synthetic_detailed_df(
    dataset: DatasetName,
    *,
    model_title: str = "synthetic-model",
    n_items: int = 8,
) -> pd.DataFrame:
    """DataFrame matching production detailed.csv schema for plot/aggregate tests."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    conditions = ["baseline", "ablate_entropy", "ablate_random"]
    branches = ["none", "te", "de_ln"]
    decode_modes = ["greedy", "sampling"]

    for item_i in range(n_items):
        item_id = f"{dataset}-item-{item_i}"
        base_entropy = float(rng.uniform(1.5, 4.0))
        base_act = float(rng.uniform(0.1, 2.0))
        base_metric = float(rng.uniform(0.2, 0.9))

        for decode_mode in decode_modes:
            for condition, branch in itertools.product(conditions, branches):
                if condition == "baseline" and branch != "none":
                    continue
                if condition != "baseline" and branch == "none":
                    continue

                shift = 0.0
                if condition == "ablate_entropy":
                    shift = -0.08 if branch == "te" else -0.04
                elif condition == "ablate_random":
                    shift = -0.01

                entropy = base_entropy + shift + rng.normal(0, 0.05)
                act = base_act + shift * 2 + rng.normal(0, 0.05)

                row: dict = {
                    "item_id": item_id,
                    "dataset": dataset,
                    "split_name": "sanity",
                    "condition": condition,
                    "ablation_branch": branch,
                    "ablation_mode": "mean_all_entropy",
                    "decode_mode": decode_mode,
                    "component_name": "ALL" if condition != "baseline" else "",
                    "prompt": f"[{model_title}] prompt for {item_id}",
                    "gold_answer": "Paris" if dataset == "lama" else "France",
                    "gold_aliases": '["Paris"]' if dataset == "lama" else '["France", "france"]',
                    "prediction": "Paris" if dataset == "lama" else "France",
                    "correct": True,
                    "entropy_at_answer": entropy,
                    "top1_prob_at_answer": float(rng.uniform(0.1, 0.95)),
                    "mean_entropy_neuron_act": act,
                    "n_ablated_neurons": 10 if condition != "baseline" else 0,
                    "generated_text": None if dataset == "lama" else "France",
                    "rank": 0,
                }

                if dataset == "lama":
                    m = max(0.0, min(1.0, base_metric + shift + rng.normal(0, 0.03)))
                    row.update(
                        {
                            "acc_at_1": m,
                            "acc_at_5": min(1.0, m + 0.15),
                            "mrr": m,
                            "em": None,
                            "f1": None,
                        }
                    )
                else:
                    m = max(0.0, min(1.0, base_metric + shift + rng.normal(0, 0.03)))
                    row.update(
                        {
                            "acc_at_1": None,
                            "acc_at_5": None,
                            "mrr": None,
                            "em": m,
                            "f1": m,
                        }
                    )
                rows.append(row)

    df = pd.DataFrame(rows)
    df["delta_entropy"] = np.nan
    df["delta_mean_entropy_neuron_act"] = np.nan
    for (item_id, decode_mode), grp in df.groupby(["item_id", "decode_mode"]):
        base = grp[grp["condition"] == "baseline"].iloc[0]
        mask = (df["item_id"] == item_id) & (df["decode_mode"] == decode_mode) & (
            df["condition"] != "baseline"
        )
        df.loc[mask, "delta_entropy"] = df.loc[mask, "entropy_at_answer"] - base["entropy_at_answer"]
        df.loc[mask, "delta_mean_entropy_neuron_act"] = (
            df.loc[mask, "mean_entropy_neuron_act"] - base["mean_entropy_neuron_act"]
        )
    return df
