"""Base vs instruct: do we find more or fewer entropy neurons?

Phase-1 selection uses fixed top-k (min(1%, n=10/12)), so ``n_selected`` is
equal by construction. This plot answers the question with:

1. Threshold counts — how many last-MLP neurons have ρ ≥ τ
2. Selected-set overlap — shared vs base-only vs instruct-only IDs
3. LN-mediated fraction (Phase 2) — causal signature beside the count story

Writes PNG (+ optional HTML) under ``outputs/base_vs_instruct_entropy/``.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))

OUT_DIR = ROOT / "outputs" / "base_vs_instruct_entropy"
PLOTS_DIR = OUT_DIR / "plots"

PAIRS = [
    {
        "family": "Llama 1B",
        "base_slug": "llama_3.2_1B_base",
        "inst_slug": "llama_3.2_1B_instruct",
    },
    {
        "family": "Qwen 2B",
        "base_slug": "qwen_3.5_2B_base",
        "inst_slug": "qwen_3.5_2B_instruct",
    },
]

THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
C_BASE = "#4C78A8"
C_INST = "#E45756"
C_SHARED = "#54A24B"
C_BASE_ONLY = "#4C78A8"
C_INST_ONLY = "#E45756"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase1(slug: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = ROOT / "outputs" / slug / "exp1" / "phase1"
    df = pd.read_csv(root / "static_neuron_metrics.csv")
    summary = _read_json(root / "summary.json")
    return df, summary


def _ln_mediated(slug: str) -> float | None:
    path = ROOT / "outputs" / slug / "exp1" / "phase2" / "summary.json"
    if not path.exists():
        return None
    entropy_set = (_read_json(path).get("entropy_set") or {})
    val = entropy_set.get("mean_ln_mediated_fraction")
    return float(val) if val is not None else None


def collect_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    ln_rows: list[dict[str, Any]] = []

    for pair in PAIRS:
        base_df, base_sum = _phase1(pair["base_slug"])
        inst_df, inst_sum = _phase1(pair["inst_slug"])
        base_sel = set((base_sum.get("selection") or {}).get("neurons") or [])
        inst_sel = set((inst_sum.get("selection") or {}).get("neurons") or [])
        shared = base_sel & inst_sel
        base_only = base_sel - inst_sel
        inst_only = inst_sel - base_sel

        for tau in THRESHOLDS:
            count_rows.append(
                {
                    "family": pair["family"],
                    "variant": "Base",
                    "tau": tau,
                    "n": int((base_df["rho"] >= tau).sum()),
                    "n_mlp": len(base_df),
                }
            )
            count_rows.append(
                {
                    "family": pair["family"],
                    "variant": "Instruct",
                    "tau": tau,
                    "n": int((inst_df["rho"] >= tau).sum()),
                    "n_mlp": len(inst_df),
                }
            )

        overlap_rows.append(
            {
                "family": pair["family"],
                "shared": len(shared),
                "base_only": len(base_only),
                "inst_only": len(inst_only),
                "n_selected_base": len(base_sel),
                "n_selected_inst": len(inst_sel),
                "jaccard": (
                    len(shared) / len(base_sel | inst_sel) if (base_sel | inst_sel) else 0.0
                ),
                "mean_rho_base": float(
                    np.mean([s["rho"] for s in (base_sum.get("selected_stats") or [])])
                )
                if base_sum.get("selected_stats")
                else None,
                "mean_rho_inst": float(
                    np.mean([s["rho"] for s in (inst_sum.get("selected_stats") or [])])
                )
                if inst_sum.get("selected_stats")
                else None,
            }
        )

        for slug, variant in (
            (pair["base_slug"], "Base"),
            (pair["inst_slug"], "Instruct"),
        ):
            ln_rows.append(
                {
                    "family": pair["family"],
                    "variant": variant,
                    "ln_mediated": _ln_mediated(slug),
                }
            )

    return pd.DataFrame(count_rows), pd.DataFrame(overlap_rows), pd.DataFrame(ln_rows)


def plot_figure(counts: pd.DataFrame) -> pathlib.Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    families = [p["family"] for p in PAIRS]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(THRESHOLDS))
    n_fam = len(families)
    pair_width = 0.22
    gap = 0.04
    group_width = n_fam * pair_width + (n_fam - 1) * gap
    start = -group_width / 2 + pair_width / 2

    for fi, family in enumerate(families):
        center = start + fi * (pair_width + gap)
        for vi, (variant, color) in enumerate((("Base", C_BASE), ("Instruct", C_INST))):
            sub = counts[(counts["family"] == family) & (counts["variant"] == variant)]
            ys = [int(sub.loc[sub["tau"] == t, "n"].iloc[0]) for t in THRESHOLDS]
            xpos = x + center + (vi - 0.5) * (pair_width / 2)
            ax.bar(
                xpos,
                ys,
                width=pair_width / 2 * 0.92,
                color=color,
                edgecolor="#222",
                linewidth=0.5,
                hatch="//" if variant == "Instruct" else "",
                label=variant if fi == 0 else None,
                alpha=0.92,
            )
            for xi, yi in zip(xpos, ys):
                ax.text(xi, yi + 0.6, str(yi), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"ρ ≥ {t:g}" for t in THRESHOLDS])
    ax.set_ylabel("# last-MLP neurons")
    ax.set_title("Do instruct-tuned variants show more or fewer entropy neurons?")
    ax.set_ylim(0, float(counts["n"].max()) * 1.18)
    ax.legend(fontsize=9, loc="upper right", frameon=False, title="Variant")
    ax.text(
        0.02,
        0.98,
        "Groups L→R: Llama 1B · Qwen 2B\n"
        "(counts with ρ ≥ τ; not fixed top-k selection)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#555",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = PLOTS_DIR / "base_vs_instruct_entropy_counts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts, overlap, ln_df = collect_rows()
    counts.to_csv(OUT_DIR / "rho_threshold_counts.csv", index=False)
    overlap.to_csv(OUT_DIR / "selected_overlap.csv", index=False)
    ln_df.to_csv(OUT_DIR / "ln_mediated.csv", index=False)

    path = plot_figure(counts)
    print(f"wrote {path}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
