"""
Per Phase-1 entropy candidate: prediction-change % after mean ablation (Phase 2).

Reads existing outputs (no model forward):
  outputs/llama_3.2_1B_base/exp1/phase1/top1pct_rho_neurons.json
  outputs/llama_3.2_1B_base/exp1/phase2/ablation.feather

Writes next to this script:
  entropy_pred_change.csv
  entropy_pred_change.html

Usage (from repo root, venv active):
  python src/temp_experiments/llama_3.2_1B_base/entropy_pred_change.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px

MODEL_SLUG = "llama_3.2_1B_base"

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE
while _ROOT != _ROOT.parent and not (_ROOT / "outputs").is_dir():
    _ROOT = _ROOT.parent
if not (_ROOT / "outputs").is_dir():
    raise SystemExit(f"Could not find repo root with outputs/ from {_HERE}")

_PHASE1 = _ROOT / "outputs" / MODEL_SLUG / "exp1" / "phase1"
_PHASE2 = _ROOT / "outputs" / MODEL_SLUG / "exp1" / "phase2"
_TOP1 = _PHASE1 / "top1pct_rho_neurons.json"
_FEATHER = _PHASE2 / "ablation.feather"


def main() -> None:
    if not _TOP1.exists():
        raise FileNotFoundError(f"Missing Phase 1 list: {_TOP1}")
    if not _FEATHER.exists():
        raise FileNotFoundError(f"Missing Phase 2 feather: {_FEATHER}")

    neurons: list[str] = json.loads(_TOP1.read_text(encoding="utf-8"))["neurons"]
    df = pd.read_feather(_FEATHER)
    df = df[~df["component_name"].astype(str).str.startswith("joint.")].copy()

    df["pred_change"] = (df["top_logits_post_ablation"] != df["pred"]).astype(float)
    df["abs_delta_h"] = (df["entropy_post_ablation"] - df["entropy"]).abs()

    rows: list[dict] = []
    for name in neurons:
        sub = df[df["component_name"] == name]
        loss_post = sub["loss_post_ablation"].astype(float)
        finite_frac = float(np.isfinite(loss_post).mean()) if len(sub) else 0.0
        # NaN post-ablation logits make != comparisons meaningless; leave metrics empty.
        if finite_frac < 0.99:
            pred_pct = float("nan")
            abs_dh = float("nan")
        else:
            pred_pct = float(sub["pred_change"].mean() * 100.0)
            abs_dh = float(sub["abs_delta_h"].mean())
        rows.append(
            {
                "component_name": name,
                "n_tokens": int(len(sub)),
                "pred_change_pct": pred_pct,
                "abs_delta_h": abs_dh,
                "finite_frac": finite_frac,
            }
        )

    agg = pd.DataFrame(rows).sort_values("pred_change_pct", ascending=True)

    rnd = df[~df["component_name"].isin(neurons)]
    rnd_pred_pct = float((rnd["top_logits_post_ablation"] != rnd["pred"]).mean() * 100.0)
    rnd_abs_dh = float((rnd["entropy_post_ablation"] - rnd["entropy"]).abs().mean())

    out_csv = _HERE / "entropy_pred_change.csv"
    agg.to_csv(out_csv, index=False)
    print(agg.to_string(index=False))
    print()
    print(f"random controls: pred_change_pct={rnd_pred_pct:.4f}  abs_delta_h={rnd_abs_dh:.6f}")
    print(f"saved {out_csv}")

    plot_df = agg.copy()
    plot_df["status"] = np.where(plot_df["finite_frac"] > 0.99, "finite", "nan_ablation")
    fig = px.bar(
        plot_df,
        x="component_name",
        y="pred_change_pct",
        color="status",
        hover_data=["abs_delta_h", "finite_frac", "n_tokens"],
        title=f"{MODEL_SLUG} — Phase-1 entropy candidates: prediction change (%)",
        labels={
            "component_name": "Entropy candidate",
            "pred_change_pct": "Prediction Change on Ablation (%)",
            "status": "Ablation",
        },
        color_discrete_map={"finite": "#EF553B", "nan_ablation": "#AAAAAA"},
    )
    fig.add_hline(
        y=rnd_pred_pct,
        line_dash="dash",
        line_color="#636EFA",
        annotation_text=f"random mean ({rnd_pred_pct:.2f}%)",
        annotation_position="top left",
    )
    fig.update_layout(
        width=720,
        height=420,
        title_font_size=15,
        xaxis_tickangle=-45,
        margin=dict(b=100),
    )
    out_html = _HERE / "entropy_pred_change.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"saved {out_html}")


if __name__ == "__main__":
    main()
