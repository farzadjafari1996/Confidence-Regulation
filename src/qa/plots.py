"""Plotly figures for Phase 3b QA ablation results."""

from __future__ import annotations

import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gpt2.core import _save_fig

_PLOT_LAYOUT = dict(width=900, height=450, title_font_size=14)


def _metric_col(dataset: str) -> str:
    return "mrr" if dataset == "lama" else "f1"


def _resolve_metric(df: pd.DataFrame, dataset: str) -> str:
    metric = _metric_col(dataset)
    if metric in df.columns and df[metric].notna().any():
        return metric
    for fallback in ("acc_at_1", "em", "f1"):
        if fallback in df.columns and df[fallback].notna().any():
            return fallback
    return metric


def _baseline_table(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """One baseline row per (item_id, decode_mode); baseline always uses branch 'none'."""
    cols = ["item_id", "decode_mode", *value_cols]
    cols = [c for c in cols if c in df.columns]
    base = df[df["condition"] == "baseline"][cols].copy()
    if base.empty:
        return base
    return base.groupby(["item_id", "decode_mode"], as_index=False).first()


def _delta_vs_baseline(
    df: pd.DataFrame,
    metric: str,
    *,
    conditions: tuple[str, ...] = ("ablate_entropy", "ablate_random"),
) -> pd.DataFrame:
    """Pair each ablated row with the matching baseline (same item, decode_mode)."""
    value_cols = [metric, "entropy_at_answer", "top1_prob_at_answer", "mean_entropy_neuron_act"]
    value_cols = [c for c in value_cols if c in df.columns]
    base = _baseline_table(df, value_cols)
    if base.empty:
        return pd.DataFrame()

    base = base.rename(
        columns={
            metric: "base_metric",
            "entropy_at_answer": "base_entropy",
            "top1_prob_at_answer": "base_top1_prob",
            "mean_entropy_neuron_act": "base_neuron_act",
        }
    )

    parts: list[pd.DataFrame] = []
    for condition in conditions:
        abl = df[df["condition"] == condition].copy()
        if abl.empty:
            continue
        merged = abl.merge(base, on=["item_id", "decode_mode"], how="inner")
        if merged.empty:
            continue
        merged["delta_metric"] = merged[metric] - merged["base_metric"]
        if "entropy_at_answer" in merged.columns:
            merged["delta_entropy"] = merged["entropy_at_answer"] - merged["base_entropy"]
        if "top1_prob_at_answer" in merged.columns:
            merged["delta_top1_prob"] = merged["top1_prob_at_answer"] - merged["base_top1_prob"]
        if "mean_entropy_neuron_act" in merged.columns:
            merged["delta_neuron_act"] = merged["mean_entropy_neuron_act"] - merged["base_neuron_act"]
        parts.append(merged)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _zoom_yaxis(fig: go.Figure, values: pd.Series) -> None:
    """Tighten y-axis when task metrics are small (e.g. LAMA MRR on GPT-2)."""
    clean = values.dropna()
    if clean.empty:
        return
    y_min, y_max = float(clean.min()), float(clean.max())
    if y_max - y_min < 0.05:
        pad = max((y_max - y_min) * 0.2, y_max * 0.05, 1e-4)
        fig.update_yaxes(range=[y_min - pad, y_max + pad])


def plot_metrics_by_condition(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    *,
    dataset: str,
    model_title: str,
    split_name: str,
) -> None:
    metric = _resolve_metric(df, dataset)
    agg = (
        df.groupby(["decode_mode", "condition", "ablation_branch"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "metric_value"})
    )
    # Baseline uses branch "none"; show it as a single series for readability.
    agg.loc[agg["condition"] == "baseline", "ablation_branch"] = "baseline"

    fig = px.bar(
        agg,
        x="condition",
        y="metric_value",
        color="ablation_branch",
        facet_col="decode_mode",
        barmode="group",
        category_orders={
            "condition": ["baseline", "ablate_entropy", "ablate_random"],
            "ablation_branch": ["baseline", "te", "de_ln"],
        },
        title=f"{model_title} — {dataset} {split_name}: mean {metric} by condition",
    )
    fig.update_layout(**_PLOT_LAYOUT)
    _zoom_yaxis(fig, agg["metric_value"])
    _save_fig(fig, out_dir / "metrics_by_condition")


def plot_metric_deltas_by_condition(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    *,
    dataset: str,
    model_title: str,
    split_name: str,
) -> None:
    metric = _resolve_metric(df, dataset)
    merged = _delta_vs_baseline(df, metric)
    if merged.empty:
        return

    agg = (
        merged.groupby(["decode_mode", "condition", "ablation_branch"], as_index=False)["delta_metric"]
        .mean()
        .rename(columns={"delta_metric": "mean_delta"})
    )
    fig = px.bar(
        agg,
        x="condition",
        y="mean_delta",
        color="ablation_branch",
        facet_col="decode_mode",
        barmode="group",
        category_orders={
            "condition": ["ablate_entropy", "ablate_random"],
            "ablation_branch": ["te", "de_ln"],
        },
        title=f"{model_title} — {dataset} {split_name}: Δ{metric} vs baseline",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(**_PLOT_LAYOUT)
    _save_fig(fig, out_dir / "metric_deltas_by_condition")


def plot_entropy_by_condition(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    *,
    dataset: str,
    model_title: str,
    split_name: str,
) -> None:
    if "entropy_at_answer" not in df.columns or df["entropy_at_answer"].isna().all():
        return

    agg = df.groupby(["decode_mode", "condition", "ablation_branch"], as_index=False)[
        "entropy_at_answer"
    ].mean()
    agg.loc[agg["condition"] == "baseline", "ablation_branch"] = "baseline"

    fig = px.bar(
        agg,
        x="condition",
        y="entropy_at_answer",
        color="ablation_branch",
        facet_col="decode_mode",
        barmode="group",
        category_orders={
            "condition": ["baseline", "ablate_entropy", "ablate_random"],
            "ablation_branch": ["baseline", "te", "de_ln"],
        },
        title=f"{model_title} — {dataset} {split_name}: entropy at answer by condition",
    )
    fig.update_layout(**_PLOT_LAYOUT)
    _save_fig(fig, out_dir / "entropy_by_condition")


def plot_delta_histogram(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    *,
    dataset: str,
    model_title: str,
    split_name: str,
) -> None:
    metric = _resolve_metric(df, dataset)
    merged = _delta_vs_baseline(df, metric, conditions=("ablate_entropy",))
    if merged.empty:
        return

    fig = px.histogram(
        merged,
        x="delta_metric",
        color="ablation_branch",
        facet_col="decode_mode",
        nbins=30,
        category_orders={"ablation_branch": ["te", "de_ln"]},
        title=f"{model_title} — {dataset} {split_name}: Δ{metric} (entropy ablation − baseline)",
    )
    fig.update_layout(**_PLOT_LAYOUT)
    _save_fig(fig, out_dir / "delta_metric_histogram")

    if "delta_entropy" not in merged.columns:
        return

    fig2 = px.scatter(
        merged,
        x="delta_entropy",
        y="delta_metric",
        color="ablation_branch",
        facet_col="decode_mode",
        category_orders={"ablation_branch": ["te", "de_ln"]},
        title=f"{model_title} — {dataset} {split_name}: Δentropy vs Δ{metric}",
    )
    fig2.update_layout(**_PLOT_LAYOUT)
    _save_fig(fig2, out_dir / "entropy_vs_metric_change")


def plot_per_neuron_ablation(
    full_df: pd.DataFrame,
    out_dir: pathlib.Path,
    *,
    dataset: str,
    model_title: str,
    split_name: str,
) -> None:
    sub = full_df[full_df["ablation_mode"] == "per_neuron_avg"].copy()
    if sub.empty or "component_name" not in sub.columns:
        return

    metric = _resolve_metric(sub, dataset)
    agg = (
        sub.groupby(["decode_mode", "ablation_branch", "component_name"], as_index=False)[metric]
        .mean()
        .sort_values(["decode_mode", "ablation_branch", metric])
    )

    fig = px.bar(
        agg,
        x="component_name",
        y=metric,
        color="ablation_branch",
        facet_col="decode_mode",
        barmode="group",
        category_orders={"ablation_branch": ["te", "de_ln"]},
        title=f"{model_title} — {dataset} {split_name}: per-neuron mean {metric}",
    )
    fig.update_layout(**_PLOT_LAYOUT, xaxis_tickangle=-45)
    _save_fig(fig, out_dir / "per_neuron_metric")


def write_ext1_plots(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    *,
    dataset: str,
    model_title: str,
    split_name: str,
    full_df: pd.DataFrame | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics_by_condition(df, out_dir, dataset=dataset, model_title=model_title, split_name=split_name)
    plot_metric_deltas_by_condition(df, out_dir, dataset=dataset, model_title=model_title, split_name=split_name)
    plot_entropy_by_condition(df, out_dir, dataset=dataset, model_title=model_title, split_name=split_name)
    plot_delta_histogram(df, out_dir, dataset=dataset, model_title=model_title, split_name=split_name)
    if full_df is not None:
        plot_per_neuron_ablation(
            full_df, out_dir, dataset=dataset, model_title=model_title, split_name=split_name
        )


write_phase3b_plots = write_ext1_plots
