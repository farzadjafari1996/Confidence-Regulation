"""Plotly figures for paper Fig 4 and Fig 5 (GPT-2 Phase 3)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative

from gpt2.core import PLOT_COLORS, _save_fig


def plot_fig4a_delta_loss_vs_rank(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    neuron_name: str,
) -> None:
    """Fig 4(a): Δloss after ablation vs reciprocal rank of correct token."""
    plot_df = df.copy()
    plot_df["delta_loss"] = plot_df["loss_post_ablation"] - plot_df["loss"]
    plot_df["reciprocal_rank"] = 1.0 / (plot_df["rank_of_correct_token"] + 1)
    plot_df = plot_df[plot_df["rank_of_correct_token"] >= 0]

    fig = px.scatter(
        plot_df,
        x="loss",
        y="delta_loss",
        color="reciprocal_rank",
        color_continuous_scale="Viridis",
        title=f"{model_title} — (a) {neuron_name}: ΔLoss Upon Ablation",
        labels={"loss": "Loss", "delta_loss": "ΔLoss Post Ablation", "reciprocal_rank": "1/RR"},
    )
    fig.update_layout(width=520, height=420, title_font_size=16)
    _save_fig(fig, out_dir / "fig4a_delta_loss_vs_rank")


def plot_fig4b_activation_trace(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    neuron_name: str,
    batch_id: int,
) -> None:
    """Fig 4(b): activation trace on one C4 example."""
    row_df = df[df["batch"] == batch_id].sort_values("pos")
    act_col = f"{neuron_name}_activation"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=row_df["pos"],
            y=row_df[act_col],
            mode="lines+markers",
            name=f"{neuron_name} activation",
            line=dict(color=qualitative.Plotly[1], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=row_df["pos"],
            y=row_df["loss"],
            mode="lines",
            name="Loss",
            yaxis="y2",
            line=dict(color=qualitative.Plotly[0], width=2),
        )
    )
    fig.update_layout(
        title=f"{model_title} — (b) {neuron_name}: Example Activation",
        xaxis_title="Position in sequence",
        yaxis_title="Activation",
        yaxis2=dict(title="Loss", overlaying="y", side="right"),
        width=620,
        height=420,
        title_font_size=16,
    )
    _save_fig(fig, out_dir / "fig4b_activation_trace")


def plot_fig5a_induction_activations(
    agg_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    entropy_neurons: list[str],
    induction_start: int,
) -> None:
    """Fig 5(a): mean entropy, loss, and entropy-neuron activations vs position."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg_df["pos"],
            y=agg_df["entropy"],
            mode="lines",
            name="Entropy",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg_df["pos"],
            y=agg_df["loss"],
            mode="lines",
            name="Loss",
            line=dict(width=3),
        )
    )
    for name in entropy_neurons:
        col = f"{name}_activation"
        if col in agg_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=agg_df["pos"],
                    y=agg_df[col],
                    mode="lines",
                    name=name,
                    line=dict(width=2, dash="dot"),
                )
            )
    fig.add_vline(
        x=induction_start,
        line_dash="dash",
        line_color="black",
        annotation_text="start of induction",
    )
    fig.update_layout(
        title=f"{model_title} — (a) Induction: Activations, Entropy, Loss",
        xaxis_title="Position in Sequence",
        width=700,
        height=420,
        title_font_size=16,
    )
    _save_fig(fig, out_dir / "fig5a_induction_activations")


def plot_fig5b_entropy_post_ablation(
    agg_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    entropy_neurons: list[str],
    induction_start: int,
) -> None:
    """Fig 5(b): entropy_post_ablation / entropy after clip ablation."""
    fig = go.Figure()
    palette = qualitative.Plotly
    for i, name in enumerate(entropy_neurons):
        sub = agg_df[agg_df["component_name"] == name].sort_values("pos")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["pos"],
                y=sub["entropy_ratio"],
                mode="lines",
                name=name,
                line=dict(color=palette[i % len(palette)], width=2),
            )
        )
    fig.add_vline(
        x=induction_start,
        line_dash="dash",
        line_color="black",
        annotation_text="start of induct.",
    )
    fig.update_layout(
        title=f"{model_title} — (b) Change in Entropy Upon Ablation",
        xaxis_title="Position in Sequence",
        yaxis_title="Entropy Post Ablation / Entropy",
        width=700,
        height=420,
        title_font_size=16,
    )
    _save_fig(fig, out_dir / "fig5b_entropy_post_ablation")


def plot_fig5c_bos_ablation(
    agg_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    target_neuron: str,
    induction_start: int,
    induction_components: list[str],
    baseline_label: str = "Baselines",
) -> None:
    """Fig 5(c): target neuron activation after BOS ablation of induction heads."""
    act_col = f"{target_neuron}_activation"
    post_col = f"{target_neuron}_activation_post_abl"
    fig = go.Figure()

    orig = agg_df.groupby("pos", as_index=False)[act_col].mean()
    fig.add_trace(
        go.Scatter(
            x=orig["pos"],
            y=orig[act_col],
            mode="lines",
            name="Original Act.",
            line=dict(color="black", width=2),
        )
    )

    baselines = [c for c in agg_df["component_name"].unique() if c not in induction_components]
    baseline_shown = False
    for comp in induction_components + baselines:
        sub = agg_df[agg_df["component_name"] == comp].groupby("pos", as_index=False)[post_col].mean()
        if sub.empty:
            continue
        is_baseline = comp in baselines
        fig.add_trace(
            go.Scatter(
                x=sub["pos"],
                y=sub[post_col],
                mode="lines",
                name=baseline_label if is_baseline and not baseline_shown else comp,
                showlegend=not (is_baseline and baseline_shown),
                line=dict(dash="dot" if is_baseline else "solid", width=1.5),
            )
        )
        if is_baseline:
            baseline_shown = True

    fig.add_vline(
        x=induction_start,
        line_dash="dash",
        line_color="black",
        annotation_text="start of ind.",
    )
    fig.update_layout(
        title=f"{model_title} — (c) {target_neuron} Activation after BOS Ablation",
        xaxis_title="Position in Sequence",
        yaxis_title=f"{target_neuron} Activation",
        width=700,
        height=420,
        title_font_size=16,
        legend_title="Attn heads",
    )
    _save_fig(fig, out_dir / "fig5c_bos_ablation")
