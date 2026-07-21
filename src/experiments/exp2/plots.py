"""Plots for Exp 2 inter-layer ρ analysis."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from exp2.metrics import cosines_with_basis

_SAVE_PNG = False

ALIGN_COLORS = {
    "high_rho_individual": "rgba(255, 170, 170, 0.9)",
    "high_rho_mean": "#D62728",
    "random_individual": "rgba(170, 200, 255, 0.85)",
    "random_mean": "#1F77B4",
}


def set_save_png(enabled: bool) -> None:
    global _SAVE_PNG
    _SAVE_PNG = enabled


def _save_fig(fig: go.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html_path = path.with_suffix(".html")
    fig.write_html(str(html_path))
    print(f"  saved {html_path}")
    if _SAVE_PNG:
        try:
            fig.write_image(str(path.with_suffix(".png")), scale=2)
            print(f"  saved {path.with_suffix('.png')}")
        except Exception as exc:
            print(f"  PNG skipped: {exc}")


def plot_rho_heatmap(summary_df: pd.DataFrame, out_dir: pathlib.Path, model_title: str) -> None:
    """Heatmap of ρ stats across writer→reader pairs."""
    plot_df = summary_df.copy()
    plot_df["pair"] = plot_df.apply(
        lambda r: f"MLP{int(r['writer_layer'])}->{r['reader']}", axis=1
    )
    metrics = ["rho_mean", "rho_p95", "rho_max", "n_rho_gt_0_5"]
    long = plot_df.melt(
        id_vars=["pair"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    fig = px.imshow(
        long.pivot(index="metric", columns="pair", values="value"),
        aspect="auto",
        title=f"{model_title} — Ext2 ρ summary across MLP→reader pairs",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(width=max(720, 90 * len(plot_df)), height=360)
    _save_fig(fig, out_dir / "rho_pair_heatmap")


def plot_rho_histograms(all_df: pd.DataFrame, out_dir: pathlib.Path, model_title: str) -> None:
    """Overlay ρ histograms for each pair (faceted)."""
    plot_df = all_df.copy()
    plot_df["pair"] = plot_df.apply(
        lambda r: f"MLP{int(r['writer_layer'])}->{r['reader']}", axis=1
    )
    fig = px.histogram(
        plot_df,
        x="rho",
        facet_col="pair",
        facet_col_wrap=4,
        nbins=40,
        title=f"{model_title} — Ext2 ρ distributions",
    )
    fig.update_layout(height=220 * max(1, (plot_df["pair"].nunique() + 3) // 4), width=900)
    fig.update_xaxes(range=[0, 1.05])
    _save_fig(fig, out_dir / "rho_histograms")


def plot_write_alignment(
    *,
    w_out: torch.Tensor,
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    pair_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    pair_tag: str,
    n_trace: int,
    n_random: int,
    seed: int,
) -> None:
    """Singular values + |cos(w_out, v_j)| for top-ρ vs random neurons."""
    indices = np.arange(len(singular_values))
    top = pair_df.nlargest(n_trace, "rho")
    pool = pair_df[~pair_df["component_name"].isin(top["component_name"])]
    n_pick = min(n_random, len(pool))
    random_df = pool.sample(n_pick, random_state=seed) if n_pick else pool

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=singular_values.detach().cpu().numpy(),
            mode="lines",
            name="Singular values",
            line=dict(color="#B0B0B0", width=2.5),
            yaxis="y",
        )
    )

    top_idx = top["neuron_index"].to_numpy()
    top_cos = cosines_with_basis(w_out[top_idx], basis)
    for cosines in top_cos:
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=cosines,
                mode="lines",
                line=dict(color=ALIGN_COLORS["high_rho_individual"], width=1.0),
                opacity=0.55,
                showlegend=False,
                yaxis="y2",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=top_cos.mean(axis=0),
            mode="lines",
            name=f"Top-ρ mean (n={len(top)})",
            line=dict(color=ALIGN_COLORS["high_rho_mean"], width=2.8),
            yaxis="y2",
        )
    )

    if len(random_df):
        rand_idx = random_df["neuron_index"].to_numpy()
        rand_cos = cosines_with_basis(w_out[rand_idx], basis)
        for cosines in rand_cos:
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=cosines,
                    mode="lines",
                    line=dict(color=ALIGN_COLORS["random_individual"], width=1.0),
                    opacity=0.5,
                    showlegend=False,
                    yaxis="y2",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=rand_cos.mean(axis=0),
                mode="lines",
                name=f"Random mean (n={len(random_df)})",
                line=dict(color=ALIGN_COLORS["random_mean"], width=2.8, dash="dash"),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=(
            f"{model_title} — Ext2 write alignment {pair_tag} "
            f"(|cos(w<sub>out</sub>, v<sub>j</sub>)|)"
        ),
        xaxis=dict(title="Singular vector index (large → small σ)"),
        yaxis=dict(title="Singular value", type="log"),
        yaxis2=dict(
            title="|cos(w<sub>out</sub>, v<sub>j</sub>)|",
            overlaying="y",
            side="right",
            range=[0, 1.05],
        ),
        width=720,
        height=440,
        legend=dict(x=1.12, y=1),
    )
    safe = pair_tag.replace("->", "_to_").replace(".", "_")
    _save_fig(fig, out_dir / f"align_{safe}")
