"""Plots for Exp 2 inter-layer ρ analysis."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from exp2.metrics import cosines_with_basis
from plotly_style import save_plotly_html

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
    html_path = path.with_suffix(".html")
    save_plotly_html(fig, html_path)
    print(f"  saved {html_path}")
    if _SAVE_PNG:
        try:
            fig.write_image(str(path.with_suffix(".png")), scale=2)
            print(f"  saved {path.with_suffix('.png')}")
        except Exception as exc:
            print(f"  PNG skipped: {exc}")


def _n_layers_from_summary(summary_df: pd.DataFrame) -> int:
    return int(summary_df["writer_layer"].max()) + 1


def _is_wu(df: pd.DataFrame) -> pd.Series:
    return df["reader"].astype(str).eq("W_U") | df["reader_kind"].astype(str).eq("W_U")


def _writer_reader_matrix(summary_df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, list[str], list[str]]:
    n = _n_layers_from_summary(summary_df)
    cols = [f"mlp{j}" for j in range(n)] + ["W_U"]
    rows = [f"MLP{i}" for i in range(n)]
    mat = np.full((n, n + 1), np.nan, dtype=float)
    for r in summary_df.itertuples():
        writer = int(r.writer_layer)
        value = float(getattr(r, value_col))
        if str(r.reader) == "W_U" or str(r.reader_kind) == "W_U":
            mat[writer, n] = value
            continue
        reader_layer = getattr(r, "reader_layer", None)
        if reader_layer is None or (isinstance(reader_layer, float) and np.isnan(reader_layer)):
            continue
        reader = int(reader_layer)
        if reader > writer:
            mat[writer, reader] = value
    return mat, rows, cols


def plot_writer_reader_heatmaps(
    summary_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    expected_rho: float,
) -> None:
    """Writer × reader heatmaps; lower triangle (reader <= writer) is masked."""
    metrics = [
        ("rho_mean", "ρ mean (unread)"),
        ("rho_p95", "ρ p95 (unread)"),
    ]
    n = _n_layers_from_summary(summary_df)
    width = max(720, 28 * (n + 1))
    height = max(480, 28 * n)
    for col, label in metrics:
        mat, rows, cols = _writer_reader_matrix(summary_df, col)
        fig = px.imshow(
            mat,
            x=cols,
            y=rows,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels={"color": label},
            title=(
                f"{model_title} — Ext2 {label} (writer MLP × later reader); "
                f"isotropic E[ρ]≈{expected_rho:.3f}"
            ),
        )
        fig.update_layout(width=width, height=height, xaxis_title="Reader", yaxis_title="Writer")
        fig.update_yaxes(autorange="reversed")
        _save_fig(fig, out_dir / f"heatmap_{col}")


def plot_hop_profile(
    summary_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    expected_rho: float,
) -> None:
    """Mean pair stats vs MLP hop (W_U excluded; that is a separate plot)."""
    mlp = summary_df.loc[~_is_wu(summary_df)].copy()
    if mlp.empty:
        return
    grouped = (
        mlp.groupby("hop", sort=True)
        .agg(
            rho_mean=("rho_mean", "mean"),
            rho_p95=("rho_p95", "mean"),
            n_pairs=("writer_layer", "size"),
        )
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=grouped["hop"], y=grouped["rho_mean"], mode="lines+markers", name="ρ mean")
    )
    fig.add_trace(
        go.Scatter(x=grouped["hop"], y=grouped["rho_p95"], mode="lines+markers", name="ρ p95")
    )
    fig.add_hline(
        y=expected_rho,
        line_dash="dash",
        line_color="#888",
        annotation_text=f"isotropic √(k/d)={expected_rho:.3f}",
    )
    fig.update_layout(
        title=f"{model_title} — Ext2 MLP→MLP ρ vs hop (avg over writers)",
        xaxis_title="Hop (reader layer − writer layer)",
        yaxis_title="Pair-level statistic",
        yaxis=dict(range=[0, 1.05]),
        width=720,
        height=440,
    )
    _save_fig(fig, out_dir / "hop_profile")


def plot_layers_to_wu(
    summary_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    expected_rho: float,
) -> None:
    """Every writer MLP → W_U (RQ3: unembedding-null geometry across depth)."""
    wu = summary_df.loc[_is_wu(summary_df)].sort_values("writer_layer")
    if wu.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=wu["writer_layer"], y=wu["rho_mean"], mode="lines+markers", name="ρ mean")
    )
    fig.add_trace(
        go.Scatter(x=wu["writer_layer"], y=wu["rho_p95"], mode="lines+markers", name="ρ p95")
    )
    fig.add_hline(
        y=expected_rho,
        line_dash="dash",
        line_color="#888",
        annotation_text=f"isotropic √(k/d)={expected_rho:.3f}",
    )
    fig.update_layout(
        title=f"{model_title} — Ext2 every MLP → W_U (unembedding null space)",
        xaxis_title="Writer MLP layer",
        yaxis_title="Pair-level statistic",
        yaxis=dict(range=[0, 1.05]),
        width=720,
        height=440,
    )
    _save_fig(fig, out_dir / "layers_to_wu")


def plot_write_vs_read_udark(
    summary_df: pd.DataFrame,
    read_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    expected_rho: float,
) -> None:
    """Write ρ into U_dark(W_U) vs each MLP W_in read of the same subspace."""
    wu = summary_df.loc[_is_wu(summary_df)].sort_values("writer_layer")
    if wu.empty or read_df.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wu["writer_layer"],
            y=wu["rho_mean"],
            mode="lines+markers",
            name="write ρ mean → W_U",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wu["writer_layer"],
            y=wu["rho_p95"],
            mode="lines+markers",
            name="write ρ p95 → W_U",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=read_df["layer"],
            y=read_df["read_rho"],
            mode="lines+markers",
            name="read r_ℓ of U_dark",
            line=dict(dash="dot"),
        )
    )
    fig.add_hline(
        y=expected_rho,
        line_dash="dash",
        line_color="#888",
        annotation_text=f"isotropic √(k/d)={expected_rho:.3f}",
    )
    fig.update_layout(
        title=f"{model_title} — Ext2 write vs read on U_dark(W_U)",
        xaxis_title="MLP layer",
        yaxis_title="ρ / read r_ℓ",
        yaxis=dict(range=[0, 1.05]),
        width=720,
        height=440,
    )
    _save_fig(fig, out_dir / "write_vs_read_udark")


def plot_top_wu_visibility_heatmap(
    vis_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    n_layers: int,
) -> None:
    """ρ vs later MLP W_in dark tails for top-ρ→W_U neurons (masked m≤ℓ)."""
    if vis_df.empty:
        return
    plot_df = vis_df.copy()
    plot_df["row"] = plot_df["component_name"].astype(str)
    cols = [f"mlp{j}" for j in range(n_layers)]
    rows = (
        plot_df[["writer_layer", "row"]]
        .drop_duplicates()
        .sort_values(["writer_layer", "row"])["row"]
        .tolist()
    )
    mat = np.full((len(rows), n_layers), np.nan, dtype=float)
    row_index = {name: i for i, name in enumerate(rows)}
    for r in plot_df.itertuples():
        i = row_index.get(str(r.component_name))
        if i is None:
            continue
        m = int(r.reader_layer)
        if 0 <= m < n_layers:
            mat[i, m] = float(r.rho_to_reader_dark)
    fig = px.imshow(
        mat,
        x=cols,
        y=rows,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"color": "ρ to later W_in dark"},
        title=(
            f"{model_title} — Ext2 top-ρ→W_U vs later MLP W_in "
            "(high ρ = later MLP does not read the write)"
        ),
        zmin=0,
        zmax=1,
    )
    fig.update_layout(
        width=max(720, 28 * n_layers),
        height=max(480, 18 * max(len(rows), 1)),
        xaxis_title="Later MLP reader",
        yaxis_title="Writer neuron",
    )
    fig.update_yaxes(autorange="reversed")
    _save_fig(fig, out_dir / "top_wu_visibility_heatmap")


def plot_rho_histograms(
    neuron_df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    expected_rho: float,
) -> None:
    """ρ histograms for the per-neuron file (consecutive MLP + every layer→W_U)."""
    df = neuron_df.copy()
    is_wu = _is_wu(df)
    consecutive = (~is_wu) & (df["hop"] == 1) if "hop" in df.columns else ~is_wu
    _facet_rho_hist(
        df.loc[consecutive],
        out_dir / "rho_histograms_consecutive",
        f"{model_title} — Ext2 ρ (consecutive MLP→MLP); isotropic E[ρ]≈{expected_rho:.3f}",
        expected_rho,
    )
    _facet_rho_hist(
        df.loc[is_wu],
        out_dir / "rho_histograms_to_wu",
        f"{model_title} — Ext2 ρ (every MLP → W_U); isotropic E[ρ]≈{expected_rho:.3f}",
        expected_rho,
    )


def _facet_rho_hist(plot_df: pd.DataFrame, path: pathlib.Path, title: str, expected_rho: float) -> None:
    if plot_df.empty:
        return
    plot_df = plot_df.copy()
    plot_df["pair"] = plot_df.apply(
        lambda r: f"MLP{int(r['writer_layer'])}->{r['reader']}", axis=1
    )
    n_pairs = int(plot_df["pair"].nunique())
    fig = px.histogram(
        plot_df,
        x="rho",
        facet_col="pair",
        facet_col_wrap=4,
        nbins=40,
        title=title,
    )
    fig.update_layout(height=220 * max(1, (n_pairs + 3) // 4), width=900)
    fig.update_xaxes(range=[0, 1.05])
    fig.add_vline(x=expected_rho, line_dash="dash", line_color="#888")
    _save_fig(fig, path)


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
