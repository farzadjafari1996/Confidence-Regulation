"""
GPT-2 family reproduction logic (Stolfo et al., NeurIPS 2024).

Metrics, plots, ablation engine, and phase runners for src/gpt2/ scripts.
"""

from __future__ import annotations

import json
import math
import pathlib
import random

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import tqdm
from datasets import load_dataset
from plotly.express.colors import qualitative
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT / "reference_code"
PLOT_COLORS = {"Normal": "#636EFA", "Entropy": "#EF553B"}

from hyperparameters import (  # noqa: E402
    DEFAULT_DATASET,
    GPT2_PAPER_ENTROPY_NEURONS,
    ModelRunConfig,
    count_from_pct_and_n,
)

_SAVE_PNG = False


def resolve_tl_model_name(name: str) -> str:
    """Map checkpoint name to a TransformerLens official name."""
    from transformer_lens.loading_from_pretrained import get_official_model_name

    try:
        return get_official_model_name(name)
    except ValueError as exc:
        raise ValueError(
            f"{name} is not a TransformerLens model name. "
            "See https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html"
        ) from exc


def _read_hf_token() -> str | None:
    """HF token: hf_token.txt → huggingface-cli cache → HF_TOKEN env (incl. Registry user env on Windows)."""
    token_path = REF / "ablations" / "hf_token.txt"
    if token_path.exists():
        text = token_path.read_text(encoding="utf-8").strip()
        if text:
            return text

    try:
        from huggingface_hub import HfFolder

        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass

    import os

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def load_model(cfg: ModelRunConfig, device: str) -> tuple[HookedTransformer, object]:
    """Default GPT-2 family loader (TransformerLens, no HF gating)."""
    torch.set_grad_enabled(False)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    tl_name = resolve_tl_model_name(cfg.name)
    model = HookedTransformer.from_pretrained(tl_name, device=device)
    model.eval()
    return model, model.tokenizer


def neuron_label(layer: int, idx: int) -> str:
    return f"{layer}.{idx}"


def top1pct_rho_neurons(
    static_df: pd.DataFrame,
    pct: float = 0.01,
    *,
    fixed_n: int | None = None,
) -> list[str]:
    """Top neurons by rho; count = min(pct * n, fixed_n) when fixed_n set (paper §3.3)."""
    n = count_from_pct_and_n(len(static_df), pct, fixed_n)
    return static_df.nlargest(n, "rho")["component_name"].tolist()


def annotate_top1pct_rho(
    static_df: pd.DataFrame,
    pct: float = 0.01,
    *,
    fixed_n: int | None = None,
) -> pd.DataFrame:
    df = static_df.copy()
    selected = set(top1pct_rho_neurons(df, pct, fixed_n=fixed_n))
    df["is_top1pct_rho"] = df["component_name"].isin(selected)
    return df


def compute_static_neuron_df(model: HookedTransformer, cfg: ModelRunConfig) -> pd.DataFrame:
    layer = cfg.last_layer(model)
    # SVD / variance on float32 — half precision is unsupported on CUDA for svd.
    w_out = model.W_out[-1].float()
    w_u = model.W_U.float()
    norm = w_out.norm(dim=1)

    normalized_comp = (w_out / norm.unsqueeze(1)) @ (w_u / w_u.norm(dim=0))
    logit_var = normalized_comp.var(dim=1)

    u, _, _ = torch.linalg.svd(w_u, full_matrices=False)
    u_dark = u[:, cfg.udark_start(model.cfg.d_model):]
    rho = (w_out @ u_dark).norm(dim=-1) / norm

    return pd.DataFrame(
        {
            "neuron_index": np.arange(model.cfg.d_mlp),
            "component_name": [neuron_label(layer, i) for i in range(model.cfg.d_mlp)],
            "norm": norm.cpu().numpy(),
            "logit_var": logit_var.cpu().numpy(),
            "rho": rho.cpu().numpy(),
        }
    )


def select_random_control_neurons(
    static_df: pd.DataFrame,
    entropy_labels: list[str],
    cfg: ModelRunConfig,
) -> list[str]:
    """Sample random final-layer neurons excluding entropy candidates (paper: 100 controls)."""
    static_df = normalize_component_names(static_df)
    pool = static_df[~static_df["component_name"].isin(entropy_labels)]
    n_pick = cfg.random_control_count(len(pool))
    return pool.sample(n_pick, random_state=cfg.seed)["component_name"].tolist()


def select_phase2_neurons(
    static_df: pd.DataFrame,
    entropy_labels: list[str],
    cfg: ModelRunConfig,
) -> list[str]:
    """Phase 2 ablation set: all entropy candidates + random controls."""
    random_labels = select_random_control_neurons(static_df, entropy_labels, cfg)
    names = set(entropy_labels) | set(random_labels)
    return sorted(names, key=lambda x: int(x.split(".")[1]))


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


def plot_b1_norm_vs_logitvar(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    highlight_col: str = "is_entropy",
    filename: str = "b1_norm_vs_logitvar",
) -> None:
    plot_df = df.copy()
    plot_df["neuron_type"] = np.where(plot_df[highlight_col], "Entropy", "Normal")
    plot_df["label"] = plot_df.apply(
        lambda r: str(r["neuron_index"]) if r[highlight_col] else "", axis=1
    )
    fig = px.scatter(
        plot_df,
        x="norm",
        y="logit_var",
        color="neuron_type",
        log_y=True,
        title=f"{model_title} — (a) Norm vs. LogitVar",
        labels={"norm": "||w<sub>out</sub>||", "logit_var": "LogitVar(w<sub>out</sub>)"},
        color_discrete_map=PLOT_COLORS,
        hover_data=["component_name"],
        marginal_x="histogram",
        marginal_y="box",
    )
    for idx in plot_df.loc[plot_df[highlight_col], "neuron_index"]:
        row = plot_df[plot_df["neuron_index"] == idx]
        fig.add_trace(
            go.Scatter(
                x=row["norm"],
                y=row["logit_var"],
                mode="text",
                text=row["label"],
                textposition="top center",
                showlegend=False,
                textfont=dict(color="#EF553B", size=10),
            )
        )
    fig.update_layout(width=500, height=420, title_font_size=16)
    _save_fig(fig, out_dir / filename)


def plot_b2_rho(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    highlight_col: str = "is_entropy",
    filename: str = "b2_rho_vs_norm",
) -> None:
    plot_df = df.copy()
    plot_df["neuron_type"] = np.where(plot_df[highlight_col], "Entropy", "Normal")
    fig = px.scatter(
        plot_df,
        x="norm",
        y="rho",
        color="neuron_type",
        title=f"{model_title} — (d) ||w_out|| vs rho",
        labels={"norm": "||w<sub>out</sub>||", "rho": "rho"},
        color_discrete_map=PLOT_COLORS,
        hover_data=["component_name"],
    )
    fig.update_layout(width=500, height=400, title_font_size=16)
    _save_fig(fig, out_dir / filename)


def plot_fig2d_singular_values(
    model: HookedTransformer,
    static_df: pd.DataFrame,
    selected: list[str],
    out_dir: pathlib.Path,
    model_title: str,
    *,
    n_trace_neurons: int = 6,
) -> None:
    """Fig 2d: W_U singular values + |cos(w_out, u_j)| for top-rho neurons in the selected set."""
    w_u = model.W_U.float()
    u, singular_values, _ = torch.linalg.svd(w_u, full_matrices=False)
    w_out = model.W_out[-1].float()
    indices = np.arange(len(singular_values))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=singular_values.cpu().numpy(),
            mode="lines",
            name="Singular values",
            line=dict(color="#636EFA", width=3),
        )
    )

    sel_df = static_df[static_df["component_name"].isin(selected)].nlargest(n_trace_neurons, "rho")
    palette = qualitative.Plotly
    for i, row in enumerate(sel_df.itertuples()):
        idx = int(row.neuron_index)
        w = w_out[idx]
        w_norm = w.norm().clamp(min=1e-12)
        cosines = (w @ u).abs().cpu().numpy() / w_norm.item()
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=cosines,
                mode="lines",
                name=str(row.component_name),
                line=dict(color=palette[i % len(palette)], width=1.5),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=f"{model_title} — (d) W_U singular values & neuron–singular-vector alignment",
        xaxis=dict(title="Singular vector index"),
        yaxis=dict(title="Singular value", type="log"),
        yaxis2=dict(
            title="|cos(w<sub>out</sub>, u<sub>j</sub>)|",
            overlaying="y",
            side="right",
            range=[0, 1.05],
        ),
        width=620,
        height=420,
        title_font_size=16,
        legend=dict(x=1.08, y=1),
    )
    _save_fig(fig, out_dir / "fig2d_singular_values")


def _remove_te_de_outliers(
    melted: pd.DataFrame,
    *,
    category_col: str = "label",
    effect_col: str = "effect",
    value_col: str = "abs_delta_loss",
) -> pd.DataFrame:
    """IQR outlier removal per category × effect (paper load_results.py Fig 2c)."""
    kept: list[pd.DataFrame] = []
    for (_, _), group in melted.groupby([category_col, effect_col], sort=False):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        kept.append(group[(group[value_col] >= lower) & (group[value_col] <= upper)])
    if not kept:
        return melted.iloc[0:0]
    return pd.concat(kept, ignore_index=True)


def _melt_te_de_boxplot_df(
    df: pd.DataFrame,
    entropy_labels: list[str],
    random_labels: list[str] | None = None,
    *,
    label_col: str = "label",
) -> pd.DataFrame:
    """Token-level |Δloss| rows for TE vs DE_LN box plots (paper Fig 2c).

    Entropy neurons appear individually; all random controls are pooled as "Random".
    """
    plot_df = df.copy()
    if random_labels is not None:
        keep = set(entropy_labels) | set(random_labels)
        plot_df = plot_df[plot_df["component_name"].isin(keep)]
    plot_df["abs_te"] = (plot_df["loss_post_ablation"] - plot_df["loss"]).abs()
    plot_df["abs_de_ln"] = (plot_df["loss_post_ablation_with_frozen_ln"] - plot_df["loss"]).abs()
    plot_df["is_entropy"] = plot_df["component_name"].isin(entropy_labels)
    plot_df[label_col] = np.where(plot_df["is_entropy"], plot_df["component_name"], "Random")
    melted = plot_df.melt(
        id_vars=label_col,
        value_vars=["abs_te", "abs_de_ln"],
        var_name="effect",
        value_name="abs_delta_loss",
    )
    melted["effect"] = melted["effect"].map({"abs_te": "Total", "abs_de_ln": "Direct"})
    return melted


def _fig2c_label_order(entropy_labels: list[str]) -> list[str]:
    """Entropy neurons left-to-right, pooled random control last (paper Fig 2c)."""
    ordered = sorted(entropy_labels, key=lambda x: int(x.split(".")[1]))
    ordered.append("Random")
    return ordered


def plot_b2_rho_vs_ln_fraction(
    agg: pd.DataFrame,
    out_dir: pathlib.Path,
    model_title: str,
    *,
    highlight_col: str = "is_entropy",
    filename: str = "fig2e_rho_vs_ln_fraction",
) -> None:
    plot_df = agg.copy()
    plot_df["neuron_type"] = np.where(plot_df[highlight_col], "Entropy", "Random")
    fig = px.scatter(
        plot_df,
        x="rho",
        y="ln_mediated_fraction",
        color="neuron_type",
        title=f"{model_title} — (e) rho vs LN-mediated fraction",
        labels={"ln_mediated_fraction": "1 - DE<sub>LN</sub> / TE"},
        color_discrete_map=PLOT_COLORS,
        hover_data=["component_name"],
    )
    fig.update_layout(width=500, height=400, title_font_size=16)
    _save_fig(fig, out_dir / filename)


def plot_fig2c_te_de_box(
    final_df: pd.DataFrame,
    out_dir: pathlib.Path,
    entropy_labels: list[str],
    model_title: str,
    *,
    random_labels: list[str] | None = None,
) -> None:
    """Fig 2c: TE vs DE_LN — each entropy neuron + one pooled Random box (paper method)."""
    melted = _melt_te_de_boxplot_df(final_df, entropy_labels, random_labels)
    melted = _remove_te_de_outliers(melted)
    label_order = _fig2c_label_order(entropy_labels)
    melted["label"] = pd.Categorical(melted["label"], categories=label_order, ordered=True)
    melted = melted.sort_values(["label", "effect"], ascending=[True, False])

    fig = px.box(
        melted,
        x="label",
        y="abs_delta_loss",
        color="effect",
        title=f"{model_title} — (c) Total & Direct Effects",
        labels={"abs_delta_loss": "|ΔLoss|", "label": "Neuron Name"},
        color_discrete_map={"Total": qualitative.Plotly[0], "Direct": qualitative.Plotly[2]},
        points=False,
        category_orders={"label": label_order},
    )
    n_ent = len(entropy_labels)
    if n_ent:
        fig.add_vline(x=n_ent - 0.5, line_dash="dash", line_color="black")
    width = max(700, 40 + n_ent * 22)
    fig.update_layout(xaxis_tickangle=-45, width=width, height=420, title_font_size=16)
    _save_fig(fig, out_dir / "fig2c_te_vs_de_ln")


def plot_b3_te_de_box(
    final_df: pd.DataFrame,
    out_dir: pathlib.Path,
    entropy_labels: list[str],
    model_title: str,
    n_random: int = 10,
) -> None:
    df = final_df.copy()
    df["is_entropy"] = df["component_name"].isin(entropy_labels)

    non_entropy = df.loc[~df["is_entropy"], "component_name"].unique().tolist()
    random_neurons = random.Random(42).sample(non_entropy, min(n_random, len(non_entropy)))
    keep = [n for n in entropy_labels if n in df["component_name"].values] + random_neurons
    filtered = df[df["component_name"].isin(keep)].copy()

    melted = _melt_te_de_boxplot_df(filtered, entropy_labels)
    melted = _remove_te_de_outliers(melted)
    melted = melted.sort_values(["label", "effect"], ascending=[True, False])

    fig = px.box(
        melted,
        x="label",
        y="abs_delta_loss",
        color="effect",
        title=f"{model_title} — (c) Total & Direct Effects",
        labels={"abs_delta_loss": "Average Effect", "label": "Neuron Name"},
        color_discrete_map={"Total": qualitative.Plotly[0], "Direct": qualitative.Plotly[2]},
        points=False,
    )
    n_ent = melted.loc[melted["label"] != "Random", "label"].nunique()
    if n_ent:
        fig.add_vline(x=n_ent - 0.5, line_dash="dash", line_color="black")
    fig.update_layout(xaxis_tickangle=-45, width=700, height=400, title_font_size=16)
    _save_fig(fig, out_dir / "b3_te_vs_de_ln")


def plot_fig1_entropy_vs_argmax(
    final_df: pd.DataFrame,
    out_dir: pathlib.Path,
    entropy_labels: list[str],
    model_title: str,
) -> None:
    df = final_df.copy()
    df["delta_entropy"] = df["entropy_post_ablation"] - df["entropy"]
    df["pred_change"] = (df["top_logits_post_ablation"] != df["pred"]).astype(int)
    agg = (
        df.groupby("component_name", as_index=False)
        .agg(
            abs_delta_entropy=("delta_entropy", lambda s: s.abs().mean()),
            pred_change=("pred_change", "mean"),
        )
    )
    agg["is_entropy"] = agg["component_name"].isin(entropy_labels)
    agg["neuron_type"] = np.where(agg["is_entropy"], "Entropy", "Random")
    fig = px.scatter(
        agg,
        x="pred_change",
        y="abs_delta_entropy",
        color="neuron_type",
        hover_data=["component_name"],
        title=f"{model_title} — (Fig 1) |ΔH| vs argmax change rate",
        color_discrete_map=PLOT_COLORS,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig.update_layout(width=520, height=420, title_font_size=16)
    _save_fig(fig, out_dir / "fig1_entropy_vs_argmax")


def plot_b4_entropy_vs_rank(
    final_df: pd.DataFrame, out_dir: pathlib.Path, entropy_labels: list[str], model_title: str
) -> None:
    df = final_df.copy()
    df["delta_entropy"] = df["entropy_post_ablation"] - df["entropy"]
    df["pred_change"] = (df["top_logits_post_ablation"] != df["pred"]).astype(int)
    agg = (
        df.groupby("component_name", as_index=False)
        .agg(
            abs_delta_entropy=("delta_entropy", lambda s: s.abs().mean()),
            pred_change=("pred_change", "mean"),
        )
    )
    agg["is_entropy"] = agg["component_name"].isin(entropy_labels)
    agg["neuron_type"] = np.where(agg["is_entropy"], "Entropy", "Normal")
    fig = px.scatter(
        agg,
        x="pred_change",
        y="abs_delta_entropy",
        color="neuron_type",
        hover_data=["component_name"],
        title=f"{model_title} — |delta H| vs argmax change",
        color_discrete_map=PLOT_COLORS,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig.update_layout(width=520, height=420, title_font_size=16)
    _save_fig(fig, out_dir / "b4_entropy_vs_argmax_change")


def plot_b4_te_de_scatter(final_df: pd.DataFrame, out_dir: pathlib.Path, model_title: str) -> None:
    agg = (
        final_df.assign(
            te=lambda d: (d["loss_post_ablation"] - d["loss"]).abs(),
            de_ln=lambda d: (d["loss_post_ablation_with_frozen_ln"] - d["loss"]).abs(),
            delta_entropy=lambda d: d["entropy_post_ablation"] - d["entropy"],
        )
        .groupby("component_name", as_index=False)
        .agg({"te": "mean", "de_ln": "mean", "delta_entropy": "mean"})
    )
    fig = px.scatter(
        agg,
        x="te",
        y="de_ln",
        color="delta_entropy",
        hover_data=["component_name"],
        title=f"{model_title} — |TE| vs |DE_LN|",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(width=500, height=400, title_font_size=16)
    _save_fig(fig, out_dir / "b4_te_vs_de_scatter")


def _apply_ln_final(
    model: HookedTransformer, res_stream: torch.Tensor, ln_scales: torch.Tensor
) -> torch.Tensor:
    if model.cfg.normalization_type in ("RMSPre",) or getattr(model.cfg, "final_rms", False):
        return res_stream / ln_scales.unsqueeze(-1)
    if model.cfg.normalization_type in ("LNPre", "LN"):
        centered = res_stream - res_stream.mean(dim=-1, keepdim=True)
        return centered / ln_scales.unsqueeze(-1)
    raise ValueError(f"Unsupported normalization: {model.cfg.normalization_type}")


def get_entropy(logits: torch.Tensor, use_log2: bool = False) -> torch.Tensor:
    entropy = logits.softmax(dim=-1) * logits.log_softmax(dim=-1)
    if use_log2:
        entropy = entropy / torch.log(torch.tensor(2.0))
    return -einops.einsum(entropy, "batch seq d_vocab -> batch seq")


def neuron_str_to_neuron(neuron_str: str) -> tuple[int, int]:
    layer, index = neuron_str.split(".")
    return int(layer), int(index)


def get_entropy_activation_df(
    neuron_names: list[str],
    tokenized_data,
    token_df: pd.DataFrame,
    model,
    *,
    batch_size: int = 1,
    device: str = "cuda",
    residuals_layer: int | None = None,
) -> pd.DataFrame:
    entropy_rows: list[np.ndarray] = []
    top_logit: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    top_p: list[np.ndarray] = []
    token_ids: list[np.ndarray] = []
    ln_final_scale: list[np.ndarray] = []
    ranks_of_correct_token: list[np.ndarray] = []
    neuron_activations: dict[str, list[np.ndarray]] = {n: [] for n in neuron_names}

    num_batches = math.ceil(len(tokenized_data) / batch_size)
    for i in tqdm.tqdm(range(num_batches), desc="collect activations"):
        start = i * batch_size
        end = min(start + batch_size, len(tokenized_data))
        batch = tokenized_data["tokens"][start:end]
        token_ids.append(batch.cpu().numpy())

        logits, cache = model.run_with_cache(batch.to(device))
        top_logit.append(logits.max(dim=-1).values.cpu().numpy())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        top_p.append(np.log(logits.softmax(dim=-1).max(dim=-1).values.cpu().numpy()))

        logits_ranking = logits.argsort(dim=-1, descending=True).cpu()
        rank_of_correct = (
            (logits_ranking[:, :-1] == batch[:, 1:].unsqueeze(-1).cpu())
            .nonzero(as_tuple=True)[2]
            .view(batch[:, :-1].shape)
            .numpy()
        )
        rank_of_correct = np.concatenate(
            (rank_of_correct, np.zeros((rank_of_correct.shape[0], 1))), axis=1
        )
        ranks_of_correct_token.append(rank_of_correct)

        loss_short = model.loss_fn(logits, batch.to(device), per_token=True).cpu().numpy()
        losses.append(np.concatenate((loss_short, np.zeros((loss_short.shape[0], 1))), axis=1))

        for neuron_name in neuron_names:
            layer, idx = neuron_str_to_neuron(neuron_name)
            act = cache[tl_utils.get_act_name("post", layer)][..., idx].cpu().numpy()
            neuron_activations[neuron_name].append(act)

        entropy_rows.append(get_entropy(logits).cpu().numpy())
        ln_final_scale.append(cache["ln_final.hook_scale"].cpu().numpy())
        del logits, cache

    token_df = token_df.copy()
    token_df["token_id"] = np.concatenate(token_ids, axis=0).flatten()
    token_df["entropy"] = np.concatenate(entropy_rows, axis=0).flatten()
    token_df["top_logit"] = np.concatenate(top_logit, axis=0).flatten()
    token_df["pred"] = np.concatenate(preds, axis=0).flatten()
    token_df["loss"] = np.concatenate(losses, axis=0).flatten()
    token_df["top_logp"] = np.concatenate(top_p, axis=0).flatten()
    token_df["ln_final_scale"] = np.concatenate(ln_final_scale, axis=0).flatten()
    token_df["rank_of_correct_token"] = np.concatenate(ranks_of_correct_token, axis=0).flatten()
    token_df["correct_token_rank"] = token_df["rank_of_correct_token"]
    token_df["pred_in_top1"] = token_df["correct_token_rank"] == 0
    token_df["pred_in_top5"] = token_df["correct_token_rank"] < 5

    for neuron_name, chunks in neuron_activations.items():
        token_df[f"{neuron_name}_activation"] = np.concatenate(chunks, axis=0).flatten()

    return token_df


def filter_entropy_activation_df(
    entropy_df: pd.DataFrame,
    *,
    model_name: str | None = None,
    tokenizer=None,
    start_pos: int = 3,
    end_pos: int = -1,
) -> pd.DataFrame:
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    newline_token_id = None

    if model_name and "gpt" in model_name.lower():
        newline_token_id = tokenizer.encode("\n")[0]

    bos_mask = entropy_df["token_id"] == bos_token_id
    eos_mask = entropy_df["token_id"] == eos_token_id
    combined_mask = bos_mask | eos_mask

    if newline_token_id is not None:
        combined_mask = combined_mask | (entropy_df["token_id"] == newline_token_id)

    preceding_token_mask = combined_mask.iloc[1:].reset_index(drop=True)
    combined_mask = combined_mask | preceding_token_mask

    filtered = entropy_df[~combined_mask]

    if end_pos < 0:
        end_pos_val = entropy_df["pos"].max() + 1 + end_pos
    else:
        end_pos_val = end_pos

    return filtered[filtered["pos"].between(start_pos, end_pos_val, inclusive="left")]


def mean_ablate_neurons(
    model: HookedTransformer,
    cfg: ModelRunConfig,
    tokenized_data,
    entropy_df: pd.DataFrame,
    neuron_names: list[str],
    *,
    device: str,
    chunk_size: int = 20,
) -> pd.DataFrame:
    layer_idx = cfg.last_layer(model)
    neuron_indices = [int(n.split(".")[1]) for n in neuron_names]
    activation_means = torch.tensor(
        entropy_df[[f"{n}_activation" for n in neuron_names]].mean().values
    )
    batch_ids = np.random.choice(entropy_df["batch"].unique(), cfg.k, replace=False)
    filtered = entropy_df[entropy_df["batch"].isin(batch_ids)].copy()
    frames: list[pd.DataFrame] = []

    for batch_n in tqdm.tqdm(batch_ids, desc="ablation batches"):
        tok_seq = tokenized_data["tokens"][batch_n]
        inp = tok_seq.unsqueeze(0).to(device)
        _, cache = model.run_with_cache(inp)
        rows = filtered[filtered["batch"] == batch_n]
        ln_scales = torch.tensor(rows["ln_final_scale"].values, device=device)
        res_stream = cache[tl_utils.get_act_name("resid_post", layer_idx)][0]
        prev_act = cache[tl_utils.get_act_name("post", layer_idx)][0, :, neuron_indices]
        del cache
        act_deltas = activation_means.to(prev_act.device) - prev_act

        te_loss, te_entropy, te_top = [], [], []
        de_loss, de_entropy, de_top = [], [], []

        for start in range(0, len(neuron_indices), chunk_size):
            idx_chunk = neuron_indices[start : start + chunk_size]
            delta_chunk = act_deltas[:, start : start + chunk_size]
            res_delta = delta_chunk.unsqueeze(-1) * model.W_out[layer_idx, idx_chunk, :]
            res_delta = res_delta.permute(1, 0, 2)
            n_chunk = res_delta.shape[0]
            updated = res_stream.repeat(n_chunk, 1, 1) + res_delta

            te_res = model.ln_final(updated)
            te_logits = te_res @ model.W_U + model.b_U
            te_loss.append(_per_token_loss(model, te_logits, inp, n_chunk))
            te_entropy.append(get_entropy(te_logits).cpu().numpy())
            te_top.append(te_logits.argmax(dim=-1).cpu().numpy())
            del te_res, te_logits

            de_res = _apply_ln_final(model, updated, ln_scales)
            de_logits = de_res @ model.W_U + model.b_U
            de_loss.append(_per_token_loss(model, de_logits, inp, n_chunk))
            de_entropy.append(get_entropy(de_logits).cpu().numpy())
            de_top.append(de_logits.argmax(dim=-1).cpu().numpy())
            del de_res, de_logits, updated, res_delta

        te_loss = np.concatenate(te_loss, axis=0)
        te_entropy = np.concatenate(te_entropy, axis=0)
        te_top = np.concatenate(te_top, axis=0)
        de_loss = np.concatenate(de_loss, axis=0)
        de_entropy = np.concatenate(de_entropy, axis=0)
        de_top = np.concatenate(de_top, axis=0)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        for i, name in enumerate(neuron_names):
            chunk = rows.copy()
            drop_cols = [f"{n}_activation" for n in neuron_names if n != name]
            chunk = chunk.drop(columns=drop_cols, errors="ignore")
            chunk = chunk.rename(columns={f"{name}_activation": "activation"})
            chunk["component_name"] = name
            chunk["loss_post_ablation"] = te_loss[i]
            chunk["entropy_post_ablation"] = te_entropy[i]
            chunk["top_logits_post_ablation"] = te_top[i]
            chunk["loss_post_ablation_with_frozen_ln"] = de_loss[i]
            chunk["entropy_post_ablation_with_frozen_ln"] = de_entropy[i]
            chunk["top_logits_post_ablation_with_frozen_ln"] = de_top[i]
            frames.append(chunk)

    return pd.concat(frames, ignore_index=True)


def _per_token_loss(
    model: HookedTransformer, logits: torch.Tensor, inp: torch.Tensor, n_repeat: int
) -> np.ndarray:
    loss = model.loss_fn(logits, inp.repeat(n_repeat, 1), per_token=True).cpu().numpy()
    return np.concatenate((loss, np.zeros((loss.shape[0], 1))), axis=1)


def load_tokenized_data(tokenizer, cfg: ModelRunConfig):
    data = load_dataset(cfg.dataset, split="train")
    subset = data.select(range(min(cfg.data_end, len(data))))
    tokenized = tl_utils.tokenize_and_concatenate(
        subset, tokenizer, max_length=cfg.max_seq_length, column_name="text"
    )
    return tokenized.shuffle(cfg.seed)


def _json_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(x)


def write_phase1_summary(
    static_df: pd.DataFrame,
    cfg: ModelRunConfig,
    out_dir: pathlib.Path,
    selected: list[str],
    *,
    paper_neurons: list[str] | None = None,
    model_meta: dict | None = None,
) -> None:
    top10 = static_df.nlargest(10, "rho")["component_name"].tolist()
    summary = {
        "model": cfg.name,
        "phase": 1,
        "selection": {
            "method": f"min({cfg.entropy_top_pct:.0%}, n={cfg.entropy_top_n}) rho",
            "entropy_top_pct": cfg.entropy_top_pct,
            "entropy_top_n": cfg.entropy_top_n,
            "nullspace_top_pct": cfg.nullspace_top_pct,
            "nullspace_k_fixed": cfg.udark_k,
            "nullspace_k_effective": cfg.nullspace_k((model_meta or {}).get("d_model", 768)),
            "n_selected": len(selected),
            "neurons": selected,
        },
        "top10_rho": [
            {"component_name": r["component_name"], "rho": _json_float(r["rho"])}
            for r in static_df.nlargest(10, "rho")[["component_name", "rho"]].to_dict("records")
        ],
        "selected_stats": [
            {
                "component_name": r["component_name"],
                "norm": _json_float(r["norm"]),
                "logit_var": _json_float(r["logit_var"]),
                "rho": _json_float(r["rho"]),
            }
            for r in static_df[static_df["component_name"].isin(selected)][
                ["component_name", "norm", "logit_var", "rho"]
            ].sort_values("rho", ascending=False)
            .to_dict("records")
        ],
    }
    if model_meta:
        summary["model_meta"] = model_meta

    if paper_neurons:
        overlap = [n for n in paper_neurons if n in selected]
        summary["paper_verification"] = {
            "paper_neurons": paper_neurons,
            "overlap_with_selected": overlap,
            "overlap_count": len(overlap),
            "paper_in_top10_rho": [n for n in paper_neurons if n in top10],
            "paper_in_top10_rho_count": sum(1 for n in paper_neurons if n in top10),
        }

    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved {path}")


def run_phase1_identify(
    cfg: ModelRunConfig,
    out_dir: pathlib.Path,
    *,
    device: str,
    load_model_fn=None,
    paper_neurons: list[str] | None = None,
) -> pd.DataFrame:
    """Phase 1: SVD/rho metrics, top-1% selection, Fig 2d plots, Fig 2a last."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _load = load_model_fn or load_model

    print(f"\n{'='*60}\nPhase 1 — {cfg.name} on {device}\n{'='*60}")
    model, _ = _load(cfg, device)
    meta = {
        "n_layers": model.cfg.n_layers,
        "d_mlp": model.cfg.d_mlp,
        "d_model": model.cfg.d_model,
        "d_vocab": model.cfg.d_vocab,
        "normalization": model.cfg.normalization_type,
        "udark_start": cfg.udark_start(model.cfg.d_model),
        "nullspace_top_pct": cfg.nullspace_top_pct,
        "nullspace_k_fixed": cfg.udark_k,
        "nullspace_k": cfg.nullspace_k(model.cfg.d_model),
    }
    print(
        f"  layers={meta['n_layers']} d_mlp={meta['d_mlp']} "
        f"d_model={meta['d_model']} norm={meta['normalization']} udark_start={meta['udark_start']}"
    )

    static_df = compute_static_neuron_df(model, cfg)
    static_df = annotate_top1pct_rho(
        static_df, cfg.entropy_top_pct, fixed_n=cfg.entropy_top_n
    )
    selected = top1pct_rho_neurons(
        static_df, cfg.entropy_top_pct, fixed_n=cfg.entropy_top_n
    )

    static_df.to_csv(out_dir / "static_neuron_metrics.csv", index=False)
    print(f"  saved {out_dir / 'static_neuron_metrics.csv'}")

    top1_path = out_dir / "top1pct_rho_neurons.json"
    top1_payload = {
        "model": cfg.name,
        "method": f"min({cfg.entropy_top_pct:.0%}, n={cfg.entropy_top_n}) rho",
        "n_selected": len(selected),
        "neurons": selected,
        "stats": static_df[static_df["is_top1pct_rho"]]
        .sort_values("rho", ascending=False)[["component_name", "norm", "logit_var", "rho"]]
        .to_dict(orient="records"),
    }
    top1_path.write_text(json.dumps(top1_payload, indent=2), encoding="utf-8")
    print(f"  saved {top1_path}")

    print("  Fig 2d: singular values + neuron alignment")
    plot_fig2d_singular_values(
        model, static_df, selected, out_dir, cfg.name, n_trace_neurons=cfg.n_trace_neurons
    )

    print("  Fig 2a: Norm vs LogitVar")
    plot_b1_norm_vs_logitvar(
        static_df,
        out_dir,
        cfg.name,
        highlight_col="is_top1pct_rho",
        filename="fig2a_norm_vs_logitvar",
    )

    write_phase1_summary(static_df, cfg, out_dir, selected, paper_neurons=paper_neurons, model_meta=meta)

    if device.startswith("cuda"):
        del model
        torch.cuda.empty_cache()

    print(f"Phase 1 done: {len(selected)} neurons -> {out_dir.resolve()}")
    return static_df


def normalize_component_names(static_df: pd.DataFrame) -> pd.DataFrame:
    """Restore layer.index labels (CSV may coerce e.g. 11.480 → float)."""
    df = static_df.copy()
    layer = int(str(df["component_name"].iloc[0]).split(".")[0])
    df["component_name"] = df["neuron_index"].apply(lambda i: f"{layer}.{int(i)}")
    return df


def load_phase1_outputs(phase1_dir: pathlib.Path) -> tuple[list[str], pd.DataFrame]:
    top1_path = phase1_dir / "top1pct_rho_neurons.json"
    static_path = phase1_dir / "static_neuron_metrics.csv"
    if not top1_path.exists():
        raise FileNotFoundError(f"Run Phase 1 first — missing {top1_path}")
    if not static_path.exists():
        raise FileNotFoundError(f"Run Phase 1 first — missing {static_path}")

    payload = json.loads(top1_path.read_text(encoding="utf-8"))
    static_df = normalize_component_names(pd.read_csv(static_path))
    if "is_top1pct_rho" not in static_df.columns:
        static_df = annotate_top1pct_rho(static_df, fixed_n=None)
    return payload["neurons"], static_df


def write_phase2_summary(
    ablation_df: pd.DataFrame,
    cfg: ModelRunConfig,
    out_dir: pathlib.Path,
    entropy_labels: list[str],
    neuron_names: list[str],
    *,
    elapsed_sec: float,
    paper_neurons: list[str] | None = None,
) -> None:
    agg = (
        ablation_df.assign(
            abs_te=lambda d: (d["loss_post_ablation"] - d["loss"]).abs(),
            abs_de_ln=lambda d: (d["loss_post_ablation_with_frozen_ln"] - d["loss"]).abs(),
            abs_delta_h=lambda d: (d["entropy_post_ablation"] - d["entropy"]).abs(),
            pred_change=lambda d: (d["top_logits_post_ablation"] != d["pred"]).astype(int),
        )
        .groupby("component_name", as_index=False)
        .agg(
            abs_te=("abs_te", "mean"),
            abs_de_ln=("abs_de_ln", "mean"),
            abs_delta_h=("abs_delta_h", "mean"),
            pred_change=("pred_change", "mean"),
        )
    )
    ent = agg[agg["component_name"].isin(entropy_labels)]
    rnd = agg[~agg["component_name"].isin(entropy_labels)]
    ent_ln_frac = 1 - ent["abs_de_ln"] / ent["abs_te"].replace(0, np.nan)

    summary = {
        "model": cfg.name,
        "phase": 2,
        "runtime_sec": round(elapsed_sec, 1),
        "ablation_config": {
            "k_batches": cfg.k,
            "data_end": cfg.data_end,
            "batch_size": cfg.batch_size,
            "chunk_size": cfg.chunk_size,
            "max_seq_length": cfg.max_seq_length,
            "n_random": cfg.n_random,
            "n_random_pct": cfg.n_random_pct,
            "seed": cfg.seed,
            "filter_start_pos": cfg.filter_start_pos,
            "filter_end_pos": cfg.filter_end_pos,
            "n_entropy": len(entropy_labels),
            "n_random_selected": len(neuron_names) - len(entropy_labels),
            "n_total": len(neuron_names),
        },
        "entropy_set": {
            "mean_abs_te": _json_float(ent["abs_te"].mean()),
            "mean_abs_de_ln": _json_float(ent["abs_de_ln"].mean()),
            "mean_ln_mediated_fraction": _json_float(ent_ln_frac.mean()),
            "mean_abs_delta_h": _json_float(ent["abs_delta_h"].mean()),
            "mean_pred_change_rate": _json_float(ent["pred_change"].mean()),
        },
        "random_controls": {
            "mean_abs_te": _json_float(rnd["abs_te"].mean()),
            "mean_abs_de_ln": _json_float(rnd["abs_de_ln"].mean()),
            "mean_abs_delta_h": _json_float(rnd["abs_delta_h"].mean()),
            "mean_pred_change_rate": _json_float(rnd["pred_change"].mean()),
        },
        "paper_expectation": "entropy: TE >> DE_LN, high |ΔH|, low argmax change",
    }

    if paper_neurons:
        paper_agg = agg[agg["component_name"].isin(paper_neurons)]
        if len(paper_agg):
            summary["paper_neurons_ablation"] = {
                "neurons": paper_neurons,
                "mean_abs_te": _json_float(paper_agg["abs_te"].mean()),
                "mean_abs_de_ln": _json_float(paper_agg["abs_de_ln"].mean()),
                "mean_abs_delta_h": _json_float(paper_agg["abs_delta_h"].mean()),
            }

    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved {path}")


def run_phase2_ablation(
    cfg: ModelRunConfig,
    phase1_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    device: str,
    load_model_fn=None,
    paper_neurons: list[str] | None = None,
) -> pd.DataFrame:
    """Phase 2: mean ablation TE/DE on Phase 1 candidates + random controls."""
    import time

    import neel.utils as nutils

    out_dir.mkdir(parents=True, exist_ok=True)
    entropy_labels, static_df = load_phase1_outputs(phase1_dir)
    random_labels = select_random_control_neurons(static_df, entropy_labels, cfg)
    neuron_names = select_phase2_neurons(static_df, entropy_labels, cfg)

    print(f"\n{'='*60}\nPhase 2 — {cfg.name} on {device}\n{'='*60}")
    print(
        f"  entropy neurons: {len(entropy_labels)}  "
        f"random controls: {len(random_labels)}  ablated: {len(neuron_names)}"
    )
    print(f"  k={cfg.k} batches  data_end={cfg.data_end} sequences")

    t0 = time.perf_counter()
    _load = load_model_fn or load_model
    model, tokenizer = _load(cfg, device)
    model.set_use_attn_result(False)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    tokenized = load_tokenized_data(tokenizer, cfg)
    token_df = nutils.make_token_df(tokenized["tokens"], model=model)
    entropy_df = get_entropy_activation_df(
        neuron_names,
        tokenized,
        token_df,
        model,
        batch_size=cfg.batch_size,
        device=device,
        residuals_layer=cfg.last_layer(model),
    )
    raw = mean_ablate_neurons(
        model,
        cfg,
        tokenized,
        entropy_df,
        neuron_names,
        device=device,
        chunk_size=cfg.chunk_size,
    )
    ablation_df = filter_entropy_activation_df(
        raw,
        model_name=cfg.name,
        tokenizer=tokenizer,
        start_pos=cfg.filter_start_pos,
        end_pos=cfg.filter_end_pos,
    )

    feather_path = out_dir / "ablation.feather"
    ablation_df.reset_index(drop=True).to_feather(feather_path)
    print(f"  saved {feather_path}")

    print("  Fig 2c: TE vs DE_LN")
    plot_fig2c_te_de_box(
        ablation_df, out_dir, entropy_labels, cfg.name, random_labels=random_labels
    )

    agg = (
        ablation_df.assign(
            te=lambda d: (d["loss_post_ablation"] - d["loss"]).abs(),
            de_ln=lambda d: (d["loss_post_ablation_with_frozen_ln"] - d["loss"]).abs(),
        )
        .groupby("component_name", as_index=False)
        .agg({"te": "mean", "de_ln": "mean"})
    )
    agg = agg.merge(static_df[["component_name", "rho"]], on="component_name")
    agg["is_entropy"] = agg["component_name"].isin(entropy_labels)
    agg["ln_mediated_fraction"] = 1 - agg["de_ln"] / agg["te"].replace(0, np.nan)
    agg.to_csv(out_dir / "rho_vs_ln_fraction.csv", index=False)

    print("  Fig 2e: rho vs LN-mediated fraction")
    plot_b2_rho_vs_ln_fraction(agg, out_dir, cfg.name, highlight_col="is_entropy")

    print("  Fig 1: |delta H| vs argmax change")
    plot_fig1_entropy_vs_argmax(ablation_df, out_dir, entropy_labels, cfg.name)

    elapsed = time.perf_counter() - t0
    write_phase2_summary(
        ablation_df,
        cfg,
        out_dir,
        entropy_labels,
        neuron_names,
        elapsed_sec=elapsed,
        paper_neurons=paper_neurons,
    )

    if device.startswith("cuda"):
        del model
        torch.cuda.empty_cache()

    print(f"Phase 2 done in {elapsed:.0f}s -> {out_dir.resolve()}")
    return ablation_df
