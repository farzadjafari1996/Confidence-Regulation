"""
GPT-2 core reproduction — Track 1B (Stolfo et al., NeurIPS 2024).

Produces plots comparable to paper Fig 1 / Fig 2a,c,d,e in outputs/gpt2-rep/.

Usage:
  python src/gpt2-rep.py                  # all steps (default)
  python src/gpt2-rep.py --step b1 b2    # static plots only (~1 min)
  python src/gpt2-rep.py --quick          # smaller ablation set for smoke test
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Iterable

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

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "reference_code"
sys.path.insert(0, str(REF))
from utils import (  # noqa: E402
    filter_entropy_activation_df,
    get_entropy,
    get_entropy_activation_df,
)

try:
    import neel.utils as nutils
except ImportError as exc:
    raise ImportError(
        "Install neelutils: pip install git+https://github.com/neelnanda-io/neelutils.git"
    ) from exc

# ---------------------------------------------------------------------------
# Constants (paper GPT-2-small)
# ---------------------------------------------------------------------------

MODEL_NAME = "gpt2-small"
LAST_LAYER = 11
ENTROPY_NEURONS = ["11.584", "11.2378", "11.2870", "11.2123", "11.1611", "11.2910"]
ENTROPY_INDICES = [int(n.split(".")[1]) for n in ENTROPY_NEURONS]
UDARK_START = -12
DEFAULT_OUT = ROOT / "outputs" / "gpt2-rep"
DEFAULT_DATASET = "stas/c4-en-10k"
PLOT_COLORS = {"Normal": "#636EFA", "Entropy": "#EF553B"}
_SAVE_PNG = False


# ---------------------------------------------------------------------------
# Model & static metrics
# ---------------------------------------------------------------------------


def load_gpt2(device: str) -> tuple[HookedTransformer, object]:
    torch.set_grad_enabled(False)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Install cu121 torch or pass --device cpu.")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()
    return model, model.tokenizer


def neuron_label(layer: int, idx: int) -> str:
    return f"{layer}.{idx}"


def compute_static_neuron_df(model: HookedTransformer) -> pd.DataFrame:
    """Per-neuron ||w_out||, LogitVar, and null-space fraction rho (matches reference_code/utils.py)."""
    w_out = model.W_out[-1]
    norm = w_out.norm(dim=1)

    normalized_comp = (w_out / norm.unsqueeze(1)) @ (model.W_U / model.W_U.norm(dim=0))
    logit_var = normalized_comp.var(dim=1)

    u, _, _ = torch.linalg.svd(model.W_U, full_matrices=False)
    u_dark = u[:, UDARK_START:]  # bottom ~1% left singular vectors of W_U
    rho = (w_out @ u_dark).norm(dim=-1) / norm

    df = pd.DataFrame(
        {
            "neuron_index": np.arange(model.cfg.d_mlp),
            "component_name": [neuron_label(LAST_LAYER, i) for i in range(model.cfg.d_mlp)],
            "norm": norm.cpu().numpy(),
            "logit_var": logit_var.cpu().numpy(),
            "rho": rho.cpu().numpy(),
        }
    )
    df["is_entropy"] = df["component_name"].isin(ENTROPY_NEURONS)
    return df


def select_ablation_neurons(
    static_df: pd.DataFrame,
    *,
    neuron_range: tuple[int, int] | None = None,
    top_rho_n: int = 15,
    n_random: int = 50,
    quick: bool = False,
) -> list[str]:
    """Neurons to ablate: paper entropy set + high-rho + random baseline."""
    names: set[str] = set(ENTROPY_NEURONS)

    if neuron_range is not None:
        lo, hi = neuron_range
        names.update(neuron_label(LAST_LAYER, i) for i in range(lo, hi))
    else:
        names.update(static_df.nlargest(top_rho_n, "rho")["component_name"])
        pool = static_df[~static_df["component_name"].isin(names)]
        n_pick = min(n_random if not quick else 10, len(pool))
        names.update(pool.sample(n_pick, random_state=42)["component_name"])

    if quick:
        names = set(list(names)[:25]) | set(ENTROPY_NEURONS)

    return sorted(names, key=lambda x: int(x.split(".")[1]))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


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


def plot_b1_norm_vs_logitvar(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    plot_df = df.copy()
    plot_df["neuron_type"] = np.where(plot_df["is_entropy"], "Entropy", "Normal")
    plot_df["label"] = plot_df.apply(
        lambda r: str(r["neuron_index"]) if r["is_entropy"] else "", axis=1
    )

    fig = px.scatter(
        plot_df,
        x="norm",
        y="logit_var",
        color="neuron_type",
        log_y=True,
        title="(a) Norm vs. LogitVar",
        labels={"norm": "||w<sub>out</sub>||", "logit_var": "LogitVar(w<sub>out</sub>)"},
        color_discrete_map=PLOT_COLORS,
        hover_data=["component_name"],
        marginal_x="histogram",
        marginal_y="box",
    )
    for idx in ENTROPY_INDICES:
        row = plot_df[plot_df["neuron_index"] == idx]
        if row.empty:
            continue
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
    fig.update_layout(width=500, height=420, title_font_size=16, margin=dict(l=40, r=20, t=40, b=40))
    _save_fig(fig, out_dir / "b1_norm_vs_logitvar")


def plot_b2_rho(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    plot_df = df.copy()
    plot_df["neuron_type"] = np.where(plot_df["is_entropy"], "Entropy", "Normal")
    fig = px.scatter(
        plot_df,
        x="norm",
        y="rho",
        color="neuron_type",
        title="(d) ||w_out|| vs rho",
        labels={"norm": "||w<sub>out</sub>||", "rho": "rho"},
        color_discrete_map=PLOT_COLORS,
        hover_data=["component_name"],
    )
    fig.update_layout(width=500, height=400, title_font_size=16)
    _save_fig(fig, out_dir / "b2_rho_vs_norm")


def plot_b2_rho_vs_ln_fraction(agg: pd.DataFrame, out_dir: pathlib.Path) -> None:
    plot_df = agg.copy()
    plot_df["neuron_type"] = np.where(plot_df["is_entropy"], "Entropy", "Normal")
    fig = px.scatter(
        plot_df,
        x="rho",
        y="ln_mediated_fraction",
        color="neuron_type",
        title="(e) rho vs LN-mediated fraction",
        labels={"ln_mediated_fraction": "1 - DE<sub>LN</sub> / TE"},
        color_discrete_map=PLOT_COLORS,
        hover_data=["component_name"],
    )
    fig.update_layout(width=500, height=400, title_font_size=16)
    _save_fig(fig, out_dir / "b2_rho_vs_ln_fraction")


def plot_b3_te_de_box(final_df: pd.DataFrame, out_dir: pathlib.Path, n_random: int = 10) -> None:
    """Fig 2c: Total vs Direct effect box plot."""
    df = final_df.copy()
    df["abs_te"] = (df["loss_post_ablation"] - df["loss"]).abs()
    df["abs_de_ln"] = (df["loss_post_ablation_with_frozen_ln"] - df["loss"]).abs()
    df["is_entropy"] = df["component_name"].isin(ENTROPY_NEURONS)

    non_entropy = df.loc[~df["is_entropy"], "component_name"].unique().tolist()
    rng = random.Random(42)
    random_neurons = rng.sample(non_entropy, min(n_random, len(non_entropy)))
    keep = [n for n in ENTROPY_NEURONS if n in df["component_name"].values] + random_neurons
    filtered = df[df["component_name"].isin(keep)].copy()
    filtered["label"] = np.where(filtered["is_entropy"], filtered["component_name"], "Random")

    melted = filtered.melt(
        id_vars="label",
        value_vars=["abs_te", "abs_de_ln"],
        var_name="effect",
        value_name="abs_delta_loss",
    )
    melted["effect"] = melted["effect"].map({"abs_te": "Total", "abs_de_ln": "Direct"})

    fig = px.box(
        melted,
        x="label",
        y="abs_delta_loss",
        color="effect",
        title="(c) Total & Direct Effects",
        labels={"abs_delta_loss": "Average Effect", "label": "Neuron Name", "effect": "Effect Type"},
        color_discrete_map={"Total": qualitative.Plotly[0], "Direct": qualitative.Plotly[2]},
        points=False,
    )
    n_entropy = sum(filtered["is_entropy"].unique()) if len(filtered) else 0
    if n_entropy:
        fig.add_vline(x=n_entropy - 0.5, line_dash="dash", line_color="black")
    fig.update_layout(xaxis_tickangle=-45, width=700, height=400, title_font_size=16)
    _save_fig(fig, out_dir / "b3_te_vs_de_ln")


def plot_b4_entropy_vs_rank(final_df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Fig 1 style: delta entropy vs change in correct-token rank (confidence)."""
    df = final_df.copy()
    df["delta_entropy"] = df["entropy_post_ablation"] - df["entropy"]
    df["pred_change"] = (df["top_logits_post_ablation"] != df["pred"]).astype(int)

    agg = (
        df.groupby("component_name", as_index=False)
        .agg(
            abs_delta_entropy=("delta_entropy", lambda s: s.abs().mean()),
            pred_change=("pred_change", "mean"),
            is_entropy=("component_name", lambda s: s.iloc[0] in ENTROPY_NEURONS),
        )
    )
    agg["neuron_type"] = np.where(agg["is_entropy"], "Entropy", "Normal")

    fig = px.scatter(
        agg,
        x="pred_change",
        y="abs_delta_entropy",
        color="neuron_type",
        hover_data=["component_name"],
        title="Fig 1: |delta H| vs argmax change rate",
        labels={"pred_change": "P(argmax changes)", "abs_delta_entropy": "mean |delta H|"},
        color_discrete_map=PLOT_COLORS,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig.update_layout(width=520, height=420, title_font_size=16)
    _save_fig(fig, out_dir / "b4_entropy_vs_argmax_change")


def plot_b4_te_de_scatter(final_df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """TE vs DE_LN per neuron (aggregated), colored by delta entropy."""
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
        title="Direct vs Total effect (per-neuron means)",
        labels={"te": "|TE|", "de_ln": "|DE_LN|", "delta_entropy": "mean delta H"},
        color_continuous_scale="Viridis",
    )
    fig.update_layout(width=500, height=400, title_font_size=16)
    _save_fig(fig, out_dir / "b4_te_vs_de_scatter")


# ---------------------------------------------------------------------------
# B3: mean ablation
# ---------------------------------------------------------------------------


def _apply_ln_final(
    model: HookedTransformer, res_stream: torch.Tensor, ln_scales: torch.Tensor
) -> torch.Tensor:
    if model.cfg.normalization_type in ("RMSPre",) or model.cfg.final_rms:
        return res_stream / ln_scales.unsqueeze(-1)
    if model.cfg.normalization_type in ("LNPre", "LN"):
        centered = res_stream - res_stream.mean(dim=-1, keepdim=True)
        return centered / ln_scales.unsqueeze(-1)
    raise ValueError(f"Unsupported normalization: {model.cfg.normalization_type}")


def mean_ablate_neurons(
    model: HookedTransformer,
    tokenized_data,
    entropy_df: pd.DataFrame,
    neuron_names: list[str],
    *,
    k: int = 10,
    device: str = "cuda",
    chunk_size: int = 20,
) -> pd.DataFrame:
    layer_idx = LAST_LAYER
    neuron_indices = [int(n.split(".")[1]) for n in neuron_names]
    activation_means = torch.tensor(
        entropy_df[[f"{n}_activation" for n in neuron_names]].mean().values
    )

    batch_ids = np.random.choice(entropy_df["batch"].unique(), k, replace=False)
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


def load_tokenized_data(
    tokenizer,
    *,
    dataset: str = DEFAULT_DATASET,
    data_start: int = 0,
    data_end: int = 1000,
    seed: int = 42,
):
    data = load_dataset(dataset, split="train")
    subset = data.select(range(data_start, min(data_end, len(data))))
    tokenized = tl_utils.tokenize_and_concatenate(
        subset, tokenizer, max_length=256, column_name="text"
    )
    return tokenized.shuffle(seed)


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def run_b1(model: HookedTransformer, out_dir: pathlib.Path) -> pd.DataFrame:
    print("B1: Norm vs LogitVar")
    df = compute_static_neuron_df(model)
    plot_b1_norm_vs_logitvar(df, out_dir)
    df.to_csv(out_dir / "static_neuron_metrics.csv", index=False)
    return df


def run_b2(model: HookedTransformer, out_dir: pathlib.Path, static_df: pd.DataFrame | None = None) -> pd.DataFrame:
    print("B2: Null-space fraction rho")
    df = static_df if static_df is not None else compute_static_neuron_df(model)
    plot_b2_rho(df, out_dir)
    top = df.nlargest(10, "rho")["component_name"].tolist()
    print(f"  top-10 rho neurons: {top}")
    hit = [n for n in ENTROPY_NEURONS if n in top]
    print(f"  paper entropy neurons in top-10 rho: {hit}")
    return df


def run_b2_ln_plot(static_df: pd.DataFrame, ablation_df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    agg = (
        ablation_df.assign(
            te=lambda d: (d["loss_post_ablation"] - d["loss"]).abs(),
            de_ln=lambda d: (d["loss_post_ablation_with_frozen_ln"] - d["loss"]).abs(),
        )
        .groupby("component_name", as_index=False)
        .agg({"te": "mean", "de_ln": "mean"})
    )
    agg = agg.merge(static_df[["component_name", "rho", "is_entropy"]], on="component_name")
    agg["ln_mediated_fraction"] = 1 - agg["de_ln"] / agg["te"].replace(0, np.nan)
    plot_b2_rho_vs_ln_fraction(agg, out_dir)
    agg.to_csv(out_dir / "rho_vs_ln_fraction.csv", index=False)


def run_b3(
    model: HookedTransformer,
    tokenizer,
    static_df: pd.DataFrame,
    out_dir: pathlib.Path,
    *,
    device: str,
    k: int = 10,
    batch_size: int = 1,
    data_end: int = 1000,
    neuron_range: tuple[int, int] | None = None,
    quick: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    print(f"B3: Mean ablation (k={k} sequences, data_end={data_end})")
    torch.manual_seed(seed)
    np.random.seed(seed)

    neuron_names = select_ablation_neurons(
        static_df, neuron_range=neuron_range, quick=quick
    )
    print(f"  ablating {len(neuron_names)} neurons: {neuron_names[:5]}...")

    tokenized = load_tokenized_data(tokenizer, data_end=data_end, seed=seed)
    token_df = nutils.make_token_df(tokenized["tokens"], model=model)

    print("  collecting activations...")
    entropy_df = get_entropy_activation_df(
        neuron_names,
        tokenized,
        token_df,
        model,
        batch_size=batch_size,
        device=device,
        cache_residuals=False,
        cache_pre_activations=False,
        residuals_layer=LAST_LAYER,
        residuals_dict={},
    )

    print("  running mean ablation...")
    model.set_use_attn_result(False)
    raw = mean_ablate_neurons(
        model, tokenized, entropy_df, neuron_names, k=k, device=device
    )
    final_df = filter_entropy_activation_df(
        raw, model_name=MODEL_NAME, tokenizer=tokenizer, start_pos=3, end_pos=-1
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    feather_path = out_dir / f"ablation_k{k}.feather"
    final_df.reset_index(drop=True).to_feather(feather_path)
    print(f"  saved {feather_path} ({len(final_df)} rows)")
    return final_df


def run_b4(final_df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    print("B4: Confidence / TE-DE plots")
    plot_b3_te_de_box(final_df, out_dir)
    plot_b4_entropy_vs_rank(final_df, out_dir)
    plot_b4_te_de_scatter(final_df, out_dir)


def _json_float(x):
    return float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else None


def write_summary(
    static_df: pd.DataFrame,
    ablation_df: pd.DataFrame | None,
    out_dir: pathlib.Path,
) -> None:
    summary = {
        "model": MODEL_NAME,
        "entropy_neurons_paper": ENTROPY_NEURONS,
        "top10_rho": [
            {"component_name": r["component_name"], "rho": _json_float(r["rho"])}
            for r in static_df.nlargest(10, "rho")[["component_name", "rho"]].to_dict("records")
        ],
        "entropy_neuron_stats": [
            {
                "component_name": r["component_name"],
                "norm": _json_float(r["norm"]),
                "logit_var": _json_float(r["logit_var"]),
                "rho": _json_float(r["rho"]),
            }
            for r in static_df[static_df["is_entropy"]][
                ["component_name", "norm", "logit_var", "rho"]
            ].to_dict("records")
        ],
    }
    if ablation_df is not None:
        agg = (
            ablation_df.assign(
                abs_te=lambda d: (d["loss_post_ablation"] - d["loss"]).abs(),
                abs_de_ln=lambda d: (d["loss_post_ablation_with_frozen_ln"] - d["loss"]).abs(),
                delta_h=lambda d: d["entropy_post_ablation"] - d["entropy"],
                pred_change=lambda d: (d["top_logits_post_ablation"] != d["pred"]).astype(int),
            )
            .groupby("component_name", as_index=False)
            .agg({"abs_te": "mean", "abs_de_ln": "mean", "delta_h": "mean", "pred_change": "mean"})
        )
        ent = agg[agg["component_name"].isin(ENTROPY_NEURONS)]
        rnd = agg[~agg["component_name"].isin(ENTROPY_NEURONS)]
        summary["ablation"] = {
            "entropy_neurons_mean_abs_te": _json_float(ent["abs_te"].mean()),
            "random_neurons_mean_abs_te": _json_float(rnd["abs_te"].mean()),
            "entropy_neurons_mean_abs_de_ln": _json_float(ent["abs_de_ln"].mean()),
            "random_neurons_mean_abs_de_ln": _json_float(rnd["abs_de_ln"].mean()),
            "entropy_neurons_mean_delta_h": _json_float(ent["delta_h"].mean()),
            "entropy_neurons_mean_pred_change": _json_float(ent["pred_change"].mean()),
            "random_neurons_mean_pred_change": _json_float(rnd["pred_change"].mean()),
        }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 Track 1B reproduction")
    p.add_argument(
        "--step",
        nargs="+",
        choices=["all", "b1", "b2", "b3", "b4"],
        default=["all"],
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    p.add_argument("--k", type=int, default=10, help="Sequences for ablation")
    p.add_argument("--data-end", type=int, default=1000, help="C4 sequences for activation cache")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--neuron-range", type=str, default=None, help="Override: e.g. 0-3072 for full MLP")
    p.add_argument("--quick", action="store_true", help="Smaller ablation set + data_end=200")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--png", action="store_true", help="Also export PNG (slower; needs kaleido)")
    p.add_argument("--ablation-path", type=pathlib.Path, default=None)
    return p.parse_args()


def _resolve_steps(steps: Iterable[str]) -> set[str]:
    return {"b1", "b2", "b3", "b4"} if "all" in steps else set(steps)


def main() -> None:
    global _SAVE_PNG
    args = parse_args()
    _SAVE_PNG = args.png
    steps = _resolve_steps(args.step)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_end = 200 if args.quick else args.data_end
    neuron_range = None
    if args.neuron_range:
        lo, hi = args.neuron_range.split("-")
        neuron_range = (int(lo), int(hi))

    need_model = bool(steps & {"b1", "b2", "b3"})
    model, tokenizer = None, None
    if need_model:
        print(f"Loading {MODEL_NAME} on {args.device}...")
        model, tokenizer = load_gpt2(args.device)

    static_df: pd.DataFrame | None = None
    ablation_df: pd.DataFrame | None = None

    if "b1" in steps:
        static_df = run_b1(model, out_dir)

    if "b2" in steps:
        static_df = run_b2(model, out_dir, static_df)

    if "b3" in steps:
        if static_df is None:
            static_df = compute_static_neuron_df(model)
        ablation_df = run_b3(
            model,
            tokenizer,
            static_df,
            out_dir,
            device=args.device,
            k=args.k,
            batch_size=args.batch_size,
            data_end=data_end,
            neuron_range=neuron_range,
            quick=args.quick,
            seed=args.seed,
        )

    if "b4" in steps:
        if ablation_df is None:
            path = args.ablation_path or out_dir / f"ablation_k{args.k}.feather"
            if not path.exists():
                raise FileNotFoundError(f"No ablation at {path}. Run b3 first.")
            ablation_df = pd.read_feather(path)
        run_b4(ablation_df, out_dir)

    if static_df is not None and ablation_df is not None and "b2" in steps:
        run_b2_ln_plot(static_df, ablation_df, out_dir)

    if static_df is not None:
        write_summary(static_df, ablation_df, out_dir)

    print(f"\nDone. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
