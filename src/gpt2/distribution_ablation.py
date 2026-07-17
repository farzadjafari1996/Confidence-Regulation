"""
Phase 2b — logit distribution comparison after mean ablation.

Compares baseline vs ablated softmax distributions (JS/KL, affine shift vs shape)
for entropy neurons vs random controls on C4 batches.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import tqdm
from transformer_lens import utils as tl_utils

PLOT_COLORS_DIST = {"Entropy": "#EF553B", "Random": "#636EFA"}

from gpt2.core import (
    _apply_ln_final,
    _save_fig,
    filter_entropy_activation_df,
    get_entropy_activation_df,
    load_phase1_outputs,
    load_tokenized_data,
    select_phase2_neurons,
    select_random_control_neurons,
    tokenized_tokens_tensor,
)
from hyperparameters import ModelRunConfig

AblationBranch = Literal["te", "de_ln", "both"]


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(P || Q) for row-wise probability vectors (..., vocab)."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen–Shannon divergence (symmetric, bounded in [0, ln 2])."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, eps=eps) + 0.5 * _kl_divergence(q, m, eps=eps)


def _affine_r2(baseline: torch.Tensor, ablated: torch.Tensor) -> torch.Tensor:
    """R² of ablated ≈ α·baseline + β (per row)."""
    b_mean = baseline.mean(dim=-1, keepdim=True)
    a_mean = ablated.mean(dim=-1, keepdim=True)
    b_c = baseline - b_mean
    a_c = ablated - a_mean
    var_b = (b_c * b_c).sum(dim=-1).clamp(min=1e-12)
    cov = (b_c * a_c).sum(dim=-1)
    alpha = cov / var_b
    beta = (a_mean - alpha.unsqueeze(-1) * b_mean).squeeze(-1)
    fitted = alpha.unsqueeze(-1) * baseline + beta.unsqueeze(-1)
    ss_res = ((ablated - fitted) ** 2).sum(dim=-1)
    ss_tot = ((ablated - a_mean) ** 2).sum(dim=-1).clamp(min=1e-12)
    return (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)


def _constant_shift_r2(baseline: torch.Tensor, ablated: torch.Tensor) -> torch.Tensor:
    """R² of ablated ≈ baseline + c (pure logit shift; softmax-invariant if R²≈1)."""
    delta = ablated - baseline
    c = delta.mean(dim=-1, keepdim=True)
    fitted = baseline + c
    ss_res = ((ablated - fitted) ** 2).sum(dim=-1)
    ss_tot = ((ablated - ablated.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1).clamp(min=1e-12)
    return (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)


def _spearman_batch(baseline: torch.Tensor, ablated: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation per row (vocab axis). Stays on the input device."""
    rank_b = baseline.argsort(dim=-1).argsort(dim=-1).float()
    rank_a = ablated.argsort(dim=-1).argsort(dim=-1).float()
    rb = rank_b - rank_b.mean(dim=-1, keepdim=True)
    ra = rank_a - rank_a.mean(dim=-1, keepdim=True)
    num = (rb * ra).sum(dim=-1)
    den = (rb.square().sum(dim=-1).sqrt() * ra.square().sum(dim=-1).sqrt()).clamp(min=1e-12)
    return (num / den).clamp(-1.0, 1.0)


def _topk_jaccard_batch(
    baseline: torch.Tensor, ablated: torch.Tensor, k: int = 5
) -> torch.Tensor:
    """Jaccard overlap of top-k token indices per row (vectorized; topk ids are unique)."""
    top_b = baseline.topk(k, dim=-1).indices
    top_a = ablated.topk(k, dim=-1).indices
    # (N, k, k) equality → intersection size; |union| = 2k - |intersection|
    inter = (top_b.unsqueeze(-1) == top_a.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()
    return inter / (2 * k - inter).clamp(min=1.0)


# Full-vocab Spearman on ~248k tokens is the Phase-2b bottleneck on Qwen (argsort
# dominated the crashed full run at ~40 min/batch). Subsample keeps rank signal.
_SPEARMAN_LARGE_VOCAB_SAMPLE = 8192
_SPEARMAN_SUBSAMPLE_SEED = 0


def _spearman_large_vocab(baseline: torch.Tensor, ablated: torch.Tensor) -> torch.Tensor:
    """Spearman on a fixed vocab subsample — fast + memory-safe for Qwen-scale vocabs."""
    d_vocab = baseline.shape[-1]
    n_sample = min(_SPEARMAN_LARGE_VOCAB_SAMPLE, d_vocab)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SPEARMAN_SUBSAMPLE_SEED)
    idx = torch.randperm(d_vocab, generator=gen)[:n_sample]
    if baseline.is_cuda:
        idx = idx.to(baseline.device, non_blocking=True)
    return _spearman_batch(baseline.index_select(-1, idx), ablated.index_select(-1, idx))


def _spearman_chunked(baseline: torch.Tensor, ablated: torch.Tensor) -> torch.Tensor:
    """Spearman that stays memory-safe on ~248k vocabs (8GB GPUs)."""
    d_vocab = baseline.shape[-1]
    if d_vocab < 100_000:
        return _spearman_batch(baseline, ablated)
    return _spearman_large_vocab(baseline, ablated)


def batch_distribution_metrics(
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    *,
    topk: int = 5,
) -> dict[str, np.ndarray]:
    """Metrics comparing baseline vs ablated logit vectors (N, vocab).

    Softmax runs after logit-space metrics so peak VRAM holds fewer large tensors.
    Large-vocab Spearman uses a fixed 8k subsample (full argsort is impractical).
    """
    base = baseline_logits.float()
    abl = ablated_logits.float()

    out: dict[str, np.ndarray] = {
        "argmax_flip": (base.argmax(dim=-1) != abl.argmax(dim=-1))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "affine_r2": _affine_r2(base, abl).detach().cpu().numpy(),
        "constant_shift_r2": _constant_shift_r2(base, abl).detach().cpu().numpy(),
        f"top{topk}_jaccard": _topk_jaccard_batch(base, abl, k=topk).detach().cpu().numpy(),
        "spearman_rho": _spearman_chunked(base, abl).detach().cpu().numpy(),
    }

    p = torch.softmax(base, dim=-1)
    del base
    q = torch.softmax(abl, dim=-1)
    del abl

    h_base = -(p * p.clamp(min=1e-12).log()).sum(dim=-1)
    h_abl = -(q * q.clamp(min=1e-12).log()).sum(dim=-1)
    out["kl_divergence"] = _kl_divergence(p, q).detach().cpu().numpy()
    out["js_divergence"] = _js_divergence(p, q).detach().cpu().numpy()
    out["delta_entropy"] = (h_abl - h_base).detach().cpu().numpy()
    del p, q
    return out


def _seq_ln_scales(ln_scales: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Normalize cached LN/RMS scales to shape (seq_len,) for batched (n_chunk, seq, d_model) tensors."""
    s = ln_scales.detach().float()
    if s.dim() >= 2 and s.shape[0] == 1:
        s = s[0]
    s = s.reshape(-1)
    if s.numel() != seq_len:
        raise ValueError(f"Expected ln_scales length {seq_len}, got {s.numel()}")
    return s


def _project_to_logits(model, resid: torch.Tensor) -> torch.Tensor:
    """Unembed residual stream. Use fp16 matmul on CUDA for large-vocab models (Qwen ~248k)."""
    d_vocab = int(getattr(model.cfg, "d_vocab", model.W_U.shape[-1]))
    if resid.is_cuda and d_vocab >= 100_000:
        w_u = model.W_U if model.W_U.dtype == torch.float16 else model.W_U.half()
        b_u = model.b_U if model.b_U.dtype == torch.float16 else model.b_U.half()
        return resid.half() @ w_u + b_u
    w_u = model.W_U.float() if hasattr(model.W_U, "float") else model.W_U
    b_u = model.b_U.float() if hasattr(model.b_U, "float") else model.b_U
    return resid.float() @ w_u + b_u


def _de_normed(model, updated: torch.Tensor, ln_scales: torch.Tensor) -> torch.Tensor:
    """Apply frozen final LN/RMSNorm (no unembed)."""
    seq_len = updated.shape[-2]
    scales = _seq_ln_scales(ln_scales, seq_len)
    if callable(getattr(model, "apply_ln_final_frozen", None)):
        return model.apply_ln_final_frozen(updated, scales)
    scales_bc = scales.view(1, -1, 1)
    if model.cfg.normalization_type in ("RMSPre",) or getattr(model.cfg, "final_rms", False):
        return updated.float() / scales_bc
    if model.cfg.normalization_type in ("LNPre", "LN"):
        centered = updated - updated.mean(dim=-1, keepdim=True)
        return centered.float() / scales_bc
    return _apply_ln_final(model, updated, scales)


def _de_logits(model, updated: torch.Tensor, ln_scales: torch.Tensor) -> torch.Tensor:
    """DE_LN logits with frozen per-position LN/RMSNorm scales."""
    return _project_to_logits(model, _de_normed(model, updated, ln_scales))


def _te_logits(model, res_stream: torch.Tensor) -> torch.Tensor:
    return _project_to_logits(model, model.ln_final(res_stream))


def _w_out_last(model, layer_idx: int) -> torch.Tensor:
    w_out = model.W_out[layer_idx]
    return w_out.float() if hasattr(w_out, "float") else w_out


def _pos_microbatch_size(model, n_chunk: int, seq_len: int, *, n_logit_bufs: int = 2) -> int:
    """Keep GPU logit+metric buffers tiny on large-vocab / 8GB cards.

    Hard-caps microbatch so we never spill into system RAM (which hangs/kills
    the process on Windows without a Python traceback).
    """
    d_vocab = int(getattr(model.cfg, "d_vocab", model.W_U.shape[-1]))
    if d_vocab < 100_000:
        return max(seq_len, 1)
    bytes_per_row = d_vocab * (2 + 4 * max(n_logit_bufs, 1))
    micro = max(1, int(0.25 * (1024**3) / (max(n_chunk, 1) * bytes_per_row)))
    return min(micro, seq_len, 4)  # hard cap: at most 4 positions per step


def mean_ablate_neurons_distribution(
    model,
    cfg: ModelRunConfig,
    tokenized_data,
    entropy_df: pd.DataFrame,
    neuron_names: list[str],
    *,
    device: str,
    chunk_size: int = 20,
    softmax_top_k: int = 50,
    n_example_tokens: int = 5,
    example_candidates: list[dict] | None = None,
    branch: AblationBranch = "both",
) -> tuple[pd.DataFrame, list[dict]]:
    """Mean-ablate neurons; record per-token distribution metrics (TE and/or DE_LN)."""
    layer_idx = cfg.last_layer(model)
    neuron_indices = [int(n.split(".")[1]) for n in neuron_names]
    # Per-column float64 mean: float16 pandas/numpy reductions overflow on long C4 runs
    act_cols = [f"{n}_activation" for n in neuron_names]
    act_means_np = np.stack(
        [np.asarray(entropy_df[c], dtype=np.float64).mean() for c in act_cols]
    )
    if not np.isfinite(act_means_np).all():
        bad = [act_cols[i] for i, v in enumerate(act_means_np) if not np.isfinite(v)]
        raise RuntimeError(f"Non-finite activation means (dtype overflow?): {bad[:5]}")
    activation_means = torch.tensor(act_means_np, dtype=torch.float32)
    do_te = branch in ("te", "both")
    do_de = branch in ("de_ln", "both")
    # Metrics stage at most 2 large fp32 tensors at once (see batch_distribution_metrics)
    n_logit_bufs = 2
    batch_ids = np.random.choice(entropy_df["batch"].unique(), cfg.k, replace=False)
    filtered = entropy_df[entropy_df["batch"].isin(batch_ids)].copy()
    frames: list[pd.DataFrame] = []
    examples: list[dict] = list(example_candidates or [])
    w_out = _w_out_last(model, layer_idx)

    for batch_n in tqdm.tqdm(batch_ids, desc="distribution ablation batches"):
        tok_seq = tokenized_data["tokens"][batch_n]
        inp = tok_seq.unsqueeze(0).to(device)
        _, cache = model.run_with_cache(inp)
        rows = filtered[filtered["batch"] == batch_n]
        pos_idx = rows["pos"].values.astype(np.int64)
        n_pos = len(pos_idx)
        res_stream = cache[tl_utils.get_act_name("resid_post", layer_idx)][0]
        seq_len = int(res_stream.shape[0])
        ln_scales = _seq_ln_scales(
            torch.tensor(rows["ln_final_scale"].values, device=device, dtype=torch.float32),
            seq_len,
        )
        prev_act = cache[tl_utils.get_act_name("post", layer_idx)][0, :, neuron_indices].float()
        del cache
        act_deltas = activation_means.to(prev_act.device) - prev_act

        # Baseline logits only at needed positions (avoids full-seq vocab materialization).
        baseline_rows = _te_logits(model, res_stream.unsqueeze(0)[:, pos_idx, :]).squeeze(0)

        for start in range(0, len(neuron_indices), chunk_size):
            idx_chunk = neuron_indices[start : start + chunk_size]
            delta_chunk = act_deltas[:, start : start + chunk_size]
            res_delta = delta_chunk.unsqueeze(-1) * w_out[idx_chunk, :]
            res_delta = res_delta.permute(1, 0, 2)
            n_chunk = res_delta.shape[0]
            updated = res_stream.repeat(n_chunk, 1, 1) + res_delta
            del res_delta

            te_metric_parts: dict[str, list[np.ndarray]] = {}
            de_metric_parts: dict[str, list[np.ndarray]] = {}
            example_te_rows: dict[int, tuple[float, torch.Tensor, int]] = {}
            pos_micro = _pos_microbatch_size(model, n_chunk, n_pos, n_logit_bufs=n_logit_bufs)

            large_vocab = int(getattr(model.cfg, "d_vocab", baseline_rows.shape[-1])) >= 100_000
            for p0 in range(0, n_pos, pos_micro):
                p_slice = slice(p0, min(p0 + pos_micro, n_pos))
                p_abs = pos_idx[p_slice]
                n_micro = len(p_abs)

                updated_sel = updated[:, p_abs, :]
                base_rows = baseline_rows[p_slice]
                # Expand baseline once per microbatch; keep on GPU (CPU full-vocab
                # softmax/spearman was the ~40 min/batch hang on Qwen).
                base_flat = (
                    base_rows.unsqueeze(0)
                    .expand(n_chunk, -1, -1)
                    .reshape(-1, base_rows.shape[-1])
                )

                if do_te:
                    te_rows = _te_logits(model, updated_sel)
                    te_flat = te_rows.reshape(-1, te_rows.shape[-1])
                    te_m = batch_distribution_metrics(base_flat, te_flat)
                    for key, arr in te_m.items():
                        te_metric_parts.setdefault(key, []).append(arr.reshape(n_chunk, n_micro))

                    if len(examples) < n_example_tokens * 2:
                        js = te_m["js_divergence"].reshape(n_chunk, n_micro)
                        for i in range(n_chunk):
                            local_j = int(np.argmax(js[i]))
                            js_val = float(js[i, local_j])
                            prev = example_te_rows.get(i)
                            if prev is None or js_val > prev[0]:
                                example_te_rows[i] = (
                                    js_val,
                                    te_rows[i, local_j].detach().float().cpu(),
                                    int(p_abs[local_j]),
                                )
                    del te_rows, te_flat, te_m

                if do_de:
                    de_rows = _de_logits(model, updated_sel, ln_scales[p_abs])
                    de_flat = de_rows.reshape(-1, de_rows.shape[-1])
                    del de_rows
                    de_m = batch_distribution_metrics(base_flat, de_flat)
                    for key, arr in de_m.items():
                        de_metric_parts.setdefault(key, []).append(arr.reshape(n_chunk, n_micro))
                    del de_flat, de_m

                del updated_sel, base_flat
                if large_vocab and device.startswith("cuda") and (p0 // pos_micro) % 8 == 7:
                    torch.cuda.empty_cache()

            del updated
            te_metrics = (
                {k: np.concatenate(v, axis=1) for k, v in te_metric_parts.items()}
                if te_metric_parts
                else {}
            )
            de_metrics = (
                {k: np.concatenate(v, axis=1) for k, v in de_metric_parts.items()}
                if de_metric_parts
                else {}
            )

            for i, name in enumerate(neuron_names[start : start + chunk_size]):
                chunk = rows.copy()
                drop_cols = [f"{n}_activation" for n in neuron_names if n != name]
                chunk = chunk.drop(columns=drop_cols, errors="ignore")
                chunk = chunk.rename(columns={f"{name}_activation": "activation"})
                chunk["component_name"] = name
                chunk["batch_id"] = batch_n

                for key, arr in te_metrics.items():
                    chunk[f"{key}_te"] = arr[i]
                for key, arr in de_metrics.items():
                    chunk[f"{key}_de_ln"] = arr[i]

                frames.append(chunk)

            if do_te and len(examples) < n_example_tokens * 2:
                for i, name in enumerate(neuron_names[start : start + chunk_size]):
                    if i not in example_te_rows:
                        continue
                    _, te_vec, abs_pos = example_te_rows[i]
                    # baseline_rows is aligned to pos_idx, not absolute positions
                    base_j = int(np.where(pos_idx == abs_pos)[0][0])
                    examples.append(
                        _softmax_example_record(
                            name,
                            int(batch_n),
                            abs_pos,
                            baseline_rows[base_j],
                            te_vec,
                            top_k=softmax_top_k,
                        )
                    )
                    if len(examples) >= n_example_tokens * 2:
                        break

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        del baseline_rows
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return pd.concat(frames, ignore_index=True), examples


def _softmax_example_record(
    neuron: str,
    batch_id: int,
    pos: int,
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    *,
    top_k: int,
) -> dict:
    base = torch.softmax(baseline_logits.float().cpu(), dim=-1)
    abl = torch.softmax(ablated_logits.float().cpu(), dim=-1)
    top_vals, top_ids = base.topk(top_k)
    return {
        "neuron": str(neuron),
        "batch_id": int(batch_id),
        "pos": int(pos),
        "token_ids": [int(t) for t in top_ids.tolist()],
        "baseline_probs": [float(p) for p in top_vals.tolist()],
        "ablated_probs": [float(p) for p in abl[top_ids].tolist()],
    }


def _branch_suffix(branch: AblationBranch) -> str:
    if branch == "both":
        return "te"
    return branch


def _metric_col(base: str, branch: AblationBranch) -> str:
    suffix = _branch_suffix(branch)
    return f"{base}_{suffix}" if suffix in ("te", "de_ln") else f"{base}_te"


def plot_dist_deltaH_vs_argmax_flip(
    df: pd.DataFrame,
    entropy_labels: list[str],
    out_dir: pathlib.Path,
    model_title: str,
    *,
    branch: AblationBranch = "te",
) -> None:
    """Per-neuron: entropy change vs modal (argmax) stability — separates shift vs flip."""
    de_col = _metric_col("delta_entropy", branch)
    flip_col = _metric_col("argmax_flip", branch)
    agg = df.groupby("component_name", as_index=False).agg(
        **{
            "abs_delta_entropy": (de_col, lambda s: np.abs(s).mean()),
            "argmax_flip_rate": (flip_col, "mean"),
        }
    )
    agg["neuron_type"] = np.where(agg["component_name"].isin(entropy_labels), "Entropy", "Random")
    branch_label = branch.upper() if branch != "both" else "TE"
    fig = px.scatter(
        agg,
        x="argmax_flip_rate",
        y="abs_delta_entropy",
        color="neuron_type",
        hover_data=["component_name"],
        title=f"{model_title} — |ΔH| vs argmax flip ({branch_label})",
        color_discrete_map=PLOT_COLORS_DIST,
        labels={
            "argmax_flip_rate": "Argmax flip rate",
            "abs_delta_entropy": "Mean |Δentropy|",
        },
    )
    fig.update_layout(width=560, height=440, title_font_size=15)
    _save_fig(fig, out_dir / f"dist_deltaH_vs_argmax_flip_{_branch_suffix(branch)}")


def plot_dist_affine_r2_vs_js(
    df: pd.DataFrame,
    entropy_labels: list[str],
    out_dir: pathlib.Path,
    model_title: str,
    *,
    branch: AblationBranch = "te",
) -> None:
    js_col = _metric_col("js_divergence", branch)
    r2_col = _metric_col("affine_r2", branch)
    agg = df.groupby("component_name", as_index=False).agg(
        **{js_col: (js_col, "mean"), r2_col: (r2_col, "mean")}
    )
    agg["neuron_type"] = np.where(agg["component_name"].isin(entropy_labels), "Entropy", "Random")
    branch_label = branch.upper() if branch != "both" else "TE"
    fig = px.scatter(
        agg,
        x=r2_col,
        y=js_col,
        color="neuron_type",
        hover_data=["component_name"],
        title=f"{model_title} — affine R² vs JS ({branch_label})",
        color_discrete_map=PLOT_COLORS_DIST,
        labels={r2_col: "Mean affine R² (shift/scale fit)", js_col: "Mean JS divergence"},
    )
    fig.update_layout(width=560, height=440, title_font_size=15)
    _save_fig(fig, out_dir / f"dist_affine_r2_vs_js_{_branch_suffix(branch)}")


def plot_dist_delta_entropy_vs_js(
    df: pd.DataFrame,
    entropy_labels: list[str],
    out_dir: pathlib.Path,
    model_title: str,
    *,
    branch: AblationBranch = "te",
) -> None:
    js_col = _metric_col("js_divergence", branch)
    de_col = _metric_col("delta_entropy", branch)
    agg = df.groupby("component_name", as_index=False).agg(
        **{
            js_col: (js_col, "mean"),
            "abs_delta_entropy": (de_col, lambda s: np.abs(s).mean()),
        }
    )
    agg["neuron_type"] = np.where(agg["component_name"].isin(entropy_labels), "Entropy", "Random")
    branch_label = branch.upper() if branch != "both" else "TE"
    fig = px.scatter(
        agg,
        x=js_col,
        y="abs_delta_entropy",
        color="neuron_type",
        hover_data=["component_name"],
        title=f"{model_title} — |ΔH| vs JS ({branch_label})",
        color_discrete_map=PLOT_COLORS_DIST,
        labels={"abs_delta_entropy": "Mean |Δentropy|", js_col: "Mean JS divergence"},
    )
    fig.update_layout(width=560, height=440, title_font_size=15)
    _save_fig(fig, out_dir / f"dist_delta_entropy_vs_js_{_branch_suffix(branch)}")


def plot_dist_entropy_vs_random_facets(
    df: pd.DataFrame,
    entropy_labels: list[str],
    out_dir: pathlib.Path,
    model_title: str,
    *,
    branch: AblationBranch = "te",
) -> None:
    """2×2 facet box plots: JS, shape change, rank correlation, top-5 overlap."""
    js_col = _metric_col("js_divergence", branch)
    shift_col = _metric_col("constant_shift_r2", branch)
    rho_col = _metric_col("spearman_rho", branch)
    jaccard_col = _metric_col("top5_jaccard", branch)

    plot_df = df.copy()
    plot_df["neuron_type"] = np.where(
        plot_df["component_name"].isin(entropy_labels), "Entropy", "Random"
    )
    plot_df["shape_change"] = 1.0 - plot_df[shift_col]

    branch_label = branch.upper() if branch != "both" else "TE"
    metric_order = [
        "JS divergence",
        "Softmax-visible shape change",
        "Spearman rank correlation",
        "Top-5 Jaccard overlap",
    ]
    melted = plot_df.melt(
        id_vars=["neuron_type"],
        value_vars=["shape_change", js_col, rho_col, jaccard_col],
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].map(
        {
            js_col: "JS divergence",
            "shape_change": "Softmax-visible shape change",
            rho_col: "Spearman rank correlation",
            jaccard_col: "Top-5 Jaccard overlap",
        }
    )
    melted["metric"] = pd.Categorical(melted["metric"], categories=metric_order, ordered=True)

    fig = px.box(
        melted,
        x="neuron_type",
        y="value",
        color="neuron_type",
        facet_col="metric",
        facet_col_wrap=2,
        category_orders={"metric": metric_order, "neuron_type": ["Entropy", "Random"]},
        title=f"{model_title} — entropy vs random controls ({branch_label})",
        color_discrete_map=PLOT_COLORS_DIST,
        labels={"value": "Metric value", "neuron_type": ""},
    )
    fig.update_xaxes(title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(width=900, height=620, title_font_size=15, showlegend=False)
    _save_fig(fig, out_dir / f"dist_entropy_vs_random_facets_{_branch_suffix(branch)}")


def plot_dist_te_vs_de_ln_js(
    df: pd.DataFrame,
    entropy_labels: list[str],
    out_dir: pathlib.Path,
    model_title: str,
) -> None:
    agg = (
        df.groupby("component_name", as_index=False)
        .agg(
            js_te=("js_divergence_te", "mean"),
            js_de=("js_divergence_de_ln", "mean"),
        )
    )
    agg["neuron_type"] = np.where(agg["component_name"].isin(entropy_labels), "Entropy", "Random")
    fig = px.scatter(
        agg,
        x="js_te",
        y="js_de",
        color="neuron_type",
        hover_data=["component_name"],
        title=f"{model_title} — JS divergence: TE vs DE_LN",
        labels={"js_te": "Mean JS (TE)", "js_de": "Mean JS (DE_LN)"},
        color_discrete_map=PLOT_COLORS_DIST,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, agg[["js_te", "js_de"]].max().max()],
            y=[0, agg[["js_te", "js_de"]].max().max()],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        )
    )
    fig.update_layout(width=560, height=440, title_font_size=15)
    _save_fig(fig, out_dir / "dist_te_vs_de_ln_js")


def plot_all_distribution_ablation(
    df: pd.DataFrame,
    static_df: pd.DataFrame,
    entropy_labels: list[str],
    out_dir: pathlib.Path,
    model_title: str,
    *,
    branch: AblationBranch = "te",
) -> None:
    _ = static_df  # reserved for future static-metric overlays
    branches: list[AblationBranch] = ["te", "de_ln"] if branch == "both" else [branch]
    for b in branches:
        plot_dist_deltaH_vs_argmax_flip(df, entropy_labels, out_dir, model_title, branch=b)
        plot_dist_affine_r2_vs_js(df, entropy_labels, out_dir, model_title, branch=b)
        plot_dist_delta_entropy_vs_js(df, entropy_labels, out_dir, model_title, branch=b)
        plot_dist_entropy_vs_random_facets(df, entropy_labels, out_dir, model_title, branch=b)
    if "js_divergence_te" in df.columns and "js_divergence_de_ln" in df.columns:
        plot_dist_te_vs_de_ln_js(df, entropy_labels, out_dir, model_title)


def write_phase2b_summary(
    df: pd.DataFrame,
    cfg: ModelRunConfig,
    out_dir: pathlib.Path,
    entropy_labels: list[str],
    neuron_names: list[str],
    *,
    branch: AblationBranch,
    elapsed_sec: float,
) -> None:
    js_col = _metric_col("js_divergence", branch)
    is_ent = df["component_name"].isin(entropy_labels)
    summary = {
        "model": cfg.name,
        "branch_primary": branch,
        "elapsed_sec": elapsed_sec,
        "n_entropy_neurons": len(entropy_labels),
        "n_random_neurons": len(neuron_names) - len(entropy_labels),
        "n_tokens": int(len(df) / max(len(neuron_names), 1)),
        "mean_js_entropy": float(df.loc[is_ent, js_col].mean()) if is_ent.any() else None,
        "mean_js_random": float(df.loc[~is_ent, js_col].mean()) if (~is_ent).any() else None,
        "mean_affine_r2_entropy": float(df.loc[is_ent, _metric_col("affine_r2", branch)].mean())
        if is_ent.any()
        else None,
        "mean_constant_shift_r2_entropy": float(
            df.loc[is_ent, _metric_col("constant_shift_r2", branch)].mean()
        )
        if is_ent.any()
        else None,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved {path}")


def run_phase2b_distribution_ablation(
    cfg: ModelRunConfig,
    phase1_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    device: str,
    load_model_fn,
    branch: AblationBranch = "te",
    softmax_top_k: int = 50,
    n_example_tokens: int = 5,
    paper_neurons: list[str] | None = None,
    load_tokenized_fn=None,
    get_entropy_df_fn=None,
    filter_df_fn=None,
) -> pd.DataFrame:
    """Run Phase 2b. Optional *_fn hooks let Qwen/Llama reuse family-specific data paths."""
    import neel.utils as nutils

    _load_tokenized = load_tokenized_fn or load_tokenized_data
    _get_entropy_df = get_entropy_df_fn or get_entropy_activation_df
    _filter_df = filter_df_fn or filter_entropy_activation_df

    out_dir.mkdir(parents=True, exist_ok=True)
    entropy_labels, static_df = load_phase1_outputs(phase1_dir)
    random_labels = select_random_control_neurons(static_df, entropy_labels, cfg)
    neuron_names = select_phase2_neurons(static_df, entropy_labels, cfg)

    print(f"\n{'='*60}\nPhase 2b (distribution) — {cfg.name} on {device}\n{'='*60}")
    print(
        f"  entropy neurons: {len(entropy_labels)}  "
        f"random controls: {len(random_labels)}  ablated: {len(neuron_names)}"
    )
    print(f"  k={cfg.k} batches  data_end={cfg.data_end}  branch={branch}")
    print(
        f"  token filter: pos in [{cfg.filter_start_pos}, {cfg.filter_end_pos}] "
        f"(BOS/EOS/newline excluded)"
    )

    t0 = time.perf_counter()
    model, tokenizer = load_model_fn(cfg, device)
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(False)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    tokenized = _load_tokenized(tokenizer, cfg)
    token_df = nutils.make_token_df(tokenized_tokens_tensor(tokenized), model=model)
    entropy_df = _get_entropy_df(
        neuron_names,
        tokenized,
        token_df,
        model,
        batch_size=cfg.batch_size,
        device=device,
        residuals_layer=cfg.last_layer(model),
    )
    raw, examples = mean_ablate_neurons_distribution(
        model,
        cfg,
        tokenized,
        entropy_df,
        neuron_names,
        device=device,
        chunk_size=cfg.chunk_size,
        softmax_top_k=softmax_top_k,
        n_example_tokens=n_example_tokens,
        branch=branch,
    )
    dist_df = _filter_df(
        raw,
        model_name=cfg.name,
        tokenizer=tokenizer,
        start_pos=cfg.filter_start_pos,
        end_pos=cfg.filter_end_pos,
    )

    feather_path = out_dir / "distribution_metrics.feather"
    dist_df.reset_index(drop=True).to_feather(feather_path)
    print(f"  saved {feather_path}")

    examples_path = out_dir / "softmax_examples.json"
    examples_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"  saved {examples_path}")

    plot_all_distribution_ablation(
        dist_df,
        static_df,
        entropy_labels,
        out_dir,
        cfg.name,
        branch=branch,
    )

    elapsed = time.perf_counter() - t0
    write_phase2b_summary(
        dist_df,
        cfg,
        out_dir,
        entropy_labels,
        neuron_names,
        branch=branch,
        elapsed_sec=elapsed,
    )

    if device.startswith("cuda"):
        del model
        torch.cuda.empty_cache()

    print(f"Phase 2b done in {elapsed:.0f}s -> {out_dir.resolve()}")
    return dist_df
