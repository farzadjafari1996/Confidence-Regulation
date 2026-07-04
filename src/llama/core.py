"""
Llama 3.2 family — Phase 1/2 logic (Stolfo et al., NeurIPS 2024).

SwiGLU last-layer neurons via down_proj rows; HuggingFace backend with gpt2/core plots.
"""

from __future__ import annotations

import math
import pathlib

import einops
import numpy as np
import pandas as pd
import torch
import tqdm
from transformer_lens import utils as tl_utils

from gpt2.core import (
    ModelRunConfig,
    annotate_top1pct_rho,
    compute_static_neuron_df,
    load_phase1_outputs,
    load_tokenized_data,
    neuron_label,
    normalize_component_names,
    plot_b1_norm_vs_logitvar,
    plot_b2_rho,
    plot_b2_rho_vs_ln_fraction,
    plot_fig1_entropy_vs_argmax,
    plot_fig2c_te_de_box,
    plot_fig2d_singular_values,
    run_phase1_identify as _gpt2_run_phase1_identify,
    select_phase2_neurons,
    top1pct_rho_neurons,
    write_phase1_summary,
    write_phase2_summary,
)
from llama.wrapper import LlamaWrapper

LLAMA_NULLSPACE_K = 20


def get_entropy(logits: torch.Tensor, use_log2: bool = False) -> torch.Tensor:
    entropy = logits.softmax(dim=-1) * logits.log_softmax(dim=-1)
    if use_log2:
        entropy = entropy / torch.log(torch.tensor(2.0, device=logits.device))
    return -einops.einsum(entropy, "batch seq d_vocab -> batch seq")


def neuron_str_to_neuron(neuron_str: str) -> tuple[int, int]:
    layer, index = neuron_str.split(".")
    return int(layer), int(index)


def filter_entropy_activation_df(
    entropy_df: pd.DataFrame,
    *,
    model_name: str | None = None,
    tokenizer=None,
    start_pos: int = 3,
    end_pos: int = -1,
) -> pd.DataFrame:
    del model_name  # plain C4; no GPT-specific newline filtering

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        for tok in ("<|begin_of_text|>", "<s>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                bos_token_id = tid
                break

    eos_token_id = tokenizer.eos_token_id
    combined_mask = pd.Series(False, index=entropy_df.index)

    if bos_token_id is not None:
        combined_mask = combined_mask | (entropy_df["token_id"] == bos_token_id)
    if eos_token_id is not None:
        combined_mask = combined_mask | (entropy_df["token_id"] == eos_token_id)

    if combined_mask.any():
        preceding_token_mask = combined_mask.iloc[1:].reset_index(drop=True)
        combined_mask = combined_mask | preceding_token_mask

    filtered = entropy_df[~combined_mask]

    if end_pos < 0:
        end_pos_val = entropy_df["pos"].max() + 1 + end_pos
    else:
        end_pos_val = end_pos

    return filtered[filtered["pos"].between(start_pos, end_pos_val, inclusive="left")]


def get_entropy_activation_df(
    neuron_names: list[str],
    tokenized_data,
    token_df: pd.DataFrame,
    model: LlamaWrapper,
    *,
    batch_size: int = 1,
    device: str = "cuda",
    residuals_layer: int | None = None,
) -> pd.DataFrame:
    del residuals_layer
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

        entropy_rows.append(get_entropy(logits.float()).cpu().numpy())
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


def _per_token_loss(
    model: LlamaWrapper, logits: torch.Tensor, inp: torch.Tensor, n_repeat: int
) -> np.ndarray:
    loss = model.loss_fn(logits, inp.repeat(n_repeat, 1), per_token=True).cpu().numpy()
    return np.concatenate((loss, np.zeros((loss.shape[0], 1))), axis=1)


def mean_ablate_neurons(
    model: LlamaWrapper,
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
        entropy_df[[f"{n}_activation" for n in neuron_names]].mean().values,
        dtype=torch.float32,
    )
    batch_ids = np.random.choice(entropy_df["batch"].unique(), cfg.k, replace=False)
    filtered = entropy_df[entropy_df["batch"].isin(batch_ids)].copy()
    frames: list[pd.DataFrame] = []
    w_out = model.W_out[layer_idx].float()

    for batch_n in tqdm.tqdm(batch_ids, desc="ablation batches"):
        tok_seq = tokenized_data["tokens"][batch_n]
        inp = tok_seq.unsqueeze(0).to(device)
        _, cache = model.run_with_cache(inp)
        rows = filtered[filtered["batch"] == batch_n]
        ln_scales = torch.tensor(rows["ln_final_scale"].values, device=device, dtype=torch.float32)
        res_stream = cache[tl_utils.get_act_name("resid_post", layer_idx)][0]
        prev_act = cache[tl_utils.get_act_name("post", layer_idx)][0, :, neuron_indices].float()
        del cache
        act_deltas = activation_means.to(prev_act.device) - prev_act

        te_loss, te_entropy, te_top = [], [], []
        de_loss, de_entropy, de_top = [], [], []

        for start in range(0, len(neuron_indices), chunk_size):
            idx_chunk = neuron_indices[start : start + chunk_size]
            delta_chunk = act_deltas[:, start : start + chunk_size]
            res_delta = delta_chunk.unsqueeze(-1) * w_out[idx_chunk, :]
            res_delta = res_delta.permute(1, 0, 2)
            n_chunk = res_delta.shape[0]
            updated = res_stream.repeat(n_chunk, 1, 1) + res_delta

            te_res = model.ln_final(updated)
            te_logits = te_res.float() @ model.W_U + model.b_U
            te_loss.append(_per_token_loss(model, te_logits, inp, n_chunk))
            te_entropy.append(get_entropy(te_logits).cpu().numpy())
            te_top.append(te_logits.argmax(dim=-1).cpu().numpy())
            del te_res, te_logits

            de_res = model.apply_ln_final_frozen(updated, ln_scales)
            de_logits = de_res.float() @ model.W_U + model.b_U
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


def run_phase1_identify(
    cfg: ModelRunConfig,
    out_dir: pathlib.Path,
    *,
    device: str,
    load_model_fn=None,
    paper_neurons: list[str] | None = None,
) -> pd.DataFrame:
    return _gpt2_run_phase1_identify(
        cfg,
        out_dir,
        device=device,
        load_model_fn=load_model_fn,
        paper_neurons=paper_neurons,
    )


def run_phase2_ablation(
    cfg: ModelRunConfig,
    phase1_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    device: str,
    load_model_fn=None,
    n_random: int = 30,
    seed: int = 42,
    paper_neurons: list[str] | None = None,
    chunk_size: int = 20,
) -> pd.DataFrame:
    import time

    import neel.utils as nutils

    out_dir.mkdir(parents=True, exist_ok=True)
    entropy_labels, static_df = load_phase1_outputs(phase1_dir)
    neuron_names = select_phase2_neurons(static_df, entropy_labels, n_random=n_random, seed=seed)

    print(f"\n{'='*60}\nPhase 2 — {cfg.name} on {device}\n{'='*60}")
    print(f"  entropy neurons: {len(entropy_labels)}  random: {len(neuron_names) - len(entropy_labels)}")
    print(f"  k={cfg.k} batches  data_end={cfg.data_end} sequences")

    t0 = time.perf_counter()
    _load = load_model_fn
    if _load is None:
        raise ValueError("load_model_fn is required for Llama phase 2")
    model, tokenizer = _load(cfg, device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenized = load_tokenized_data(tokenizer, cfg, seed=seed)
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
        chunk_size=chunk_size,
    )
    ablation_df = filter_entropy_activation_df(
        raw, model_name=cfg.name, tokenizer=tokenizer, start_pos=3, end_pos=-1
    )

    feather_path = out_dir / "ablation.feather"
    ablation_df.reset_index(drop=True).to_feather(feather_path)
    print(f"  saved {feather_path}")

    print("  Fig 2c: TE vs DE_LN")
    plot_fig2c_te_de_box(ablation_df, out_dir, entropy_labels, cfg.name)

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


__all__ = [
    "LLAMA_NULLSPACE_K",
    "filter_entropy_activation_df",
    "get_entropy_activation_df",
    "mean_ablate_neurons",
    "run_phase1_identify",
    "run_phase2_ablation",
    "compute_static_neuron_df",
    "neuron_label",
    "top1pct_rho_neurons",
    "annotate_top1pct_rho",
    "normalize_component_names",
    "load_phase1_outputs",
    "select_phase2_neurons",
    "write_phase1_summary",
    "write_phase2_summary",
    "plot_b1_norm_vs_logitvar",
    "plot_b2_rho",
    "plot_b2_rho_vs_ln_fraction",
    "plot_fig1_entropy_vs_argmax",
    "plot_fig2c_te_de_box",
    "plot_fig2d_singular_values",
]
