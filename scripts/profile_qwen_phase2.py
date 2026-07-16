"""Quick timing breakdown for Qwen phase-2 hot paths (smoke / profile)."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyperparameters import QWEN_08B_HPARAMS, ModelRunConfig
from gpt2.core import load_tokenized_data
from qwen.load import load_qwen
from qwen.core import get_entropy, _per_token_metrics


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelRunConfig.from_hparams(
        "Qwen/Qwen3.5-0.8B-Base",
        QWEN_08B_HPARAMS,
        data_end=8,
    )
    chunk_size = cfg.chunk_size
    n_neurons = 100
    n_chunks = math.ceil(n_neurons / chunk_size)

    print("Loading model...")
    model, tokenizer = load_qwen(cfg, device)
    c = model.cfg
    print(
        f"  n_layers={c.n_layers} d_model={c.d_model} d_mlp={c.d_mlp} "
        f"d_vocab={c.d_vocab} W_U device={model.W_U.device} dtype={model.W_U.dtype}"
    )

    tokenized = load_tokenized_data(tokenizer, cfg)
    n_seq = len(tokenized)
    print(f"  tokenized rows: {n_seq} (data_end={cfg.data_end})")

    batch = tokenized["tokens"][0:1].to(device=device, dtype=torch.long)
    seq_len = batch.shape[1]

    _sync()
    t0 = time.perf_counter()
    logits, cache = model.run_with_cache(batch)
    _sync()
    t_fwd = time.perf_counter() - t0
    print(f"\nrun_with_cache (1 seq): {t_fwd:.3f}s  logits {tuple(logits.shape)}")

    _sync()
    t0 = time.perf_counter()
    logits_ranking = logits.argsort(dim=-1, descending=True).cpu()
    _ = (
        (logits_ranking[:, :-1] == batch[:, 1:].unsqueeze(-1).cpu())
        .nonzero(as_tuple=True)[2]
        .view(batch[:, :-1].shape)
    )
    _sync()
    t_argsort = time.perf_counter() - t0
    print(f"argsort rank path: {t_argsort:.3f}s")

    _sync()
    t0 = time.perf_counter()
    correct_logits = logits[:, :-1].gather(-1, batch[:, 1:].unsqueeze(-1))
    _ = (logits[:, :-1] > correct_logits).sum(dim=-1)
    _sync()
    t_rank_cmp = time.perf_counter() - t0
    print(f"comparison rank path: {t_rank_cmp:.3f}s")

    from transformer_lens import utils as tl_utils

    layer_idx = cfg.last_layer(model)
    w_out = model.W_out[layer_idx].float()
    w_u = model.W_U
    b_u = model.b_U
    res_stream = cache[tl_utils.get_act_name("resid_post", layer_idx)][0]
    neuron_indices = list(range(n_neurons))
    prev_act = cache[tl_utils.get_act_name("post", layer_idx)][0, :, neuron_indices].float()
    act_deltas = prev_act.mean(dim=0, keepdim=True).expand_as(prev_act) - prev_act
    n_chunk = chunk_size
    delta_chunk = act_deltas[:, :n_chunk]
    res_delta = delta_chunk.unsqueeze(-1) * w_out[neuron_indices[:n_chunk], :]
    res_delta = res_delta.permute(1, 0, 2)
    updated = res_stream.repeat(n_chunk, 1, 1) + res_delta

    _sync()
    t0 = time.perf_counter()
    te_res = model.ln_final(updated)
    te_logits = te_res.float() @ w_u + b_u
    _sync()
    t_matmul = time.perf_counter() - t0

    _sync()
    t0 = time.perf_counter()
    _ = get_entropy(te_logits)
    _sync()
    t_entropy_old = time.perf_counter() - t0

    _sync()
    t0 = time.perf_counter()
    _ = model.loss_fn(te_logits, batch.repeat(n_chunk, 1), per_token=True)
    _sync()
    t_loss_old = time.perf_counter() - t0

    _sync()
    t0 = time.perf_counter()
    _, _, _ = _per_token_metrics(te_logits, batch.repeat(n_chunk, 1))
    _sync()
    t_fused = time.perf_counter() - t0

    chunk_inner_old = t_matmul + t_entropy_old + t_loss_old
    chunk_inner_new = t_matmul + t_fused
    print(
        f"\nOne ablation chunk (n_chunk={n_chunk}, seq={seq_len}): "
        f"ln+matmul={t_matmul:.3f}s"
    )
    print(
        f"  OLD entropy+loss separate: {t_entropy_old:.3f}s + {t_loss_old:.3f}s "
        f"= {t_entropy_old + t_loss_old:.3f}s"
    )
    print(f"  NEW fused metrics: {t_fused:.3f}s")
    print(
        f"  te_logits shape {tuple(te_logits.shape)} "
        f"elements={te_logits.numel():,} (~{te_logits.numel() * 4 / 1e9:.2f} GB fp32)"
    )

    est_batch_old = t_fwd + n_chunks * 2 * chunk_inner_old
    est_batch_new = t_fwd + n_chunks * 2 * chunk_inner_new
    print(f"\nEstimated 1 ablation batch OLD: {est_batch_old:.1f}s ({est_batch_old/60:.1f} min)")
    print(f"Estimated 1 ablation batch NEW: {est_batch_new:.1f}s ({est_batch_new/60:.1f} min)")
    print(f"Estimated 4 ablation batches NEW: {est_batch_new * 4:.1f}s ({est_batch_new * 4 / 60:.1f} min)")
    print(f"Estimated activation collection 719 forwards: {t_fwd * 719:.1f}s ({t_fwd * 719 / 60:.1f} min)")


if __name__ == "__main__":
    main()
