"""Induction datasets and hook-based ablations for paper Fig 5."""

from __future__ import annotations

import random
import re
from functools import partial

import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import Dataset, load_dataset
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils

from gpt2.core import get_entropy, neuron_str_to_neuron


def build_c4_dup_induction_dataset(
    tokenized_data,
    *,
    num_sequences: int,
    seq_length: int,
    seed: int,
) -> Dataset:
    """Duplicate first seq_length tokens (paper §6 synthetic induction)."""
    rng = random.Random(seed)
    tokens_list: list[np.ndarray] = []
    n_src = len(tokenized_data["tokens"])
    for _ in range(num_sequences):
        index = rng.randint(0, n_src - 1)
        sequence = tokenized_data["tokens"][index]
        first = sequence[:seq_length]
        second = sequence[1:seq_length]
        tokens_list.append(torch.cat((first, second), dim=0).cpu().numpy())
    ds = Dataset.from_dict({"tokens": tokens_list})
    ds.set_format(type="torch", columns=["tokens"])
    return ds


def generate_induction_examples(
    model: HookedTransformer,
    tokenizer,
    *,
    seq_length: int,
    num_examples: int,
    seed: int,
    device: str,
    use_natural_text: bool = True,
) -> torch.Tensor:
    """Build duplicated sequences (BOS + prefix + repeat prefix)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if use_natural_text:
        data = load_dataset("stas/c4-en-10k", split="train")
        subset = data.select(range(4000, min(8000, len(data))))

        def tok_fn(examples):
            return {
                "tokens": tokenizer(
                    examples["text"],
                    add_special_tokens=False,
                    max_length=256,
                    truncation=True,
                ).input_ids
            }

        tok_ds = subset.map(tok_fn, batched=True, remove_columns=["text"])
        filtered = tok_ds.filter(lambda ex: len(ex["tokens"]) >= 256)
        filtered = filtered.shuffle(seed=seed)
        filtered.set_format(type="torch", columns=["tokens"])
        sequences = filtered["tokens"][:num_examples, :seq_length].to(device)
    else:
        sequences = torch.randint(
            0, model.cfg.d_vocab, size=(num_examples, seq_length), generator=gen, device=device
        )

    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([tokenizer.bos_token_id] * num_examples, device=device).unsqueeze(1)
        tokens = torch.cat([bos, sequences, sequences], dim=-1)
    else:
        tokens = torch.cat([sequences, sequences], dim=-1)
    return tokens


def get_induction_data_and_token_df(
    model: HookedTransformer,
    tokenizer,
    token_df_fn,
    *,
    seq_length: int,
    num_examples: int,
    seed: int,
    device: str,
    use_natural_text: bool = True,
) -> tuple[Dataset, pd.DataFrame]:
    tokens = generate_induction_examples(
        model,
        tokenizer,
        seq_length=seq_length,
        num_examples=num_examples,
        seed=seed,
        device=device,
        use_natural_text=use_natural_text,
    )
    ds = Dataset.from_dict({"tokens": [row.cpu().numpy() for row in tokens]})
    ds.set_format(type="torch", columns=["tokens"])
    return ds, token_df_fn(ds["tokens"])


def compute_weighted_activation_diffs(
    entropy_df: pd.DataFrame,
    model: HookedTransformer,
    neuron_names: list[str],
    *,
    half_pos: int | None = None,
) -> dict[str, float]:
    """Induction minus first-half mean activation, weighted by ||w_out||."""
    if half_pos is None:
        half_pos = int(entropy_df["pos"].max() // 2)
    diffs: dict[str, float] = {}
    for name in neuron_names:
        layer, idx = neuron_str_to_neuron(name)
        col = f"{name}_activation"
        mean_first = entropy_df.loc[entropy_df["pos"] < half_pos, col].mean()
        mean_second = entropy_df.loc[entropy_df["pos"] >= half_pos, col].mean()
        diff = mean_second - mean_first
        diffs[name] = diff * model.W_out[layer, idx].norm().item()
    return diffs


def _strict_clip_hook(value, hook, neuron_indices, ablation_values, weighted_diffs, layer: int):
    for neuron_idx, ablation_value in zip(neuron_indices, ablation_values):
        name = f"{layer}.{neuron_idx}"
        if weighted_diffs.get(name, 0.0) < 0:
            value[0, :, neuron_idx] = torch.max(ablation_value, value[0, :, neuron_idx])
        else:
            value[0, :, neuron_idx] = torch.min(ablation_value, value[0, :, neuron_idx])
    return value


def clip_ablate_neurons(
    neuron_names: list[str],
    tokenized_data,
    entropy_df: pd.DataFrame,
    model: HookedTransformer,
    weighted_diffs: dict[str, float],
    *,
    k: int,
    device: str,
    seed: int,
) -> pd.DataFrame:
    """Clip/mean ablation on induction sequences (paper Fig 5b)."""
    layer = int(neuron_names[0].split(".")[0])
    half_pos = int(entropy_df["pos"].max() // 2)
    first_half = entropy_df[entropy_df["pos"] < half_pos]
    batch_ids = np.random.default_rng(seed).choice(
        entropy_df["batch"].unique(), size=min(k, entropy_df["batch"].nunique()), replace=False
    )
    frames: list[pd.DataFrame] = []

    for name in tqdm.tqdm(neuron_names, desc="clip ablate"):
        _, neuron_idx = neuron_str_to_neuron(name)
        ablation_value = torch.tensor(
            first_half[[f"{name}_activation"]].mean().values, device=device
        )
        filtered = entropy_df[entropy_df["batch"].isin(batch_ids)].copy()

        for batch_n in filtered["batch"].unique():
            tok_seq = tokenized_data["tokens"][batch_n].to(device)
            inp = tok_seq.unsqueeze(0)
            rows = filtered[filtered["batch"] == batch_n]
            hook = (
                tl_utils.get_act_name("post", layer),
                partial(
                    _strict_clip_hook,
                    neuron_indices=[neuron_idx],
                    ablation_values=ablation_value,
                    weighted_diffs=weighted_diffs,
                    layer=layer,
                ),
            )
            model.reset_hooks()
            with model.hooks(fwd_hooks=[hook]):
                logits, _ = model(inp, return_type="both", loss_per_token=True)
            loss = model.loss_fn(logits, inp, per_token=True).cpu().numpy()[0]
            ent = get_entropy(logits).cpu().numpy()[0]
            n = len(rows)

            def _align(arr: np.ndarray, size: int) -> np.ndarray:
                out = arr[:size]
                if len(out) < size:
                    out = np.pad(out, (0, size - len(out)))
                return out

            chunk = rows.copy()
            chunk["component_name"] = name
            chunk["loss_post_ablation"] = _align(loss, n)
            chunk["entropy_post_ablation"] = _align(ent, n)
            frames.append(chunk)

    return pd.concat(frames, ignore_index=True)


def _parse_attn_head(name: str) -> tuple[int, int]:
    match = re.search(r"L(\d+)H(\d+)", name)
    if not match:
        raise ValueError(f"Invalid attention head label: {name}")
    return int(match.group(1)), int(match.group(2))


def _bos_ablation_hook(value, hook, position: int, head: int):
    value[..., head, position, :] = 0.0
    value[..., head, position, 0] = 1.0
    return value


def bos_ablate_attn_heads(
    attn_head_names: list[str],
    tokenized_data,
    entropy_df: pd.DataFrame,
    model: HookedTransformer,
    neuron_names: list[str],
    *,
    device: str,
) -> pd.DataFrame:
    """BOS attention ablation per token (paper Fig 5c)."""
    rows_out: list[dict] = []
    for _, row in tqdm.tqdm(entropy_df.iterrows(), total=len(entropy_df), desc="BOS ablate"):
        inp = tokenized_data["tokens"][row.batch].to(device)
        position = int(row.pos)
        hooks = [
            (
                tl_utils.get_act_name("pattern", layer),
                partial(_bos_ablation_hook, position=position, head=head),
            )
            for layer, head in (_parse_attn_head(h) for h in attn_head_names)
        ]
        model.reset_hooks()
        with model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(inp.unsqueeze(0))
        record = {
            "batch": row.batch,
            "pos": row.pos,
            "component_name": "-".join(attn_head_names),
        }
        for name in neuron_names:
            layer, idx = neuron_str_to_neuron(name)
            act = cache[tl_utils.get_act_name("post", layer)][0, position, idx].item()
            record[f"{name}_activation_post_abl"] = act
            record[f"{name}_activation"] = row.get(f"{name}_activation")
        rows_out.append(record)
    return pd.DataFrame(rows_out)


def bos_ablate_components(
    component_lists: list[list[str]],
    tokenized_data,
    entropy_df: pd.DataFrame,
    model: HookedTransformer,
    neuron_names: list[str],
    *,
    device: str,
) -> pd.DataFrame:
    frames = [
        bos_ablate_attn_heads(
            heads, tokenized_data, entropy_df, model, neuron_names, device=device
        )
        for heads in component_lists
    ]
    return pd.concat(frames, ignore_index=True)
