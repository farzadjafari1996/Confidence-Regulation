"""Step 1: collect full last-MLP activations + label fields on LAMA T-REx."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch
import tqdm
from transformer_lens import utils as tl_utils

from exp1.ablation_engine import encode_prompt, forward_cache
from exp1.data_loaders import load_or_create_item_splits
from exp1.prompts import build_lama_prompt
from exp1.settings import Exp1Config, LamaDatasetSettings, ModelSpec
from exp3.dataset_io import (
    dataset_path,
    ensure_splits,
    load_arrays,
    meta_path,
    partial_path,
    save_arrays,
    write_collection_meta,
)
from exp3.settings import Exp3Config, SplitName
from gpt2.core import get_entropy
from hyperparameters import GPT2_HPARAMS, LLAMA_HPARAMS, ModelRunConfig, model_slug, qwen_hparams_for_model


def _model_spec(cfg: Exp3Config) -> ModelSpec:
    return ModelSpec(
        family=cfg.family,  # type: ignore[arg-type]
        model=cfg.model,
        use_chat_template=cfg.use_chat_template,
    )


def _hparams_for_family(cfg: Exp3Config):
    if cfg.family == "gpt2":
        return GPT2_HPARAMS
    if cfg.family == "llama":
        return LLAMA_HPARAMS
    if cfg.family == "qwen":
        return qwen_hparams_for_model(cfg.model)
    raise ValueError(f"Unsupported Exp3 family: {cfg.family}")


def _load_model(cfg: Exp3Config, run_cfg: ModelRunConfig, device: str):
    if cfg.family == "gpt2":
        from gpt2.load import load_gpt2

        return load_gpt2(run_cfg, device)
    if cfg.family == "llama":
        from llama.load import load_llama

        return load_llama(run_cfg, device)
    if cfg.family == "qwen":
        from qwen.load import load_qwen

        return load_qwen(run_cfg, device)
    raise ValueError(f"Unsupported Exp3 family: {cfg.family}")


def _items_for_split(cfg: Exp3Config, split_name: SplitName, tokenizer) -> list[dict[str, Any]]:
    """Reuse Ext1-style LAMA caching under data/exp3 with Ext3 sample sizes."""
    lama = LamaDatasetSettings(
        hf_path=cfg.lama.hf_path,
        config=cfg.lama.config,
        filter_single_token_gold=cfg.lama.filter_single_token_gold,
    )
    bridge = Exp1Config(
        out_dir=cfg.out_dir,
        data_dir=cfg.data_dir,
        seed=cfg.seed,
        models=[_model_spec(cfg)],
        sample_sizes=dict(cfg.sample_sizes),
        lama=lama,
    )
    slug = model_slug(cfg.model)
    splits = load_or_create_item_splits(
        bridge,
        "lama",
        slug,
        tokenizer,
        filter_single_token=cfg.lama.filter_single_token_gold,
    )
    return list(splits[split_name])


def _empty_buffers() -> tuple[dict[str, list], dict[str, list]]:
    arrays: dict[str, list] = {
        "X_full": [],
        "y_top1": [],
        "y_top5": [],
        "gold_id": [],
        "pred_id": [],
        "gold_rank": [],
        "top1_prob": [],
        "entropy": [],
        "gold_logprob": [],
    }
    lists: dict[str, list] = {
        "item_ids": [],
        "predicate_ids": [],
        "predicate_labels": [],
        "gold_answers": [],
        "pred_tokens": [],
    }
    return arrays, lists


def _load_buffers(path: pathlib.Path) -> tuple[dict[str, list], dict[str, list]]:
    data = load_arrays(path)
    arrays, lists = _empty_buffers()
    for key in arrays:
        arrays[key] = list(np.asarray(data[key]))
    # X_full stored as 2d — keep as list of 1d rows
    arrays["X_full"] = [np.asarray(row, dtype=np.float32) for row in data["X_full"]]
    for key in lists:
        lists[key] = [str(x) if x is not None else "" for x in data.get(key, [])]
    return arrays, lists


def _flush(path: pathlib.Path, arrays: dict[str, list], lists: dict[str, list]) -> None:
    packed = {
        "X_full": np.stack(arrays["X_full"], axis=0).astype(np.float32),
        "y_top1": np.asarray(arrays["y_top1"], dtype=np.int64),
        "y_top5": np.asarray(arrays["y_top5"], dtype=np.int64),
        "gold_id": np.asarray(arrays["gold_id"], dtype=np.int64),
        "pred_id": np.asarray(arrays["pred_id"], dtype=np.int64),
        "gold_rank": np.asarray(arrays["gold_rank"], dtype=np.int64),
        "top1_prob": np.asarray(arrays["top1_prob"], dtype=np.float64),
        "entropy": np.asarray(arrays["entropy"], dtype=np.float64),
        "gold_logprob": np.asarray(arrays["gold_logprob"], dtype=np.float64),
    }
    save_arrays(path, packed, lists)


def collect_item(
    model,
    tokenizer,
    item: dict[str, Any],
    *,
    device: str,
    model_spec: ModelSpec,
) -> dict[str, Any]:
    prompt = build_lama_prompt(item["masked_sentence"], model_spec, tokenizer)
    input_ids = encode_prompt(tokenizer, prompt, device)
    logits_raw, cache = forward_cache(model, input_ids)
    pos = input_ids.shape[1] - 1
    layer = model.cfg.n_layers - 1
    act = cache[tl_utils.get_act_name("post", layer)][0, pos].float().cpu().numpy()
    logits = logits_raw[0, pos].float()
    probs = torch.softmax(logits, dim=-1)
    gold_id = int(item["gold_token_id"])
    pred_id = int(torch.argmax(logits).item())
    sorted_ids = torch.argsort(logits, descending=True)
    ranks = (sorted_ids == gold_id).nonzero(as_tuple=True)[0]
    gold_rank = int(ranks[0].item()) if len(ranks) else -1
    gold_logprob = float(torch.log(probs[gold_id] + 1e-12).item())
    entropy = float(get_entropy(logits.unsqueeze(0).unsqueeze(0)).item())
    top1_prob = float(probs.max().item())
    del cache
    return {
        "X_full": act.astype(np.float32),
        "y_top1": int(pred_id == gold_id),
        "y_top5": int(0 <= gold_rank < 5),
        "gold_id": gold_id,
        "pred_id": pred_id,
        "gold_rank": gold_rank,
        "top1_prob": top1_prob,
        "entropy": entropy,
        "gold_logprob": gold_logprob,
        "item_id": str(item.get("item_id") or item.get("uuid") or ""),
        "predicate_id": str(item.get("predicate_id") or ""),
        "predicate_label": str(item.get("predicate_label") or ""),
        "gold_answer": str(item.get("gold_answer") or ""),
        "pred_token": tokenizer.decode([pred_id]),
    }


def run_collect(
    cfg: Exp3Config,
    split_name: SplitName,
    *,
    device: str,
    resume: bool = True,
    force: bool = False,
) -> pathlib.Path:
    data_dir = cfg.data_out_dir(split_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    final_path = dataset_path(data_dir)
    part_path = partial_path(data_dir)

    if final_path.exists() and not force:
        print(f"Dataset already exists: {final_path} (use --force to regenerate)")
        data = load_arrays(final_path)
        ensure_splits(cfg, split_name, data)
        return final_path

    if force:
        for p in (final_path, part_path, meta_path(data_dir)):
            if p.exists():
                p.unlink()

    run_cfg = ModelRunConfig.from_hparams(cfg.model, _hparams_for_family(cfg), dtype=cfg.dtype)
    model, tokenizer = _load_model(cfg, run_cfg, device)
    model_spec = _model_spec(cfg)
    items = _items_for_split(cfg, split_name, tokenizer)
    print(
        f"Collecting Ext3 {split_name}: {len(items)} LAMA items "
        f"on {device} ({cfg.family}/{cfg.model})"
    )

    skip_ids: set[str] = set()
    if resume and part_path.exists() and not force:
        arrays, lists = _load_buffers(part_path)
        skip_ids = {str(x) for x in lists["item_ids"]}
        print(f"  Resuming from partial ({len(skip_ids)} items)")
    else:
        arrays, lists = _empty_buffers()

    for i, item in enumerate(tqdm.tqdm(items, desc=f"ext3-collect-{split_name}")):
        item_id = str(item.get("item_id") or item.get("uuid") or f"row_{i}")
        if item_id in skip_ids:
            continue
        # Ensure gold_token_id present
        if "gold_token_id" not in item:
            ids = tokenizer.encode(item["gold_answer"], add_special_tokens=False)
            if len(ids) != 1:
                continue
            item = dict(item)
            item["gold_token_id"] = ids[0]
            item["item_id"] = item_id

        row = collect_item(
            model, tokenizer, item, device=device, model_spec=model_spec
        )
        arrays["X_full"].append(row["X_full"])
        arrays["y_top1"].append(row["y_top1"])
        arrays["y_top5"].append(row["y_top5"])
        arrays["gold_id"].append(row["gold_id"])
        arrays["pred_id"].append(row["pred_id"])
        arrays["gold_rank"].append(row["gold_rank"])
        arrays["top1_prob"].append(row["top1_prob"])
        arrays["entropy"].append(row["entropy"])
        arrays["gold_logprob"].append(row["gold_logprob"])
        lists["item_ids"].append(row["item_id"] or item_id)
        lists["predicate_ids"].append(row["predicate_id"])
        lists["predicate_labels"].append(row.get("predicate_label") or "")
        lists["gold_answers"].append(row["gold_answer"])
        lists["pred_tokens"].append(row["pred_token"])
        skip_ids.add(item_id)

        if len(arrays["X_full"]) % max(cfg.checkpoint_every, 1) == 0:
            _flush(part_path, arrays, lists)

    if not arrays["X_full"]:
        raise RuntimeError("No items collected — check LAMA filter / tokenizer")

    _flush(final_path, arrays, lists)
    if part_path.exists():
        part_path.unlink()

    data = load_arrays(final_path)
    n = int(data["X_full"].shape[0])
    d_mlp = int(data["X_full"].shape[1])
    write_collection_meta(
        meta_path(data_dir),
        {
            "model": cfg.model,
            "family": cfg.family,
            "use_chat_template": cfg.use_chat_template,
            "split": split_name,
            "n_items": n,
            "d_mlp": d_mlp,
            "layer": int(model.cfg.n_layers - 1),
            "seed": cfg.seed,
            "top1_accuracy": float(np.mean(data["y_top1"])),
            "top5_accuracy": float(np.mean(data["y_top5"])),
        },
    )
    ensure_splits(cfg, split_name, data)
    print(f"Wrote {final_path}  (N={n}, d_mlp={d_mlp}, top1_acc={np.mean(data['y_top1']):.3f})")
    return final_path
