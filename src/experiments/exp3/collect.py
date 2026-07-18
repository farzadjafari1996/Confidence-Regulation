"""Step 1: collect full last-MLP activations + label fields on BIG-bench qa_wikidata."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch
import tqdm
from transformer_lens import utils as tl_utils

from exp1.ablation_engine import encode_prompt, forward_cache
from exp1.data_loaders import _gold_token_id
from exp1.prompts import format_user_prompt
from exp1.settings import ModelSpec
from exp3.data_loaders import get_qa_wikidata_split
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
    if cfg.dataset != "qa_wikidata":
        raise ValueError(f"Unsupported Exp3 dataset: {cfg.dataset}")
    return get_qa_wikidata_split(cfg, split_name, model_slug(cfg.model), tokenizer)


def _build_prompt(item: dict[str, Any], model_spec: ModelSpec, tokenizer) -> str:
    text = str(item.get("inputs") or item.get("masked_sentence") or "").rstrip()
    if not text:
        raise ValueError(f"Item {item.get('item_id')} has empty cloze prompt")
    return format_user_prompt(text, model_spec, tokenizer)


def _gold_token_ids(item: dict[str, Any]) -> list[int]:
    if item.get("gold_token_ids"):
        return [int(x) for x in item["gold_token_ids"]]
    return [int(item["gold_token_id"])]


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
    prompt = _build_prompt(item, model_spec, tokenizer)
    input_ids = encode_prompt(tokenizer, prompt, device)
    logits_raw, cache = forward_cache(model, input_ids)
    pos = input_ids.shape[1] - 1
    layer = model.cfg.n_layers - 1
    act = cache[tl_utils.get_act_name("post", layer)][0, pos].float().cpu().numpy()
    logits = logits_raw[0, pos].float()
    probs = torch.softmax(logits, dim=-1)
    gold_ids = _gold_token_ids(item)
    gold_id = int(gold_ids[0])
    pred_id = int(torch.argmax(logits).item())
    sorted_ids = torch.argsort(logits, descending=True)
    gold_rank = -1
    for gid in gold_ids:
        ranks = (sorted_ids == gid).nonzero(as_tuple=True)[0]
        if len(ranks):
            r = int(ranks[0].item())
            gold_rank = r if gold_rank < 0 else min(gold_rank, r)
    gold_logprob = float(max(torch.log(probs[gid] + 1e-12).item() for gid in gold_ids))
    entropy = float(get_entropy(logits.unsqueeze(0).unsqueeze(0)).item())
    top1_prob = float(probs.max().item())
    del cache
    return {
        "X_full": act.astype(np.float32),
        "y_top1": int(pred_id in gold_ids),
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
        for p in (final_path, part_path, meta_path(data_dir), data_dir / "split_indices.json"):
            if p.exists():
                p.unlink()

    run_cfg = ModelRunConfig.from_hparams(cfg.model, _hparams_for_family(cfg), dtype=cfg.dtype)
    model, tokenizer = _load_model(cfg, run_cfg, device)
    model_spec = _model_spec(cfg)
    items = _items_for_split(cfg, split_name, tokenizer)
    print(
        f"Collecting Ext3 {split_name}: {len(items)} {cfg.dataset} items "
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
        if "gold_token_id" not in item:
            tid = _gold_token_id(tokenizer, item["gold_answer"])
            if tid is None:
                continue
            item = dict(item)
            item["gold_token_id"] = tid
            item["gold_token_ids"] = [tid]
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
        raise RuntimeError(f"No items collected — check {cfg.dataset} filter / tokenizer")

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
            "dataset": cfg.dataset,
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
