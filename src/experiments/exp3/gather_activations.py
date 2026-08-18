"""Step 2: feed-forward and store last-token residuals + last-MLP posts + labels.

Usage (from repo root, venv active). Omitting --dataset runs both cloze sets:
  python src/experiments/exp3/gather_activations.py --config config/llama_instruct.yaml --split main --device cuda
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import numpy as np
import torch
import tqdm
from transformer_lens import utils as tl_utils

_src = pathlib.Path(__file__).resolve().parent
while _src != _src.parent and not (
    (_src / "common").is_dir() and (_src / "experiments").is_dir() and (_src / "models").is_dir()
):
    _src = _src.parent
for _p in (_src, _src / "common", _src / "experiments", _src / "models"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from exp3.cli import add_step_args, iter_cfgs, resolve_device  # noqa: E402
from exp3.data_loaders import load_cached_split  # noqa: E402
from exp3.dataset_io import (  # noqa: E402
    CollectionWriter,
    collection_ready,
    ensure_splits,
    labels_path,
    load_arrays,
    meta_path,
    mlp_path,
    resid_path,
    write_collection_meta,
)
from exp3.model_utils import (  # noqa: E402
    ModelSpec,
    encode_prompt,
    first_param,
    format_user_prompt,
    forward_cache,
    gold_token_id,
    load_exp3_model,
    model_spec_from_cfg,
    token_id_candidates,
)
from exp3.settings import Exp3Config, SplitName  # noqa: E402
from gpt2.core import get_entropy  # noqa: E402
from hyperparameters import model_slug  # noqa: E402

LETTERS = "ABCDEFGHIJ"


def _build_prompt(item: dict[str, Any], model_spec: ModelSpec, tokenizer) -> str:
    text = str(item.get("inputs") or item.get("masked_sentence") or "").rstrip()
    if not text:
        raise ValueError(f"Item {item.get('item_id')} has empty prompt")
    return format_user_prompt(text, model_spec, tokenizer)


def _gold_token_ids(item: dict[str, Any]) -> list[int]:
    if item.get("gold_token_ids"):
        return [int(x) for x in item["gold_token_ids"] if int(x) >= 0]
    if item.get("gold_token_id") is not None and int(item["gold_token_id"]) >= 0:
        return [int(item["gold_token_id"])]
    return []


def _letter_token_ids(tokenizer) -> list[list[int]]:
    cached = getattr(tokenizer, "_exp3_letter_ids", None)
    if cached is None:
        cached = [token_id_candidates(tokenizer, LETTERS[i]) for i in range(10)]
        tokenizer._exp3_letter_ids = cached
    return cached


def _option_logprobs(probs: torch.Tensor, tokenizer, n_options: int) -> np.ndarray:
    out = np.full(10, np.nan, dtype=np.float64)
    letter_ids = _letter_token_ids(tokenizer)
    log_probs = torch.log(probs + 1e-12)
    for i in range(min(n_options, 10)):
        ids = letter_ids[i]
        if not ids:
            continue
        out[i] = float(log_probs[ids].max().item())
    return out


def _model_n_ctx(model) -> int | None:
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return None
    n_ctx = int(getattr(cfg, "n_ctx", 0) or getattr(cfg, "n_positions", 0) or 0)
    return n_ctx if n_ctx > 0 else None


def collect_item(
    model,
    tokenizer,
    item: dict[str, Any],
    *,
    device: str,
    model_spec: ModelSpec,
) -> dict[str, Any]:
    prompt = _build_prompt(item, model_spec, tokenizer)
    n_ctx = _model_n_ctx(model)
    input_ids, n_orig = encode_prompt(tokenizer, prompt, device, max_length=n_ctx)
    if n_ctx is not None and n_orig > n_ctx:
        item_id = str(item.get("item_id") or item.get("uuid") or "")
        print(
            f"  truncate {item_id or 'item'} {n_orig} -> {n_ctx} tokens (model n_ctx)",
            flush=True,
        )
    logits_raw, cache = forward_cache(model, input_ids)
    pos = input_ids.shape[1] - 1
    n_layers = int(model.cfg.n_layers)
    last = n_layers - 1
    resid_t = torch.stack(
        [cache[tl_utils.get_act_name("resid_post", layer)][0, pos] for layer in range(n_layers)]
    )
    resid = resid_t.to(dtype=torch.float16).cpu().numpy()
    mlp_last = (
        cache[tl_utils.get_act_name("post", last)][0, pos].to(dtype=torch.float16).cpu().numpy()
    )
    logits = logits_raw[0, pos].float()
    probs = torch.softmax(logits, dim=-1)
    gold_ids = _gold_token_ids(item)
    gold_id = int(gold_ids[0]) if gold_ids else -1
    pred_id = int(torch.argmax(logits).item())
    sorted_ids = torch.argsort(logits, descending=True)
    gold_rank = -1
    for gid in gold_ids:
        ranks = (sorted_ids == gid).nonzero(as_tuple=True)[0]
        if len(ranks):
            r = int(ranks[0].item())
            gold_rank = r if gold_rank < 0 else min(gold_rank, r)
    if gold_ids:
        gold_logprob = float(max(torch.log(probs[gid] + 1e-12).item() for gid in gold_ids))
    else:
        gold_logprob = float("nan")
    entropy = float(get_entropy(logits.unsqueeze(0).unsqueeze(0)).item())
    top1_prob = float(probs.max().item())
    n_options = int(item.get("n_options") or 0)
    option_logprobs = (
        _option_logprobs(probs, tokenizer, n_options) if n_options > 0 else np.full(10, np.nan)
    )
    option_pred = -1
    y_option = -1
    if n_options > 0 and np.any(np.isfinite(option_logprobs[:n_options])):
        option_pred = int(np.nanargmax(option_logprobs[:n_options]))
        gold_index = int(item.get("gold_index", -1))
        y_option = int(option_pred == gold_index) if gold_index >= 0 else -1
    del cache
    return {
        "X_resid": resid.astype(np.float16),
        "X_mlp_last": mlp_last.astype(np.float16),
        "y_top1": int(bool(gold_ids) and pred_id in gold_ids),
        "y_top5": int(0 <= gold_rank < 5),
        "y_option": y_option,
        "gold_id": gold_id,
        "pred_id": pred_id,
        "gold_rank": gold_rank,
        "n_options": n_options,
        "option_pred": option_pred,
        "option_logprobs": option_logprobs,
        "top1_prob": top1_prob,
        "entropy": entropy,
        "gold_logprob": gold_logprob,
        "item_id": str(item.get("item_id") or item.get("uuid") or ""),
        "predicate_id": str(item.get("predicate_id") or ""),
        "predicate_label": str(item.get("predicate_label") or ""),
        "gold_answer": str(item.get("gold_answer") or ""),
        "pred_token": tokenizer.decode([pred_id]),
        "category": str(item.get("category") or ""),
    }


def _clear_collection(data_dir) -> None:
    for p in (
        labels_path(data_dir),
        resid_path(data_dir),
        mlp_path(data_dir),
        meta_path(data_dir),
        data_dir / "split_indices.json",
    ):
        if p.exists():
            p.unlink()


def run_gather(
    cfg: Exp3Config,
    split_name: SplitName,
    *,
    device: str,
    resume: bool = True,
    force: bool = False,
    model=None,
    tokenizer=None,
):
    data_dir = cfg.data_out_dir(split_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    if collection_ready(data_dir) and not force:
        print(f"Dataset already exists: {data_dir} (use --force to regenerate)")
        data = load_arrays(data_dir)
        ensure_splits(cfg, split_name, data)
        return labels_path(data_dir)

    if force:
        _clear_collection(data_dir)

    if device == "cuda" and not cfg.dtype:
        cfg.dtype = "float16"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    own_model = model is None
    if own_model:
        model, tokenizer = load_exp3_model(cfg, device)
        param = first_param(model)
        print(
            f"  model dtype={param.dtype} device={param.device} "
            f"(requested={cfg.dtype or 'auto'})"
        )
        if device == "cuda" and param.dtype not in (torch.float16, torch.bfloat16):
            print(f"  warning: expected fp16/bf16 on CUDA, got {param.dtype}")

    model_spec = model_spec_from_cfg(cfg)
    items = load_cached_split(cfg, split_name, model_slug(cfg.model))
    print(
        f"Gathering Exp3 {split_name}/{cfg.dataset}: {len(items)} items "
        f"on {device} ({cfg.family}/{cfg.model}) chat={model_spec.use_chat_template}"
    )

    writer = CollectionWriter(
        data_dir,
        capacity=len(items),
        n_layers=int(model.cfg.n_layers),
        d_model=int(model.cfg.d_model),
        d_mlp=int(model.cfg.d_mlp),
        resume=resume and not force,
    )
    skip_ids = writer.item_ids() if resume and not force else set()

    for i, item in enumerate(tqdm.tqdm(items, desc=f"exp3-gather-{cfg.dataset}-{split_name}")):
        item_id = str(item.get("item_id") or item.get("uuid") or f"row_{i}")
        if item_id in skip_ids:
            continue
        if not item.get("gold_token_ids"):
            tid = gold_token_id(tokenizer, item["gold_answer"])
            if tid is None:
                continue
            item = dict(item)
            item["gold_token_id"] = tid
            item["gold_token_ids"] = [tid]
            item["item_id"] = item_id

        try:
            row = collect_item(model, tokenizer, item, device=device, model_spec=model_spec)
        except Exception as exc:
            print(f"  skip {item_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not row["item_id"]:
            row["item_id"] = item_id
        writer.append(row)
        skip_ids.add(item_id)
        if writer.n_written % max(cfg.checkpoint_every, 1) == 0:
            writer.flush()

    if writer.n_written == 0:
        writer.close()
        raise RuntimeError(f"No items collected — check {cfg.dataset} filter / tokenizer")

    n = writer.n_written
    n_layers = writer.n_layers
    d_model = writer.d_model
    d_mlp = writer.d_mlp
    writer.close()

    write_collection_meta(
        meta_path(data_dir),
        {
            "model": cfg.model,
            "family": cfg.family,
            "dataset": cfg.dataset,
            "use_chat_template": model_spec.use_chat_template,
            "split": split_name,
            "n_items": n,
            "n_layers": n_layers,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "layer": n_layers - 1,
            "seed": cfg.seed,
            "complete": True,
            "capacity": len(items),
        },
    )
    data = load_arrays(data_dir)
    y_top1 = np.asarray(data["y_top1"])
    y_option = np.asarray(data.get("y_option", np.full(n, -1)))
    extra = {
        "top1_accuracy": float(np.mean(y_top1)),
        "top5_accuracy": float(np.mean(data["y_top5"])),
    }
    if np.any(y_option >= 0):
        extra["option_accuracy"] = float(np.mean(y_option[y_option >= 0]))
    write_collection_meta(
        meta_path(data_dir),
        {
            **extra,
            "model": cfg.model,
            "family": cfg.family,
            "dataset": cfg.dataset,
            "use_chat_template": model_spec.use_chat_template,
            "split": split_name,
            "n_items": n,
            "n_layers": n_layers,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "layer": n_layers - 1,
            "seed": cfg.seed,
            "complete": True,
            "capacity": len(items),
        },
    )
    ensure_splits(cfg, split_name, data)
    acc_bits = [f"top1_acc={extra['top1_accuracy']:.3f}"]
    if "option_accuracy" in extra:
        acc_bits.append(f"option_acc={extra['option_accuracy']:.3f}")
    print(
        f"Wrote {data_dir}  (N={n}, resid={n_layers}x{d_model}, d_mlp={d_mlp}, "
        + ", ".join(acc_bits)
        + ")"
    )
    if own_model:
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
    return labels_path(data_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 3 step 2: gather activations")
    add_step_args(p)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from exp3.dataset_io import collection_ready

    model = tokenizer = None
    device = None
    for dataset, cfg in iter_cfgs(args):
        if device is None:
            device = resolve_device(args.device or cfg.device)
        print(f"\n=== gather  {dataset}  {args.split}  n={cfg.sample_size(args.split)} ===")
        need = args.force or not collection_ready(cfg.data_out_dir(args.split))
        if need and model is None:
            if device == "cuda" and not cfg.dtype:
                cfg.dtype = "float16"
            model, tokenizer = load_exp3_model(cfg, device)
            param = first_param(model)
            print(
                f"  model dtype={param.dtype} device={param.device} "
                f"(requested={cfg.dtype or 'auto'})"
            )
        run_gather(
            cfg,
            args.split,
            device=device,
            resume=not args.no_resume,
            force=args.force,
            model=model,
            tokenizer=tokenizer,
        )
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
