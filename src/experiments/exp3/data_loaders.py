"""Load and sample BIG-bench qa_wikidata items for Exp 3."""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
from typing import Any

from datasets import load_dataset

from exp3.model_utils import gold_token_id
from exp3.settings import BigBenchSettings, Exp3Config, SplitName


def _stable_item_id(inputs: str, idx: int | None = None) -> str:
    if idx is not None:
        return f"qa_wikidata-{idx}"
    digest = hashlib.sha1(inputs.encode("utf-8")).hexdigest()[:16]
    return f"qa_wikidata-{digest}"


def load_qa_wikidata_pool(cfg: BigBenchSettings) -> list[dict[str, Any]]:
    print(f"  Loading BIG-bench pool ({cfg.hf_path}, config={cfg.config})...")
    ds = load_dataset(cfg.hf_path, cfg.config)
    items: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    for split_name in ("train", "validation"):
        if split_name not in ds:
            continue
        for row in ds[split_name]:
            inputs = (row.get("inputs") or "").strip()
            targets = [str(t).strip() for t in (row.get("targets") or []) if str(t).strip()]
            if not inputs or not targets:
                continue
            if inputs in seen_inputs:
                continue
            seen_inputs.add(inputs)
            idx = row.get("idx")
            items.append(
                {
                    "item_id": _stable_item_id(inputs, int(idx) if idx is not None else None),
                    "inputs": inputs,
                    "masked_sentence": inputs,  # cloze prefix; no [MASK]
                    "gold_answer": targets[0],
                    "gold_aliases": targets,
                    "predicate_id": "",
                    "predicate_label": "",
                }
            )
    print(f"  BIG-bench qa_wikidata pool ready: {len(items)} unique prompts")
    return items


def _filter_for_tokenizer(
    pool: list[dict[str, Any]],
    tokenizer,
    *,
    filter_single_token: bool,
) -> list[dict[str, Any]]:
    if not filter_single_token:
        return list(pool)
    filtered: list[dict[str, Any]] = []
    for item in pool:
        gold_ids: list[int] = []
        gold_strs: list[str] = []
        for alias in item.get("gold_aliases") or [item["gold_answer"]]:
            tid = gold_token_id(tokenizer, alias)
            if tid is None:
                continue
            if tid not in gold_ids:
                gold_ids.append(tid)
                gold_strs.append(tokenizer.decode([tid]))
        if not gold_ids:
            continue
        out = dict(item)
        out["gold_token_id"] = gold_ids[0]
        out["gold_token_ids"] = gold_ids
        out["gold_token_str"] = gold_strs[0]
        out["gold_token_strs"] = gold_strs
        filtered.append(out)
    return filtered


def sample_items(
    pool: list[dict[str, Any]],
    *,
    n_sanity: int,
    n_main: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    # Prefer unique item_ids (pool should already be unique).
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in shuffled:
        iid = str(item.get("item_id") or "")
        if iid in seen:
            continue
        seen.add(iid)
        unique.append(item)
    need = n_sanity + n_main
    if len(unique) < need:
        raise ValueError(
            f"Need {need} unique qa_wikidata items but only {len(unique)} available after filtering"
        )
    # Main first so changing n_sanity does not shift the main item set.
    main = unique[:n_main]
    sanity = unique[n_main : n_main + n_sanity]
    return {"sanity": sanity, "main": main}


def items_cache_path(data_dir: pathlib.Path, model_slug: str) -> pathlib.Path:
    return data_dir / f"qa_wikidata_items_{model_slug}.json"


def load_or_create_qa_wikidata_splits(
    cfg: Exp3Config,
    model_slug: str,
    tokenizer,
) -> dict[str, list[dict[str, Any]]]:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    path = items_cache_path(cfg.data_dir, model_slug)
    need_sanity = cfg.sample_size("sanity")
    need_main = cfg.sample_size("main")

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        splits = {"sanity": payload["sanity"], "main": payload["main"]}
        ok = (
            len(splits["sanity"]) >= need_sanity
            and len(splits["main"]) >= need_main
            and splits["sanity"]
            and "gold_token_str" in splits["sanity"][0]
            and "gold_token_ids" in splits["sanity"][0]
            and "inputs" in splits["sanity"][0]
        )
        if ok:
            # Truncate if cache is larger than requested.
            return {
                "sanity": splits["sanity"][:need_sanity],
                "main": splits["main"][:need_main],
            }
        path.unlink()

    pool = load_qa_wikidata_pool(cfg.bigbench)
    pool = _filter_for_tokenizer(
        pool,
        tokenizer,
        filter_single_token=cfg.bigbench.filter_single_token_gold,
    )
    print(
        f"  Sampling qa_wikidata items "
        f"(sanity={need_sanity}, main={need_main}; pool={len(pool)})..."
    )
    splits = sample_items(
        pool,
        n_sanity=need_sanity,
        n_main=need_main,
        seed=cfg.seed,
    )
    path.write_text(
        json.dumps(
            {
                "seed": cfg.seed,
                "model_slug": model_slug,
                "dataset": "qa_wikidata",
                "hf_path": cfg.bigbench.hf_path,
                "config": cfg.bigbench.config,
                "sanity": splits["sanity"],
                "main": splits["main"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Cached qa_wikidata splits -> {path}")
    return splits


def get_qa_wikidata_split(
    cfg: Exp3Config,
    split_name: SplitName,
    model_slug: str,
    tokenizer,
) -> list[dict[str, Any]]:
    splits = load_or_create_qa_wikidata_splits(cfg, model_slug, tokenizer)
    return list(splits[split_name])
