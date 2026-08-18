"""Load and sample Exp 3 items (Wikidata cloze, LAMA T-REx cloze)."""

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
    name: str = "items",
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in shuffled:
        iid = str(item.get("item_id") or "")
        if not iid:
            iid = f"{item.get('inputs')}||{item.get('gold_answer')}"
            item = dict(item)
            item["item_id"] = iid
        if iid in seen:
            fallback = f"{item.get('inputs')}||{item.get('gold_answer')}#{len(unique)}"
            if fallback in seen:
                continue
            item = dict(item)
            item["item_id"] = fallback
            iid = fallback
        seen.add(iid)
        unique.append(item)
    if len(unique) < n_sanity:
        raise ValueError(
            f"Need {n_sanity} unique {name} sanity items but only {len(unique)} available after filtering"
        )
    if len(unique) < n_sanity + n_main:
        print(
            f"  warning: need {n_sanity + n_main} unique {name} items, have {len(unique)}; "
            f"main will be {len(unique) - n_sanity}"
        )
        sanity = unique[-n_sanity:]
        main = unique[:-n_sanity]
        return {"sanity": sanity, "main": main}
    main = unique[:n_main]
    sanity = unique[n_main : n_main + n_sanity]
    return {"sanity": sanity, "main": main}


def items_cache_path(
    data_dir: pathlib.Path, model_slug: str, dataset: DatasetName = "qa_wikidata"
) -> pathlib.Path:
    return data_dir / f"{dataset}_items_{model_slug}.json"


def _require_cache(
    path: pathlib.Path,
    *,
    need_sanity: int,
    need_main: int,
    required_keys: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = {"sanity": payload.get("sanity") or [], "main": payload.get("main") or []}
    if (
        len(splits["sanity"]) >= need_sanity
        and len(splits["main"]) >= need_main
        and splits["sanity"]
        and all(key in splits["sanity"][0] for key in required_keys)
    ):
        return {
            "sanity": splits["sanity"][:need_sanity],
            "main": splits["main"][:need_main],
        }
    return None


def load_or_create_qa_wikidata_splits(
    cfg: Exp3Config,
    model_slug: str,
    tokenizer,
) -> dict[str, list[dict[str, Any]]]:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    path = items_cache_path(cfg.data_dir, model_slug, "qa_wikidata")
    need_sanity = cfg.sample_size("sanity")
    need_main = cfg.sample_size("main")

    cached = _require_cache(
        path,
        need_sanity=need_sanity,
        need_main=need_main,
        required_keys=("gold_token_str", "gold_token_ids", "inputs"),
    )
    if cached is not None:
        return cached
    if path.exists():
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
        name="qa_wikidata",
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


def _causal_prefix_from_masked(sentence: str, *, require_mask_at_end: bool) -> str | None:
    text = (sentence or "").strip()
    if not text or "[MASK]" not in text:
        return None
    pre, post = text.split("[MASK]", 1)
    leftover = post.strip().strip(".!?,;:\"'")
    if require_mask_at_end and leftover:
        return None
    prefix = pre.rstrip()
    return prefix or None


def load_trex_pool(cfg: Exp3Config, *, target_n: int | None = None) -> list[dict[str, Any]]:
    print(f"  Loading LAMA T-REx ({cfg.trex.hf_path}, config={cfg.trex.config})...")
    max_keep = None if target_n is None else max(int(target_n) * 4, int(target_n) + 200)
    try:
        split = load_dataset(
            cfg.trex.hf_path,
            cfg.trex.config,
            split="train",
            streaming=max_keep is not None,
        )
    except Exception:
        ds = load_dataset(cfg.trex.hf_path, cfg.trex.config)
        split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in split:
        masked = str(row.get("masked_sentence") or "").strip()
        prefix = _causal_prefix_from_masked(
            masked, require_mask_at_end=cfg.trex.require_mask_at_end
        )
        gold = str(row.get("obj_label") or row.get("obj_surface") or "").strip()
        if not prefix or not gold:
            continue
        key = f"{prefix}||{gold}"
        if key in seen:
            continue
        seen.add(key)
        uuid = str(row.get("uuid") or "")
        items.append(
            {
                "item_id": f"trex-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}",
                "uuid": uuid,
                "inputs": prefix,
                "masked_sentence": prefix,
                "gold_answer": gold,
                "gold_aliases": [gold],
                "predicate_id": str(row.get("predicate_id") or ""),
                "predicate_label": str(row.get("label") or ""),
                "task_type": "cloze",
            }
        )
        if max_keep is not None and len(items) >= max_keep:
            break
    print(f"  LAMA T-REx pool ready: {len(items)} unique mask-at-end prompts")
    return items


def load_or_create_trex_splits(
    cfg: Exp3Config,
    model_slug: str,
    tokenizer,
) -> dict[str, list[dict[str, Any]]]:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    path = items_cache_path(cfg.data_dir, model_slug, "lama_trex")
    need_sanity = cfg.sample_size("sanity")
    need_main = cfg.sample_size("main")
    cached = _require_cache(
        path,
        need_sanity=need_sanity,
        need_main=need_main,
        required_keys=("gold_token_ids",),
    )
    if cached is not None:
        return cached
    if path.exists():
        path.unlink()

    pool = load_trex_pool(cfg, target_n=need_sanity + need_main)
    pool = _filter_for_tokenizer(
        pool,
        tokenizer,
        filter_single_token=cfg.trex.filter_single_token_gold,
    )
    print(f"  Sampling T-REx items (sanity={need_sanity}, main={need_main}; pool={len(pool)})...")
    splits = sample_items(
        pool, n_sanity=need_sanity, n_main=need_main, seed=cfg.seed, name="lama_trex"
    )
    path.write_text(
        json.dumps(
            {
                "seed": cfg.seed,
                "model_slug": model_slug,
                "dataset": "lama_trex",
                "sanity": splits["sanity"],
                "main": splits["main"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Cached T-REx splits -> {path}")
    return splits


def _load_or_create_splits(cfg: Exp3Config, model_slug: str, tokenizer):
    if cfg.dataset == "qa_wikidata":
        return load_or_create_qa_wikidata_splits(cfg, model_slug, tokenizer)
    if cfg.dataset == "lama_trex":
        return load_or_create_trex_splits(cfg, model_slug, tokenizer)
    raise ValueError(f"Unsupported Exp3 dataset: {cfg.dataset}")


def prepare_splits(
    cfg: Exp3Config,
    model_slug: str,
    tokenizer,
    *,
    force: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    path = items_cache_path(cfg.data_dir, model_slug, cfg.dataset)
    if force and path.exists():
        path.unlink()
        print(f"  Removed item cache {path}")
    return _load_or_create_splits(cfg, model_slug, tokenizer)


def load_cached_splits(
    cfg: Exp3Config,
    model_slug: str,
) -> dict[str, list[dict[str, Any]]]:
    path = items_cache_path(cfg.data_dir, model_slug, cfg.dataset)
    need_sanity = cfg.sample_size("sanity")
    need_main = cfg.sample_size("main")
    required = (
        ("gold_token_str", "gold_token_ids", "inputs")
        if cfg.dataset == "qa_wikidata"
        else ("gold_token_ids",)
    )
    cached = _require_cache(
        path, need_sanity=need_sanity, need_main=need_main, required_keys=required
    )
    if cached is None:
        raise FileNotFoundError(
            f"Missing item cache {path}. Run exp3/prepare_datasets.py first."
        )
    return cached


def load_cached_split(
    cfg: Exp3Config,
    split_name: SplitName,
    model_slug: str,
) -> list[dict[str, Any]]:
    return list(load_cached_splits(cfg, model_slug)[split_name])


def get_exp3_split(
    cfg: Exp3Config,
    split_name: SplitName,
    model_slug: str,
    tokenizer=None,
) -> list[dict[str, Any]]:
    if tokenizer is None:
        return load_cached_split(cfg, split_name, model_slug)
    splits = _load_or_create_splits(cfg, model_slug, tokenizer)
    return list(splits[split_name])
