"""Load and sample LAMA T-REx and TriviaQA items for Phase 3b."""

from __future__ import annotations

import json
import pathlib
import random
from typing import Any

from datasets import load_dataset

from exp1.settings import Exp1Config, LamaDatasetSettings, SplitName, TriviaQADatasetSettings


def _valid_lama_cloze(item: dict[str, Any]) -> bool:
    """LAMA cloze needs non-empty text before [MASK] for causal LM scoring."""
    masked = item.get("masked_sentence") or ""
    if "[MASK]" not in masked:
        return False
    return bool(masked.split("[MASK]")[0].strip())


def _gold_token_id(tokenizer, label: str) -> int | None:
    """Return a single vocab id for the gold answer, preferring GPT-2 space-prefixed form."""
    candidates: list[str] = []
    label = (label or "").strip()
    if not label:
        return None
    # Causal LAMA cloze almost always predicts after whitespace → BPE " Word".
    candidates.append(" " + label)
    candidates.append(label)
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) == 1:
            return ids[0]
    return None


def load_lama_pool(cfg: LamaDatasetSettings, *, max_rows: int = 200_000) -> list[dict[str, Any]]:
    # T-REx is ~1.3M rows; slice keeps HF load tractable on consumer hardware.
    split = f"train[:{max_rows}]"
    ds = load_dataset(cfg.hf_path, cfg.config, split=split)
    items: list[dict[str, Any]] = []
    for row in ds:
        label = (row.get("obj_label") or row.get("obj_surface") or "").strip()
        if not label:
            continue
        masked = row.get("masked_sentence") or ""
        if "[MASK]" not in masked:
            continue
        if not masked.split("[MASK]")[0].strip():
            continue
        items.append(
            {
                "item_id": row.get("uuid") or row.get("id") or str(len(items)),
                "masked_sentence": masked,
                "gold_answer": label,
                "gold_aliases": [label],
                "predicate_id": row.get("predicate_id"),
                # Wikidata relation name, e.g. P37 → "official language"
                "predicate_label": row.get("label") or row.get("description") or "",
                "template": row.get("template") or "",
            }
        )
    return items


def load_trivia_pool(cfg: TriviaQADatasetSettings, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    split = cfg.split if max_rows is None else f"{cfg.split}[:{max_rows}]"
    print(f"  Loading TriviaQA pool ({cfg.hf_path}, split={split})...")
    ds = load_dataset(cfg.hf_path, cfg.config, split=split)
    items: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        question = row.get("question")
        if not question:
            continue
        answer = row.get("answer") or {}
        aliases = list(answer.get("normalized_aliases") or [])
        value = answer.get("normalized_value") or answer.get("value") or ""
        if value and value not in aliases:
            aliases.append(value)
        aliases = [a for a in aliases if a]
        if not aliases:
            continue
        items.append(
            {
                "item_id": row.get("question_id") or row.get("question_source") or str(len(items)),
                "question": question,
                "gold_answer": aliases[0],
                "gold_aliases": aliases,
            }
        )
        if (i + 1) % 5000 == 0:
            print(f"    scanned {i + 1} rows, kept {len(items)} items")
    print(f"  TriviaQA pool ready: {len(items)} items")
    return items


def _filter_lama_for_tokenizer(
    pool: list[dict[str, Any]],
    tokenizer,
    *,
    filter_single_token: bool,
) -> list[dict[str, Any]]:
    if not filter_single_token:
        return [i for i in pool if _valid_lama_cloze(i)]
    filtered: list[dict[str, Any]] = []
    for item in pool:
        if not _valid_lama_cloze(item):
            continue
        tid = _gold_token_id(tokenizer, item["gold_answer"])
        if tid is None:
            continue
        out = dict(item)
        out["gold_token_id"] = tid
        # Surface form that maps to gold_token_id (often leading-space for GPT-2).
        decoded = tokenizer.decode([tid])
        out["gold_token_str"] = decoded
        filtered.append(out)
    return filtered


def sample_items(
    pool: list[dict[str, Any]],
    *,
    n_sanity: int,
    n_main: int,
    seed: int,
    tokenizer=None,
    filter_single_token: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    if filter_single_token and tokenizer is not None:
        pool = _filter_lama_for_tokenizer(pool, tokenizer, filter_single_token=True)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    need = n_sanity + n_main
    if len(shuffled) < need:
        raise ValueError(f"Need {need} items but only {len(shuffled)} available after filtering")
    picked = shuffled[:need]
    return {"sanity": picked[:n_sanity], "main": picked[n_sanity : n_sanity + n_main]}


def items_cache_path(
    data_dir: pathlib.Path,
    dataset: str,
    model_slug: str,
) -> pathlib.Path:
    return data_dir / f"{dataset}_items_{model_slug}.json"


def load_or_create_item_splits(
    p3_cfg: Exp1Config,
    dataset: str,
    model_slug: str,
    tokenizer,
    *,
    filter_single_token: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    p3_cfg.data_dir.mkdir(parents=True, exist_ok=True)
    path = items_cache_path(p3_cfg.data_dir, dataset, model_slug)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        splits = {"sanity": payload["sanity"], "main": payload["main"]}
        if dataset == "lama":
            splits = {
                k: [i for i in v if _valid_lama_cloze(i)] for k, v in splits.items()
            }
            # Rebuild if cache predates predicate_label or gold_token space-prefix fix.
            needs_rebuild = (
                len(splits["sanity"]) < p3_cfg.sample_size("sanity")
                or len(splits["main"]) < p3_cfg.sample_size("main")
                or (
                    splits["sanity"]
                    and (
                        "predicate_label" not in splits["sanity"][0]
                        or "gold_token_str" not in splits["sanity"][0]
                    )
                )
            )
            if needs_rebuild:
                path.unlink()
            else:
                return splits
        else:
            return splits

    if dataset == "lama":
        pool = load_lama_pool(p3_cfg.lama)
    elif dataset == "triviaqa":
        pool = load_trivia_pool(p3_cfg.triviaqa)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    print(
        f"  Sampling {dataset} items "
        f"(sanity={p3_cfg.sample_size('sanity')}, main={p3_cfg.sample_size('main')})..."
    )
    splits = sample_items(
        pool,
        n_sanity=p3_cfg.sample_size("sanity"),
        n_main=p3_cfg.sample_size("main"),
        seed=p3_cfg.seed,
        tokenizer=tokenizer,
        filter_single_token=filter_single_token,
    )
    path.write_text(
        json.dumps(
            {
                "seed": p3_cfg.seed,
                "model_slug": model_slug,
                "dataset": dataset,
                "sanity": splits["sanity"],
                "main": splits["main"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Cached {dataset} splits -> {path}")
    return splits


def get_split_items(
    p3_cfg: Exp1Config,
    dataset: str,
    split_name: SplitName,
    model_slug: str,
    tokenizer,
) -> list[dict[str, Any]]:
    filter_single = dataset == "lama" and p3_cfg.lama.filter_single_token_gold
    splits = load_or_create_item_splits(
        p3_cfg,
        dataset,
        model_slug,
        tokenizer,
        filter_single_token=filter_single,
    )
    return splits[split_name]
