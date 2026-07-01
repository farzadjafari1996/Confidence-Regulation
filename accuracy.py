from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from typing import Any
import torch
from datasets import load_dataset

TRIVIA_QA_DATASET = "mandarjoshi/trivia_qa"
TRIVIA_QA_CONFIG  = "rc.nocontext"
TRIVIA_QA_SPLIT   = "validation"

LAMA_DATASET = "facebook/lama"
LAMA_CONFIG  = "trex"
LAMA_SPLIT   = "train"


# ── Text helpers ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_correct(generated: str, gold_answers: list[str]) -> bool:
    haystack = f" {_normalize(generated)} "
    return any(f" {_normalize(ans)} " in haystack for ans in gold_answers if ans)


# ── Dataset loading (cached so the loop in main doesn't re-download) ──────────

@lru_cache(maxsize=None)
def _load_triviaqa():
    print("[dataset] Loading TriviaQA (rc.nocontext / validation) ...")
    ds = load_dataset(TRIVIA_QA_DATASET, TRIVIA_QA_CONFIG, split=TRIVIA_QA_SPLIT)
    print(f"[dataset] TriviaQA loaded — {len(ds):,} examples")
    return ds


from huggingface_hub import HfApi

@lru_cache(maxsize=None)
def _load_lama(config_name: str):
    print(f"[dataset] Loading LAMA ({config_name}) ...")
    api = HfApi()
    files = api.list_repo_files(
        "facebook/lama",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    urls = [
        f"https://huggingface.co/datasets/facebook/lama/resolve/refs%2Fconvert%2Fparquet/{path}"
        for path in files
        if path.startswith(f"{config_name}/") and path.endswith(".parquet")
    ]
    print(f"[dataset] Found {len(urls)} parquet file(s) for config '{config_name}'")
    ds = load_dataset("parquet", data_files={"train": urls}, split="train")
    print(f"[dataset] LAMA ({config_name}) loaded — {len(ds):,} examples")
    return ds

# ── Prompt builders ───────────────────────────────────────────────────────────

def _triviaqa_prompt(example: dict) -> str:
    q = example.get("question", "")
    return f"Question: {q}\nAnswer:" if q else ""


def _lama_prompt(example: dict) -> str:
    masked = example.get("masked_sentence", "")
    return masked.replace("[MASK]", "").strip() if masked else ""


# ── Gold answer extractors ────────────────────────────────────────────────────

def _triviaqa_gold(example: dict) -> list[str]:
    answer = example.get("answer", {})
    return [answer.get("value", "")] + list(answer.get("aliases", []))


def _lama_gold(example: dict) -> list[str]:
    label = example.get("obj_label", "")
    return [label] if label else []


# ── Generation ────────────────────────────────────────────────────────────────

def _generate(model: Any, tokenizer: Any, prompt: str) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Core evaluation loop ──────────────────────────────────────────────────────

LOG_INTERVAL = 100

def _evaluate(model, tokenizer, dataset, prompt_fn, gold_fn, max_examples, dataset_name="dataset"):
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    total_examples = len(dataset)
    print(f"[eval] Starting evaluation on {dataset_name} — {total_examples:,} examples")

    correct = total = skipped = 0
    for i, example in enumerate(dataset):
        prompt = prompt_fn(example)
        gold   = gold_fn(example)
        if not prompt or not gold:
            skipped += 1
            continue
        if _is_correct(_generate(model, tokenizer, prompt), gold):
            correct += 1
        total += 1

        if total % LOG_INTERVAL == 0:
            running_acc = correct / total
            print(f"[eval] {total:>{len(str(total_examples))}}/{total_examples} evaluated "
                  f"| correct: {correct} | running accuracy: {running_acc:.4f}"
                  + (f" | skipped: {skipped}" if skipped else ""))

    final_acc = correct / total if total else 0.0
    print(f"[eval] Done — {total:,} evaluated, {correct:,} correct, "
          f"{skipped:,} skipped | final accuracy: {final_acc:.4f}")
    return final_acc


# ── Public API (signatures unchanged from main.py) ────────────────────────────

def evaluate_model_on_dataset_trivia_qa(
    tokenizer: Any,
    model: Any,
    *,
    max_examples: int | None = None,
) -> float:
    return _evaluate(model, tokenizer, _load_triviaqa(),
                     _triviaqa_prompt, _triviaqa_gold, max_examples,
                     dataset_name="TriviaQA")


def evaluate_model_on_dataset_lama(
    tokenizer: Any,
    model: Any,
    *,
    config_name: str = LAMA_CONFIG,
    max_examples: int | None = None,
) -> float:
    return _evaluate(model, tokenizer, _load_lama(config_name),
                     _lama_prompt, _lama_gold, max_examples,
                     dataset_name=f"LAMA/{config_name}")