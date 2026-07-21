"""Small LM helpers shared by Exp 3 collect / labeling (no LLM training)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Family = Literal["gpt2", "llama", "qwen"]


@dataclass
class ModelSpec:
    family: Family
    model: str
    use_chat_template: bool = False


def encode_prompt(tokenizer, prompt: str, device: str) -> torch.Tensor:
    """Tokenize prompt to long input_ids on device (TL tokenizer quirk on empty input)."""
    if not prompt.strip():
        raise ValueError("Cannot encode empty prompt")
    ids = tokenizer.encode(prompt, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor([ids], dtype=torch.long)
    return ids.to(device=device, dtype=torch.long)


def forward_cache(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
    with torch.no_grad():
        return model.run_with_cache(input_ids.long())


def gold_token_id(tokenizer, label: str) -> int | None:
    """Return a single vocab id for the gold answer, preferring GPT-2 space-prefixed form."""
    candidates: list[str] = []
    label = (label or "").strip()
    if not label:
        return None
    # Causal cloze almost always predicts after whitespace → BPE " Word".
    candidates.append(" " + label)
    candidates.append(label)
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) == 1:
            return ids[0]
    return None


def format_user_prompt(text: str, model_spec: ModelSpec, tokenizer) -> str:
    if not model_spec.use_chat_template:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
