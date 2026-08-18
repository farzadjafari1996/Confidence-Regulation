"""Small LM helpers shared by Exp 3 collect / labeling (no LLM training)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transformers import AutoTokenizer

from hyperparameters import (
    GEMMA_HPARAMS,
    GPT2_HPARAMS,
    LLAMA_HPARAMS,
    SMOLLM_HPARAMS,
    ModelRunConfig,
    pythia_hparams_for_model,
    qwen_hparams_for_model,
)

Family = Literal["gpt2", "llama", "qwen", "pythia", "smollm", "gemma"]

_TOKENIZER_HF_IDS = {
    "gpt2-small": "gpt2",
    "gpt2": "gpt2",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}


@dataclass
class ModelSpec:
    family: Family
    model: str
    use_chat_template: bool = False


def encode_prompt(tokenizer, prompt: str, device: str, *, max_length: int | None = None):
    """Tokenize prompt to long input_ids on device (TL tokenizer quirk on empty input).

    If `max_length` is set and the prompt is longer, keep the suffix so the last
    token (e.g. MMLU "Answer:") still sits in context.
    """
    import torch

    if not prompt.strip():
        raise ValueError("Cannot encode empty prompt")
    ids = tokenizer.encode(prompt, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor([ids], dtype=torch.long)
    n_orig = int(ids.shape[-1])
    if max_length is not None and max_length > 0 and n_orig > max_length:
        ids = ids[..., -max_length:]
    return ids.to(device=device, dtype=torch.long), n_orig


def forward_cache(model, input_ids):
    import torch

    with torch.inference_mode():
        return model.run_with_cache(input_ids.long())


def token_id_candidates(tokenizer, text: str) -> list[int]:
    """Single-token vocab ids for `text` and a leading-space variant."""
    text = (text or "").strip()
    if not text:
        return []
    ids_out: list[int] = []
    for cand in (" " + text, text):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) == 1 and ids[0] not in ids_out:
            ids_out.append(ids[0])
    return ids_out


def gold_token_id(tokenizer, label: str) -> int | None:
    """Return a single vocab id for the gold answer, preferring GPT-2 space-prefixed form."""
    ids = token_id_candidates(tokenizer, label)
    return ids[0] if ids else None


def format_user_prompt(text: str, model_spec: ModelSpec, tokenizer) -> str:
    if not model_spec.use_chat_template:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def hparams_for_family(family: str, model: str):
    if family == "gpt2":
        return GPT2_HPARAMS
    if family == "llama":
        return LLAMA_HPARAMS
    if family == "qwen":
        return qwen_hparams_for_model(model)
    if family == "pythia":
        return pythia_hparams_for_model(model)
    if family == "smollm":
        return SMOLLM_HPARAMS
    if family == "gemma":
        return GEMMA_HPARAMS
    raise ValueError(f"Unsupported Exp3 family: {family}")


def model_spec_from_cfg(cfg) -> ModelSpec:
    return ModelSpec(
        family=cfg.family,  # type: ignore[arg-type]
        model=cfg.model,
        use_chat_template=cfg.effective_chat_template(),
    )


def tokenizer_pretrained_id(model: str) -> str:
    return _TOKENIZER_HF_IDS.get(model, model)


def _read_hf_token() -> str | None:
    import os
    import pathlib

    token_path = pathlib.Path(__file__).resolve().parents[3] / "reference_code" / "ablations" / "hf_token.txt"
    if token_path.exists():
        text = token_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    try:
        from huggingface_hub import HfFolder

        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def load_tokenizer(cfg) -> object:
    """Tokenizer only (no LM weights). Vocab must match the gather-step tokenizer."""
    pretrained = tokenizer_pretrained_id(cfg.model)
    token = _read_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(pretrained, token=token)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_exp3_model(cfg, device: str):
    """Load family LM + tokenizer for activation gathering."""
    run_cfg = ModelRunConfig.from_hparams(
        cfg.model, hparams_for_family(cfg.family, cfg.model), dtype=cfg.dtype
    )
    if cfg.family == "gpt2":
        from gpt2.load import load_gpt2

        return load_gpt2(run_cfg, device)
    if cfg.family == "llama":
        from llama.load import load_llama

        return load_llama(run_cfg, device)
    if cfg.family == "qwen":
        from qwen.load import load_qwen

        return load_qwen(run_cfg, device)
    if cfg.family == "pythia":
        from pythia.load import load_pythia

        return load_pythia(run_cfg, device)
    if cfg.family == "smollm":
        from smollm.load import load_smollm

        return load_smollm(run_cfg, device)
    if cfg.family == "gemma":
        from gemma.load import load_gemma

        return load_gemma(run_cfg, device)
    raise ValueError(f"Unsupported Exp3 family: {cfg.family}")


def first_param(model):
    inner = getattr(model, "model", model)
    return next(inner.parameters())
