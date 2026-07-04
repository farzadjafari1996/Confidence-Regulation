"""Llama 3.2 model loading via HuggingFace."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.core import ModelRunConfig, _read_hf_token
from llama.wrapper import LlamaWrapper

LLAMA_32_1B_BASE = "meta-llama/Llama-3.2-1B"
LLAMA_32_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(cfg: ModelRunConfig, device: str) -> torch.dtype:
    if cfg.dtype:
        key = cfg.dtype.lower()
        if key not in _DTYPE_MAP:
            raise ValueError(f"Unknown dtype {cfg.dtype!r}; use float16, bfloat16, or float32")
        return _DTYPE_MAP[key]
    return torch.float16 if device == "cuda" else torch.float32


def load_llama(cfg: ModelRunConfig, device: str) -> tuple[LlamaWrapper, AutoTokenizer]:
    torch.set_grad_enabled(False)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    token = _read_hf_token()
    dtype = _resolve_dtype(cfg, device)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        token=token,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.name, token=token)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return LlamaWrapper(model, tokenizer, device), tokenizer
