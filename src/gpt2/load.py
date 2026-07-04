"""GPT-2 family model loading (TransformerLens, no HF gating)."""

from __future__ import annotations

from gpt2.core import ModelRunConfig, load_model


def load_gpt2(cfg: ModelRunConfig, device: str):
    return load_model(cfg, device)
