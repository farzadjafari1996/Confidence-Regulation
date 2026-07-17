"""Prompt builders for LAMA cloze and TriviaQA."""

from __future__ import annotations

from exp1.settings import ModelSpec, TriviaQADatasetSettings


def lama_cloze_prompt(masked_sentence: str) -> str:
    """Text before [MASK]; causal LM predicts the object at the end."""
    if "[MASK]" in masked_sentence:
        return masked_sentence.split("[MASK]")[0]
    return masked_sentence


def format_user_prompt(text: str, model_spec: ModelSpec, tokenizer) -> str:
    if not model_spec.use_chat_template:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_lama_prompt(masked_sentence: str, model_spec: ModelSpec, tokenizer) -> str:
    return format_user_prompt(lama_cloze_prompt(masked_sentence), model_spec, tokenizer)


def build_trivia_prompt(
    question: str,
    model_spec: ModelSpec,
    tokenizer,
    trivia_cfg: TriviaQADatasetSettings,
) -> str:
    text = trivia_cfg.prompt_template.format(question=question)
    return format_user_prompt(text, model_spec, tokenizer)
