"""TriviaQA-style answer normalization and scoring."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles, collapse whitespace."""
    text = text.lower()
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def trivia_em_f1(prediction: str, ground_truths: list[str]) -> tuple[float, float]:
    em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return em, f1


def lama_rank_metrics(gold_token_id: int, logits) -> dict:
    """Compute @1, @5, MRR for a single gold token id from vocab logits."""
    import torch

    if logits.dim() > 1:
        logits = logits.squeeze()
    sorted_ids = torch.argsort(logits, descending=True)
    ranks = (sorted_ids == gold_token_id).nonzero(as_tuple=True)[0]
    if len(ranks) == 0:
        return {"rank": -1, "acc_at_1": 0.0, "acc_at_5": 0.0, "mrr": 0.0, "correct": False}
    rank = int(ranks[0].item())
    return {
        "rank": rank,
        "acc_at_1": float(rank == 0),
        "acc_at_5": float(rank < 5),
        "mrr": 1.0 / (rank + 1),
        "correct": rank == 0,
    }


def sample_token_from_logits(logits, *, temperature: float, top_p: float) -> int:
    """Sample one token; temperature=1, top_p=1 reproduces full softmax sampling."""
    import torch

    if logits.dim() > 1:
        logits = logits.squeeze()
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative <= top_p
        mask[0] = True
        filtered_probs = sorted_probs * mask
        filtered_probs = filtered_probs / filtered_probs.sum()
        choice = torch.multinomial(filtered_probs, 1).item()
        return int(sorted_idx[choice].item())
    return int(torch.multinomial(probs, 1).item())
