"""Mean-ablation logit computation and generation (TE / DE_LN) for QA."""

from __future__ import annotations

from typing import Literal

import torch
import tqdm
from datasets import load_dataset
from transformer_lens import utils as tl_utils
from transformer_lens.utils import tokenize_and_concatenate

from gpt2.core import _apply_ln_final, get_entropy
from exp1.metrics import sample_token_from_logits
from exp1.settings import DecodeGreedySettings, DecodeSamplingSettings, Exp1Config

AblationBranch = Literal["none", "te", "de_ln"]


def _layer_idx(model) -> int:
    return model.cfg.n_layers - 1


def _w_out_last(model) -> torch.Tensor:
    layer = _layer_idx(model)
    w = model.W_out[layer]
    return w.float() if w.dtype != torch.float32 else w


def load_calibration_tokenized(tokenizer, p3_cfg: Exp1Config):
    cal = p3_cfg.calibration
    data = load_dataset(cal.dataset, split="train")
    subset = data.select(range(min(cal.data_end, len(data))))
    return tokenize_and_concatenate(
        subset,
        tokenizer,
        max_length=cal.max_seq_length,
        column_name="text",
    )


def compute_activation_means_from_tokenized(
    model,
    tokenized,
    neuron_names: list[str],
    *,
    device: str,
) -> torch.Tensor:
    """Average post-MLP activations over all token positions in tokenized batches."""
    layer = _layer_idx(model)
    indices = [int(n.split(".")[1]) for n in neuron_names]
    sums = torch.zeros(len(indices), dtype=torch.float64)
    count = 0

    for i in tqdm.tqdm(range(len(tokenized)), desc="calibration activations"):
        tok_seq = tokenized["tokens"][i].unsqueeze(0).to(device)
        _, cache = model.run_with_cache(tok_seq)
        act = cache[tl_utils.get_act_name("post", layer)][0, :, indices].float().cpu()
        sums += act.sum(dim=0).double()
        count += act.shape[0]
        del cache

    return (sums / max(count, 1)).float()


def logits_at_position(
    model,
    cache: dict,
    pos: int,
    neuron_indices: list[int] | None,
    activation_means: torch.Tensor | None,
    branch: AblationBranch,
    *,
    device: str,
) -> torch.Tensor:
    """Next-token logits at position `pos` (predicts token pos+1)."""
    layer = _layer_idx(model)
    w_out = _w_out_last(model)
    res = cache[tl_utils.get_act_name("resid_post", layer)][0, pos].float()

    if branch == "none" or not neuron_indices:
        normed = model.ln_final(res.unsqueeze(0).unsqueeze(0))
        logits = normed.float() @ model.W_U + model.b_U
        return logits.squeeze()

    prev_act = cache[tl_utils.get_act_name("post", layer)][0, pos, neuron_indices].float()
    means = activation_means.to(prev_act.device).float()
    deltas = means - prev_act
    res_delta = (deltas.unsqueeze(-1) * w_out[neuron_indices]).sum(dim=0)
    updated = res + res_delta

    if branch == "te":
        normed = model.ln_final(updated.unsqueeze(0).unsqueeze(0))
    else:
        ln_scale = cache["ln_final.hook_scale"][0, pos].to(updated.device).float()
        x = updated.unsqueeze(0).unsqueeze(0)
        if hasattr(model, "apply_ln_final_frozen"):
            normed = model.apply_ln_final_frozen(x, ln_scale.unsqueeze(0))
        else:
            normed = _apply_ln_final(model, x, ln_scale.unsqueeze(0))

    logits = normed.float() @ model.W_U + model.b_U
    return logits.squeeze()


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


def entropy_and_top1(logits: torch.Tensor) -> tuple[float, float]:
    h = get_entropy(logits.unsqueeze(0).unsqueeze(0)).item()
    probs = torch.softmax(logits.float(), dim=-1)
    top1 = float(probs.max().item())
    return h, top1


def mean_neuron_activation(
    cache: dict,
    layer: int,
    pos: int,
    neuron_indices: list[int],
) -> float:
    act = cache[tl_utils.get_act_name("post", layer)][0, pos, neuron_indices].float()
    return float(act.mean().item())


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    *,
    neuron_indices: list[int] | None,
    activation_means: torch.Tensor | None,
    branch: AblationBranch,
    max_new_tokens: int,
    greedy_cfg: DecodeGreedySettings,
    sampling_cfg: DecodeSamplingSettings,
    decode_mode: str,
    device: str,
) -> str:
    """Autoregressive decode with optional mean ablation at each step."""
    input_ids = encode_prompt(tokenizer, prompt, device)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits_raw, cache = forward_cache(model, input_ids)
        pos = input_ids.shape[1] - 1
        if branch == "none" or not neuron_indices:
            next_logits = logits_raw[0, pos].float()
        else:
            next_logits = logits_at_position(
                model,
                cache,
                pos,
                neuron_indices,
                activation_means,
                branch,
                device=device,
            )
        del cache

        if decode_mode == "greedy" or not sampling_cfg.do_sample:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_id = sample_token_from_logits(
                next_logits,
                temperature=sampling_cfg.temperature,
                top_p=sampling_cfg.top_p,
            )

        generated.append(next_id)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
        next_tok = torch.tensor([[next_id]], device=device, dtype=torch.long)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

        piece = tokenizer.decode(generated, skip_special_tokens=True)
        if "\n" in piece:
            piece = piece.split("\n")[0]
            generated = tokenizer.encode(piece, add_special_tokens=False)
            break

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_lama_item(
    model,
    tokenizer,
    prompt: str,
    gold_token_id: int,
    *,
    neuron_indices: list[int] | None,
    activation_means: torch.Tensor | None,
    branch: AblationBranch,
    decode_mode: str,
    sampling_cfg: DecodeSamplingSettings,
    device: str,
) -> tuple[dict, dict]:
    """Return (metrics_dict, feature_dict) for one LAMA cloze item."""
    from exp1.metrics import lama_rank_metrics

    input_ids = encode_prompt(tokenizer, prompt, device)
    logits_raw, cache = forward_cache(model, input_ids)
    pos = input_ids.shape[1] - 1
    layer = _layer_idx(model)

    if branch == "none" or not neuron_indices:
        logits = logits_raw[0, pos].float()
    else:
        logits = logits_at_position(
            model, cache, pos, neuron_indices, activation_means, branch, device=device
        )

    if decode_mode == "sampling" and sampling_cfg.do_sample:
        pred_id = sample_token_from_logits(
            logits,
            temperature=sampling_cfg.temperature,
            top_p=sampling_cfg.top_p,
        )
        metrics = lama_rank_metrics(gold_token_id, logits)
        metrics["prediction_id"] = pred_id
        metrics["prediction"] = tokenizer.decode([pred_id])
        metrics["correct"] = pred_id == gold_token_id
        metrics["acc_at_1"] = float(pred_id == gold_token_id)
    else:
        metrics = lama_rank_metrics(gold_token_id, logits)
        pred_id = int(torch.argmax(logits).item())
        metrics["prediction_id"] = pred_id
        metrics["prediction"] = tokenizer.decode([pred_id])

    ent, top1 = entropy_and_top1(logits)
    neuron_act = None
    if neuron_indices:
        neuron_act = mean_neuron_activation(cache, layer, pos, neuron_indices)

    features = {
        "entropy_at_answer": ent,
        "top1_prob_at_answer": top1,
        "mean_entropy_neuron_act": neuron_act,
    }
    del cache
    return metrics, features
