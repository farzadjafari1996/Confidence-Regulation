"""Ext 1 QA ablation runner."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import tqdm

from gpt2.core import load_phase1_outputs, normalize_component_names
from hyperparameters import ModelRunConfig, model_slug
from qa.ablation_engine import (
    AblationBranch,
    compute_activation_means_from_tokenized,
    encode_prompt,
    evaluate_lama_item,
    generate_answer,
    load_calibration_tokenized,
    forward_cache,
    entropy_and_top1,
    mean_neuron_activation,
    _layer_idx,
)
from qa.data_loaders import get_split_items
from qa.metrics import trivia_em_f1
from qa.plots import write_ext1_plots
from qa.prompts import build_lama_prompt, build_trivia_prompt
from qa.settings import (
    AblationMode,
    DatasetName,
    Ext1Config,
    ModelSpec,
    SplitName,
    ext1_dataset_dir,
)


def _slug(model_name: str) -> str:
    return model_slug(model_name)


def _pick_random_controls(static_df: pd.DataFrame, entropy_labels: list[str], seed: int) -> list[str]:
    pool = static_df[~static_df["component_name"].isin(entropy_labels)]
    n = min(len(entropy_labels), len(pool))
    return pool.sample(n, random_state=seed)["component_name"].tolist()


def _indices(names: list[str]) -> list[int]:
    return [int(n.split(".")[1]) for n in names]


def _json_float(x) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(x)


def _aggregate_metrics(df: pd.DataFrame, dataset: DatasetName) -> dict[str, Any]:
    """Aggregate mean metrics grouped by decode / condition / branch."""
    out: dict[str, Any] = {}
    metric_cols = (
        ["acc_at_1", "acc_at_5", "mrr"]
        if dataset == "lama"
        else ["em", "f1"]
    )
    for (decode_mode, condition, branch), grp in df.groupby(
        ["decode_mode", "condition", "ablation_branch"]
    ):
        out.setdefault(decode_mode, {}).setdefault(condition, {})[branch] = {
            col: _json_float(grp[col].mean()) for col in metric_cols if col in grp.columns
        }
    return out


def _deltas_vs_baseline(agg: dict, dataset: DatasetName) -> dict[str, Any]:
    primary = "mrr" if dataset == "lama" else "f1"
    deltas: dict[str, Any] = {}
    for decode_mode, conds in agg.items():
        base_block = conds.get("baseline", {})
        base = base_block.get("none") or base_block.get("te") or {}
        base_val = base.get(primary, 0.0) or 0.0
        deltas[decode_mode] = {}
        for condition in ("ablate_entropy", "ablate_random"):
            if condition not in conds:
                continue
            for branch, metrics in conds[condition].items():
                val = metrics.get(primary, 0.0) or 0.0
                deltas[decode_mode][f"{condition}_{branch}"] = {primary: _json_float(val - base_val)}
    return deltas


def _run_lama_item(
    model,
    tokenizer,
    item: dict,
    model_spec: ModelSpec,
    p3_cfg: Ext1Config,
    *,
    condition: str,
    branch: AblationBranch,
    neuron_names: list[str] | None,
    activation_means: torch.Tensor | None,
    decode_mode: str,
    ablation_mode: AblationMode,
    component_name: str,
    device: str,
) -> dict[str, Any]:
    prompt = build_lama_prompt(item["masked_sentence"], model_spec, tokenizer)
    metrics, features = evaluate_lama_item(
        model,
        tokenizer,
        prompt,
        item["gold_token_id"],
        neuron_indices=_indices(neuron_names) if neuron_names else None,
        activation_means=activation_means,
        branch=branch,
        decode_mode=decode_mode,
        sampling_cfg=p3_cfg.decoding.sampling,
        device=device,
    )
    return {
        "item_id": item["item_id"],
        "dataset": "lama",
        "condition": condition,
        "ablation_branch": branch,
        "ablation_mode": ablation_mode,
        "decode_mode": decode_mode,
        "component_name": component_name,
        "prompt": prompt,
        "gold_answer": item["gold_answer"],
        "gold_aliases": json.dumps(item.get("gold_aliases", [])),
        "prediction": metrics.get("prediction"),
        "correct": metrics.get("correct"),
        "acc_at_1": metrics.get("acc_at_1"),
        "acc_at_5": metrics.get("acc_at_5"),
        "mrr": metrics.get("mrr"),
        "rank": metrics.get("rank"),
        "em": None,
        "f1": None,
        "entropy_at_answer": features.get("entropy_at_answer"),
        "top1_prob_at_answer": features.get("top1_prob_at_answer"),
        "mean_entropy_neuron_act": features.get("mean_entropy_neuron_act"),
        "n_ablated_neurons": len(neuron_names) if neuron_names else 0,
        "generated_text": None,
    }


def _run_trivia_item(
    model,
    tokenizer,
    item: dict,
    model_spec: ModelSpec,
    p3_cfg: Ext1Config,
    *,
    condition: str,
    branch: AblationBranch,
    neuron_names: list[str] | None,
    activation_means: torch.Tensor | None,
    decode_mode: str,
    ablation_mode: AblationMode,
    component_name: str,
    device: str,
) -> dict[str, Any]:
    prompt = build_trivia_prompt(item["question"], model_spec, tokenizer, p3_cfg.triviaqa)
    indices = _indices(neuron_names) if neuron_names else None

    input_ids = encode_prompt(tokenizer, prompt, device)
    logits_raw, cache = forward_cache(model, input_ids)
    pos = input_ids.shape[1] - 1
    layer = _layer_idx(model)
    if branch == "none" or not indices:
        ans_logits = logits_raw[0, pos].float()
    else:
        from qa.ablation_engine import logits_at_position

        ans_logits = logits_at_position(
            model, cache, pos, indices, activation_means, branch, device=device
        )
    ent, top1 = entropy_and_top1(ans_logits)
    neuron_act = mean_neuron_activation(cache, layer, pos, indices) if indices else None
    del cache

    prediction = generate_answer(
        model,
        tokenizer,
        prompt,
        neuron_indices=indices,
        activation_means=activation_means,
        branch=branch,
        max_new_tokens=p3_cfg.triviaqa.max_new_tokens,
        greedy_cfg=p3_cfg.decoding.greedy,
        sampling_cfg=p3_cfg.decoding.sampling,
        decode_mode=decode_mode,
        device=device,
    )
    aliases = item.get("gold_aliases") or [item["gold_answer"]]
    em, f1 = trivia_em_f1(prediction, aliases)
    return {
        "item_id": item["item_id"],
        "dataset": "triviaqa",
        "condition": condition,
        "ablation_branch": branch,
        "ablation_mode": ablation_mode,
        "decode_mode": decode_mode,
        "component_name": component_name,
        "prompt": prompt,
        "gold_answer": item["gold_answer"],
        "gold_aliases": json.dumps(aliases),
        "prediction": prediction,
        "correct": bool(em),
        "em": em,
        "f1": f1,
        "acc_at_1": None,
        "acc_at_5": None,
        "mrr": None,
        "rank": None,
        "entropy_at_answer": ent,
        "top1_prob_at_answer": top1,
        "mean_entropy_neuron_act": neuron_act,
        "n_ablated_neurons": len(neuron_names) if neuron_names else 0,
        "generated_text": prediction,
    }


def _apply_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_entropy"] = np.nan
    df["delta_mean_entropy_neuron_act"] = np.nan

    baseline_df = df[df["condition"] == "baseline"]
    for (item_id, decode_mode), base_rows in baseline_df.groupby(["item_id", "decode_mode"]):
        base = base_rows.iloc[0]
        mask = (
            (df["item_id"] == item_id)
            & (df["decode_mode"] == decode_mode)
            & (df["condition"] != "baseline")
        )
        if pd.notna(base.get("entropy_at_answer")):
            df.loc[mask, "delta_entropy"] = df.loc[mask, "entropy_at_answer"] - base["entropy_at_answer"]
        if pd.notna(base.get("mean_entropy_neuron_act")):
            df.loc[mask, "delta_mean_entropy_neuron_act"] = (
                df.loc[mask, "mean_entropy_neuron_act"] - base["mean_entropy_neuron_act"]
            )
    return df


def _eval_item(
    model,
    tokenizer,
    item: dict,
    model_spec: ModelSpec,
    p3_cfg: Ext1Config,
    dataset: DatasetName,
    **kwargs,
) -> dict[str, Any]:
    fn = _run_lama_item if dataset == "lama" else _run_trivia_item
    return fn(model, tokenizer, item, model_spec, p3_cfg, **kwargs)


def run_phase3b_split(
    model,
    tokenizer,
    run_cfg: ModelRunConfig,
    p3_cfg: Ext1Config,
    model_spec: ModelSpec,
    *,
    dataset: DatasetName,
    split_name: SplitName,
    phase1_dir: pathlib.Path,
    device: str,
) -> pd.DataFrame:
    entropy_labels, static_df = load_phase1_outputs(phase1_dir)
    static_df = normalize_component_names(static_df)
    random_labels = (
        _pick_random_controls(static_df, entropy_labels, p3_cfg.seed)
        if p3_cfg.ablation.include_random_control
        else []
    )

    cal_tok = load_calibration_tokenized(tokenizer, p3_cfg)
    all_neurons = sorted(set(entropy_labels) | set(random_labels))
    print(f"  Computing calibration activations for {len(all_neurons)} neurons...")
    activation_means = compute_activation_means_from_tokenized(
        model, cal_tok, all_neurons, device=device
    )
    mean_map = {n: activation_means[i] for i, n in enumerate(all_neurons)}

    def means_for(names: list[str]) -> torch.Tensor:
        return torch.stack([mean_map[n] for n in names])

    items = get_split_items(p3_cfg, dataset, split_name, _slug(run_cfg.name), tokenizer)
    print(f"  Evaluating {len(items)} {dataset} items ({split_name})...")
    rows: list[dict[str, Any]] = []
    modes = p3_cfg.ablation_modes(split_name)

    for decode_mode in p3_cfg.decoding.modes:
        for item in tqdm.tqdm(
            items,
            desc=f"{dataset}/{split_name}/{decode_mode}",
            leave=False,
        ):
            rows.append(
                _eval_item(
                    model,
                    tokenizer,
                    item,
                    model_spec,
                    p3_cfg,
                    dataset,
                    condition="baseline",
                    branch="none",
                    neuron_names=None,
                    activation_means=None,
                    decode_mode=decode_mode,
                    ablation_mode="mean_all_entropy",
                    component_name="",
                    device=device,
                )
            )

            for ablation_mode in modes:
                for branch in p3_cfg.ablation.branches:
                    if ablation_mode == "mean_all_entropy":
                        sets: list[tuple[str, list[str], str]] = [
                            ("ablate_entropy", entropy_labels, "ALL")
                        ]
                        if random_labels:
                            sets.append(("ablate_random", random_labels, "ALL"))
                        for condition, names, comp in sets:
                            rows.append(
                                _eval_item(
                                    model,
                                    tokenizer,
                                    item,
                                    model_spec,
                                    p3_cfg,
                                    dataset,
                                    condition=condition,
                                    branch=branch,  # type: ignore[arg-type]
                                    neuron_names=names,
                                    activation_means=means_for(names),
                                    decode_mode=decode_mode,
                                    ablation_mode=ablation_mode,
                                    component_name=comp,
                                    device=device,
                                )
                            )
                    elif ablation_mode == "per_neuron_avg":
                        for neuron in entropy_labels:
                            rows.append(
                                _eval_item(
                                    model,
                                    tokenizer,
                                    item,
                                    model_spec,
                                    p3_cfg,
                                    dataset,
                                    condition="ablate_entropy",
                                    branch=branch,  # type: ignore[arg-type]
                                    neuron_names=[neuron],
                                    activation_means=means_for([neuron]),
                                    decode_mode=decode_mode,
                                    ablation_mode=ablation_mode,
                                    component_name=neuron,
                                    device=device,
                                )
                            )

    return _apply_deltas(pd.DataFrame(rows))


def _per_neuron_summary(df: pd.DataFrame, dataset: DatasetName) -> dict[str, Any]:
    sub = df[df["ablation_mode"] == "per_neuron_avg"]
    if sub.empty:
        return {}
    metric_cols = ["acc_at_1", "acc_at_5", "mrr"] if dataset == "lama" else ["em", "f1"]
    avg = (
        sub.groupby(["decode_mode", "ablation_branch"], as_index=False)[metric_cols]
        .mean()
        .to_dict(orient="records")
    )
    return {"ablate_entropy": avg}


def write_split_outputs(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    *,
    run_cfg: ModelRunConfig,
    p3_cfg: Ext1Config,
    dataset: DatasetName,
    split_name: SplitName,
    elapsed_sec: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "detailed.csv", index=False)

    main_df = df[df["ablation_mode"] == "mean_all_entropy"]
    agg = _aggregate_metrics(main_df, dataset)
    summary = {
        "model": run_cfg.name,
        "dataset": dataset,
        "split_name": split_name,
        "n_items": int(main_df["item_id"].nunique()),
        "metrics": agg,
        "deltas_vs_baseline": _deltas_vs_baseline(agg, dataset),
        "per_neuron_avg": _per_neuron_summary(df, dataset),
        "elapsed_sec": elapsed_sec,
        "config_seed": p3_cfg.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_ext1_plots(
        main_df,
        out_dir,
        dataset=dataset,
        model_title=run_cfg.name,
        split_name=split_name,
        full_df=df,
    )


def run_model_dataset(
    model,
    tokenizer,
    run_cfg: ModelRunConfig,
    p3_cfg: Ext1Config,
    model_spec: ModelSpec,
    *,
    dataset: DatasetName,
    phase1_dir: pathlib.Path,
    out_base: pathlib.Path,
    device: str,
    splits: list[SplitName] | None = None,
) -> None:
    splits = splits or ["sanity", "main"]
    for split_name in splits:
        t0 = time.perf_counter()
        print(f"\n{'='*60}\nPhase 3b — {run_cfg.name} / {dataset} / {split_name}\n{'='*60}")
        df = run_phase3b_split(
            model,
            tokenizer,
            run_cfg,
            p3_cfg,
            model_spec,
            dataset=dataset,
            split_name=split_name,
            phase1_dir=phase1_dir,
            device=device,
        )
        out_dir = ext1_dataset_dir(out_base, split_name, dataset)
        elapsed = time.perf_counter() - t0
        write_split_outputs(
            df,
            out_dir,
            run_cfg=run_cfg,
            p3_cfg=p3_cfg,
            dataset=dataset,
            split_name=split_name,
            elapsed_sec=elapsed,
        )
        print(f"  done in {elapsed:.0f}s -> {out_dir.resolve()}")


run_ext1_split = run_phase3b_split
