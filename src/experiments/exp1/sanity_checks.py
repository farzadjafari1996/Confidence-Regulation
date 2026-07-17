"""Sanity checks for Exp 1 QA modules; writes minimal plots to exp1_sanity."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable

import pandas as pd
import torch

from hyperparameters import ROOT
from exp1.data_loaders import sample_items
from exp1.fixtures import make_synthetic_detailed_df
from exp1.metrics import (
    exact_match_score,
    f1_score,
    lama_rank_metrics,
    normalize_answer,
    sample_token_from_logits,
    trivia_em_f1,
)
from exp1.plots import write_exp1_plots
from exp1.prompts import build_lama_prompt, build_trivia_prompt, lama_cloze_prompt
from exp1.runner import _aggregate_metrics, _deltas_vs_baseline
from exp1.settings import Exp1Config, ModelSpec, exp1_dataset_dir, load_exp1_config


def check_settings(config_path: pathlib.Path, cfg: Exp1Config) -> dict[str, Any]:
    spec = cfg.model_spec()
    return {
        "check": "settings",
        "config_path": str(config_path),
        "model": spec.model,
        "family": spec.family,
        "use_chat_template": spec.use_chat_template,
        "sanity_size": cfg.sample_size("sanity"),
        "main_size": cfg.sample_size("main"),
        "decoding_modes": cfg.decoding.modes,
        "passed": spec.model is not None and cfg.sample_size("sanity") > 0,
    }


def check_metrics() -> dict[str, Any]:
    assert normalize_answer("The U.S.A.") == "usa"
    em, f1 = trivia_em_f1("France", ["France", "france", "French Republic"])
    assert em == 1.0 and f1 == 1.0

    logits = torch.zeros(100)
    logits[7] = 10.0
    lama = lama_rank_metrics(7, logits)
    assert lama["acc_at_1"] == 1.0 and lama["mrr"] == 1.0

    sampled = sample_token_from_logits(logits, temperature=1.0, top_p=1.0)
    assert 0 <= sampled < 100

    return {
        "check": "metrics",
        "passed": True,
    }


def check_prompts(cfg: Exp1Config) -> dict[str, Any]:
    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return f"<chat>{messages[0]['content']}</chat>"

    base_spec = ModelSpec(family="gpt2", model="gpt2-small", use_chat_template=False)
    instruct_spec = ModelSpec(
        family="llama", model="meta-llama/Llama-3.2-1B-Instruct", use_chat_template=True
    )
    masked = "Paris is the capital of [MASK]."
    cloze = lama_cloze_prompt(masked)
    plain = build_lama_prompt(masked, base_spec, _Tok())
    chat = build_lama_prompt(masked, instruct_spec, _Tok())
    trivia_plain = build_trivia_prompt("What is 2+2?", base_spec, _Tok(), cfg.triviaqa)
    trivia_chat = build_trivia_prompt("What is 2+2?", instruct_spec, _Tok(), cfg.triviaqa)

    return {
        "check": "prompts",
        "passed": cloze in plain and "<chat>" in chat and "Question:" in trivia_plain,
    }


def check_datasets() -> dict[str, Any]:
    pool = [
        {
            "item_id": f"id-{i}",
            "masked_sentence": f"Subject {i} was born in [MASK].",
            "gold_answer": "Paris",
            "gold_aliases": ["Paris"],
        }
        for i in range(50)
    ]

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return [100] if text == "Paris" else [1, 2]

    splits = sample_items(pool, n_sanity=5, n_main=10, seed=42, tokenizer=_Tok(), filter_single_token=True)
    return {
        "check": "datasets",
        "sanity_n": len(splits["sanity"]),
        "main_n": len(splits["main"]),
        "passed": len(splits["sanity"]) == 5 and len(splits["main"]) == 10,
    }


def check_ablation_engine() -> dict[str, Any]:
    from exp1.ablation_engine import entropy_and_top1, logits_at_position

    class _Cfg:
        n_layers = 2
        d_mlp = 4
        d_model = 8
        d_vocab = 16
        normalization_type = "LN"

    class _FakeModel:
        cfg = _Cfg()

        def ln_final(self, x):
            return x

        @property
        def W_out(self):
            w = torch.zeros(2, 4, 8)
            w[1] = torch.eye(4, 8)[:4]
            return w

        @property
        def W_U(self):
            return torch.eye(8, 16)

        @property
        def b_U(self):
            return torch.zeros(16)

    model = _FakeModel()
    layer = 1
    cache = {
        "resid_post": torch.randn(1, 3, 8),
        f"blocks.{layer}.mlp.hook_post": torch.randn(1, 3, 4),
        "ln_final.hook_scale": torch.ones(1, 3),
    }
    from transformer_lens import utils as tl_utils

    cache[tl_utils.get_act_name("resid_post", layer)] = cache["resid_post"]
    cache[tl_utils.get_act_name("post", layer)] = cache[f"blocks.{layer}.mlp.hook_post"]

    means = torch.zeros(2)
    logits_te = logits_at_position(model, cache, 1, [0, 1], means, "te", device="cpu")
    logits_de = logits_at_position(model, cache, 1, [0, 1], means, "de_ln", device="cpu")
    ent, top1 = entropy_and_top1(logits_te)

    return {
        "check": "ablation_engine",
        "passed": logits_te.shape == (16,) and ent >= 0 and not torch.allclose(logits_te, logits_de),
    }


def _write_sanity_plots(model_name: str, out_dir: pathlib.Path) -> dict[str, Any]:
    """Write one set of synthetic plots per dataset under exp1_sanity."""
    plot_paths: list[str] = []
    for dataset in ("lama", "triviaqa"):
        df = make_synthetic_detailed_df(dataset, model_title=model_name)
        main_df = df[df["ablation_mode"] == "mean_all_entropy"]
        ds_out = exp1_dataset_dir(out_dir, "sanity", dataset)  # type: ignore[arg-type]
        write_exp1_plots(
            main_df,
            ds_out,
            dataset=dataset,
            model_title=model_name,
            split_name="sanity",
        )
        plot_paths.append(str(ds_out / "metrics_by_condition.html"))

    return {
        "check": "plots",
        "plot_paths": plot_paths,
        "passed": all(pathlib.Path(p).exists() for p in plot_paths),
    }


def check_runner_aggregation(model_name: str) -> dict[str, Any]:
    results: dict[str, Any] = {"check": "runner", "datasets": {}}
    for dataset in ("lama", "triviaqa"):
        df = make_synthetic_detailed_df(dataset, model_title=model_name)
        main_df = df[df["ablation_mode"] == "mean_all_entropy"]
        agg = _aggregate_metrics(main_df, dataset)  # type: ignore[arg-type]
        deltas = _deltas_vs_baseline(agg, dataset)  # type: ignore[arg-type]
        results["datasets"][dataset] = {"has_metrics": bool(agg), "has_deltas": bool(deltas)}

    results["passed"] = all(d["has_metrics"] for d in results["datasets"].values())
    return results


CHECKS: list[tuple[str, Callable[..., dict[str, Any]]]] = [
    ("settings", lambda cfg, path, _: check_settings(path, cfg)),
    ("metrics", lambda cfg, path, _: check_metrics()),
    ("prompts", lambda cfg, path, _: check_prompts(cfg)),
    ("datasets", lambda cfg, path, _: check_datasets()),
    ("ablation_engine", lambda cfg, path, _: check_ablation_engine()),
    ("runner", lambda cfg, path, model: check_runner_aggregation(model)),
    ("plots", lambda cfg, path, model: _write_sanity_plots(model, cfg.out_dir)),
]


def run_all_sanity_checks(
    config_path: pathlib.Path,
    *,
    model_name: str | None = None,
    out_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    cfg = load_exp1_config(config_path)
    spec = cfg.model_spec()
    model = model_name or spec.model
    out_root = out_root or cfg.out_dir

    print(f"\n{'='*60}\nPhase 3 sanity checks — {model}\n{'='*60}")
    summary: dict[str, Any] = {"model": model, "config": str(config_path), "checks": {}}
    all_passed = True

    for name, fn in CHECKS:
        print(f"  running {name}...")
        try:
            result = fn(cfg, config_path, model)
            summary["checks"][name] = result
            all_passed = all_passed and bool(result.get("passed", False))
        except Exception as exc:
            summary["checks"][name] = {"passed": False, "error": str(exc)}
            all_passed = False
            print(f"  [FAIL] {name}: {exc}")

    summary["all_passed"] = all_passed
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "exp1_sanity" / "checks_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSanity checks {'PASSED' if all_passed else 'FAILED'} -> {out_root.resolve()}")
    return summary
