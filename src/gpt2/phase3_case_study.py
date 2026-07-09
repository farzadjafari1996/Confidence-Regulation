"""
GPT-2 — Phase 3: neuron activity & induction case studies (paper Fig 4–5).

Requires Phase 1 outputs in outputs/<model-slug>/phase1/.

Usage:
  python src/gpt2/phase3_case_study.py --device cuda
  python src/gpt2/phase3_case_study.py --plots-only
  python src/gpt2/phase3_case_study.py --section induction

Outputs: outputs/gpt2-small/phase3/
  fig4a_delta_loss_vs_rank.html
  fig4b_activation_trace.html
  fig5a_induction_activations.html
  fig5b_entropy_post_ablation.html
  fig5c_bos_ablation.html
  neuron_activity.feather
  induction_entropy.feather
  induction_clip_ablation.feather
  bos_ablation.feather
  summary.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import neel.utils as nutils
import numpy as np
import pandas as pd
import torch

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from gpt2.core import (  # noqa: E402
    filter_entropy_activation_df,
    get_entropy_activation_df,
    load_phase1_outputs,
    load_tokenized_data,
    mean_ablate_neurons,
)
from gpt2.induction import (  # noqa: E402
    bos_ablate_components,
    build_c4_dup_induction_dataset,
    clip_ablate_neurons,
    compute_weighted_activation_diffs,
    get_induction_data_and_token_df,
)
from gpt2.load import load_gpt2  # noqa: E402
from gpt2.phase3_plots import (  # noqa: E402
    plot_fig4a_delta_loss_vs_rank,
    plot_fig4b_activation_trace,
    plot_fig5a_induction_activations,
    plot_fig5b_entropy_post_ablation,
    plot_fig5c_bos_ablation,
)
from gpt2.phase3_settings import Phase3Config, load_phase3_config  # noqa: E402
from hyperparameters import ModelRunConfig  # noqa: E402
from run_config import add_config_argument, build_model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 — Phase 3 case studies (Fig 4–5)")
    add_config_argument(p)
    p.add_argument("--model", default=None)
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--phase1-dir", type=pathlib.Path, default=None)
    p.add_argument(
        "--section",
        choices=["all", "neuron_activity", "induction"],
        default="all",
        help="Run one section or all",
    )
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--png", action="store_true")
    return p.parse_args()


def _pick_fig4b_batch(entropy_df: pd.DataFrame, neuron_name: str) -> int:
    """Choose a sequence with high loss and strong target-neuron activation."""
    act_col = f"{neuron_name}_activation"
    scored = entropy_df.copy()
    scored["score"] = scored["loss"] * scored[act_col]
    best = scored.loc[scored["score"].idxmax()]
    return int(best["batch"])


def run_neuron_activity(
    cfg: ModelRunConfig,
    p3_cfg: Phase3Config,
    phase1_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    device: str,
) -> dict:
    target = p3_cfg.neuron_activity.target_neuron
    entropy_labels, _ = load_phase1_outputs(phase1_dir)

    model, tokenizer = load_gpt2(cfg, device)
    model.set_use_attn_result(False)
    torch.manual_seed(p3_cfg.seed)
    np.random.seed(p3_cfg.seed)

    tokenized = load_tokenized_data(tokenizer, cfg)
    token_df = nutils.make_token_df(tokenized["tokens"], model=model)
    track = sorted(set(entropy_labels) | {target})
    entropy_df = get_entropy_activation_df(
        track,
        tokenized,
        token_df,
        model,
        batch_size=cfg.batch_size,
        device=device,
        residuals_layer=cfg.last_layer(model),
    )

    ablation_df = mean_ablate_neurons(
        model,
        cfg,
        tokenized,
        entropy_df,
        [target],
        device=device,
        chunk_size=cfg.chunk_size,
    )
    ablation_df = filter_entropy_activation_df(
        ablation_df,
        model_name=cfg.name,
        tokenizer=tokenizer,
        start_pos=cfg.filter_start_pos,
        end_pos=cfg.filter_end_pos,
    )
    ablation_df.reset_index(drop=True).to_feather(out_dir / "neuron_activity.feather")
    entropy_df.reset_index(drop=True).to_feather(out_dir / "neuron_activity_traces.feather")

    plot_fig4a_delta_loss_vs_rank(ablation_df, out_dir, cfg.name, neuron_name=target)
    batch_id = _pick_fig4b_batch(entropy_df, target)
    plot_fig4b_activation_trace(entropy_df, out_dir, cfg.name, neuron_name=target, batch_id=batch_id)

    if device.startswith("cuda"):
        del model
        torch.cuda.empty_cache()

    return {"target_neuron": target, "fig4b_batch": batch_id, "n_tokens": len(ablation_df)}


def run_induction(
    cfg: ModelRunConfig,
    p3_cfg: Phase3Config,
    phase1_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    device: str,
) -> dict:
    ind = p3_cfg.induction
    entropy_labels, _ = load_phase1_outputs(phase1_dir)
    trace_neurons = list(dict.fromkeys(ind.trace_neurons))

    model, tokenizer = load_gpt2(cfg, device)
    model.set_use_attn_result(False)
    torch.manual_seed(p3_cfg.seed)
    np.random.seed(p3_cfg.seed)

    tokenized = load_tokenized_data(tokenizer, cfg)
    induction_ds = build_c4_dup_induction_dataset(
        tokenized,
        num_sequences=ind.num_sequences,
        seq_length=ind.seq_length,
        seed=p3_cfg.seed,
    )
    induction_token_df = nutils.make_token_df(induction_ds["tokens"], model=model)
    induction_start = ind.seq_length

    entropy_df = get_entropy_activation_df(
        trace_neurons,
        induction_ds,
        induction_token_df,
        model,
        batch_size=cfg.batch_size,
        device=device,
        residuals_layer=cfg.last_layer(model),
    )
    entropy_df.reset_index(drop=True).to_feather(out_dir / "induction_entropy.feather")

    agg_cols = ["entropy", "loss"] + [f"{n}_activation" for n in trace_neurons]
    agg5a = entropy_df.groupby("pos", as_index=False)[agg_cols].mean()
    plot_fig5a_induction_activations(
        agg5a,
        out_dir,
        cfg.name,
        entropy_neurons=trace_neurons,
        induction_start=induction_start,
    )

    weighted_diffs = compute_weighted_activation_diffs(
        entropy_df, model, trace_neurons, half_pos=induction_start
    )
    clip_df = clip_ablate_neurons(
        trace_neurons,
        induction_ds,
        entropy_df,
        model,
        weighted_diffs,
        k=ind.ablation_k,
        device=device,
        seed=p3_cfg.seed,
    )
    clip_df["entropy_ratio"] = clip_df["entropy_post_ablation"] / clip_df["entropy"].replace(0, np.nan)
    clip_df.reset_index(drop=True).to_feather(out_dir / "induction_clip_ablation.feather")

    agg5b = (
        clip_df.groupby(["pos", "component_name"], as_index=False)
        .agg(entropy_ratio=("entropy_ratio", "mean"))
    )
    plot_fig5b_entropy_post_ablation(
        agg5b,
        out_dir,
        cfg.name,
        entropy_neurons=trace_neurons,
        induction_start=induction_start,
    )

    bos_ds, bos_token_df = get_induction_data_and_token_df(
        model,
        tokenizer,
        lambda tokens, m=model: nutils.make_token_df(tokens, model=m),
        seq_length=ind.bos_seq_length,
        num_examples=ind.bos_num_examples,
        seed=p3_cfg.seed,
        device=device,
        use_natural_text=True,
    )
    bos_track = list(dict.fromkeys(trace_neurons + [ind.bos_target_neuron]))
    bos_entropy_df = get_entropy_activation_df(
        bos_track,
        bos_ds,
        bos_token_df,
        model,
        batch_size=cfg.batch_size,
        device=device,
        residuals_layer=cfg.last_layer(model),
    )
    bos_half = ind.bos_seq_length + 1
    bos_entropy_df = bos_entropy_df[bos_entropy_df["pos"] >= bos_half].copy()

    induction_components = ind.induction_heads[:3] + ind.joint_induction_heads
    baseline_combos = [[h] for h in ind.bos_baselines]
    bos_components = [[h] for h in induction_components] + baseline_combos

    bos_df = bos_ablate_components(
        bos_components,
        bos_ds,
        bos_entropy_df,
        model,
        bos_track,
        device=device,
    )
    bos_df.reset_index(drop=True).to_feather(out_dir / "bos_ablation.feather")

    agg5c = bos_df.groupby(["pos", "component_name"], as_index=False).mean(numeric_only=True)
    plot_fig5c_bos_ablation(
        agg5c,
        out_dir,
        cfg.name,
        target_neuron=ind.bos_target_neuron,
        induction_start=bos_half,
        induction_components=induction_components,
    )

    if device.startswith("cuda"):
        del model
        torch.cuda.empty_cache()

    return {
        "num_sequences": ind.num_sequences,
        "seq_length": ind.seq_length,
        "trace_neurons": trace_neurons,
        "induction_heads": induction_components,
    }


def _plots_only(cfg: ModelRunConfig, p3_cfg: Phase3Config, out_dir: pathlib.Path) -> None:
    target = p3_cfg.neuron_activity.target_neuron
    ind = p3_cfg.induction

    na_path = out_dir / "neuron_activity.feather"
    traces_path = out_dir / "neuron_activity_traces.feather"
    if na_path.exists():
        ablation_df = pd.read_feather(na_path)
        plot_fig4a_delta_loss_vs_rank(ablation_df, out_dir, cfg.name, neuron_name=target)
    if traces_path.exists():
        entropy_df = pd.read_feather(traces_path)
        batch_id = _pick_fig4b_batch(entropy_df, target)
        plot_fig4b_activation_trace(
            entropy_df, out_dir, cfg.name, neuron_name=target, batch_id=batch_id
        )

    if (out_dir / "induction_entropy.feather").exists():
        entropy_df = pd.read_feather(out_dir / "induction_entropy.feather")
        trace = list(dict.fromkeys(ind.trace_neurons))
        agg5a = entropy_df.groupby("pos", as_index=False)[
            ["entropy", "loss"] + [f"{n}_activation" for n in trace if f"{n}_activation" in entropy_df.columns]
        ].mean()
        plot_fig5a_induction_activations(
            agg5a, out_dir, cfg.name, entropy_neurons=trace, induction_start=ind.seq_length
        )

    if (out_dir / "induction_clip_ablation.feather").exists():
        clip_df = pd.read_feather(out_dir / "induction_clip_ablation.feather")
        if "entropy_ratio" not in clip_df.columns:
            clip_df["entropy_ratio"] = clip_df["entropy_post_ablation"] / clip_df["entropy"].replace(
                0, np.nan
            )
        agg5b = clip_df.groupby(["pos", "component_name"], as_index=False).agg(
            entropy_ratio=("entropy_ratio", "mean")
        )
        plot_fig5b_entropy_post_ablation(
            agg5b,
            out_dir,
            cfg.name,
            entropy_neurons=list(dict.fromkeys(ind.trace_neurons)),
            induction_start=ind.seq_length,
        )

    if (out_dir / "bos_ablation.feather").exists():
        bos_df = pd.read_feather(out_dir / "bos_ablation.feather")
        agg5c = bos_df.groupby(["pos", "component_name"], as_index=False).mean(numeric_only=True)
        plot_fig5c_bos_ablation(
            agg5c,
            out_dir,
            cfg.name,
            target_neuron=ind.bos_target_neuron,
            induction_start=ind.bos_seq_length + 1,
            induction_components=ind.induction_heads[:3] + ind.joint_induction_heads,
        )


def main() -> None:
    import gpt2.core as core

    args = parse_args()
    core._SAVE_PNG = args.png

    cfg, out_base = build_model_config(args, "gpt2")
    p3_cfg = load_phase3_config(args.config)
    phase1_dir = args.phase1_dir or cfg.output_dir(out_base) / "phase1"
    out_dir = args.out_dir or cfg.output_dir(out_base) / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        _plots_only(cfg, p3_cfg, out_dir)
        print(f"Phase 3 plots regenerated -> {out_dir.resolve()}")
        return

    t0 = time.perf_counter()
    summary: dict = {"model": cfg.name, "phase": 3, "sections": {}}

    if args.section in ("all", "neuron_activity"):
        print(f"\n{'='*60}\nPhase 3 — neuron activity (Fig 4)\n{'='*60}")
        summary["sections"]["neuron_activity"] = run_neuron_activity(
            cfg, p3_cfg, phase1_dir, out_dir, device=args.device
        )

    if args.section in ("all", "induction"):
        print(f"\n{'='*60}\nPhase 3 — induction (Fig 5)\n{'='*60}")
        summary["sections"]["induction"] = run_induction(
            cfg, p3_cfg, phase1_dir, out_dir, device=args.device
        )

    summary["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved {summary_path}")
    print(f"Phase 3 done in {summary['elapsed_sec']:.0f}s -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
