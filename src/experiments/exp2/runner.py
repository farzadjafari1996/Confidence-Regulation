"""Runner for Ext 2 static inter-layer weight projection."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable

import pandas as pd
import torch

from exp2.metrics import (
    ReaderSVD,
    alignment_pair_indices,
    mlp_to_mlp_pairs,
    pair_metrics_df,
    summarize_pair_rho,
    svd_reader_basis,
)
from exp2.plots import (
    plot_rho_heatmap,
    plot_rho_histograms,
    plot_write_alignment,
    set_save_png,
)
from exp2.sanity import compare_to_phase1, run_unit_checks, write_sanity_report
from exp2.settings import Exp2Config


def _json_float(x: float) -> float:
    return float(x)


def _get_w_out(model, layer: int) -> torch.Tensor:
    if hasattr(model, "get_W_out"):
        return model.get_W_out(layer).float()
    return model.W_out[layer].float()


def _get_w_in(model, layer: int) -> torch.Tensor:
    if hasattr(model, "get_W_in"):
        return model.get_W_in(layer).float()
    return model.W_in[layer].float()


def _get_w_u(model) -> torch.Tensor:
    return model.W_U.float()


def compute_all_pair_frames(
    model,
    cfg: Exp2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, ReaderSVD]]:
    """Compute ρ for all MLP→MLP pairs and last→W_U."""
    n_layers = int(model.cfg.n_layers)
    d_model = int(model.cfg.d_model)
    k = cfg.effective_k(d_model)
    print(f"  n_layers={n_layers} d_model={d_model} nullspace_k={k}", flush=True)

    frames: list[pd.DataFrame] = []
    svd_cache: dict[str, ReaderSVD] = {}

    for writer_layer, reader_layer in mlp_to_mlp_pairs(n_layers):
        key = f"mlp{writer_layer}_to_mlp{reader_layer}"
        print(f"  pair {key}: SVD(W_in[{reader_layer}])...", flush=True)
        basis, svals = svd_reader_basis(_get_w_in(model, reader_layer), "W_in")
        svd = ReaderSVD(
            basis=basis,
            singular_values=svals,
            reader_name="W_in",
            writer_layer=writer_layer,
            reader_layer=reader_layer,
        )
        svd_cache[key] = svd
        frames.append(pair_metrics_df(_get_w_out(model, writer_layer), writer_layer, svd, k))

    last = n_layers - 1
    key = f"mlp{last}_to_W_U"
    print(f"  pair {key}: SVD(W_U)...", flush=True)
    basis, svals = svd_reader_basis(_get_w_u(model), "W_U")
    svd = ReaderSVD(
        basis=basis,
        singular_values=svals,
        reader_name="W_U",
        writer_layer=last,
        reader_layer=None,
    )
    svd_cache[key] = svd
    frames.append(pair_metrics_df(_get_w_out(model, last), last, svd, k))

    all_df = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame([summarize_pair_rho(g) for _, g in all_df.groupby(
        ["writer_layer", "reader"], sort=False
    )])
    return all_df, summary_df, svd_cache


def _pair_tag(writer_layer: int, reader_layer: int | None) -> str:
    if reader_layer is None:
        return f"MLP{writer_layer}->W_U"
    return f"MLP{writer_layer}->MLP{reader_layer}.W_in"


def _cache_key(writer_layer: int, reader_layer: int | None) -> str:
    if reader_layer is None:
        return f"mlp{writer_layer}_to_W_U"
    return f"mlp{writer_layer}_to_mlp{reader_layer}"


def write_alignment_plots(
    model,
    all_df: pd.DataFrame,
    svd_cache: dict[str, ReaderSVD],
    cfg: Exp2Config,
    out_dir: pathlib.Path,
) -> None:
    n_layers = int(model.cfg.n_layers)
    align_dir = out_dir / "alignment"
    align_dir.mkdir(parents=True, exist_ok=True)
    for writer_layer, reader_layer in alignment_pair_indices(n_layers):
        key = _cache_key(writer_layer, reader_layer)
        svd = svd_cache[key]
        tag = _pair_tag(writer_layer, reader_layer)
        print(f"  alignment plot: {tag}", flush=True)
        reader_label = "W_U" if reader_layer is None else f"mlp{reader_layer}.W_in"
        pair_df = all_df[
            (all_df["writer_layer"] == writer_layer) & (all_df["reader"] == reader_label)
        ]
        plot_write_alignment(
            w_out=_get_w_out(model, writer_layer),
            basis=svd.basis,
            singular_values=svd.singular_values,
            pair_df=pair_df,
            out_dir=align_dir,
            model_title=cfg.model,
            pair_tag=tag,
            n_trace=cfg.n_trace_neurons,
            n_random=cfg.n_align_random,
            seed=cfg.seed,
        )


def run_exp2(
    cfg: Exp2Config,
    *,
    device: str,
    load_model_fn: Callable,
    save_png: bool = False,
    plots_only: bool = False,
) -> pathlib.Path:
    """Run Ext 2 analysis; returns output directory."""
    set_save_png(save_png)
    out_dir = cfg.study_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    from hyperparameters import Family, ModelRunConfig, load_run_settings

    family: Family = cfg.family if cfg.family in ("gpt2", "llama", "qwen") else "gpt2"
    settings = load_run_settings(cfg.config_path, family)
    run_cfg = ModelRunConfig.from_hparams(cfg.model, settings["hparams"], dtype=cfg.dtype)

    print(f"Ext2 weight projection — {cfg.model} on {device}", flush=True)
    model, _tokenizer = load_model_fn(run_cfg, device)
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(False)

    metrics_path = out_dir / "pair_neuron_metrics.csv"
    summary_path = out_dir / "pair_rho_summary.csv"

    if plots_only:
        if not metrics_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                f"plots-only requires {metrics_path.name} and {summary_path.name} in {out_dir}"
            )
        all_df = pd.read_csv(metrics_path)
        summary_df = pd.read_csv(summary_path)
        # Recompute SVDs for alignment plots only (cheap).
        _, _, svd_cache = compute_all_pair_frames(model, cfg)
    else:
        all_df, summary_df, svd_cache = compute_all_pair_frames(model, cfg)
        all_df.to_csv(metrics_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        print(f"  saved {metrics_path}", flush=True)
        print(f"  saved {summary_path}", flush=True)

        # Top-ρ neurons per pair
        top_rows = []
        for (writer_layer, reader), g in all_df.groupby(["writer_layer", "reader"], sort=False):
            top = g.nlargest(cfg.n_trace_neurons, "rho")[
                ["writer_layer", "reader", "component_name", "neuron_index", "norm", "rho"]
            ]
            top_rows.append(top)
        top_df = pd.concat(top_rows, ignore_index=True)
        top_path = out_dir / "top_rho_per_pair.csv"
        top_df.to_csv(top_path, index=False)
        print(f"  saved {top_path}", flush=True)

    print("  plots...", flush=True)
    plot_rho_heatmap(summary_df, out_dir, cfg.model)
    plot_rho_histograms(all_df, out_dir, cfg.model)
    write_alignment_plots(model, all_df, svd_cache, cfg, out_dir)

    # Sanity vs Phase 1
    last = int(model.cfg.n_layers) - 1
    last_wu = all_df[(all_df["writer_layer"] == last) & (all_df["reader"] == "W_U")]
    phase1_dir = cfg.study_dir().parent / "phase1"
    sanity_results: list[dict[str, Any]] = run_unit_checks()
    sanity_results.append(compare_to_phase1(last_wu, phase1_dir))
    write_sanity_report(out_dir, sanity_results)

    meta = {
        "model": cfg.model,
        "family": cfg.family,
        "device": device,
        "n_layers": int(model.cfg.n_layers),
        "d_model": int(model.cfg.d_model),
        "d_mlp": int(model.cfg.d_mlp),
        "nullspace_top_pct": cfg.nullspace_top_pct,
        "nullspace_k_fixed": cfg.nullspace_k,
        "nullspace_k_effective": cfg.effective_k(int(model.cfg.d_model)),
        "n_mlp_to_mlp_pairs": len(mlp_to_mlp_pairs(int(model.cfg.n_layers))),
        "includes_last_to_W_U": True,
        "seed": cfg.seed,
        "pair_summary": [
            {
                "writer_layer": int(r.writer_layer),
                "reader": r.reader,
                "rho_mean": _json_float(r.rho_mean),
                "rho_p95": _json_float(r.rho_p95),
                "rho_max": _json_float(r.rho_max),
                "n_rho_gt_0_5": int(r.n_rho_gt_0_5),
                "top_neuron": r.top_neuron,
            }
            for r in summary_df.itertuples()
        ],
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  saved {summary_json}", flush=True)
    print(f"Ext2 done -> {out_dir}", flush=True)
    return out_dir
