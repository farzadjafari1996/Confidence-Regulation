"""Runner for Exp 2 static inter-layer weight projection."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable

import pandas as pd
import torch

from exp2.metrics import (
    ReaderSVD,
    alignment_pair_indices,
    all_writer_reader_pairs,
    compute_rho,
    dark_basis,
    expected_rho_isotropic,
    filter_per_neuron_df,
    layer_udark_read_df,
    n_all_pairs,
    pair_metrics_df,
    reader_cache_key,
    summarize_pair_rho,
    svd_reader_basis,
)
from exp2.plots import (
    plot_hop_profile,
    plot_layers_to_wu,
    plot_rho_histograms,
    plot_top_wu_visibility_heatmap,
    plot_write_alignment,
    plot_write_vs_read_udark,
    plot_writer_reader_heatmaps,
    set_save_png,
)
from exp2.sanity import compare_to_phase1, run_unit_checks, write_sanity_report
from exp2.settings import Exp2Config


def _json_float(x: float) -> float:
    return float(x)


def _get_w_out(model, layer: int) -> torch.Tensor:
    """Native weight dtype (fp16/bf16 on CUDA HF models). SVD/ρ still upcast in metrics."""
    if hasattr(model, "get_W_out"):
        return model.get_W_out(layer)
    return model.W_out[layer]


def _get_w_in(model, layer: int) -> torch.Tensor:
    if hasattr(model, "get_W_in"):
        return model.get_W_in(layer)
    return model.W_in[layer]


def _get_w_u(model) -> torch.Tensor:
    return model.W_U


def _pair_tag(writer_layer: int, reader_layer: int | None) -> str:
    if reader_layer is None:
        return f"MLP{writer_layer}->W_U"
    return f"MLP{writer_layer}->MLP{reader_layer}.W_in"


def build_reader_svd_cache(
    model,
    reader_layers: set[int | None],
) -> dict[str, ReaderSVD]:
    """SVD each requested reader once (MLP W_in layers and/or W_U)."""
    cache: dict[str, ReaderSVD] = {}
    for reader_layer in sorted(reader_layers, key=lambda r: (-1 if r is None else r)):
        key = reader_cache_key(reader_layer)
        if reader_layer is None:
            print("  SVD(W_U)...", flush=True)
            basis, svals = svd_reader_basis(_get_w_u(model), "W_U")
            cache[key] = ReaderSVD(
                basis=basis,
                singular_values=svals,
                reader_name="W_U",
                reader_layer=None,
            )
        else:
            print(f"  SVD(W_in[{reader_layer}])...", flush=True)
            basis, svals = svd_reader_basis(_get_w_in(model, reader_layer), "W_in")
            cache[key] = ReaderSVD(
                basis=basis,
                singular_values=svals,
                reader_name="W_in",
                reader_layer=reader_layer,
            )
    return cache


def compute_all_pair_frames(
    model,
    cfg: Exp2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, ReaderSVD]]:
    """Compute ρ for every writer vs later MLP W_in and vs W_U."""
    n_layers = int(model.cfg.n_layers)
    d_model = int(model.cfg.d_model)
    k = cfg.effective_k(d_model)
    n_pairs = n_all_pairs(n_layers)
    print(
        f"  n_layers={n_layers} d_model={d_model} nullspace_k={k} n_pairs={n_pairs}",
        flush=True,
    )

    pairs = all_writer_reader_pairs(n_layers)
    svd_cache = build_reader_svd_cache(model, set(range(n_layers)) | {None})

    frames: list[pd.DataFrame] = []
    w_out_cache: dict[int, torch.Tensor] = {}
    for writer_layer, reader_layer in pairs:
        if writer_layer not in w_out_cache:
            w_out_cache[writer_layer] = _get_w_out(model, writer_layer)
        svd = svd_cache[reader_cache_key(reader_layer)]
        frames.append(
            pair_metrics_df(
                w_out_cache[writer_layer],
                writer_layer,
                svd,
                k,
                n_layers=n_layers,
                d_model=d_model,
            )
        )

    all_df = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(
        [
            summarize_pair_rho(g)
            for _, g in all_df.groupby(["writer_layer", "reader"], sort=False)
        ]
    )
    return all_df, summary_df, svd_cache


def compute_layer_udark_read(
    model,
    svd_cache: dict[str, ReaderSVD],
    k: int,
    d_model: int,
    n_layers: int,
) -> pd.DataFrame:
    """How much each MLP W_in listens to U_dark of W_U."""
    u_dark = dark_basis(svd_cache[reader_cache_key(None)].basis, k)
    w_in_by_layer = {ell: _get_w_in(model, ell) for ell in range(n_layers)}
    return layer_udark_read_df(w_in_by_layer, u_dark, k, d_model)


def compute_top_wu_visibility(
    model,
    neuron_df: pd.DataFrame,
    svd_cache: dict[str, ReaderSVD],
    k: int,
    n_layers: int,
    n_trace: int,
) -> pd.DataFrame:
    """Top-ρ → W_U neurons vs later MLP W_in dark tails (ρ; high = unread)."""
    wu = neuron_df.loc[
        neuron_df["reader"].astype(str).eq("W_U")
        | neuron_df["reader_kind"].astype(str).eq("W_U")
    ]
    rows: list[dict] = []
    for writer_layer in range(n_layers):
        top = wu.loc[wu["writer_layer"] == writer_layer].nlargest(n_trace, "rho")
        if top.empty:
            continue
        idx = top["neuron_index"].to_numpy(dtype=int)
        names = top["component_name"].astype(str).to_numpy()
        rho_wu = top["rho"].to_numpy()
        writes = _get_w_out(model, int(writer_layer))[idx]
        for reader_layer in range(writer_layer + 1, n_layers):
            u_dark = dark_basis(svd_cache[reader_cache_key(reader_layer)].basis, k)
            rho, _ = compute_rho(writes, u_dark)
            rho_np = rho.detach().cpu().numpy()
            for i, name in enumerate(names):
                r = float(rho_np[i])
                rows.append(
                    {
                        "writer_layer": int(writer_layer),
                        "component_name": name,
                        "neuron_index": int(idx[i]),
                        "rho_to_wu": float(rho_wu[i]),
                        "reader_layer": int(reader_layer),
                        "rho_to_reader_dark": r,
                    }
                )
    return pd.DataFrame(rows)


def write_alignment_plots(
    model,
    neuron_df: pd.DataFrame,
    svd_cache: dict[str, ReaderSVD],
    cfg: Exp2Config,
    out_dir: pathlib.Path,
) -> None:
    n_layers = int(model.cfg.n_layers)
    align_dir = out_dir / "alignment"
    align_dir.mkdir(parents=True, exist_ok=True)
    for writer_layer, reader_layer in alignment_pair_indices(n_layers):
        key = reader_cache_key(reader_layer)
        svd = svd_cache[key]
        tag = _pair_tag(writer_layer, reader_layer)
        print(f"  alignment plot: {tag}", flush=True)
        reader = "W_U" if reader_layer is None else f"mlp{reader_layer}.W_in"
        pair_df = neuron_df[
            (neuron_df["writer_layer"] == writer_layer) & (neuron_df["reader"] == reader)
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
    """Run Exp 2 analysis; returns output directory."""
    set_save_png(save_png)
    out_dir = cfg.study_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    from hyperparameters import Family, ModelRunConfig, load_run_settings

    family: Family = cfg.family if cfg.family in ("gpt2", "llama", "qwen", "pythia", "smollm", "gemma") else "gpt2"
    settings = load_run_settings(cfg.config_path, family)
    run_cfg = ModelRunConfig.from_hparams(cfg.model, settings["hparams"], dtype=cfg.dtype)

    print(f"Ext2 weight projection — {cfg.model} on {device}", flush=True)
    model, _tokenizer = load_model_fn(run_cfg, device)
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(False)

    metrics_path = out_dir / "pair_neuron_metrics.csv"
    summary_path = out_dir / "pair_rho_summary.csv"
    read_path = out_dir / "layer_udark_read.csv"
    vis_path = out_dir / "top_wu_visibility.csv"
    n_layers = int(model.cfg.n_layers)
    d_model = int(model.cfg.d_model)
    k = cfg.effective_k(d_model)

    if plots_only:
        if not metrics_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                f"plots-only requires {metrics_path.name} and {summary_path.name} in {out_dir}"
            )
        neuron_df = pd.read_csv(metrics_path)
        summary_df = pd.read_csv(summary_path)
        svd_cache = build_reader_svd_cache(model, set(range(n_layers)) | {None})
        read_df = compute_layer_udark_read(model, svd_cache, k, d_model, n_layers)
        vis_df = compute_top_wu_visibility(
            model, neuron_df, svd_cache, k, n_layers, cfg.n_trace_neurons
        )
        read_df.to_csv(read_path, index=False)
        vis_df.to_csv(vis_path, index=False)
        print(f"  saved {read_path}", flush=True)
        print(f"  saved {vis_path}", flush=True)
    else:
        all_df, summary_df, svd_cache = compute_all_pair_frames(model, cfg)
        neuron_df = filter_per_neuron_df(all_df)
        neuron_df.to_csv(metrics_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        print(f"  saved {metrics_path} ({len(neuron_df)} rows)", flush=True)
        print(f"  saved {summary_path} ({len(summary_df)} pairs)", flush=True)

        top_rows = []
        for (_writer_layer, _reader), g in neuron_df.groupby(["writer_layer", "reader"], sort=False):
            top = g.nlargest(cfg.n_trace_neurons, "rho")[
                ["writer_layer", "reader", "component_name", "neuron_index", "norm", "rho"]
            ]
            top_rows.append(top)
        top_df = pd.concat(top_rows, ignore_index=True)
        top_path = out_dir / "top_rho_per_pair.csv"
        top_df.to_csv(top_path, index=False)
        print(f"  saved {top_path}", flush=True)

        read_df = compute_layer_udark_read(model, svd_cache, k, d_model, n_layers)
        vis_df = compute_top_wu_visibility(
            model, neuron_df, svd_cache, k, n_layers, cfg.n_trace_neurons
        )
        read_df.to_csv(read_path, index=False)
        vis_df.to_csv(vis_path, index=False)
        print(f"  saved {read_path}", flush=True)
        print(f"  saved {vis_path} ({len(vis_df)} rows)", flush=True)

    print("  plots...", flush=True)
    expected_rho = expected_rho_isotropic(k, d_model)
    plot_writer_reader_heatmaps(summary_df, out_dir, cfg.model, expected_rho=expected_rho)
    plot_hop_profile(summary_df, out_dir, cfg.model, expected_rho=expected_rho)
    plot_layers_to_wu(summary_df, out_dir, cfg.model, expected_rho=expected_rho)
    plot_rho_histograms(neuron_df, out_dir, cfg.model, expected_rho=expected_rho)
    plot_write_vs_read_udark(
        summary_df, read_df, out_dir, cfg.model, expected_rho=expected_rho
    )
    plot_top_wu_visibility_heatmap(vis_df, out_dir, cfg.model, n_layers=n_layers)
    write_alignment_plots(model, neuron_df, svd_cache, cfg, out_dir)

    last = n_layers - 1
    last_wu = neuron_df[(neuron_df["writer_layer"] == last) & (neuron_df["reader"] == "W_U")]
    phase1_dir = cfg.study_dir().parent / "exp1" / "phase1"
    sanity_results: list[dict[str, Any]] = run_unit_checks()
    sanity_results.append(compare_to_phase1(last_wu, phase1_dir))
    write_sanity_report(out_dir, sanity_results)

    last_wu_summary = summary_df[
        (summary_df["writer_layer"] == last) & (summary_df["reader"] == "W_U")
    ]
    last_wu_row = last_wu_summary.iloc[0] if len(last_wu_summary) else None

    meta = {
        "model": cfg.model,
        "family": cfg.family,
        "device": device,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_mlp": int(model.cfg.d_mlp),
        "nullspace_top_pct": cfg.nullspace_top_pct,
        "nullspace_k_fixed": cfg.nullspace_k,
        "nullspace_k_effective": k,
        "expected_rho": _json_float(expected_rho),
        "n_all_pairs": n_all_pairs(n_layers),
        "n_mlp_to_mlp_pairs": n_layers * (n_layers - 1) // 2,
        "n_writer_to_W_U": n_layers,
        "includes_every_layer_to_W_U": True,
        "includes_udark_read": True,
        "per_neuron_pairs": "consecutive_mlp_and_all_to_W_U",
        "seed": cfg.seed,
        "layer_udark_read": [
            {
                "layer": int(r.layer),
                "read_rho": _json_float(r.read_rho),
            }
            for r in read_df.itertuples()
        ],
        "last_to_W_U": (
            {
                "rho_mean": _json_float(last_wu_row.rho_mean),
                "rho_p95": _json_float(last_wu_row.rho_p95),
                "rho_max": _json_float(last_wu_row.rho_max),
                "n_rho_gt_0_5": int(last_wu_row.n_rho_gt_0_5),
                "top_neuron": last_wu_row.top_neuron,
            }
            if last_wu_row is not None
            else None
        ),
        "pair_summary": [
            {
                "writer_layer": int(r.writer_layer),
                "reader": r.reader,
                "hop": int(r.hop) if pd.notna(r.hop) else None,
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
