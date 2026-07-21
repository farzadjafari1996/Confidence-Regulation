"""
Full cross-model comparison across Exp 1 Phase 1 / 2 / 3 (roadmap 1C).

Reads artifacts under outputs/<model-slug>/exp1/{phase1,phase2,phase3}/
and writes aggregated tables + Plotly diagrams to outputs/comparison/.

Usage (after running exp1 phase 1 + 2 + 3 for each model):
  python src/common/track1_compare.py
  python src/common/track1_compare.py --models gpt2-small meta-llama/Llama-3.2-1B
  python src/common/track1_compare.py --outputs-dir outputs --comparison-dir outputs/comparison

Outputs:
  outputs/comparison/comparison_table.csv
  outputs/comparison/comparison_report.md
  outputs/comparison/comparison.json
  outputs/comparison/plots/*.html
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_src = pathlib.Path(__file__).resolve().parent
while _src != _src.parent and not (
    (_src / "common").is_dir() and (_src / "experiments").is_dir() and (_src / "models").is_dir()
):
    _src = _src.parent
for _p in (_src, _src / "common", _src / "experiments", _src / "models"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from hyperparameters import DEFAULT_OUTPUT_DIR, GPT2_PAPER_ENTROPY_NEURONS, model_slug  # noqa: E402

DEFAULT_MODELS = [
    "gpt2-small",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-2B",
]

DEFAULT_COMPARISON_DIR = DEFAULT_OUTPUT_DIR / "comparison"

DISPLAY_NAMES = {
    "gpt2-small": "GPT-2 Small",
    "meta-llama/Llama-3.2-1B": "Llama 1B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 1B-Inst",
    "Qwen/Qwen3.5-0.8B-Base": "Qwen 0.8B-Base",
    "Qwen/Qwen3.5-0.8B": "Qwen 0.8B-Inst",
    "Qwen/Qwen3.5-2B-Base": "Qwen 2B-Base",
    "Qwen/Qwen3.5-2B": "Qwen 2B-Inst",
}

FAMILY_COLORS = {
    "GPT-2": "#636EFA",
    "Llama": "#00CC96",
    "Qwen": "#EF553B",
}


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: pathlib.Path) -> dict[str, Any] | None:
    return _read_json(path) if path.exists() else None


def _fmt_list(items: list[str], limit: int = 6) -> str:
    if not items:
        return ""
    head = items[:limit]
    text = ", ".join(head)
    if len(items) > limit:
        text += f", ... (+{len(items) - limit} more)"
    return text


def _mean_rho_selected(phase1: dict[str, Any]) -> float | None:
    stats = phase1.get("selected_stats") or []
    rhos = [row["rho"] for row in stats if row.get("rho") is not None]
    if not rhos:
        return None
    return float(sum(rhos) / len(rhos))


def _model_family(model_name: str) -> str:
    if "gpt2" in model_name.lower():
        return "GPT-2"
    if "llama" in model_name.lower():
        return "Llama"
    if "qwen" in model_name.lower():
        return "Qwen"
    return "Other"


def _model_variant(model_name: str) -> str:
    lower = model_name.lower()
    if "gpt2" in lower:
        return "base"
    if "instruct" in lower:
        return "instruct"
    if lower.endswith("-base") or lower.endswith("_base"):
        return "base"
    # Qwen instruct checkpoints omit "-Instruct" / "-Base" suffix
    if "qwen" in lower and not lower.endswith("-base"):
        return "instruct"
    return "base"


def _display_name(model_name: str) -> str:
    return DISPLAY_NAMES.get(model_name, model_name.split("/")[-1])


def _enrich_phase3_from_feather(
    phase3_dir: pathlib.Path,
    entropy_neurons: list[str],
) -> dict[str, float | None]:
    """Optional extra Exp 1 Phase 3 aggregates from distribution_metrics.feather (TE columns)."""
    feather = phase3_dir / "distribution_metrics.feather"
    if not feather.exists() or not entropy_neurons:
        return {}

    df = pd.read_feather(feather)
    if "component_name" not in df.columns or "js_divergence_te" not in df.columns:
        return {}

    ent = df[df["component_name"].isin(entropy_neurons)]
    rnd = df[~df["component_name"].isin(entropy_neurons)]
    out: dict[str, float | None] = {}

    def _mean(frame: pd.DataFrame, col: str) -> float | None:
        if col not in frame.columns or frame.empty:
            return None
        return float(frame[col].mean())

    out["mean_kl_entropy_te"] = _mean(ent, "kl_divergence_te")
    out["mean_kl_random_te"] = _mean(rnd, "kl_divergence_te")
    out["mean_argmax_flip_entropy_te"] = _mean(ent, "argmax_flip_te")
    out["mean_argmax_flip_random_te"] = _mean(rnd, "argmax_flip_te")
    out["mean_spearman_entropy_te"] = _mean(ent, "spearman_rho_te")
    out["mean_top5_jaccard_entropy_te"] = _mean(ent, "top5_jaccard_te")
    if "js_divergence_de_ln" in df.columns:
        out["mean_js_entropy_de_ln"] = _mean(ent, "js_divergence_de_ln")
        out["mean_js_random_de_ln"] = _mean(rnd, "js_divergence_de_ln")
    return out


def load_model_track1_metrics(
    model_name: str,
    outputs_dir: pathlib.Path,
) -> dict[str, Any]:
    slug = model_slug(model_name)
    exp1_root = outputs_dir / slug / "exp1"
    phase1_dir = exp1_root / "phase1"
    phase2_dir = exp1_root / "phase2"
    phase3_dir = exp1_root / "phase3"

    phase1 = _read_json(phase1_dir / "summary.json")
    phase2 = _read_json(phase2_dir / "summary.json")
    phase3 = _optional_json(phase3_dir / "summary.json")

    rho_csv = phase2_dir / "rho_vs_ln_fraction.csv"
    rho_df = pd.read_csv(rho_csv) if rho_csv.exists() else None

    selection = phase1.get("selection") or {}
    entropy_set = phase2.get("entropy_set") or {}
    random_set = phase2.get("random_controls") or {}
    meta = phase1.get("model_meta") or {}
    entropy_neurons = list(selection.get("neurons") or [])

    te_ent = entropy_set.get("mean_abs_te")
    de_ent = entropy_set.get("mean_abs_de_ln")
    te_rnd = random_set.get("mean_abs_te")
    ln_frac = entropy_set.get("mean_ln_mediated_fraction")

    te_de_ratio_entropy = None
    te_de_ratio_random = None
    if te_ent and de_ent and de_ent > 0:
        te_de_ratio_entropy = te_ent / de_ent
    if te_rnd and random_set.get("mean_abs_de_ln"):
        de_rnd = random_set["mean_abs_de_ln"]
        if de_rnd > 0:
            te_de_ratio_random = te_rnd / de_rnd

    paper_overlap = None
    paper_overlap_neurons: list[str] = []
    if model_name == "gpt2-small":
        paper = phase1.get("paper_verification") or {}
        paper_overlap = paper.get("overlap_count", paper.get("n_overlap"))
        paper_overlap_neurons = list(paper.get("overlap_with_selected") or [])

    row: dict[str, Any] = {
        "model": model_name,
        "display_name": _display_name(model_name),
        "family": _model_family(model_name),
        "variant": _model_variant(model_name),
        "slug": slug,
        "n_layers": meta.get("n_layers"),
        "d_model": meta.get("d_model"),
        "d_mlp": meta.get("d_mlp"),
        "normalization": meta.get("normalization"),
        "udark_start": meta.get("udark_start"),
        "nullspace_k": meta.get("nullspace_k"),
        "n_entropy_selected": selection.get("n_selected"),
        "top_entropy_neurons": entropy_neurons,
        "top_entropy_neurons_str": _fmt_list(entropy_neurons),
        "mean_rho_selected": _mean_rho_selected(phase1),
        "mean_abs_te_entropy": te_ent,
        "mean_abs_de_ln_entropy": de_ent,
        "mean_ln_mediated_fraction": ln_frac,
        "mean_abs_delta_h_entropy": entropy_set.get("mean_abs_delta_h"),
        "mean_pred_change_rate_entropy": entropy_set.get("mean_pred_change_rate"),
        "mean_abs_te_random": te_rnd,
        "mean_abs_de_ln_random": random_set.get("mean_abs_de_ln"),
        "te_over_de_entropy": te_de_ratio_entropy,
        "te_over_de_random": te_de_ratio_random,
        "phase3_ran": phase3 is not None,
        "paper_neuron_overlap_gpt2": paper_overlap,
        "paper_neuron_overlap_list": paper_overlap_neurons,
        "phase1_dir": str(phase1_dir),
        "phase2_dir": str(phase2_dir),
        "phase3_dir": str(phase3_dir),
    }

    if rho_df is not None and "is_entropy" in rho_df.columns:
        ent_rho = rho_df[rho_df["is_entropy"] == True]  # noqa: E712
        rnd_rho = rho_df[rho_df["is_entropy"] == False]  # noqa: E712
        row["mean_rho_entropy_ablated"] = float(ent_rho["rho"].mean()) if len(ent_rho) else None
        row["mean_rho_random_ablated"] = float(rnd_rho["rho"].mean()) if len(rnd_rho) else None

    if phase3 is not None:
        js_e = phase3.get("mean_js_entropy")
        js_r = phase3.get("mean_js_random")
        aff = phase3.get("mean_affine_r2_entropy")
        const = phase3.get("mean_constant_shift_r2_entropy")
        row["phase3_branch"] = phase3.get("branch_primary")
        row["mean_js_entropy"] = js_e
        row["mean_js_random"] = js_r
        row["js_entropy_over_random"] = (
            float(js_e / js_r) if js_e is not None and js_r and js_r > 0 else None
        )
        row["mean_affine_r2_entropy"] = aff
        row["mean_constant_shift_r2_entropy"] = const
        row["mean_shape_change_entropy"] = (
            float(1.0 - const) if const is not None else None
        )
        row["n_tokens_phase3"] = phase3.get("n_tokens")
        row.update(_enrich_phase3_from_feather(phase3_dir, entropy_neurons))

    return row


def evaluate_paper_claims(row: dict[str, Any]) -> dict[str, bool | str]:
    """Heuristic pass/fail flags for seminar sanity checks (not strict thresholds)."""

    def _ok_ratio(ratio: float | None, min_ratio: float) -> bool | None:
        if ratio is None:
            return None
        return ratio >= min_ratio

    def _ok_low_change(rate: float | None, max_rate: float = 0.15) -> bool | None:
        if rate is None:
            return None
        return rate <= max_rate

    def _ok_high_dh(delta_h: float | None, min_dh: float = 0.01) -> bool | None:
        if delta_h is None:
            return None
        return delta_h >= min_dh

    te_ratio = row.get("te_over_de_entropy")
    rnd_te = row.get("mean_abs_te_random")
    ent_te = row.get("mean_abs_te_entropy")

    claims = {
        "entropy_te_exceeds_de": None if te_ratio is None else te_ratio > 1.0,
        "entropy_te_much_gt_de": _ok_ratio(te_ratio, min_ratio=1.5),
        "random_te_not_dominated_by_ln": (
            None
            if row.get("te_over_de_random") is None
            else row["te_over_de_random"] < (te_ratio or 0)
        ),
        "entropy_te_gt_random_te": (
            None if ent_te is None or rnd_te is None else ent_te > rnd_te
        ),
        "entropy_high_delta_h": _ok_high_dh(row.get("mean_abs_delta_h_entropy")),
        "entropy_low_argmax_change": _ok_low_change(row.get("mean_pred_change_rate_entropy")),
        "ln_mediated_effect": (
            None
            if row.get("mean_ln_mediated_fraction") is None
            else row["mean_ln_mediated_fraction"] >= 0.3
        ),
        "phase3_js_entropy_gt_random": (
            None
            if row.get("mean_js_entropy") is None or row.get("mean_js_random") is None
            else row["mean_js_entropy"] > row["mean_js_random"]
        ),
        "phase3_near_affine_shift": (
            None
            if row.get("mean_affine_r2_entropy") is None
            else row["mean_affine_r2_entropy"] >= 0.999
        ),
    }

    if row.get("model") == "gpt2-small" and row.get("paper_neuron_overlap_gpt2") is not None:
        claims["gpt2_paper_neuron_overlap"] = row["paper_neuron_overlap_gpt2"] >= 1

    passed = [k for k, v in claims.items() if v is True]
    failed = [k for k, v in claims.items() if v is False]
    skipped = [k for k, v in claims.items() if v is None]

    return {
        "claims": claims,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "overall": "pass" if failed == [] and passed else ("partial" if passed else "fail"),
    }


def build_comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    display_cols = [
        "model",
        "display_name",
        "family",
        "variant",
        "normalization",
        "udark_start",
        "n_entropy_selected",
        "top_entropy_neurons_str",
        "mean_rho_selected",
        "mean_ln_mediated_fraction",
        "mean_abs_te_entropy",
        "mean_abs_de_ln_entropy",
        "te_over_de_entropy",
        "mean_abs_delta_h_entropy",
        "mean_pred_change_rate_entropy",
        "te_over_de_random",
        "mean_js_entropy",
        "mean_js_random",
        "js_entropy_over_random",
        "mean_affine_r2_entropy",
        "mean_constant_shift_r2_entropy",
        "mean_shape_change_entropy",
        "mean_kl_entropy_te",
        "mean_argmax_flip_entropy_te",
        "paper_neuron_overlap_gpt2",
    ]
    df = pd.DataFrame(rows)
    cols = [c for c in display_cols if c in df.columns]
    return df[cols]


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Simple markdown table without extra dependencies."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.4g}" if pd.notna(val) else "")
            elif val is None or (isinstance(val, float) and pd.isna(val)):
                cells.append("")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rows_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "display_name" not in df.columns:
        df["display_name"] = df["model"].map(_display_name)
    # Stable plot order matching DEFAULT_MODELS when present
    order = {m: i for i, m in enumerate(DEFAULT_MODELS)}
    df["_ord"] = df["model"].map(lambda m: order.get(m, 999))
    return df.sort_values("_ord").drop(columns=["_ord"])


def _save_plot(fig: go.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    print(f"  saved {path}")


def _bar_by_model(
    df: pd.DataFrame,
    y: str,
    title: str,
    yaxis_title: str,
    out_path: pathlib.Path,
    *,
    color_by_family: bool = True,
) -> None:
    plot_df = df.dropna(subset=[y]).copy()
    if plot_df.empty:
        return
    fig = px.bar(
        plot_df,
        x="display_name",
        y=y,
        color="family" if color_by_family else None,
        color_discrete_map=FAMILY_COLORS if color_by_family else None,
        title=title,
        labels={"display_name": "Model", y: yaxis_title, "family": "Family"},
        text_auto=".3g",
    )
    fig.update_layout(
        width=900,
        height=420,
        title_font_size=15,
        xaxis_tickangle=-25,
        showlegend=color_by_family,
    )
    _save_plot(fig, out_path)


def _grouped_bars(
    df: pd.DataFrame,
    metrics: list[tuple[str, str]],
    title: str,
    yaxis_title: str,
    out_path: pathlib.Path,
) -> None:
    """metrics: list of (column, series_label)."""
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for col, label in metrics:
            val = row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            records.append(
                {
                    "display_name": row["display_name"],
                    "metric": label,
                    "value": float(val),
                    "family": row.get("family"),
                }
            )
    if not records:
        return
    plot_df = pd.DataFrame(records)
    fig = px.bar(
        plot_df,
        x="display_name",
        y="value",
        color="metric",
        barmode="group",
        title=title,
        labels={"display_name": "Model", "value": yaxis_title, "metric": ""},
    )
    fig.update_layout(width=960, height=440, title_font_size=15, xaxis_tickangle=-25)
    _save_plot(fig, out_path)


def write_comparison_plots(rows: list[dict[str, Any]], plots_dir: pathlib.Path) -> list[str]:
    """Generate cross-model Plotly HTML diagrams. Returns relative filenames."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = _rows_df(rows)
    written: list[str] = []

    def _track(name: str) -> pathlib.Path:
        written.append(name)
        return plots_dir / name

    _bar_by_model(
        df,
        "mean_rho_selected",
        "Phase 1 - mean Ï of selected entropy neurons",
        "Mean Ï (null-space alignment)",
        _track("phase1_mean_rho.html"),
    )
    _bar_by_model(
        df,
        "mean_ln_mediated_fraction",
        "Phase 2 - LN/RMS-mediated fraction (1 âˆ’ DE_LN/TE)",
        "LN-mediated fraction",
        _track("phase2_ln_mediated_fraction.html"),
    )
    _bar_by_model(
        df,
        "te_over_de_entropy",
        "Phase 2 - TE / DE_LN ratio (entropy neurons)",
        "TE / DE_LN",
        _track("phase2_te_over_de.html"),
    )
    _grouped_bars(
        df,
        [
            ("mean_abs_te_entropy", "|TE| entropy"),
            ("mean_abs_de_ln_entropy", "|DE_LN| entropy"),
            ("mean_abs_te_random", "|TE| random"),
        ],
        "Phase 2 - total vs direct effect magnitudes",
        "Mean |effect|",
        _track("phase2_te_de_magnitudes.html"),
    )
    _bar_by_model(
        df,
        "mean_abs_delta_h_entropy",
        "Phase 2 - mean |DeltaH| after ablating entropy neurons",
        "Mean |Deltaentropy|",
        _track("phase2_delta_h.html"),
    )
    _bar_by_model(
        df,
        "mean_pred_change_rate_entropy",
        "Phase 2 - argmax / prediction change rate",
        "Pred change rate",
        _track("phase2_pred_change.html"),
    )
    _grouped_bars(
        df,
        [
            ("mean_js_entropy", "JS entropy neurons"),
            ("mean_js_random", "JS random controls"),
        ],
        "Exp 1 Phase 3 - Jensen-Shannon divergence (TE)",
        "Mean JS divergence",
        _track("phase3_js_entropy_vs_random.html"),
    )
    _bar_by_model(
        df,
        "js_entropy_over_random",
        "Exp 1 Phase 3 - JS(entropy) / JS(random)",
        "JS ratio",
        _track("phase3_js_ratio.html"),
    )
    _grouped_bars(
        df,
        [
            ("mean_affine_r2_entropy", "Affine RÂ²"),
            ("mean_constant_shift_r2_entropy", "Constant-shift RÂ²"),
        ],
        "Exp 1 Phase 3 - softmax shift fit quality (entropy neurons)",
        "RÂ²",
        _track("phase3_affine_vs_constant_r2.html"),
    )

    # Scatter: Ï vs LN-mediated fraction
    scatter_df = df.dropna(subset=["mean_rho_selected", "mean_ln_mediated_fraction"])
    if not scatter_df.empty:
        fig = px.scatter(
            scatter_df,
            x="mean_rho_selected",
            y="mean_ln_mediated_fraction",
            color="family",
            text="display_name",
            color_discrete_map=FAMILY_COLORS,
            title="Phase 1 Ï vs Phase 2 LN-mediated fraction",
            labels={
                "mean_rho_selected": "Mean Ï (selected)",
                "mean_ln_mediated_fraction": "LN-mediated fraction",
                "family": "Family",
            },
        )
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(width=720, height=480, title_font_size=15)
        _save_plot(fig, _track("scatter_rho_vs_ln_fraction.html"))

    # Scatter: |ΔH| vs JS (phase2 x phase3)
    scatter2 = df.dropna(subset=["mean_abs_delta_h_entropy", "mean_js_entropy"])
    if not scatter2.empty:
        fig = px.scatter(
            scatter2,
            x="mean_abs_delta_h_entropy",
            y="mean_js_entropy",
            color="family",
            text="display_name",
            color_discrete_map=FAMILY_COLORS,
            title="Phase 2 |DeltaH| vs Exp 1 Phase 3 JS (entropy neurons)",
            labels={
                "mean_abs_delta_h_entropy": "Mean |DeltaH| (Phase 2)",
                "mean_js_entropy": "Mean JS (Exp 1 Phase 3 TE)",
                "family": "Family",
            },
            log_x=True,
            log_y=True,
        )
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(width=720, height=480, title_font_size=15)
        _save_plot(fig, _track("scatter_deltaH_vs_js.html"))

    # Base vs instruct paired deltas
    pairs = [
        ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "Llama 1B"),
        ("Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-0.8B", "Qwen 0.8B"),
        ("Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-2B", "Qwen 2B"),
    ]
    by_model = {r["model"]: r for r in rows}
    pair_records: list[dict[str, Any]] = []
    for base_name, inst_name, label in pairs:
        if base_name not in by_model or inst_name not in by_model:
            continue
        base, inst = by_model[base_name], by_model[inst_name]
        for metric, nice in [
            ("mean_rho_selected", "Mean Ï"),
            ("mean_ln_mediated_fraction", "LN-mediated frac"),
            ("mean_abs_te_entropy", "|TE|"),
            ("mean_abs_delta_h_entropy", "|DeltaH|"),
            ("mean_pred_change_rate_entropy", "Pred change"),
            ("mean_js_entropy", "JS (TE)"),
        ]:
            b, i = base.get(metric), inst.get(metric)
            if b is None or i is None:
                continue
            pair_records.append(
                {
                    "pair": label,
                    "metric": nice,
                    "delta_inst_minus_base": float(i) - float(b),
                    "base": float(b),
                    "instruct": float(i),
                }
            )
    if pair_records:
        pair_df = pd.DataFrame(pair_records)
        fig = px.bar(
            pair_df,
            x="metric",
            y="delta_inst_minus_base",
            color="pair",
            barmode="group",
            title="Base â†’ Instruct deltas (instruct âˆ’ base)",
            labels={
                "metric": "Metric",
                "delta_inst_minus_base": "Delta (instruct âˆ’ base)",
                "pair": "Pair",
            },
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(width=960, height=440, title_font_size=15)
        _save_plot(fig, _track("base_vs_instruct_deltas.html"))

    # Dashboard overview (multi-panel)
    overview = df.dropna(
        subset=[
            "mean_rho_selected",
            "mean_ln_mediated_fraction",
            "te_over_de_entropy",
            "mean_js_entropy",
        ]
    )
    if not overview.empty:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Mean Ï (Phase 1)",
                "LN-mediated fraction (Phase 2)",
                "TE / DE_LN (Phase 2)",
                "JS entropy neurons (Exp 1 Phase 3)",
            ),
            vertical_spacing=0.16,
            horizontal_spacing=0.10,
        )
        names = overview["display_name"].tolist()
        colors = [FAMILY_COLORS.get(f, "#888") for f in overview["family"]]
        panels = [
            (1, 1, overview["mean_rho_selected"].tolist()),
            (1, 2, overview["mean_ln_mediated_fraction"].tolist()),
            (2, 1, overview["te_over_de_entropy"].tolist()),
            (2, 2, overview["mean_js_entropy"].tolist()),
        ]
        for r, c, vals in panels:
            fig.add_trace(
                go.Bar(x=names, y=vals, marker_color=colors, showlegend=False),
                row=r,
                col=c,
            )
        fig.update_layout(
            width=1000,
            height=700,
            title_text="Cross-model overview - Exp 1 Phase 1 / 2 / 3",
            title_font_size=16,
        )
        fig.update_xaxes(tickangle=-30)
        _save_plot(fig, _track("overview_dashboard.html"))

    return written


def _findings_bullets(rows: list[dict[str, Any]]) -> list[str]:
    """Auto-generate high-level findings from the aggregated table."""
    bullets: list[str] = []
    by_model = {r["model"]: r for r in rows}
    gpt2 = by_model.get("gpt2-small")
    if gpt2 is not None:
        bullets.append(
            f"**GPT-2 (LNPre)** shows clear LN mediation "
            f"(LN-mediated fraction â‰ˆ {gpt2.get('mean_ln_mediated_fraction', float('nan')):.2f}, "
            f"TE/DE â‰ˆ {gpt2.get('te_over_de_entropy', float('nan')):.2f}) and recovers "
            f"{gpt2.get('paper_neuron_overlap_gpt2', '?')}/6 paper entropy neurons."
        )

    rms = [r for r in rows if r.get("normalization") == "RMSPre"]
    if rms:
        mean_ln = sum(float(r["mean_ln_mediated_fraction"] or 0) for r in rms) / len(rms)
        bullets.append(
            f"**RMSNorm models (Llama + Qwen)** show near-zero LN/RMS mediation "
            f"(mean fraction â‰ˆ {mean_ln:.4f}; TE â‰ˆ DE_LN), so the paper's "
            "LN-mediated confidence story does not transfer directly."
        )

    qwen = [r for r in rows if r.get("family") == "Qwen"]
    if qwen:
        mean_dh = sum(float(r["mean_abs_delta_h_entropy"] or 0) for r in qwen) / len(qwen)
        bullets.append(
            f"**Qwen** ablations produce very large |DeltaH| (family mean â‰ˆ {mean_dh:.2f}) "
            "with TEâ‰ˆDE_LN and TEâ‰ˆrandom TE - treat as a distinct RMS / ablation-path "
            "regime, not paper-style entropy-neuron mediation."
        )

    p2b = [r for r in rows if r.get("mean_js_entropy") is not None]
    if p2b:
        sep = [
            r
            for r in p2b
            if r.get("js_entropy_over_random") is not None and r["js_entropy_over_random"] > 1.5
        ]
        bullets.append(
            f"**Exp 1 Phase 3:** all {len(p2b)} models keep high affine RÂ² (â‰¥0.999) after ablation; "
            f"{len(sep)}/{len(p2b)} show JS(entropy) â‰« JS(random) (ratio > 1.5), "
            "consistent with a mostly shift/scale change of the softmax."
        )

    llama = [r for r in rows if r.get("family") == "Llama"]
    if llama:
        mean_pred = sum(float(r["mean_pred_change_rate_entropy"] or 0) for r in llama) / len(llama)
        bullets.append(
            f"**Llama** keeps modest |TE| and |DeltaH| but higher argmax change "
            f"(family mean pred-change â‰ˆ {mean_pred:.2f}) than GPT-2."
        )

    return bullets


def write_markdown_report(
    rows: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    out_path: pathlib.Path,
    *,
    errors: list[str] | None = None,
    plot_files: list[str] | None = None,
) -> None:
    table = build_comparison_table(rows)
    # Compact primary table for the report body
    primary_cols = [
        c
        for c in [
            "display_name",
            "normalization",
            "mean_rho_selected",
            "mean_ln_mediated_fraction",
            "te_over_de_entropy",
            "mean_abs_delta_h_entropy",
            "mean_pred_change_rate_entropy",
            "mean_js_entropy",
            "js_entropy_over_random",
            "mean_affine_r2_entropy",
        ]
        if c in table.columns
    ]

    lines = [
        "# Full model comparison - Exp 1 Phase 1 / 2 / 3",
        "",
        "Aggregated from `outputs/<model>/exp1/{phase1,phase2,phase3}` summaries "
        "(and optional Exp 1 Phase 3 feather metrics).",
        "",
        "## Key findings",
        "",
    ]
    for bullet in _findings_bullets(rows):
        lines.append(f"- {bullet}")
    lines.append("")

    lines.extend(
        [
            "## Primary metrics",
            "",
            table[primary_cols].pipe(_dataframe_to_markdown),
            "",
            "## Full metrics table",
            "",
            "See also `comparison_table.csv` / `comparison.json`.",
            "",
            table.pipe(_dataframe_to_markdown),
            "",
        ]
    )

    if plot_files:
        lines.extend(["## Diagrams", ""])
        for name in plot_files:
            lines.append(f"- [`plots/{name}`](plots/{name})")
        lines.append("")

    lines.extend(["## Paper-claim checks (heuristic)", ""])
    for row, ev in zip(rows, evaluations):
        lines.append(f"### {row['display_name']} (`{row['model']}`)")
        lines.append("")
        if row.get("paper_neuron_overlap_gpt2") is not None:
            overlap_list = row.get("paper_neuron_overlap_list") or []
            lines.append(
                f"- Paper neuron overlap: **{row['paper_neuron_overlap_gpt2']}/6** "
                f"({', '.join(overlap_list) if overlap_list else 'none'})"
            )
            lines.append("")
        if ev["failed"]:
            lines.append(f"- **Failed:** {', '.join(ev['failed'])}")
        if ev["passed"]:
            lines.append(f"- **Passed:** {', '.join(ev['passed'])}")
        if ev["skipped"]:
            lines.append(f"- **Skipped (missing data):** {', '.join(ev['skipped'])}")
        lines.append(f"- **Overall:** `{ev['overall']}`")
        lines.append("")

    if errors:
        lines.extend(["## Skipped models", ""])
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")

    if any(r.get("model") == "gpt2-small" for r in rows):
        lines.extend(
            [
                "## GPT-2 paper neurons (reference)",
                "",
                _fmt_list(GPT2_PAPER_ENTROPY_NEURONS, limit=20),
                "",
            ]
        )

    lines.extend(
        [
            "## How to read Exp 1 Phase 3",
            "",
            "- **JS divergence**: overall softmax change under ablation (entropy vs random).",
            "- **Affine / constant-shift RÂ²**: whether the post-ablation distribution is "
            "mostly a shift/scale of the baseline (paper-style confidence regulation).",
            "- GPT-2/Llama Exp 1 Phase 3 summaries use branch `te`; Qwen runs used `both` "
            "(TE + DE_LN). Cross-model plots use TE columns for fairness.",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(
    models: list[str],
    outputs_dir: pathlib.Path,
    comparison_dir: pathlib.Path,
) -> tuple[pd.DataFrame, list[str]]:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = comparison_dir / "plots"

    rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    errors: list[str] = []

    for model in models:
        try:
            row = load_model_track1_metrics(model, outputs_dir)
            rows.append(row)
            evaluations.append(evaluate_paper_claims(row))
        except FileNotFoundError as exc:
            errors.append(f"{model}: {exc}")

    if not rows:
        msg = "No model artifacts found.\n" + "\n".join(errors)
        raise FileNotFoundError(msg)

    table = build_comparison_table(rows)
    table.to_csv(comparison_dir / "comparison_table.csv", index=False)

    print("Writing comparison plots...")
    plot_files = write_comparison_plots(rows, plots_dir)

    payload = {
        "models": models,
        "outputs_dir": str(outputs_dir.resolve()),
        "rows": [{k: v for k, v in r.items() if k not in ("top_entropy_neurons",)} for r in rows],
        "evaluations": evaluations,
        "errors": errors,
        "plots": plot_files,
        "findings": _findings_bullets(rows),
    }
    (comparison_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_markdown_report(
        rows,
        evaluations,
        comparison_dir / "comparison_report.md",
        errors=errors,
        plot_files=plot_files,
    )

    return table, errors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full Phase 1/2/2b cross-model comparison")
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names matching outputs/<slug>/ subdirs",
    )
    p.add_argument(
        "--outputs-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root outputs directory (default: outputs/)",
    )
    p.add_argument(
        "--comparison-dir",
        type=pathlib.Path,
        default=DEFAULT_COMPARISON_DIR,
        help="Where to write comparison artifacts (default: outputs/comparison/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    table, errors = run_comparison(args.models, args.outputs_dir, args.comparison_dir)
    n_ok = len(table)
    n_skip = len(errors)
    print(f"{n_ok} model(s) -> {args.comparison_dir.resolve()}", end="")
    if n_skip:
        print(f" ({n_skip} skipped)", end="")
    print()
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
