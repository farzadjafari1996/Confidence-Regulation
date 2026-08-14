"""
Extra Phase 1 plots from existing static metrics (no model reload).

Per-model outputs → outputs/<model_slug>/extra/
Shared compare     → outputs/extra/exp1/

Usage (from repo root):
  python src/extra_plots/exp1/phase1_extra_plots.py
  python src/extra_plots/exp1/phase1_extra_plots.py --slug gpt2
  python src/extra_plots/exp1/phase1_extra_plots.py --outputs-dir outputs

Writes per model:
  rho_vs_logitvar.html
  rho_vs_norm.html
  rho_distribution.html
  phase1_extra_summary.json

Writes shared:
  cross_model_phase1.html / .json
  base_instruct_overlap.html / .json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_src = pathlib.Path(__file__).resolve().parent
while _src != _src.parent and not (
    (_src / "common").is_dir() and (_src / "experiments").is_dir()
):
    _src = _src.parent
for _p in (_src, _src / "common"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from hyperparameters import DEFAULT_OUTPUT_DIR, ROOT  # noqa: E402
from plotly_style import save_plotly_html  # noqa: E402

PLOT_COLORS = {"Entropy": "#EF553B", "Normal": "#636EFA"}

# (base_slug, instruct_slug) pairs present in this repo.
BASE_INSTRUCT_PAIRS: tuple[tuple[str, str], ...] = (
    ("llama_3.2_1B_base", "llama_3.2_1B_instruct"),
    ("qwen_3.5_0.8B_base", "qwen_3.5_0.8B_instruct"),
    ("qwen_3.5_2B_base", "qwen_3.5_2B_instruct"),
    ("smollm2_1.7B_base", "smollm2_1.7B_instruct"),
    ("gemma2_2b_base", "gemma2_2b_instruct"),
)


def _phase1_dir(outputs_dir: pathlib.Path, slug: str) -> pathlib.Path:
    return outputs_dir / slug / "exp1" / "phase1"


def _extra_dir(outputs_dir: pathlib.Path, slug: str) -> pathlib.Path:
    return outputs_dir / slug / "extra"


def _shared_extra_dir(outputs_dir: pathlib.Path) -> pathlib.Path:
    return outputs_dir / "extra" / "exp1"


def discover_phase1_slugs(outputs_dir: pathlib.Path) -> list[str]:
    slugs: list[str] = []
    if not outputs_dir.is_dir():
        return slugs
    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir() or child.name in {"extra", "_compare"}:
            continue
        csv_path = _phase1_dir(outputs_dir, child.name) / "static_neuron_metrics.csv"
        if csv_path.exists():
            slugs.append(child.name)
    return slugs


def load_phase1_static(phase1_dir: pathlib.Path) -> pd.DataFrame:
    csv_path = phase1_dir / "static_neuron_metrics.csv"
    df = pd.read_csv(csv_path)
    if "is_top1pct_rho" not in df.columns:
        raise ValueError(f"missing is_top1pct_rho in {csv_path}")
    # Restore string labels if CSV coerced component_name to float.
    if "neuron_index" in df.columns and "component_name" in df.columns:
        layer = str(df["component_name"].iloc[0]).split(".")[0]
        # Only rewrite when values look float-mangled (e.g. 11.48 vs 11.480).
        df = df.copy()
        df["component_name"] = df["neuron_index"].apply(lambda i: f"{layer}.{int(i)}")
    return df


def load_summary(phase1_dir: pathlib.Path) -> dict:
    path = phase1_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _neuron_type(df: pd.DataFrame) -> pd.Series:
    return np.where(df["is_top1pct_rho"], "Entropy", "Normal")


def plot_rho_vs_logitvar(df: pd.DataFrame, out_path: pathlib.Path, title: str) -> None:
    plot_df = df.copy()
    plot_df["neuron_type"] = _neuron_type(plot_df)
    fig = px.scatter(
        plot_df,
        x="rho",
        y="logit_var",
        color="neuron_type",
        log_y=True,
        title=f"{title} — ρ vs LogitVar",
        labels={"rho": "ρ (null-space alignment)", "logit_var": "LogitVar(w_out)"},
        color_discrete_map=PLOT_COLORS,
        hover_data=["component_name", "norm"],
        category_orders={"neuron_type": ["Normal", "Entropy"]},
    )
    fig.update_layout(legend_title_text="")
    save_plotly_html(fig, out_path)


def plot_rho_vs_norm(df: pd.DataFrame, out_path: pathlib.Path, title: str) -> None:
    plot_df = df.copy()
    plot_df["neuron_type"] = _neuron_type(plot_df)
    fig = px.scatter(
        plot_df,
        x="norm",
        y="rho",
        color="neuron_type",
        title=f"{title} — ‖w_out‖ vs ρ",
        labels={"norm": "‖w_out‖", "rho": "ρ (null-space alignment)"},
        color_discrete_map=PLOT_COLORS,
        hover_data=["component_name", "logit_var"],
        category_orders={"neuron_type": ["Normal", "Entropy"]},
    )
    fig.update_layout(legend_title_text="")
    save_plotly_html(fig, out_path)


def plot_rho_distribution(df: pd.DataFrame, out_path: pathlib.Path, title: str) -> None:
    plot_df = df.copy()
    plot_df["neuron_type"] = _neuron_type(plot_df)
    fig = px.histogram(
        plot_df,
        x="rho",
        color="neuron_type",
        barmode="overlay",
        nbins=60,
        opacity=0.65,
        title=f"{title} — ρ distribution",
        labels={"rho": "ρ (null-space alignment)", "count": "neurons"},
        color_discrete_map=PLOT_COLORS,
        category_orders={"neuron_type": ["Normal", "Entropy"]},
    )
    # CDF of all neurons on secondary axis.
    sorted_rho = np.sort(plot_df["rho"].to_numpy())
    cdf_y = np.arange(1, len(sorted_rho) + 1) / len(sorted_rho)
    fig.add_trace(
        go.Scatter(
            x=sorted_rho,
            y=cdf_y,
            mode="lines",
            name="CDF (all)",
            line=dict(color="#333333", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        legend_title_text="",
        yaxis=dict(title="count"),
        yaxis2=dict(title="CDF", overlaying="y", side="right", range=[0, 1]),
    )
    save_plotly_html(fig, out_path)


def model_extra_summary(df: pd.DataFrame, summary: dict, slug: str) -> dict:
    ent = df[df["is_top1pct_rho"]]
    other = df[~df["is_top1pct_rho"]]
    return {
        "slug": slug,
        "model": summary.get("model"),
        "n_neurons": int(len(df)),
        "n_selected": int(len(ent)),
        "rho_all": {
            "mean": float(df["rho"].mean()),
            "median": float(df["rho"].median()),
            "p90": float(df["rho"].quantile(0.9)),
            "p99": float(df["rho"].quantile(0.99)),
            "max": float(df["rho"].max()),
        },
        "rho_selected": {
            "mean": float(ent["rho"].mean()) if len(ent) else None,
            "min": float(ent["rho"].min()) if len(ent) else None,
            "max": float(ent["rho"].max()) if len(ent) else None,
        },
        "logit_var_selected_mean": float(ent["logit_var"].mean()) if len(ent) else None,
        "logit_var_other_mean": float(other["logit_var"].mean()) if len(other) else None,
        "norm_selected_mean": float(ent["norm"].mean()) if len(ent) else None,
        "norm_other_mean": float(other["norm"].mean()) if len(other) else None,
        "selected_neurons": ent.sort_values("rho", ascending=False)["component_name"].tolist(),
    }


def write_per_model_plots(
    outputs_dir: pathlib.Path,
    slug: str,
) -> pathlib.Path | None:
    phase1 = _phase1_dir(outputs_dir, slug)
    csv_path = phase1 / "static_neuron_metrics.csv"
    if not csv_path.exists():
        print(f"  skip {slug}: missing {csv_path}")
        return None

    df = load_phase1_static(phase1)
    summary = load_summary(phase1)
    title = summary.get("model") or slug
    out_dir = _extra_dir(outputs_dir, slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_rho_vs_logitvar(df, out_dir / "rho_vs_logitvar.html", title)
    plot_rho_vs_norm(df, out_dir / "rho_vs_norm.html", title)
    plot_rho_distribution(df, out_dir / "rho_distribution.html", title)

    payload = model_extra_summary(df, summary, slug)
    (out_dir / "phase1_extra_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"  wrote {out_dir}")
    return out_dir


def _selected_set(phase1_dir: pathlib.Path) -> set[str]:
    top1 = phase1_dir / "top1pct_rho_neurons.json"
    if top1.exists():
        payload = json.loads(top1.read_text(encoding="utf-8"))
        return set(payload.get("neurons") or [])
    df = load_phase1_static(phase1_dir)
    return set(df.loc[df["is_top1pct_rho"], "component_name"].astype(str))


def _neuron_index(label: str) -> int:
    return int(str(label).split(".")[-1])


def base_instruct_overlap_row(
    outputs_dir: pathlib.Path, base_slug: str, instruct_slug: str
) -> dict | None:
    base_p1 = _phase1_dir(outputs_dir, base_slug)
    inst_p1 = _phase1_dir(outputs_dir, instruct_slug)
    if not (base_p1 / "static_neuron_metrics.csv").exists():
        return None
    if not (inst_p1 / "static_neuron_metrics.csv").exists():
        return None

    base_set = _selected_set(base_p1)
    inst_set = _selected_set(inst_p1)
    base_idx = {_neuron_index(x) for x in base_set}
    inst_idx = {_neuron_index(x) for x in inst_set}
    inter = base_idx & inst_idx
    union = base_idx | inst_idx
    jaccard = (len(inter) / len(union)) if union else 0.0

    base_df = load_phase1_static(base_p1).set_index("neuron_index")
    inst_df = load_phase1_static(inst_p1).set_index("neuron_index")
    common_idx = sorted(set(base_df.index) & set(inst_df.index))
    if common_idx:
        rho_corr = float(
            np.corrcoef(
                base_df.loc[common_idx, "rho"].to_numpy(),
                inst_df.loc[common_idx, "rho"].to_numpy(),
            )[0, 1]
        )
    else:
        rho_corr = float("nan")

    base_sum = load_summary(base_p1)
    inst_sum = load_summary(inst_p1)
    return {
        "base_slug": base_slug,
        "instruct_slug": instruct_slug,
        "base_model": base_sum.get("model"),
        "instruct_model": inst_sum.get("model"),
        "n_base_selected": len(base_idx),
        "n_instruct_selected": len(inst_idx),
        "intersection_neuron_indices": sorted(inter),
        "intersection_size": len(inter),
        "union_size": len(union),
        "jaccard_index": jaccard,
        "rho_pearson_all_neurons": rho_corr,
        "base_only_indices": sorted(base_idx - inst_idx),
        "instruct_only_indices": sorted(inst_idx - base_idx),
    }


def write_base_instruct_overlap(outputs_dir: pathlib.Path, slugs: list[str]) -> pathlib.Path | None:
    rows: list[dict] = []
    for base_slug, instruct_slug in BASE_INSTRUCT_PAIRS:
        if base_slug not in slugs or instruct_slug not in slugs:
            continue
        row = base_instruct_overlap_row(outputs_dir, base_slug, instruct_slug)
        if row is None:
            continue
        rows.append(row)
        # Mirror a compact copy under each model’s extra/ folder.
        for slug in (base_slug, instruct_slug):
            out = _extra_dir(outputs_dir, slug)
            out.mkdir(parents=True, exist_ok=True)
            (out / "base_instruct_overlap.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )

    if not rows:
        print("  no base/instruct pairs with Phase 1 outputs")
        return None

    shared = _shared_extra_dir(outputs_dir)
    shared.mkdir(parents=True, exist_ok=True)
    json_path = shared / "base_instruct_overlap.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig = go.Figure()
    labels = [f"{r['base_slug'].replace('_base', '')}" for r in rows]
    fig.add_trace(
        go.Bar(
            name="Jaccard (selected indices)",
            x=labels,
            y=[r["jaccard_index"] for r in rows],
            marker_color="#EF553B",
            text=[f"{r['intersection_size']}/{r['union_size']}" for r in rows],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="ρ Pearson (all neurons)",
            x=labels,
            y=[r["rho_pearson_all_neurons"] for r in rows],
            mode="lines+markers",
            marker=dict(color="#2166AC", size=10),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Base vs instruct — Phase 1 entropy-neuron overlap",
        yaxis=dict(title="Jaccard index", range=[0, 1.15]),
        yaxis2=dict(title="ρ Pearson", overlaying="y", side="right", range=[0, 1.05]),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    html_path = shared / "base_instruct_overlap.html"
    save_plotly_html(fig, html_path)
    print(f"  wrote {html_path}")
    return shared


def write_cross_model_summary(outputs_dir: pathlib.Path, slugs: list[str]) -> pathlib.Path | None:
    rows: list[dict] = []
    for slug in slugs:
        summary_path = _extra_dir(outputs_dir, slug) / "phase1_extra_summary.json"
        if not summary_path.exists():
            # Fall back to computing from Phase 1 if per-model step was skipped.
            phase1 = _phase1_dir(outputs_dir, slug)
            if not (phase1 / "static_neuron_metrics.csv").exists():
                continue
            df = load_phase1_static(phase1)
            payload = model_extra_summary(df, load_summary(phase1), slug)
        else:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(payload)

    if not rows:
        return None

    shared = _shared_extra_dir(outputs_dir)
    shared.mkdir(parents=True, exist_ok=True)
    json_path = shared / "cross_model_phase1.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    plot_df = pd.DataFrame(
        [
            {
                "slug": r["slug"],
                "model": r.get("model") or r["slug"],
                "rho_selected_mean": (r.get("rho_selected") or {}).get("mean"),
                "rho_selected_max": (r.get("rho_selected") or {}).get("max"),
                "rho_p99": (r.get("rho_all") or {}).get("p99"),
                "n_selected": r.get("n_selected"),
            }
            for r in rows
        ]
    ).dropna(subset=["rho_selected_mean"])
    plot_df = plot_df.sort_values("rho_selected_mean", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="mean ρ (selected)",
            x=plot_df["slug"],
            y=plot_df["rho_selected_mean"],
            marker_color="#EF553B",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="max ρ",
            x=plot_df["slug"],
            y=plot_df["rho_selected_max"],
            mode="markers",
            marker=dict(color="#333333", size=9, symbol="diamond"),
        )
    )
    fig.add_trace(
        go.Scatter(
            name="ρ p99 (all)",
            x=plot_df["slug"],
            y=plot_df["rho_p99"],
            mode="markers+lines",
            marker=dict(color="#636EFA", size=7),
            line=dict(color="#636EFA", width=1, dash="dot"),
        )
    )
    fig.update_layout(
        title="Cross-model Phase 1 — selected ρ strength",
        xaxis_title="model slug",
        yaxis_title="ρ",
        yaxis_range=[0, 1.05],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(tickangle=-35)
    html_path = shared / "cross_model_phase1.html"
    save_plotly_html(fig, html_path)
    print(f"  wrote {html_path}")
    return shared


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extra Phase 1 plots from cached static metrics")
    p.add_argument(
        "--outputs-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR if DEFAULT_OUTPUT_DIR.is_absolute() else ROOT / "outputs",
        help="Root outputs directory (default: <repo>/outputs)",
    )
    p.add_argument(
        "--slug",
        action="append",
        default=None,
        help="Only process this model slug (repeatable). Default: all with Phase 1 CSV.",
    )
    p.add_argument(
        "--skip-shared",
        action="store_true",
        help="Skip cross-model and base/instruct shared plots",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    slugs = args.slug or discover_phase1_slugs(outputs_dir)
    if not slugs:
        raise SystemExit(f"No Phase 1 static_neuron_metrics.csv under {outputs_dir}")

    print(f"Extra Phase 1 plots — {len(slugs)} model(s) under {outputs_dir}")
    done: list[str] = []
    for slug in slugs:
        print(f"=== {slug} ===")
        if write_per_model_plots(outputs_dir, slug) is not None:
            done.append(slug)

    if not args.skip_shared and done:
        print("=== shared compare ===")
        write_cross_model_summary(outputs_dir, done)
        write_base_instruct_overlap(outputs_dir, done)

    print(f"Done ({len(done)} models).")


if __name__ == "__main__":
    main()
