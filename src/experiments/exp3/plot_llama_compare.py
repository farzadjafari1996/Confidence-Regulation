"""Slide-ready Exp3 plots for Llama 1B base vs instruct.

Reads existing probe / matched-k / ablation JSONs and writes matplotlib PNGs
(+ Plotly HTML) under outputs/exp3_llama_compare/plots/.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "common"))

OUT_DIR = ROOT / "outputs" / "exp3_llama_compare"
PLOTS_DIR = OUT_DIR / "plots"

VARIANTS = [
    {"slug": "llama_3.2_1B_base", "display": "Llama 1B-Base", "short": "Base"},
    {"slug": "llama_3.2_1B_instruct", "display": "Llama 1B-Instruct", "short": "Instruct"},
]

# Same palette as exp3/plots.py and ablation plots.
C_FULL = "#4C78A8"
C_ENT = "#E45756"
C_REST = "#54A24B"
C_BOTTOM = "#B279A2"
C_TOPK = "#F58518"
C_BASE = "#00CC96"
C_INST = "#636EFA"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _test_metrics(probe_dir: pathlib.Path) -> dict[str, Any]:
    path = probe_dir / "test_metrics.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    test = dict(payload.get("test") or {})
    test["d_in"] = payload.get("d_in")
    return test


def _savefig(fig: plt.Figure, stem: str) -> pathlib.Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")
    return path


def collect_feature_set_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        root = ROOT / "outputs" / v["slug"] / "exp3"
        for probe in ("logistic", "mlp"):
            for feat, label in (
                ("", "full"),
                ("_entropy", "entropy"),
                ("_rest", "rest"),
            ):
                m = _test_metrics(root / f"{probe}{feat}")
                if not m or m.get("auroc") is None:
                    continue
                rows.append(
                    {
                        "variant": v["display"],
                        "short": v["short"],
                        "probe": probe,
                        "feature_set": label,
                        "auroc": float(m["auroc"]),
                        "accuracy": m.get("accuracy"),
                        "f1": m.get("f1"),
                        "d_in": m.get("d_in"),
                    }
                )
    return pd.DataFrame(rows)


def plot_feature_sets(df: pd.DataFrame) -> None:
    for probe in ("logistic", "mlp"):
        sub = df[df["probe"] == probe]
        if sub.empty:
            continue
        variants = ["Base", "Instruct"]
        sets = ["full", "entropy", "rest"]
        colors = {"full": C_FULL, "entropy": C_ENT, "rest": C_REST}
        labels = {
            "full": "full (all dims)",
            "entropy": "entropy only (k=12)",
            "rest": "rest (non-entropy)",
        }
        x = np.arange(len(variants))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for i, fs in enumerate(sets):
            vals = [
                float(sub[(sub["short"] == v) & (sub["feature_set"] == fs)]["auroc"].iloc[0])
                for v in variants
            ]
            bars = ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                label=labels[fs],
                color=colors[fs],
            )
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylabel("Test AUROC")
        ax.set_ylim(0.55, 1.05)
        ax.set_title(f"Exp3 {probe} probe — test AUROC by feature set (Llama 1B)")
        ax.legend(frameon=False, fontsize=9)
        ax.axhline(0.5, color="#999", lw=0.8, ls=":")
        _savefig(fig, f"feature_set_auroc_{probe}")


def plot_matched_k_null() -> None:
    for v in VARIANTS:
        mk = _read_json(ROOT / "outputs" / v["slug"] / "exp3" / "matched_k_summary.json")
        arms = mk["arms"]
        null = arms["random_null"]
        draws = mk.get("random_draws") or []
        aurocs = [float(d["auroc"]) for d in draws if d.get("auroc") is not None]
        ent = float(arms["entropy_k12"]["auroc"])
        bot = float(arms["bottom_k12"]["auroc"])

        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.hist(aurocs, bins=10, color=C_FULL, alpha=0.85, label=f"random k=12 (n={len(aurocs)})")
        ax.axvline(
            ent,
            color=C_ENT,
            lw=2.2,
            label=f"entropy {ent:.3f} (pct {null['entropy_percentile']:.0f})",
        )
        ax.axvline(
            bot,
            color=C_BOTTOM,
            lw=2.0,
            ls="--",
            label=f"bottom-ρ {bot:.3f}",
        )
        ax.set_xlabel("Test AUROC")
        ax.set_ylabel("Random-draw count")
        ax.set_title(f"Exp3 matched-k MLP — {v['display']} (k=12)")
        ax.legend(frameon=False, fontsize=9)
        _savefig(fig, f"matched_k_null_{v['short'].lower()}")


def plot_matched_k_arm_comparison() -> None:
    arms_order = ["entropy (k=12)", "random mean", "bottom-ρ (k=12)"]
    colors = {"entropy (k=12)": C_ENT, "random mean": C_FULL, "bottom-ρ (k=12)": C_BOTTOM}
    variants = ["Base", "Instruct"]
    data: dict[str, list[float]] = {a: [] for a in arms_order}
    for v in VARIANTS:
        mk = _read_json(ROOT / "outputs" / v["slug"] / "exp3" / "matched_k_summary.json")
        arms = mk["arms"]
        null = arms["random_null"]
        data["entropy (k=12)"].append(float(arms["entropy_k12"]["auroc"]))
        data["random mean"].append(float(null["auroc_mean"]))
        data["bottom-ρ (k=12)"].append(float(arms["bottom_k12"]["auroc"]))

    x = np.arange(len(variants))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, arm in enumerate(arms_order):
        bars = ax.bar(x + (i - 1) * width, data[arm], width, label=arm, color=colors[arm])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.82, 1.0)
    ax.set_title("Exp3 matched-k MLP — entropy vs random vs bottom-ρ (k=12)")
    ax.legend(frameon=False, fontsize=9)
    _savefig(fig, "matched_k_arm_comparison")


def plot_topk_sweep() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for v, color in zip(VARIANTS, (C_BASE, C_INST)):
        mk = _read_json(ROOT / "outputs" / v["slug"] / "exp3" / "matched_k_summary.json")
        ks, vals = [], []
        for tag, arm in sorted((mk["arms"].get("topk") or {}).items()):
            k = arm.get("k")
            if k is None and tag.startswith("topk_"):
                k = int(tag.split("_")[1])
            ks.append(int(k))
            vals.append(float(arm["auroc"]))
        order = np.argsort(ks)
        ks = [ks[i] for i in order]
        vals = [vals[i] for i in order]
        ax.plot(ks, vals, marker="o", color=color, lw=2, label=v["short"])
        for k, a in zip(ks, vals):
            ax.annotate(f"{a:.3f}", (k, a), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlabel("Top-k neurons by ρ")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Exp3 matched-k MLP — top-k by ρ sweep")
    ax.set_ylim(0.70, 1.0)
    ax.legend(frameon=False)
    ax.set_xticks([4, 8, 12, 16, 20])
    _savefig(fig, "topk_rho_sweep")


def plot_ablation_sufficiency() -> None:
    for v in VARIANTS:
        for probe in ("logistic", "mlp"):
            path = ROOT / "outputs" / v["slug"] / "exp3" / probe / "entropy_feature_ablation.json"
            csv_path = path.with_suffix(".csv")
            if not path.exists() or not csv_path.exists():
                continue
            payload = _read_json(path)
            res = (payload.get("results") or {}).get("sufficiency_mean")
            if not res:
                continue
            cdf = pd.read_csv(csv_path)
            mask = (
                (cdf["intervention"] == "sufficiency")
                & (cdf["mode"] == "mean")
                & (cdf["set"].astype(str) != "entropy")
            )
            control_deltas = cdf.loc[mask, "delta_auroc"].astype(float).tolist()
            ent_delta = float(res["delta_auroc"])
            pct = (res.get("controls") or {}).get("percentile_vs_controls")

            fig, ax = plt.subplots(figsize=(8.5, 4.0))
            ax.hist(
                control_deltas,
                bins=min(12, max(5, len(control_deltas))),
                color=C_FULL,
                alpha=0.85,
                label="random controls",
            )
            ax.axvline(
                ent_delta,
                color=C_ENT,
                lw=2.2,
                label=f"entropy ΔAUROC={ent_delta:.3f}"
                + (f" (pct {pct:.0f})" if pct is not None else ""),
            )
            ax.set_xlabel("ΔAUROC (baseline − ablated)")
            ax.set_ylabel("Control count")
            ax.set_title(f"Exp3 frozen {probe} sufficiency (mean) — {v['display']}")
            ax.legend(frameon=False, fontsize=9)
            _savefig(fig, f"ablation_sufficiency_{probe}_{v['short'].lower()}")


def plot_ablation_delta_bars() -> None:
    for probe in ("logistic", "mlp"):
        variants = ["Base", "Instruct"]
        nec_vals, suf_vals = [], []
        for v in VARIANTS:
            path = ROOT / "outputs" / v["slug"] / "exp3" / probe / "entropy_feature_ablation.json"
            payload = _read_json(path)
            nec_vals.append(float(payload["results"]["necessity_mean"]["delta_auroc"]))
            suf_vals.append(float(payload["results"]["sufficiency_mean"]["delta_auroc"]))
        x = np.arange(len(variants))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        b1 = ax.bar(x - width / 2, nec_vals, width, label="necessity (mean)", color=C_FULL)
        b2 = ax.bar(x + width / 2, suf_vals, width, label="sufficiency (mean)", color=C_ENT)
        ax.bar_label(b1, fmt="%.2e", padding=2, fontsize=7)
        ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylabel("ΔAUROC (baseline − ablated)")
        ax.set_title(f"Exp3 frozen {probe} — entropy-feature ablation ΔAUROC")
        ax.legend(frameon=False, fontsize=9)
        ax.axhline(0, color="#999", lw=0.8)
        _savefig(fig, f"ablation_delta_auroc_{probe}")


def also_write_plotly_html() -> None:
    """Optional interactive HTML (no kaleido PNG — matplotlib owns slide assets)."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly_style import save_plotly_html
    except Exception as exc:
        print(f"Skipping Plotly HTML: {exc}")
        return

    feat = collect_feature_set_rows()
    for probe in ("logistic", "mlp"):
        sub = feat[feat["probe"] == probe].copy()
        sub["feature_set"] = sub["feature_set"].map(
            {
                "full": "full (all dims)",
                "entropy": "entropy only (k=12)",
                "rest": "rest (non-entropy)",
            }
        )
        fig = px.bar(
            sub,
            x="short",
            y="auroc",
            color="feature_set",
            barmode="group",
            title=f"Exp3 {probe} probe — test AUROC by feature set (Llama 1B)",
            labels={"short": "Variant", "auroc": "Test AUROC", "feature_set": ""},
            color_discrete_map={
                "full (all dims)": C_FULL,
                "entropy only (k=12)": C_ENT,
                "rest (non-entropy)": C_REST,
            },
            text_auto=".3f",
        )
        fig.update_layout(width=920, height=440, template="plotly_white", yaxis_range=[0.55, 1.05])
        save_plotly_html(fig, PLOTS_DIR / f"feature_set_auroc_{probe}.html")
        print(f"  saved feature_set_auroc_{probe}.html")

    records = []
    for v in VARIANTS:
        mk = _read_json(ROOT / "outputs" / v["slug"] / "exp3" / "matched_k_summary.json")
        for tag, arm in sorted((mk["arms"].get("topk") or {}).items()):
            k = arm.get("k") or int(tag.split("_")[1])
            records.append({"short": v["short"], "k": int(k), "auroc": float(arm["auroc"])})
    fig = px.line(
        pd.DataFrame(records),
        x="k",
        y="auroc",
        color="short",
        markers=True,
        title="Exp3 matched-k MLP — top-k by ρ sweep",
        labels={"k": "Top-k by ρ", "auroc": "Test AUROC", "short": "Variant"},
        color_discrete_map={"Base": C_BASE, "Instruct": C_INST},
    )
    fig.update_layout(width=860, height=420, template="plotly_white", yaxis_range=[0.7, 1.0])
    save_plotly_html(fig, PLOTS_DIR / "topk_rho_sweep.html")
    print("  saved topk_rho_sweep.html")

    for v in VARIANTS:
        mk = _read_json(ROOT / "outputs" / v["slug"] / "exp3" / "matched_k_summary.json")
        arms = mk["arms"]
        null = arms["random_null"]
        aurocs = [float(d["auroc"]) for d in (mk.get("random_draws") or []) if d.get("auroc") is not None]
        ent = float(arms["entropy_k12"]["auroc"])
        bot = float(arms["bottom_k12"]["auroc"])
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=aurocs, nbinsx=10, name="random k=12", marker_color=C_FULL, opacity=0.85))
        fig.add_vline(x=ent, line_color=C_ENT, line_width=2.5,
                      annotation_text=f"entropy {ent:.3f} (pct {null['entropy_percentile']:.0f})")
        fig.add_vline(x=bot, line_color=C_BOTTOM, line_width=2, line_dash="dash",
                      annotation_text=f"bottom {bot:.3f}")
        fig.update_layout(
            title=f"Exp3 matched-k MLP — {v['display']} (k=12)",
            xaxis_title="Test AUROC",
            yaxis_title="Count",
            width=900,
            height=420,
            template="plotly_white",
        )
        save_plotly_html(fig, PLOTS_DIR / f"matched_k_null_{v['short'].lower()}.html")
        print(f"  saved matched_k_null_{v['short'].lower()}.html")


def write_slide_summary(feat: pd.DataFrame) -> pathlib.Path:
    lines: list[str] = ["# Exp3 Llama probes — slide bullets\n"]
    for v in VARIANTS:
        root = ROOT / "outputs" / v["slug"] / "exp3"
        mk = _read_json(root / "matched_k_summary.json")
        arms = mk["arms"]
        null = arms["random_null"]
        lines.append(f"## {v['display']}\n")
        for probe in ("logistic", "mlp"):
            row = feat[(feat["short"] == v["short"]) & (feat["probe"] == probe)]
            full = float(row[row["feature_set"] == "full"]["auroc"].iloc[0])
            ent = float(row[row["feature_set"] == "entropy"]["auroc"].iloc[0])
            rest = float(row[row["feature_set"] == "rest"]["auroc"].iloc[0])
            lines.append(
                f"- **{probe} AUROC**: full={full:.3f}, entropy-only={ent:.3f}, rest={rest:.3f}"
            )
        ent_k = float(arms["entropy_k12"]["auroc"])
        lines.append(
            f"- **Matched-k MLP (k=12)**: entropy={ent_k:.3f}, "
            f"random={null['auroc_mean']:.3f}±{null['auroc_std']:.3f}, "
            f"bottom={float(arms['bottom_k12']['auroc']):.3f}; "
            f"entropy is {null['entropy_percentile']:.0f}th pct vs random"
        )
        for probe in ("logistic", "mlp"):
            ab = _read_json(root / probe / "entropy_feature_ablation.json")
            nec = ab["results"]["necessity_mean"]
            suf = ab["results"]["sufficiency_mean"]
            lines.append(
                f"- **Frozen {probe} ablation**: necessity ΔAUROC≈{nec['delta_auroc']:.1e} "
                f"(not uniquely necessary); sufficiency ΔAUROC={suf['delta_auroc']:.3f} "
                f"(pct {suf['controls']['percentile_vs_controls']:.0f} vs controls — "
                f"entropy dims more sufficient than random)"
            )
        lines.append("")

    lines.append("## One-liner takeaways\n")
    lines.append(
        "- Hallucination probes work well on Llama 1B (logistic AUROC ~0.95–0.97)."
    )
    lines.append(
        "- High-ρ “entropy” dims alone are informative (MLP AUROC 0.94 base / 0.89 instruct) "
        "but **not necessary**: ablating them barely changes full-probe AUROC."
    )
    lines.append(
        "- Rest-only ≈ full ⇒ signal is distributed; entropy set is a **compact sufficient** subset "
        "(matched-k: 100th percentile vs random k=12 on both variants)."
    )
    path = OUT_DIR / "slide_bullets.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Slide bullets -> {path}")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing Exp3 Llama compare plots -> {PLOTS_DIR}")
    feat = collect_feature_set_rows()
    feat.to_csv(OUT_DIR / "feature_set_auroc.csv", index=False)
    plot_feature_sets(feat)
    plot_matched_k_null()
    plot_matched_k_arm_comparison()
    plot_topk_sweep()
    plot_ablation_sufficiency()
    plot_ablation_delta_bars()
    also_write_plotly_html()
    write_slide_summary(feat)
    print("Done.")


if __name__ == "__main__":
    main()
