"""
Track 1 side-by-side comparison (roadmap 1C deliverable).

Reads phase 1/2 (and optional phase 3) artifacts under outputs/<model-slug>/
and writes aggregated tables to outputs/comparison/.

Usage (after running phase 1 + 2 for each model):
  python src/track1_compare.py
  python src/track1_compare.py --models gpt2-small meta-llama/Llama-3.2-1B
  python src/track1_compare.py --outputs-dir outputs --comparison-dir outputs/comparison

Outputs:
  outputs/comparison/comparison_table.csv
  outputs/comparison/comparison_report.md
  outputs/comparison/comparison.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import pandas as pd

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from hyperparameters import DEFAULT_OUTPUT_DIR, GPT2_PAPER_ENTROPY_NEURONS, model_slug  # noqa: E402

DEFAULT_MODELS = [
    "gpt2-small",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
]

DEFAULT_COMPARISON_DIR = DEFAULT_OUTPUT_DIR / "comparison"


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


def _induction_seq_length(phase3_dir: pathlib.Path, phase3_summary: dict[str, Any] | None) -> int | None:
    """Read seq_length from phase3 summary (matches fig5a induction_start)."""
    if phase3_summary:
        induction = (phase3_summary.get("sections") or {}).get("induction") or {}
        if induction.get("seq_length") is not None:
            return int(induction["seq_length"])
    summary_path = phase3_dir / "summary.json"
    if summary_path.exists():
        payload = _read_json(summary_path)
        induction = (payload.get("sections") or {}).get("induction") or {}
        if induction.get("seq_length") is not None:
            return int(induction["seq_length"])
    return None


def _induction_entropy_ratio(
    phase3_dir: pathlib.Path,
    phase3_summary: dict[str, Any] | None = None,
    *,
    seq_length: int | None = None,
) -> float | None:
    """
    Mean entropy (2nd half) / mean entropy (1st half) on duplicated sequences.

    Matches phase3 fig5a: group by token position, split at induction_start=seq_length.
    """
    feather = phase3_dir / "induction_entropy.feather"
    if not feather.exists():
        return None

    df = pd.read_feather(feather)
    if "entropy" not in df.columns or "pos" not in df.columns:
        return None

    induction_start = seq_length or _induction_seq_length(phase3_dir, phase3_summary)
    if induction_start is None:
        # Fallback: duplicated C4 dataset length is ~2*seq_length-1
        induction_start = (int(df["pos"].max()) + 1) // 2

    by_pos = df.groupby("pos", as_index=False)["entropy"].mean()
    first = by_pos[by_pos["pos"] < induction_start]["entropy"].mean()
    second = by_pos[by_pos["pos"] >= induction_start]["entropy"].mean()
    if first is None or second is None or first == 0 or pd.isna(first) or pd.isna(second):
        return None
    return float(second / first)


def load_model_track1_metrics(
    model_name: str,
    outputs_dir: pathlib.Path,
) -> dict[str, Any]:
    slug = model_slug(model_name)
    phase1_dir = outputs_dir / slug / "phase1"
    phase2_dir = outputs_dir / slug / "phase2"
    phase3_dir = outputs_dir / slug / "phase3"

    phase1 = _read_json(phase1_dir / "summary.json")
    phase2 = _read_json(phase2_dir / "summary.json")
    phase3 = _optional_json(phase3_dir / "summary.json")

    rho_csv = phase2_dir / "rho_vs_ln_fraction.csv"
    rho_df = pd.read_csv(rho_csv) if rho_csv.exists() else None

    selection = phase1.get("selection") or {}
    entropy_set = phase2.get("entropy_set") or {}
    random_set = phase2.get("random_controls") or {}
    meta = phase1.get("model_meta") or {}

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

    induction_start = _induction_seq_length(phase3_dir, phase3)
    induction_ratio = _induction_entropy_ratio(phase3_dir, phase3, seq_length=induction_start)

    row: dict[str, Any] = {
        "model": model_name,
        "slug": slug,
        "n_layers": meta.get("n_layers"),
        "d_model": meta.get("d_model"),
        "d_mlp": meta.get("d_mlp"),
        "normalization": meta.get("normalization"),
        "udark_start": meta.get("udark_start"),
        "nullspace_k": meta.get("nullspace_k"),
        "n_entropy_selected": selection.get("n_selected"),
        "top_entropy_neurons": selection.get("neurons") or [],
        "top_entropy_neurons_str": _fmt_list(selection.get("neurons") or []),
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
        "induction_entropy_ratio_2nd_over_1st": induction_ratio,
        "induction_split_pos": induction_start,
        "phase3_ran": phase3 is not None,
        "paper_neuron_overlap_gpt2": paper_overlap,
        "paper_neuron_overlap_list": paper_overlap_neurons,
        "phase1_dir": str(phase1_dir),
        "phase2_dir": str(phase2_dir),
    }

    if rho_df is not None and "is_entropy" in rho_df.columns:
        ent_rho = rho_df[rho_df["is_entropy"] == True]  # noqa: E712
        rnd_rho = rho_df[rho_df["is_entropy"] == False]  # noqa: E712
        row["mean_rho_entropy_ablated"] = float(ent_rho["rho"].mean()) if len(ent_rho) else None
        row["mean_rho_random_ablated"] = float(rnd_rho["rho"].mean()) if len(rnd_rho) else None

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
        # TE > DE_LN (basic paper claim; ratio > 1)
        "entropy_te_exceeds_de": None if te_ratio is None else te_ratio > 1.0,
        # Strong separation (paper figures often show TE >> DE_LN)
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
        "induction_entropy_rises": (
            None
            if row.get("induction_entropy_ratio_2nd_over_1st") is None
            else row["induction_entropy_ratio_2nd_over_1st"] > 1.05
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
        "induction_entropy_ratio_2nd_over_1st",
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


def write_markdown_report(
    rows: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    out_path: pathlib.Path,
    *,
    errors: list[str] | None = None,
) -> None:
    lines = [
        "# Track 1 — GPT-2 vs modern models (1C comparison)",
        "",
        "Aggregated from `outputs/<model>/phase1|phase2` summaries.",
        "",
        "## Summary table",
        "",
        build_comparison_table(rows).pipe(_dataframe_to_markdown),
        "",
        "## Paper-claim checks (heuristic)",
        "",
    ]

    for row, ev in zip(rows, evaluations):
        lines.append(f"### {row['model']}")
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

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(
    models: list[str],
    outputs_dir: pathlib.Path,
    comparison_dir: pathlib.Path,
) -> pd.DataFrame:
    comparison_dir.mkdir(parents=True, exist_ok=True)

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

    payload = {
        "models": models,
        "outputs_dir": str(outputs_dir.resolve()),
        "rows": [{k: v for k, v in r.items() if k not in ("top_entropy_neurons",)} for r in rows],
        "evaluations": evaluations,
        "errors": errors,
    }
    (comparison_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(rows, evaluations, comparison_dir / "comparison_report.md", errors=errors)

    return table, errors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track 1 side-by-side model comparison (1C)")
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
