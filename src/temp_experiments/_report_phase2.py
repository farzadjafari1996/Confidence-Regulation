"""Quick Phase2 summary dump for re-run reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "outputs"


def report(slug: str) -> None:
    s = json.loads((ROOT / slug / "exp1" / "phase2" / "summary.json").read_text())
    e, r, j = s["entropy_set"], s["random_controls"], s["joint"]
    print(f"=== {slug} (runtime {s['runtime_sec']}s) ===")
    print(
        f"  entropy: |TE|={e['mean_abs_te']:.5g} |DE|={e['mean_abs_de_ln']:.5g} "
        f"LN-frac={e['mean_ln_mediated_fraction']:.5g} |dH|={e['mean_abs_delta_h']:.5g} "
        f"pred={e['mean_pred_change_rate']:.5g}"
    )
    print(
        f"  random:  |TE|={r['mean_abs_te']:.5g} |DE|={r['mean_abs_de_ln']:.5g} "
        f"|dH|={r['mean_abs_delta_h']:.5g} pred={r['mean_pred_change_rate']:.5g}"
    )
    print(
        f"  joint.ent |TE|={j['entropy']['mean_abs_te']:.5g} "
        f"|dH|={j['entropy']['mean_abs_delta_h']:.5g} "
        f"pred={j['entropy']['mean_pred_change_rate']:.5g}"
    )
    print(
        f"  joint.rnd |TE|={j['random']['mean_abs_te']:.5g} "
        f"|dH|={j['random']['mean_abs_delta_h']:.5g}"
    )
    te_ratio = e["mean_abs_te"] / max(r["mean_abs_te"], 1e-12)
    dh_ratio = e["mean_abs_delta_h"] / max(r["mean_abs_delta_h"], 1e-12)
    print(f"  ratio TE ent/rnd={te_ratio:.2f}  |dH| ent/rnd={dh_ratio:.2f}")


if __name__ == "__main__":
    for slug in sys.argv[1:]:
        report(slug)
