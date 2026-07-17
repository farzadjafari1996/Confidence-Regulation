"""Sanity checks for Ext 2 (math + Phase-1 last→W_U agreement)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import torch

from weight_proj.metrics import compute_rho, dark_basis, svd_reader_basis


def _ok(name: str, passed: bool, detail: str) -> dict[str, Any]:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return {"name": name, "passed": passed, "detail": detail}


def run_unit_checks() -> list[dict[str, Any]]:
    """Pure-tensor checks (no model load)."""
    results: list[dict[str, Any]] = []
    torch.manual_seed(0)

    d_model, d_mlp, k = 8, 16, 2
    scales = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.05, 0.01])
    # TL layout: W_in is (d_model, d_mlp); resid @ W_in.
    v_rand, _ = torch.linalg.qr(torch.randn(d_mlp, d_model), mode="reduced")
    w_in = torch.diag(scales) @ v_rand[:, :d_model].T

    basis, svals = svd_reader_basis(w_in, "W_in")
    results.append(
        _ok(
            "svd_W_in_basis_shape",
            basis.shape[0] == d_model and svals.shape[0] == d_model,
            f"basis={tuple(basis.shape)} svals={tuple(svals.shape)}",
        )
    )
    results.append(
        _ok(
            "svd_W_in_spectrum",
            torch.allclose(svals.cpu(), scales.sort(descending=True).values, rtol=1e-4, atol=1e-4),
            f"svals={svals.cpu().tolist()}",
        )
    )

    # Neurons writing into darkest direction → ρ ≈ 1; bright axis → ρ ≈ 0.
    dark = dark_basis(basis, k)
    w_dark = dark[:, -1].unsqueeze(0).repeat(3, 1)
    w_bright = basis[:, :1].T
    rho, _ = compute_rho(torch.cat([w_dark, w_bright], dim=0), dark)
    results.append(
        _ok(
            "rho_dark_vs_bright",
            float(rho[:3].min()) > 0.99 and float(rho[-1]) < 0.05,
            f"rho={rho.tolist()}",
        )
    )

    w_u = torch.randn(d_model, 32)
    u, s_u = svd_reader_basis(w_u, "W_U")
    results.append(
        _ok(
            "svd_W_U_basis_shape",
            u.shape[0] == d_model and s_u.ndim == 1,
            f"u={tuple(u.shape)} s={tuple(s_u.shape)}",
        )
    )
    return results


def compare_to_phase1(
    last_wu_df: pd.DataFrame,
    phase1_dir: pathlib.Path,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    min_corr: float = 0.999,
) -> dict[str, Any]:
    """Compare last-layer→W_U ρ against Phase 1 static_neuron_metrics.csv.

    Tolerances allow small float differences between full SVD and the Gram/eigh
    path used for wide W_U (Qwen).
    """
    csv_path = phase1_dir / "static_neuron_metrics.csv"
    if not csv_path.exists():
        detail = f"missing {csv_path}"
        print(f"  [SKIP] phase1_rho_agreement: {detail}")
        return {"name": "phase1_rho_agreement", "passed": None, "detail": detail}

    phase1 = pd.read_csv(csv_path, dtype={"component_name": str})
    left = last_wu_df.copy()
    left["component_name"] = left["component_name"].astype(str)
    right = phase1[["component_name", "rho"]].rename(columns={"rho": "rho_phase1"})
    right["component_name"] = right["component_name"].astype(str)
    merged = left.merge(right, on="component_name", how="inner")
    if merged.empty:
        detail = "no overlapping component_name rows"
        print(f"  [FAIL] phase1_rho_agreement: {detail}")
        return {"name": "phase1_rho_agreement", "passed": False, "detail": detail}

    a = merged["rho"].to_numpy()
    b = merged["rho_phase1"].to_numpy()
    max_abs = float(np.max(np.abs(a - b)))
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
    close = bool(np.allclose(a, b, rtol=rtol, atol=atol))
    corr_ok = bool(corr >= min_corr) if corr == corr else False
    passed = close or corr_ok
    detail = f"n={len(merged)} max_abs_diff={max_abs:.3e} corr={corr:.6f}"
    print(f"  [{'PASS' if passed else 'FAIL'}] phase1_rho_agreement: {detail}")
    return {
        "name": "phase1_rho_agreement",
        "passed": passed,
        "detail": detail,
        "max_abs_diff": max_abs,
        "corr": corr,
        "n": int(len(merged)),
    }


def write_sanity_report(out_dir: pathlib.Path, results: list[dict[str, Any]]) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sanity_report.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  saved {path}")
    return path
