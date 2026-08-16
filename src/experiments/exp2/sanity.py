"""Sanity checks for Exp 2 (math + Phase-1 last→W_U agreement)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import torch

from exp2.metrics import (
    all_writer_reader_pairs,
    compute_rho,
    dark_basis,
    expected_rho_isotropic,
    filter_per_neuron_df,
    n_all_pairs,
    pair_metrics_df,
    ReaderSVD,
    read_frobenius,
    svd_reader_basis,
)


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

    for n_layers in (1, 2, 12, 16, 24, 26):
        got = len(all_writer_reader_pairs(n_layers))
        want = n_all_pairs(n_layers)
        formula = n_layers * (n_layers + 1) // 2
        results.append(
            _ok(
                f"pair_count_n{n_layers}",
                got == want == formula,
                f"got={got} n(n+1)/2={formula}",
            )
        )

    svd = ReaderSVD(basis=basis, singular_values=svals, reader_name="W_in", reader_layer=1)
    w_out = torch.randn(5, d_model)
    pair_df = pair_metrics_df(w_out, writer_layer=0, svd=svd, k=k, n_layers=2, d_model=d_model)
    rho_ok = bool(
        "rho" in pair_df.columns
        and "bright" not in pair_df.columns
        and float(pair_df["rho"].min()) >= 0.0
        and float(pair_df["rho"].max()) <= 1.0
    )
    results.append(
        _ok(
            "pair_metrics_rho_in_unit_interval",
            rho_ok and int(pair_df["hop"].iloc[0]) == 1,
            f"hop={int(pair_df['hop'].iloc[0])} rho_ok={rho_ok}",
        )
    )

    skip = pair_metrics_df(w_out, writer_layer=0, svd=svd, k=k, n_layers=3, d_model=d_model)
    # reader_layer=1 is consecutive; synthesize a hop-2 row and a W_U row
    skip2 = skip.copy()
    skip2["reader_layer"] = 2
    skip2["reader"] = "mlp2.W_in"
    skip2["hop"] = 2
    wu = skip.copy()
    wu["reader_layer"] = pd.NA
    wu["reader"] = "W_U"
    wu["reader_kind"] = "W_U"
    wu["hop"] = 3
    mixed = pd.concat([skip, skip2, wu], ignore_index=True)
    kept = filter_per_neuron_df(mixed)
    kept_readers = set(kept["reader"].unique())
    results.append(
        _ok(
            "filter_keeps_consecutive_and_wu",
            kept_readers == {"mlp1.W_in", "W_U"} and "mlp2.W_in" not in kept_readers,
            f"readers={sorted(kept_readers)}",
        )
    )

    torch.manual_seed(1)
    d_big, k_big = 32, 4
    u_rand, _ = torch.linalg.qr(torch.randn(d_big, d_big), mode="reduced")
    u_dark_rand = u_rand[:, -k_big:]
    w_in_rand = torch.randn(d_big, 64)
    read_rho = read_frobenius(w_in_rand, u_dark_rand)
    chance = expected_rho_isotropic(k_big, d_big)
    results.append(
        _ok(
            "read_rho_near_sqrt_k_over_d",
            abs(read_rho - chance) < 0.08,
            f"read_rho={read_rho:.4f} chance={chance:.4f}",
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
