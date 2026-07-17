"""Static ρ metrics for MLP→MLP and last-MLP→W_U projections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class ReaderSVD:
    """Residual-space singular basis of a reader matrix (columns = singular vectors)."""

    basis: torch.Tensor  # (d_model, n_sv)
    singular_values: torch.Tensor  # (n_sv,)
    reader_name: str
    writer_layer: int
    reader_layer: int | None  # None for W_U


def neuron_label(layer: int, idx: int) -> str:
    return f"{layer}.{idx}"


def svd_reader_basis(reader: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (basis, singular_values) with basis columns in residual space.

    kind:
      - \"W_in\": MLP read map in residual space, shape (d_model, *).
        GPT-2/TL: resid @ W_in. Llama SwiGLU: stacked [gate|up] as (d_model, 2*d_mlp).
        Dark residual directions = left singular vectors with small σ.
      - \"W_U\": unembedding (d_model, d_vocab); residual basis = left singular vectors.

    For wide readers (e.g. Qwen W_U with ~250k vocab), use eigh(W W^T) instead of
    full SVD — same left singular vectors, much faster.
    """
    reader = reader.float()
    if kind not in {"W_in", "W_U"}:
        raise ValueError(f"Unknown reader kind: {kind!r}")
    m, n = reader.shape
    if n > max(2 * m, 4096):
        gram = reader @ reader.mT
        gram = 0.5 * (gram + gram.mT)
        evals, evecs = torch.linalg.eigh(gram)
        singular_values = evals.clamp(min=0).sqrt().flip(0).contiguous()
        basis = evecs.flip(1).contiguous()
        return basis, singular_values
    u, singular_values, _ = torch.linalg.svd(reader, full_matrices=False)
    return u.contiguous(), singular_values


def dark_basis(basis: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > basis.shape[1]:
        raise ValueError(f"k={k} exceeds basis width {basis.shape[1]}")
    return basis[:, -k:]


def compute_rho(w_out: torch.Tensor, u_dark: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ρ_i = ||w_out_i @ U_dark|| / ||w_out_i||."""
    w_out = w_out.float()
    norm = w_out.norm(dim=1).clamp(min=1e-12)
    rho = (w_out @ u_dark).norm(dim=-1) / norm
    return rho, norm


def cosines_with_basis(w_rows: torch.Tensor, basis: torch.Tensor) -> np.ndarray:
    """Per-neuron |cos(w_out, basis_j)|; shape (n_neurons, n_sv)."""
    norms = w_rows.norm(dim=-1).clamp(min=1e-12)
    return ((w_rows @ basis).abs() / norms.unsqueeze(-1)).cpu().numpy()


def pair_metrics_df(
    w_out: torch.Tensor,
    writer_layer: int,
    svd: ReaderSVD,
    k: int,
) -> pd.DataFrame:
    u_dark = dark_basis(svd.basis, k)
    rho, norm = compute_rho(w_out, u_dark)
    n = w_out.shape[0]
    reader_label = (
        f"mlp{svd.reader_layer}.W_in" if svd.reader_layer is not None else "W_U"
    )
    return pd.DataFrame(
        {
            "writer_layer": writer_layer,
            "reader": reader_label,
            "reader_kind": svd.reader_name,
            "neuron_index": np.arange(n),
            "component_name": [neuron_label(writer_layer, i) for i in range(n)],
            "norm": norm.detach().cpu().numpy(),
            "rho": rho.detach().cpu().numpy(),
            "nullspace_k": k,
        }
    )


def summarize_pair_rho(df: pd.DataFrame) -> dict[str, float | int | str]:
    rho = df["rho"].to_numpy()
    return {
        "writer_layer": int(df["writer_layer"].iloc[0]),
        "reader": str(df["reader"].iloc[0]),
        "reader_kind": str(df["reader_kind"].iloc[0]),
        "n_neurons": int(len(df)),
        "rho_mean": float(rho.mean()),
        "rho_std": float(rho.std()),
        "rho_p50": float(np.quantile(rho, 0.50)),
        "rho_p90": float(np.quantile(rho, 0.90)),
        "rho_p95": float(np.quantile(rho, 0.95)),
        "rho_max": float(rho.max()),
        "n_rho_gt_0_5": int((rho > 0.5).sum()),
        "n_rho_gt_0_7": int((rho > 0.7).sum()),
        "top_neuron": str(df.loc[df["rho"].idxmax(), "component_name"]),
        "top_rho": float(df["rho"].max()),
    }


def mlp_to_mlp_pairs(n_layers: int) -> list[tuple[int, int]]:
    """Consecutive MLP writer→reader pairs: (ℓ, ℓ+1) for ℓ = 0 .. n_layers-2."""
    return [(ell, ell + 1) for ell in range(n_layers - 1)]


def alignment_pair_indices(n_layers: int) -> list[tuple[int, int] | tuple[int, None]]:
    """Early / mid / late MLP→MLP pairs plus last→W_U (reader_layer None)."""
    pairs = mlp_to_mlp_pairs(n_layers)
    if not pairs:
        return [(n_layers - 1, None)]
    early = pairs[0]
    mid = pairs[len(pairs) // 2]
    late = pairs[-1]
    chosen: list[tuple[int, int] | tuple[int, None]] = []
    for p in (early, mid, late):
        if p not in chosen:
            chosen.append(p)
    chosen.append((n_layers - 1, None))
    return chosen
