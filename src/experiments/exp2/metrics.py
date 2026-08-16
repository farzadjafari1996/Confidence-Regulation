"""Static ρ metrics: every MLP W_out vs later MLP W_in and vs W_U."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class ReaderSVD:
    """Residual-space singular basis of a reader matrix (columns = singular vectors).

    Cached once per reader (an MLP W_in or W_U), not per writer→reader pair.
    """

    basis: torch.Tensor  # (d_model, n_sv)
    singular_values: torch.Tensor  # (n_sv,)
    reader_name: str  # "W_in" or "W_U"
    reader_layer: int | None  # None for W_U


def neuron_label(layer: int, idx: int) -> str:
    return f"{layer}.{idx}"


def reader_label(reader_layer: int | None) -> str:
    if reader_layer is None:
        return "W_U"
    return f"mlp{reader_layer}.W_in"


def reader_cache_key(reader_layer: int | None) -> str:
    return reader_label(reader_layer)


def svd_reader_basis(reader: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (basis, singular_values) with basis columns in residual space.

    Weights may be fp16/bf16; the factorization is always float32.

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
    """ρ_i = ||w_out_i @ U_dark|| / ||w_out_i|| (unread / dark-tail fraction)."""
    w_out = w_out.float()
    norm = w_out.norm(dim=1).clamp(min=1e-12)
    rho = (w_out @ u_dark).norm(dim=-1) / norm
    return rho, norm


def read_frobenius(w_in: torch.Tensor, u_dark: torch.Tensor) -> float:
    """||U_dark^T W_in||_F / ||W_in||_F — how much of W_in sits on U_dark."""
    w_in = w_in.float()
    u_dark = u_dark.float()
    num = (u_dark.mT @ w_in).norm()
    den = w_in.norm().clamp(min=1e-12)
    return float(num / den)


def expected_rho_isotropic(k: int, d_model: int) -> float:
    """E[ρ] ≈ √(k/d) for a random unit write in d_model dimensions."""
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    return float(np.sqrt(k / d_model))


def layer_udark_read_df(
    w_in_by_layer: dict[int, torch.Tensor],
    u_dark: torch.Tensor,
    k: int,
    d_model: int,
) -> pd.DataFrame:
    """Per-layer W_in read of the shared unembedding null space."""
    expected_rho = expected_rho_isotropic(k, d_model)
    rows = []
    for layer in sorted(w_in_by_layer):
        read_rho = read_frobenius(w_in_by_layer[layer], u_dark)
        rows.append(
            {
                "layer": int(layer),
                "read_rho": read_rho,
                "expected_rho": expected_rho,
                "nullspace_k": k,
                "d_model": d_model,
            }
        )
    return pd.DataFrame(rows)


def cosines_with_basis(w_rows: torch.Tensor, basis: torch.Tensor) -> np.ndarray:
    """Per-neuron |cos(w_out, basis_j)|; shape (n_neurons, n_sv)."""
    w_rows = w_rows.float()
    basis = basis.float()
    norms = w_rows.norm(dim=-1).clamp(min=1e-12)
    return ((w_rows @ basis).abs() / norms.unsqueeze(-1)).cpu().numpy()


def hop_distance(writer_layer: int, reader_layer: int | None, n_layers: int) -> int:
    """MLP hop = reader - writer; W_U sits after the last block (n_layers - writer)."""
    if reader_layer is None:
        return n_layers - writer_layer
    return reader_layer - writer_layer


def mlp_reader_layers(writer_layer: int, n_layers: int) -> list[int]:
    """Later MLP readers for writer ℓ: ℓ+1 … n_layers-1 (same-layer W_in excluded)."""
    return list(range(writer_layer + 1, n_layers))


def all_writer_reader_pairs(n_layers: int) -> list[tuple[int, int | None]]:
    """Every MLP writer vs later MLP W_in and vs W_U. Count = n(n+1)/2."""
    pairs: list[tuple[int, int | None]] = []
    for writer in range(n_layers):
        for reader in mlp_reader_layers(writer, n_layers):
            pairs.append((writer, reader))
        pairs.append((writer, None))
    return pairs


def n_all_pairs(n_layers: int) -> int:
    return n_layers * (n_layers + 1) // 2


def mlp_to_mlp_pairs(n_layers: int) -> list[tuple[int, int]]:
    """Consecutive MLP writer→reader pairs: (ℓ, ℓ+1) for ℓ = 0 .. n_layers-2."""
    return [(ell, ell + 1) for ell in range(n_layers - 1)]


def keep_per_neuron_pair(writer_layer: int, reader_layer: int | None) -> bool:
    """Per-neuron CSV: consecutive MLP→MLP and every layer→W_U."""
    if reader_layer is None:
        return True
    return reader_layer == writer_layer + 1


def pair_metrics_df(
    w_out: torch.Tensor,
    writer_layer: int,
    svd: ReaderSVD,
    k: int,
    n_layers: int,
    d_model: int,
) -> pd.DataFrame:
    u_dark = dark_basis(svd.basis, k)
    rho, norm = compute_rho(w_out, u_dark)
    rho_np = rho.detach().cpu().numpy()
    n = w_out.shape[0]
    hop = hop_distance(writer_layer, svd.reader_layer, n_layers)
    reader_layer_col: list[int | None] = [svd.reader_layer] * n
    return pd.DataFrame(
        {
            "writer_layer": writer_layer,
            "reader_layer": reader_layer_col,
            "reader": reader_label(svd.reader_layer),
            "reader_kind": svd.reader_name,
            "hop": hop,
            "neuron_index": np.arange(n),
            "component_name": [neuron_label(writer_layer, i) for i in range(n)],
            "norm": norm.detach().cpu().numpy(),
            "rho": rho_np,
            "nullspace_k": k,
            "d_model": d_model,
        }
    )


def filter_per_neuron_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep consecutive MLP→MLP rows and every writer→W_U row."""
    reader_layer = df["reader_layer"]
    is_wu = df["reader_kind"].eq("W_U") | df["reader"].eq("W_U")
    is_next_mlp = reader_layer.notna() & (reader_layer == df["writer_layer"] + 1)
    return df.loc[is_wu | is_next_mlp].reset_index(drop=True)


def summarize_pair_rho(df: pd.DataFrame) -> dict[str, float | int | str | None]:
    rho = df["rho"].to_numpy()
    k = int(df["nullspace_k"].iloc[0])
    d_model = int(df["d_model"].iloc[0]) if "d_model" in df.columns else None
    reader_layer_val = df["reader_layer"].iloc[0] if "reader_layer" in df.columns else None
    if reader_layer_val is not None and pd.isna(reader_layer_val):
        reader_layer_out: int | None = None
    elif reader_layer_val is None:
        reader_layer_out = None
    else:
        reader_layer_out = int(reader_layer_val)
    hop = int(df["hop"].iloc[0]) if "hop" in df.columns else None
    expected = expected_rho_isotropic(k, d_model) if d_model else None
    return {
        "writer_layer": int(df["writer_layer"].iloc[0]),
        "reader_layer": reader_layer_out,
        "reader": str(df["reader"].iloc[0]),
        "reader_kind": str(df["reader_kind"].iloc[0]),
        "hop": hop,
        "n_neurons": int(len(df)),
        "d_model": d_model,
        "nullspace_k": k,
        "rho_mean": float(rho.mean()),
        "rho_std": float(rho.std()),
        "rho_p50": float(np.quantile(rho, 0.50)),
        "rho_p90": float(np.quantile(rho, 0.90)),
        "rho_p95": float(np.quantile(rho, 0.95)),
        "rho_max": float(rho.max()),
        "expected_rho": expected,
        "n_rho_gt_0_5": int((rho > 0.5).sum()),
        "n_rho_gt_0_7": int((rho > 0.7).sum()),
        "top_neuron": str(df.loc[df["rho"].idxmax(), "component_name"]),
        "top_rho": float(df["rho"].max()),
    }


def alignment_pair_indices(n_layers: int) -> list[tuple[int, int | None]]:
    """Early / mid / late consecutive MLP→MLP pairs plus last→W_U."""
    pairs = mlp_to_mlp_pairs(n_layers)
    if not pairs:
        return [(n_layers - 1, None)]
    early = pairs[0]
    mid = pairs[len(pairs) // 2]
    late = pairs[-1]
    chosen: list[tuple[int, int | None]] = []
    for p in (early, mid, late):
        if p not in chosen:
            chosen.append(p)
    chosen.append((n_layers - 1, None))
    return chosen
