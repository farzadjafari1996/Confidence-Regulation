"""Step 2b: train a small MLP probe with randomized hyperparameter search."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from exp3.features import feature_matrix, mlp_probe_allowed
from exp3.metrics import classification_metrics, regression_metrics
from exp3.settings import Exp3Config, FeatureSet, LabelName, SplitName
from exp3.train_logistic import _class_balance, _load_xy


class ProbeMLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        *,
        hidden: int,
        n_classes: int,
        dropout: float,
        task: str,
    ):
        super().__init__()
        self.task = task
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2 if hidden > 2 else hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2 if hidden > 2 else hidden, 1 if task != "multiclass" else n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.task != "multiclass":
            return out.squeeze(-1)
        return out


def _standardize(
    X_train: np.ndarray, X_other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X_train - mean) / std, (X_other - mean) / std, mean, std


def _train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    task: str,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    device: str,
) -> tuple[ProbeMLP, dict[str, Any], np.ndarray, np.ndarray]:
    # Collect uses torch.inference_mode(); that mode is not overridden by enable_grad().
    torch.inference_mode(False).__enter__()
    torch.set_grad_enabled(True)
    torch.manual_seed(seed)
    X_tr, X_va, mean, std = _standardize(X_train, X_val)
    n_classes = int(len(np.unique(y_train))) if task == "multiclass" else 2
    model = ProbeMLP(
        X_tr.shape[1],
        hidden=hidden,
        n_classes=max(n_classes, 2),
        dropout=dropout,
        task=task,
    ).to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)

    if task == "regression":
        loss_fn: nn.Module = nn.MSELoss()
        y_tr_t = torch.tensor(y_train, dtype=torch.float32)
        y_va_t = torch.tensor(y_val, dtype=torch.float32)
    elif task == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
        y_tr_t = torch.tensor(y_train, dtype=torch.long)
        y_va_t = torch.tensor(y_val, dtype=torch.long)
    else:
        # class-balanced BCE
        pos = float(np.mean(y_train))
        pos = min(max(pos, 1e-3), 1 - 1e-3)
        weight = torch.tensor((1 - pos) / pos, dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        y_tr_t = torch.tensor(y_train, dtype=torch.float32)
        y_va_t = torch.tensor(y_val, dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), y_tr_t),
        batch_size=min(batch_size, len(X_tr)),
        shuffle=True,
    )
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_va_dev = y_va_t.to(device)

    best_state = None
    best_val = float("inf")
    bad = 0
    history: list[float] = []

    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                pred = model(xb)
                loss = loss_fn(pred, yb)
                if not loss.requires_grad:
                    raise RuntimeError(
                        "MLP loss has no grad_fn (inference_mode/grad disabled?). "
                        f"grad_enabled={torch.is_grad_enabled()} "
                        f"inference={torch.is_inference_mode_enabled()} "
                        f"param_requires_grad={next(model.parameters()).requires_grad}"
                    )
                loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_v = model(X_va_t)
            val_loss = float(loss_fn(pred_v, y_va_dev).item())
        history.append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {"best_val_loss": best_val, "epochs": len(history), "history": history}
    return model.cpu(), meta, mean, std


def _predict(
    model: ProbeMLP,
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        logits = model(Xt).numpy()
    if task == "regression":
        return logits, None
    if task == "multiclass":
        pred = logits.argmax(axis=-1)
        return pred, None
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= 0.5).astype(np.int64)
    return pred, prob


def run_train_mlp(
    cfg: Exp3Config,
    split_name: SplitName,
    *,
    label: LabelName | None = None,
    features: FeatureSet = "all",
    device: str | None = None,
    phase1_dir: pathlib.Path | None = None,
    skip_phase1: bool = False,
) -> pathlib.Path:
    label = cfg.resolved_label(label)
    if not mlp_probe_allowed(features):
        raise ValueError(
            f"MLP probes are only for last-MLP arms, not {features!r}. Use logistic for resid_all."
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.probe_out_dir(split_name, "mlp", features=features)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, y, splits, task = _load_xy(cfg, split_name, label)
    X, feat_idx, feat_meta = feature_matrix(
        data,
        cfg,
        features,
        phase1_dir=phase1_dir,
        device=device,
        skip_phase1=skip_phase1,
    )
    d_mlp = int(np.asarray(data["X_mlp_last"]).shape[1])
    print(
        f"MLP features={features}: d_in={X.shape[1]} "
        f"(source={feat_meta.get('source')} d_mlp={d_mlp}) device={device}"
    )
    train_idx = np.asarray(splits["train"], dtype=np.int64)
    val_idx = np.asarray(splits["val"], dtype=np.int64)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if task == "binary" and len(np.unique(y_train)) < 2:
        raise RuntimeError("Train set has a single class; collect more data or change --label")

    rng = np.random.RandomState(cfg.seed)
    trials = []
    best_bundle = None
    best_score = -np.inf

    for i in range(cfg.mlp.n_iter):
        hidden = int(rng.choice([64, 128, 256, 512]))
        dropout = float(rng.choice([0.0, 0.1, 0.2, 0.3]))
        lr = float(10 ** rng.uniform(-4, -2))
        weight_decay = float(10 ** rng.uniform(-6, -2))
        model, meta, mean, std = _train_one(
            X_train,
            y_train,
            X_val,
            y_val,
            task=task,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=cfg.mlp.max_epochs,
            batch_size=cfg.mlp.batch_size,
            patience=cfg.mlp.patience,
            seed=cfg.seed + i,
            device=device,
        )
        pred, prob = _predict(model, X_val, mean, std, task)
        if task == "regression":
            metrics = regression_metrics(y_val, pred)
            score = metrics["r2"]
        else:
            metrics = classification_metrics(y_val, pred, prob, task=task)
            score = metrics.get("auroc")
            if score is None:
                score = metrics.get("f1", metrics.get("f1_macro", metrics["accuracy"]))

        trial = {
            "trial": i,
            "hidden": hidden,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "best_val_loss": meta["best_val_loss"],
            "epochs": meta["epochs"],
            "val_score": float(score) if score is not None else None,
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        trials.append(trial)
        if score is not None and score > best_score:
            best_score = float(score)
            best_bundle = {
                "state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "hidden": hidden,
                "dropout": dropout,
                "n_classes": int(len(np.unique(y_train))) if task == "multiclass" else 2,
                "params": {
                    "hidden": hidden,
                    "dropout": dropout,
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
                "val_metrics": metrics,
            }
        print(f"  MLP trial {i + 1}/{cfg.mlp.n_iter}: score={score}")

    if best_bundle is None:
        raise RuntimeError("MLP search produced no valid trial")

    # Refit best architecture on train+val
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)
    # Use a holdout slice of fit for early stopping (last 15% of fit indices shuffled)
    rng2 = np.random.RandomState(cfg.seed)
    perm = rng2.permutation(len(X_fit))
    cut = max(1, int(0.15 * len(perm)))
    val_fit = perm[:cut]
    train_fit = perm[cut:]
    model, meta, mean, std = _train_one(
        X_fit[train_fit],
        y_fit[train_fit],
        X_fit[val_fit],
        y_fit[val_fit],
        task=task,
        hidden=best_bundle["hidden"],
        dropout=best_bundle["dropout"],
        lr=best_bundle["params"]["lr"],
        weight_decay=best_bundle["params"]["weight_decay"],
        max_epochs=cfg.mlp.max_epochs,
        batch_size=cfg.mlp.batch_size,
        patience=cfg.mlp.patience,
        seed=cfg.seed,
        device=device,
    )

    ckpt = {
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "label": label,
        "task": task,
        "d_mlp": d_mlp,
        "d_in": int(X.shape[1]),
        "feature_set": features,
        "feature_indices": feat_idx.tolist(),
        "feature_meta": feat_meta,
        "hidden": best_bundle["hidden"],
        "dropout": best_bundle["dropout"],
        "n_classes": best_bundle["n_classes"],
        "best_params": best_bundle["params"],
        "seed": cfg.seed,
    }
    model_path = out_dir / "best_model.pt"
    torch.save(ckpt, model_path)

    pd.DataFrame(trials).to_csv(out_dir / "search_results.csv", index=False)
    train_metrics = {
        "label": label,
        "task": task,
        "feature_set": features,
        "d_in": int(X.shape[1]),
        "d_mlp": d_mlp,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "best_params": best_bundle["params"],
        "best_val_score": float(best_score),
        "val": best_bundle["val_metrics"],
        "class_balance_train": _class_balance(y_train, task),
        "class_balance_val": _class_balance(y_val, task),
        "mlp_settings": asdict(cfg.mlp),
        "feature_meta": feat_meta,
        "device": device,
    }
    (out_dir / "train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2), encoding="utf-8"
    )
    print(f"MLP probe saved -> {model_path}")
    print(f"  best val score: {best_score}")
    return model_path
