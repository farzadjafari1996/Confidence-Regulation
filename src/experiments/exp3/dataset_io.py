"""Load/save Exp3 collected datasets (memmap activations + label sidecar)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from exp3.settings import Exp3Config, SplitName

META_LIST_KEYS = (
    "item_ids",
    "predicate_ids",
    "predicate_labels",
    "gold_answers",
    "pred_tokens",
    "categories",
)

RESID_NAME = "X_resid.dat"
MLP_NAME = "X_mlp_last.dat"
LABELS_NAME = "labels.npz"


def dataset_path(data_dir: pathlib.Path) -> pathlib.Path:
    """Ready marker: labels sidecar (memmap collection is complete)."""
    return data_dir / LABELS_NAME


def labels_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / LABELS_NAME


def resid_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / RESID_NAME


def mlp_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / MLP_NAME


def split_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "split_indices.json"


def meta_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "collection_meta.json"


def write_collection_meta(path: pathlib.Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_collection_meta(data_dir: pathlib.Path) -> dict[str, Any]:
    path = meta_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _open_memmap(
    path: pathlib.Path,
    shape: tuple[int, ...],
    *,
    mode: str,
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=np.float16, mode=mode, shape=shape)


class CollectionWriter:
    """Append-only float16 memmaps plus a JSON/npz label sidecar."""

    def __init__(
        self,
        data_dir: pathlib.Path,
        *,
        capacity: int,
        n_layers: int,
        d_model: int,
        d_mlp: int,
        resume: bool,
    ) -> None:
        self.data_dir = data_dir
        self.capacity = int(capacity)
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.d_mlp = int(d_mlp)
        self.n_written = 0
        self.lists: dict[str, list] = {k: [] for k in META_LIST_KEYS}
        self.scalars: dict[str, list] = {
            "y_top1": [],
            "y_top5": [],
            "y_option": [],
            "gold_id": [],
            "pred_id": [],
            "gold_rank": [],
            "n_options": [],
            "option_pred": [],
            "top1_prob": [],
            "entropy": [],
            "gold_logprob": [],
        }
        self.option_logprobs: list[np.ndarray] = []
        mmap_mode = "r+" if resume and resid_path(data_dir).exists() else "w+"
        self.X_resid = _open_memmap(
            resid_path(data_dir),
            (self.capacity, self.n_layers, self.d_model),
            mode=mmap_mode,
        )
        self.X_mlp = _open_memmap(
            mlp_path(data_dir),
            (self.capacity, self.d_mlp),
            mode=mmap_mode,
        )
        if resume and labels_path(data_dir).exists():
            self._load_partial_labels()

    def _load_partial_labels(self) -> None:
        data = _load_label_npz(labels_path(self.data_dir))
        n = int(len(data.get("item_ids") or []))
        self.n_written = n
        for key in self.lists:
            self.lists[key] = [str(x) if x is not None else "" for x in data.get(key, [])]
        for key in self.scalars:
            if key in data:
                self.scalars[key] = list(np.asarray(data[key]))
        opts = data.get("option_logprobs")
        if opts is not None:
            self.option_logprobs = [np.asarray(row, dtype=np.float64) for row in opts]
        print(f"  Resuming collection at n={self.n_written}")

    def item_ids(self) -> set[str]:
        return {str(x) for x in self.lists["item_ids"]}

    def append(self, row: dict[str, Any]) -> None:
        if self.n_written >= self.capacity:
            raise RuntimeError("CollectionWriter capacity exceeded")
        i = self.n_written
        self.X_resid[i] = np.asarray(row["X_resid"], dtype=np.float16)
        self.X_mlp[i] = np.asarray(row["X_mlp_last"], dtype=np.float16)
        self.lists["item_ids"].append(str(row.get("item_id") or ""))
        self.lists["predicate_ids"].append(str(row.get("predicate_id") or ""))
        self.lists["predicate_labels"].append(str(row.get("predicate_label") or ""))
        self.lists["gold_answers"].append(str(row.get("gold_answer") or ""))
        self.lists["pred_tokens"].append(str(row.get("pred_token") or ""))
        self.lists["categories"].append(str(row.get("category") or ""))
        self.scalars["y_top1"].append(int(row["y_top1"]))
        self.scalars["y_top5"].append(int(row["y_top5"]))
        self.scalars["y_option"].append(int(row.get("y_option", -1)))
        self.scalars["gold_id"].append(int(row["gold_id"]))
        self.scalars["pred_id"].append(int(row["pred_id"]))
        self.scalars["gold_rank"].append(int(row["gold_rank"]))
        self.scalars["n_options"].append(int(row.get("n_options", 0)))
        self.scalars["option_pred"].append(int(row.get("option_pred", -1)))
        self.scalars["top1_prob"].append(float(row["top1_prob"]))
        self.scalars["entropy"].append(float(row["entropy"]))
        self.scalars["gold_logprob"].append(float(row["gold_logprob"]))
        opts = row.get("option_logprobs")
        if opts is None:
            opts = np.full(10, np.nan)
        self.option_logprobs.append(np.asarray(opts, dtype=np.float64))
        self.n_written += 1

    def flush(self) -> None:
        self.X_resid.flush()
        self.X_mlp.flush()
        self._write_labels()

    def _write_labels(self) -> None:
        n = self.n_written
        packed: dict[str, Any] = {}
        for key, values in self.scalars.items():
            packed[key] = np.asarray(values[:n])
        packed["option_logprobs"] = np.stack(self.option_logprobs[:n], axis=0) if n else np.zeros((0, 10))
        save_label_npz(labels_path(self.data_dir), packed, {k: v[:n] for k, v in self.lists.items()})

    def close(self) -> None:
        self.flush()
        del self.X_resid
        del self.X_mlp


def save_label_npz(
    path: pathlib.Path,
    arrays: dict[str, np.ndarray],
    lists: dict[str, list],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {k: v for k, v in arrays.items()}
    for key, values in lists.items():
        payload[key] = np.asarray(values, dtype=object)
    np.savez_compressed(path, **payload)


def _load_label_npz(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        out: dict[str, Any] = {}
        for key in z.files:
            arr = z[key]
            if key in META_LIST_KEYS:
                out[key] = arr.tolist()
            else:
                out[key] = arr
        return out


def load_arrays(path: pathlib.Path) -> dict[str, Any]:
    """Load a finished collection. `path` may be the data dir or labels.npz."""
    data_dir = path if path.is_dir() else path.parent
    labels = _load_label_npz(labels_path(data_dir))
    meta = load_collection_meta(data_dir)
    n = int(meta.get("n_items") or len(labels["item_ids"]))
    n_layers = int(meta["n_layers"])
    d_model = int(meta["d_model"])
    d_mlp = int(meta["d_mlp"])
    X_resid = np.memmap(
        resid_path(data_dir),
        dtype=np.float16,
        mode="r",
        shape=(max(n, 1), n_layers, d_model),
    )
    X_mlp = np.memmap(
        mlp_path(data_dir),
        dtype=np.float16,
        mode="r",
        shape=(max(n, 1), d_mlp),
    )
    out = dict(labels)
    out["X_resid"] = np.asarray(X_resid[:n])
    out["X_mlp_last"] = np.asarray(X_mlp[:n])
    out["X_full"] = out["X_mlp_last"]  # alias for last-MLP probes / ablation
    return out


def collection_ready(data_dir: pathlib.Path) -> bool:
    meta_file = meta_path(data_dir)
    if not (labels_path(data_dir).exists() and resid_path(data_dir).exists() and mlp_path(data_dir).exists()):
        return False
    if not meta_file.exists():
        return False
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return bool(meta.get("complete"))


def make_splits(
    y: np.ndarray,
    *,
    seed: int,
    ratios: tuple[float, float, float],
    task: str,
) -> dict[str, list[int]]:
    """Stratified train/val/test indices when possible."""
    indices = np.arange(len(y))
    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")

    stratify = y if task in ("binary", "multiclass") and len(np.unique(y)) > 1 else None
    try:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_r + test_r),
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_r + test_r),
            random_state=seed,
            stratify=None,
        )

    temp_y = y[temp_idx] if stratify is not None else None
    rel_test = test_r / (val_r + test_r) if (val_r + test_r) > 0 else 0.5
    try:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test,
            random_state=seed,
            stratify=temp_y if stratify is not None else None,
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test,
            random_state=seed,
            stratify=None,
        )

    return {
        "train": [int(i) for i in train_idx],
        "val": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
    }


def write_splits(path: pathlib.Path, splits: dict[str, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def load_splits(path: pathlib.Path) -> dict[str, list[int]]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_splits(
    cfg: Exp3Config,
    split_name: SplitName,
    data: dict[str, Any],
    *,
    label_for_stratify: str | None = None,
) -> dict[str, list[int]]:
    from exp3.labels import build_targets

    data_dir = cfg.data_out_dir(split_name)
    path = split_path(data_dir)
    n = int(np.asarray(data["y_top1"]).shape[0])
    if path.exists():
        splits = load_splits(path)
        used = {int(i) for key in ("train", "val", "test") for i in splits.get(key, [])}
        if used == set(range(n)):
            return splits
        path.unlink()

    strat_label = label_for_stratify or cfg.resolved_label()
    y, task = build_targets(
        data,
        strat_label,  # type: ignore[arg-type]
        confident_error_top1_prob=cfg.confident_error_top1_prob,
    )
    if task == "regression":
        y = np.asarray(data["y_top1"], dtype=np.int64)
        task = "binary"

    splits = make_splits(
        y,
        seed=cfg.seed,
        ratios=(cfg.split_ratios.train, cfg.split_ratios.val, cfg.split_ratios.test),
        task=task,
    )
    write_splits(path, splits)
    return splits
