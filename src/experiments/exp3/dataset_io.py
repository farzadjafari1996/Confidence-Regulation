"""Load/save Ext3 collected datasets and train/val/test splits."""

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
)


def dataset_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "dataset.npz"


def partial_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "dataset_partial.npz"


def split_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "split_indices.json"


def meta_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "collection_meta.json"


def save_arrays(
    path: pathlib.Path,
    arrays: dict[str, np.ndarray],
    lists: dict[str, list],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {k: v for k, v in arrays.items()}
    for key, values in lists.items():
        payload[key] = np.asarray(values, dtype=object)
    np.savez_compressed(path, **payload)


def load_arrays(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        out: dict[str, Any] = {}
        for key in z.files:
            arr = z[key]
            if key in META_LIST_KEYS:
                out[key] = arr.tolist()
            else:
                out[key] = arr
        return out


def done_item_ids(data_dir: pathlib.Path) -> set[str]:
    for candidate in (dataset_path(data_dir), partial_path(data_dir)):
        if candidate.exists():
            data = load_arrays(candidate)
            return {str(x) for x in data.get("item_ids", [])}
    return set()


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
    label_for_stratify: str = "top1",
) -> dict[str, list[int]]:
    from exp3.labels import build_targets

    data_dir = cfg.data_out_dir(split_name)
    path = split_path(data_dir)
    if path.exists():
        return load_splits(path)

    y, task = build_targets(
        data,
        label_for_stratify,  # type: ignore[arg-type]
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


def write_collection_meta(path: pathlib.Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
