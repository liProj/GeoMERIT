from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def labeled_rows(df: pd.DataFrame, label_col: str = "label_idx") -> pd.DataFrame:
    return df.loc[df[label_col] >= 0]


def iter_group_folds(df: pd.DataFrame, folds: int, group_col: str = "well_id", label_col: str = "label_idx"):
    mask = df[label_col].to_numpy() >= 0
    indices = df.index.to_numpy()[mask]
    labels = df.loc[indices, label_col].to_numpy()
    groups = df.loc[indices, group_col].to_numpy()
    splitter = GroupKFold(n_splits=folds)
    X = np.zeros(len(indices))
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, labels, groups=groups)):
        yield fold, indices[train_idx], indices[valid_idx]


def train_valid_split(df: pd.DataFrame, valid_fraction: float = 0.15, group_col: str = "well_id", label_col: str = "label_idx", seed: int = 2026):
    mask = df[label_col].to_numpy() >= 0
    indices = df.index.to_numpy()[mask]
    labels = df.loc[indices, label_col].to_numpy()
    groups = df.loc[indices, group_col].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_fraction, random_state=seed)
    train_idx, valid_idx = next(splitter.split(np.zeros(len(indices)), labels, groups=groups))
    return indices[train_idx], indices[valid_idx]


def save_indices(path: str | Path, splits: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: np.asarray(value).astype(int).tolist() for key, value in splits.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
