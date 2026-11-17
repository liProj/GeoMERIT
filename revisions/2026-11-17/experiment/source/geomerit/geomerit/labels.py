from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_CODES = [65000, 30000, 65030, 70000, 80000, 99000, 88000, 70032, 90000, 74000, 86000, 93000]
LABEL_NAMES = [
    "Shale",
    "Sandstone",
    "Sandstone/Shale",
    "Limestone",
    "Marl",
    "Tuff",
    "Halite",
    "Chalk",
    "Coal",
    "Dolomite",
    "Anhydrite",
    "Basement",
]

CODE_TO_IDX = {code: idx for idx, code in enumerate(LABEL_CODES)}
IDX_TO_CODE = {idx: code for idx, code in enumerate(LABEL_CODES)}
IDX_TO_NAME = {idx: name for idx, name in enumerate(LABEL_NAMES)}
NAME_TO_IDX = {name: idx for idx, name in enumerate(LABEL_NAMES)}

TAIL_IDS = [5, 6, 8, 9, 10, 11]  # Tuff, Halite, Coal, Dolomite, Anhydrite, Basement

COARSE_GROUPS = {
    "clastic": [1, 2, 0],
    "carbonate": [3, 7, 9, 4],
    "evaporite": [6, 10],
    "coal": [8],
    "basement": [11],
    "tuff": [5],
}
IDX_TO_COARSE = {}
for coarse_idx, (_, fine_ids) in enumerate(COARSE_GROUPS.items()):
    for fine_id in fine_ids:
        IDX_TO_COARSE[fine_id] = coarse_idx


def encode_labels(values: Iterable[float | int]) -> np.ndarray:
    """Map FORCE lithology codes to contiguous class indices; missing becomes -1."""
    out = []
    for value in values:
        if pd.isna(value):
            out.append(-1)
            continue
        out.append(CODE_TO_IDX.get(int(value), -1))
    return np.asarray(out, dtype=np.int16)


def decode_labels(indices: Iterable[int]) -> np.ndarray:
    """Map contiguous class indices back to FORCE lithology codes."""
    return np.asarray([IDX_TO_CODE[int(idx)] for idx in indices], dtype=np.int32)


def coarse_labels(y: Iterable[int]) -> np.ndarray:
    """Map fine class indices to GeoMERIT coarse classes."""
    return np.asarray([IDX_TO_COARSE.get(int(v), -1) for v in y], dtype=np.int16)


def load_penalty_matrix(path: str | Path | None) -> np.ndarray:
    """Load a FORCE-style 12x12 penalty matrix from csv or the provided xlsx file."""
    if path is None:
        return default_penalty_matrix()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        arr = pd.read_csv(path, header=None).to_numpy(float)
        return _validate_penalty(arr)

    raw = pd.read_excel(path, header=None)
    names = [
        "Sandstone",
        "Sandstone/Shale",
        "Shale",
        "Marl",
        "Dolomite",
        "Limestone",
        "Chalk",
        "Halite",
        "Anhydrite",
        "Tuff",
        "Coal",
        "Crystalline Basement",
    ]
    row_lookup = {}
    for i in range(raw.shape[0]):
        for value in raw.iloc[i].tolist():
            cell = str(value).strip()
            if cell in names or cell == "Crystalline Basement":
                row_lookup[cell] = i
                break

    source_order = [
        "Shale",
        "Sandstone",
        "Sandstone/Shale",
        "Limestone",
        "Marl",
        "Tuff",
        "Halite",
        "Chalk",
        "Coal",
        "Dolomite",
        "Anhydrite",
        "Crystalline Basement",
    ]
    excel_order = [
        "Sandstone",
        "Sandstone/Shale",
        "Shale",
        "Marl",
        "Dolomite",
        "Limestone",
        "Chalk",
        "Halite",
        "Anhydrite",
        "Tuff",
        "Coal",
        "Crystalline Basement",
    ]
    if not set(excel_order).issubset(row_lookup):
        raise ValueError(f"Could not parse penalty matrix rows from {path}")

    excel_matrix = np.zeros((12, 12), dtype=float)
    for r, name in enumerate(excel_order):
        row = raw.iloc[row_lookup[name], 3:15].to_numpy(float)
        excel_matrix[r] = row

    excel_index = {name: i for i, name in enumerate(excel_order)}
    reorder = [excel_index[name] for name in source_order]
    arr = excel_matrix[np.ix_(reorder, reorder)]
    return _validate_penalty(arr)


def save_penalty_matrix(path: str | Path, matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix).to_csv(path, header=False, index=False)


def default_penalty_matrix() -> np.ndarray:
    """Fallback matrix: zero diagonal and unit off-diagonal."""
    arr = np.ones((len(LABEL_CODES), len(LABEL_CODES)), dtype=float)
    np.fill_diagonal(arr, 0.0)
    return arr


def _validate_penalty(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (len(LABEL_CODES), len(LABEL_CODES)):
        raise ValueError(f"Expected 12x12 penalty matrix, got {arr.shape}")
    return arr
