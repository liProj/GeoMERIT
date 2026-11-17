from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .labels import LABEL_NAMES, TAIL_IDS


@dataclass(frozen=True)
class MetricBundle:
    weighted_f1: float
    macro_f1: float
    boundary_f1: float
    penalty: float
    tail_mean_f1: float


def penalty_score(y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray) -> float:
    return float(-np.mean(A[np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)]))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, class_count: int = 12) -> float:
    return float(f1_score(y_true, y_pred, labels=list(range(class_count)), average="macro", zero_division=0))


def tail_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[int, float], float]:
    per = f1_score(y_true, y_pred, labels=TAIL_IDS, average=None, zero_division=0)
    per_dict = {cls: float(score) for cls, score in zip(TAIL_IDS, per)}
    return per_dict, float(np.mean(per))


def boundary_mask_by_well(y_true: np.ndarray, well_id: np.ndarray, radius: int = 1) -> np.ndarray:
    y_true = np.asarray(y_true)
    well_id = np.asarray(well_id)
    mask = np.zeros(len(y_true), dtype=bool)
    for well in np.unique(well_id):
        idx = np.flatnonzero(well_id == well)
        if len(idx) < 2:
            continue
        seq = y_true[idx]
        local = np.zeros(len(idx), dtype=bool)
        changes = np.flatnonzero(seq[:-1] != seq[1:])
        for change in changes:
            lo = max(0, change - radius)
            hi = min(len(idx), change + radius + 2)
            local[lo:hi] = True
        mask[idx] = local
    return mask


def boundary_f1(y_true: np.ndarray, y_pred: np.ndarray, well_id: np.ndarray, radius: int = 1) -> float:
    mask = boundary_mask_by_well(y_true, well_id, radius)
    if not mask.any():
        return float("nan")
    return float(f1_score(np.asarray(y_true)[mask], np.asarray(y_pred)[mask], average="weighted", zero_division=0))


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, class_count: int = 12) -> pd_like_dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(class_count)),
        zero_division=0,
    )
    return {
        i: {
            "name": LABEL_NAMES[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(class_count)
    }


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, well_id: np.ndarray, A: np.ndarray) -> MetricBundle:
    _, tail_mean = tail_f1(y_true, y_pred)
    return MetricBundle(
        weighted_f1=weighted_f1(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred),
        boundary_f1=boundary_f1(y_true, y_pred, well_id),
        penalty=penalty_score(y_true, y_pred, A),
        tail_mean_f1=tail_mean,
    )


pd_like_dict = dict
