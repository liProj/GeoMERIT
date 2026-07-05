from __future__ import annotations

from collections import Counter

import numpy as np


def class_balanced_weights(y: np.ndarray, class_count: int, rho: float = 0.9995, cap: float = 8.0) -> np.ndarray:
    """Cui et al. effective-number weighting with clipping for extreme tails."""
    y = np.asarray(y, dtype=int)
    freq = Counter(y.tolist())
    weights = {}
    for cls in range(class_count):
        n = max(freq.get(cls, 0), 1)
        weights[cls] = (1.0 - rho) / (1.0 - rho**n)
    scale = class_count / sum(weights.values())
    weights = {cls: min(value * scale, cap) for cls, value in weights.items()}
    return np.asarray([weights[int(v)] for v in y], dtype=float)


def boundary_flags(y: np.ndarray, well_id: np.ndarray, radius: int = 1) -> np.ndarray:
    from .metrics import boundary_mask_by_well

    return boundary_mask_by_well(y, well_id, radius)


def sample_weights(
    y: np.ndarray,
    confidence: np.ndarray,
    is_boundary: np.ndarray,
    class_count: int,
    rho: float = 0.9995,
    cap: float = 8.0,
    confidence_map: dict[int, float] | None = None,
    boundary_weight: float = 1.2,
    interior_weight: float = 0.7,
) -> np.ndarray:
    confidence_map = confidence_map or {1: 1.0, 2: 0.8, 3: 0.6}
    w_class = class_balanced_weights(y, class_count, rho=rho, cap=cap)
    w_boundary = np.where(is_boundary, boundary_weight, interior_weight)
    w_conf = np.asarray([confidence_map.get(int(v), 1.0) if not np.isnan(v) else 1.0 for v in confidence], dtype=float)
    return w_class * w_boundary * w_conf
