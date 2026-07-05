from __future__ import annotations

import numpy as np

from .labels import TAIL_IDS


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float32)
    denom = proba.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return proba / denom


def bayes_risk_decode(proba: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return predictions that minimize expected FORCE penalty A[true, pred]."""
    expected_cost = normalize_proba(proba) @ np.asarray(A, dtype=np.float32)
    return expected_cost.argmin(axis=1).astype(np.int16)


def logit_adjust(proba: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    logits = np.log(normalize_proba(proba) + np.float32(1e-12)) - np.float32(tau) * np.log(np.asarray(prior, dtype=np.float32) + np.float32(1e-12))
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def tail_expert_gate(
    proba: np.ndarray,
    expert_scores: dict[int, np.ndarray],
    tail_ids: list[int] | None = None,
    gamma: dict[int, float] | float = 0.5,
    theta: dict[int, float] | float = 0.5,
) -> np.ndarray:
    p = normalize_proba(proba).copy()
    tail_ids = tail_ids or TAIL_IDS
    for cls in tail_ids:
        if cls not in expert_scores:
            continue
        g = gamma.get(cls, 0.5) if isinstance(gamma, dict) else gamma
        th = theta.get(cls, 0.5) if isinstance(theta, dict) else theta
        fire = np.asarray(expert_scores[cls]) > th
        p[fire, cls] *= 1.0 + g
    return normalize_proba(p)


def estimate_transition(labels_by_well: list[np.ndarray], class_count: int, lam: float = 1.0) -> np.ndarray:
    matrix = np.full((class_count, class_count), lam, dtype=float)
    for seq in labels_by_well:
        seq = np.asarray(seq, dtype=int)
        seq = seq[seq >= 0]
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            matrix[int(a), int(b)] += 1.0
    return matrix / matrix.sum(axis=1, keepdims=True)


def apply_tail_transition_floor(P: np.ndarray, tail_ids: list[int] | None = None, floor: float = 1e-4) -> np.ndarray:
    P = np.asarray(P, dtype=float).copy()
    tail_ids = tail_ids or TAIL_IDS
    P[:, tail_ids] = np.maximum(P[:, tail_ids], floor)
    return P / P.sum(axis=1, keepdims=True)


def viterbi(logp_emit: np.ndarray, logP_trans: np.ndarray, beta: float = 1.0) -> np.ndarray:
    logp_emit = np.asarray(logp_emit, dtype=float)
    logP_trans = np.asarray(logP_trans, dtype=float)
    T, C = logp_emit.shape
    dp = np.full((T, C), -np.inf, dtype=float)
    bp = np.zeros((T, C), dtype=np.int16)
    dp[0] = logp_emit[0]
    transition = beta * logP_trans
    for t in range(1, T):
        scores = dp[t - 1][:, None] + transition
        bp[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + logp_emit[t]
    path = np.zeros(T, dtype=np.int16)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def viterbi_by_well(proba: np.ndarray, well_id: np.ndarray, transition: np.ndarray, beta: float = 1.0) -> np.ndarray:
    proba = normalize_proba(proba)
    out = np.zeros(len(proba), dtype=np.int16)
    log_transition = np.log(transition + 1e-12)
    for well in np.unique(well_id):
        idx = np.flatnonzero(well_id == well)
        log_emit = np.log(proba[idx] + 1e-12)
        out[idx] = viterbi(log_emit, log_transition, beta=beta)
    return out


def risk_viterbi_by_well(proba: np.ndarray, well_id: np.ndarray, transition: np.ndarray, A: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Viterbi where emission scores are negative expected FORCE penalties."""
    proba = normalize_proba(proba)
    emission = -(proba @ np.asarray(A, dtype=float))
    out = np.zeros(len(proba), dtype=np.int16)
    log_transition = np.log(transition + 1e-12)
    for well in np.unique(well_id):
        idx = np.flatnonzero(well_id == well)
        out[idx] = viterbi(emission[idx], log_transition, beta=beta)
    return out
