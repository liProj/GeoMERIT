from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geomerit.decode import normalize_proba
from geomerit.labels import LABEL_NAMES, TAIL_IDS, load_penalty_matrix
from geomerit.metrics import boundary_mask_by_well


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoRACS OOF experiment on an existing GeoMERIT run.")
    parser.add_argument("--table", required=True, help="Feature table parquet used by the run.")
    parser.add_argument("--run", required=True, help="Existing run directory with OOF probabilities.")
    parser.add_argument("--penalty", default="configs/penalty_matrix.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--class_count", type=int, default=12)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--retrieval-dim", type=int, default=32)
    parser.add_argument("--index", choices=["ivf", "hnsw"], default="ivf")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-search", type=int, default=96)
    parser.add_argument("--train-sample", type=int, default=250000)
    parser.add_argument("--inv-freq-power", type=float, default=0.5)
    parser.add_argument("--prior-smoothing", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--tau-grid", default="0,0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--eta-grid", default="0,0.2,0.4,0.7,1.0")
    parser.add_argument("--gamma-grid", default="0,0.5,1.0")
    parser.add_argument("--theta-grid", default="0.35,0.5")
    parser.add_argument("--stack", action="store_true", help="Cross-fit multinomial logistic stacking.")
    parser.add_argument("--meta-max-rows", type=int, default=450000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    table_path = resolve_path(args.table, project_root)
    run_dir = resolve_path(args.run, project_root)
    out_path = resolve_path(args.out, project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    A = load_penalty_matrix(resolve_path(args.penalty, project_root)).astype(np.float32)
    C = int(args.class_count)
    tau_grid = parse_float_grid(args.tau_grid)
    eta_grid = parse_float_grid(args.eta_grid)
    gamma_grid = parse_float_grid(args.gamma_grid)
    theta_grid = parse_float_grid(args.theta_grid)

    print(f"[GeoRACS] loading OOF from {run_dir}", flush=True)
    proba_base = normalize_proba(np.load(run_dir / "oof_proba.npy")).astype(np.float32)
    fold_ids = np.load(run_dir / "oof_fold.npy").astype(np.int16)
    oof_index = np.load(run_dir / "oof_index.npy")
    rows = pd.read_parquet(run_dir / "oof_rows.parquet")
    y = rows["label_idx"].to_numpy(np.int16)
    well = rows["well_id"].astype(str).to_numpy()

    meta = json.loads(table_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    retrieval_cols = choose_retrieval_columns(meta["numeric"], args.retrieval_dim)
    print(f"[GeoRACS] retrieval columns ({len(retrieval_cols)}): {retrieval_cols}", flush=True)
    df_phi = pd.read_parquet(table_path, columns=retrieval_cols).iloc[oof_index]
    phi = df_phi.to_numpy(np.float32, copy=True)
    del df_phi

    expert_scores = load_expert_scores(run_dir)
    class_prior = np.bincount(y, minlength=C).astype(np.float64)
    class_prior = class_prior / class_prior.sum()
    boundary_mask = boundary_mask_by_well(y, well, radius=1)

    report: dict[str, object] = {
        "run": str(run_dir),
        "table": str(table_path),
        "rows": int(len(y)),
        "class_counts": {LABEL_NAMES[i]: int(v) for i, v in enumerate(np.bincount(y, minlength=C))},
        "retrieval_columns": retrieval_cols,
        "candidates": [],
    }

    candidates: list[dict[str, object]] = []
    add_candidate(candidates, "base_argmax", proba_base.argmax(axis=1), y, well, A, boundary_mask)
    add_candidate(candidates, "base_bayes", bayes_decode(proba_base, A), y, well, A, boundary_mask)

    print("[GeoRACS] cross-fold temperature calibration for base proba", flush=True)
    proba_cal, temp_by_fold = crossfit_temperature(proba_base, y, fold_ids)
    report["temperature_by_fold"] = temp_by_fold
    add_candidate(candidates, "base_temp_bayes", bayes_decode(proba_cal, A), y, well, A, boundary_mask)

    print("[GeoRACS] building leave-fold-out retrieval prior", flush=True)
    prior_path = run_dir / f"georacs_retrieval_prior_k{args.k}_{args.index}.npy"
    if prior_path.exists():
        print(f"[GeoRACS] using cached retrieval prior {prior_path}", flush=True)
        proba_ret = normalize_proba(np.load(prior_path)).astype(np.float32)
    else:
        proba_ret = build_retrieval_prior(
            phi=phi,
            y=y,
            fold_ids=fold_ids,
            class_count=C,
            k=args.k,
            index_kind=args.index,
            nlist=args.nlist,
            nprobe=args.nprobe,
            train_sample=args.train_sample,
            hnsw_m=args.hnsw_m,
            ef_search=args.ef_search,
            inv_freq_power=args.inv_freq_power,
            prior_smoothing=args.prior_smoothing,
            threads=args.threads,
            rng=rng,
        )
        np.save(prior_path, proba_ret.astype(np.float32))
    add_candidate(candidates, "retrieval_bayes", bayes_decode(proba_ret, A), y, well, A, boundary_mask)

    print("[GeoRACS] evaluating calibrated retrieval blends", flush=True)
    for eta in eta_grid:
        blended = geometric_blend(proba_cal, proba_ret, eta=eta)
        pred = bayes_decode(blended, A)
        add_candidate(candidates, f"blend_eta={eta:g}_bayes", pred, y, well, A, boundary_mask, {"eta": eta})
        for tau in tau_grid:
            adjusted = logit_adjust_fast(blended, class_prior, tau)
            pred = bayes_decode(adjusted, A)
            add_candidate(candidates, f"blend_eta={eta:g}_tau={tau:g}_bayes", pred, y, well, A, boundary_mask, {"eta": eta, "tau": tau})
            if expert_scores:
                for gamma in gamma_grid:
                    if gamma <= 0:
                        continue
                    for theta in theta_grid:
                        gated = tail_gate(adjusted, expert_scores, gamma=gamma, theta=theta)
                        pred = bayes_decode(gated, A)
                        add_candidate(
                            candidates,
                            f"blend_eta={eta:g}_tau={tau:g}_gamma={gamma:g}_theta={theta:g}_bayes",
                            pred,
                            y,
                            well,
                            A,
                            boundary_mask,
                            {"eta": eta, "tau": tau, "gamma": gamma, "theta": theta},
                        )

    if args.stack:
        print("[GeoRACS] cross-fitting logistic stacking", flush=True)
        proba_stack = crossfit_stack(
            proba_base=proba_cal,
            proba_ret=proba_ret,
            expert_scores=expert_scores,
            y=y,
            fold_ids=fold_ids,
            class_count=C,
            max_rows=args.meta_max_rows,
            seed=args.seed,
        )
        stack_path = run_dir / "georacs_stack_oof.npy"
        np.save(stack_path, proba_stack.astype(np.float32))
        proba_stack_cal, stack_temp_by_fold = crossfit_temperature(proba_stack, y, fold_ids)
        report["stack_temperature_by_fold"] = stack_temp_by_fold
        add_candidate(candidates, "stack_bayes", bayes_decode(proba_stack_cal, A), y, well, A, boundary_mask)
        for tau in tau_grid:
            adjusted = logit_adjust_fast(proba_stack_cal, class_prior, tau)
            add_candidate(
                candidates,
                f"stack_tau={tau:g}_bayes",
                bayes_decode(adjusted, A),
                y,
                well,
                A,
                boundary_mask,
                {"tau": tau},
            )

    candidates.sort(key=objective_key, reverse=True)
    best = candidates[0]
    best_pred = np.asarray(best.pop("_pred"), dtype=np.int16)
    for item in candidates[1:]:
        item.pop("_pred", None)
    report["best"] = best
    report["candidates"] = candidates[:50]
    report["per_class"] = per_class_report(y, best_pred, C)

    np.save(out_path.with_suffix(".pred.npy"), best_pred)
    pd.DataFrame({"well_id": well, "depth": rows["DEPTH_MD"], "y_true": y, "y_pred": best_pred}).to_csv(
        out_path.with_suffix(".csv"), index=False
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "out": str(out_path)}, indent=2), flush=True)


def build_retrieval_prior(
    phi: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    class_count: int,
    k: int,
    index_kind: str,
    nlist: int,
    nprobe: int,
    train_sample: int,
    hnsw_m: int,
    ef_search: int,
    inv_freq_power: float,
    prior_smoothing: float,
    threads: int,
    rng: np.random.Generator,
) -> np.ndarray:
    import faiss

    faiss.omp_set_num_threads(max(1, int(threads)))
    out = np.zeros((len(y), class_count), dtype=np.float32)
    folds = sorted(int(v) for v in np.unique(fold_ids) if v >= 0)
    for fold in folds:
        t0 = time.time()
        train_mask = fold_ids != fold
        valid_mask = fold_ids == fold
        train_idx = np.flatnonzero(train_mask)
        valid_idx = np.flatnonzero(valid_mask)
        xb, xq = standardize_train_query(phi[train_idx], phi[valid_idx])
        yb = y[train_idx]
        dim = xb.shape[1]
        print(f"[GeoRACS][retrieval] fold={fold} train={len(train_idx)} valid={len(valid_idx)} dim={dim}", flush=True)

        if index_kind == "hnsw":
            index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_L2)
            index.hnsw.efConstruction = 120
            index.hnsw.efSearch = ef_search
            index.add(xb)
        else:
            nlist_fold = min(int(nlist), max(64, int(math.sqrt(len(xb))) * 4))
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist_fold, faiss.METRIC_L2)
            index.nprobe = min(int(nprobe), nlist_fold)
            if len(xb) > train_sample:
                sample_idx = rng.choice(len(xb), size=train_sample, replace=False)
                index.train(xb[sample_idx])
            else:
                index.train(xb)
            index.add(xb)

        dist, nn = index.search(xq, min(k, len(xb)))
        acc = votes_to_prior(dist, nn, yb, class_count, inv_freq_power, prior_smoothing)
        out[valid_idx] = acc
        print(f"[GeoRACS][retrieval] fold={fold} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return normalize_proba(out)


def votes_to_prior(
    dist: np.ndarray,
    nn: np.ndarray,
    y_train: np.ndarray,
    class_count: int,
    inv_freq_power: float,
    prior_smoothing: float,
) -> np.ndarray:
    valid = nn >= 0
    safe_nn = np.maximum(nn, 0)
    labels = y_train[safe_nn]
    freq = np.bincount(y_train, minlength=class_count).astype(np.float64)
    class_weight = (freq.sum() / np.maximum(freq * class_count, 1.0)) ** inv_freq_power
    class_weight = np.clip(class_weight, 0.05, 50.0)
    scale_col = min(15, dist.shape[1] - 1)
    scale = float(np.nanmedian(dist[valid[:, scale_col], scale_col])) if np.any(valid[:, scale_col]) else float(np.nanmedian(dist[valid]))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    kernel = np.exp(-np.maximum(dist, 0.0) / (2.0 * scale + 1e-12)) * valid
    kernel *= class_weight[labels]
    acc = np.full((dist.shape[0], class_count), prior_smoothing, dtype=np.float64)
    for c in range(class_count):
        acc[:, c] += (kernel * (labels == c)).sum(axis=1)
    acc /= np.maximum(acc.sum(axis=1, keepdims=True), 1e-12)
    return acc.astype(np.float32)


def standardize_train_query(xb_raw: np.ndarray, xq_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xb = xb_raw.astype(np.float32, copy=True)
    xq = xq_raw.astype(np.float32, copy=True)
    xb[~np.isfinite(xb)] = np.nan
    xq[~np.isfinite(xq)] = np.nan
    med = np.nanmedian(xb, axis=0).astype(np.float32)
    med[~np.isfinite(med)] = 0.0
    xb = np.where(np.isnan(xb), med, xb)
    xq = np.where(np.isnan(xq), med, xq)
    mean = xb.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = xb.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    xb = (xb - mean) / std
    xq = (xq - mean) / std
    np.clip(xb, -6.0, 6.0, out=xb)
    np.clip(xq, -6.0, 6.0, out=xq)
    return np.ascontiguousarray(xb, dtype=np.float32), np.ascontiguousarray(xq, dtype=np.float32)


def crossfit_temperature(proba: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    out = np.zeros_like(proba, dtype=np.float32)
    temps: dict[str, float] = {}
    logits = np.log(normalize_proba(proba) + 1e-12).astype(np.float32)
    for fold in sorted(int(v) for v in np.unique(fold_ids) if v >= 0):
        train = fold_ids != fold
        valid = fold_ids == fold
        T = fit_temperature(logits[train], y[train])
        out[valid] = apply_temperature(logits[valid], T)
        temps[str(fold)] = float(T)
        print(f"[GeoRACS][cal] fold={fold} T={T:.4f}", flush=True)
    return out, temps


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    idx = np.arange(len(y))

    def nll(temp: float) -> float:
        z = logits / temp
        z = z - z.max(axis=1, keepdims=True)
        logsum = np.log(np.exp(z).sum(axis=1))
        return float((-z[idx, y] + logsum).mean())

    res = minimize_scalar(nll, bounds=(0.05, 8.0), method="bounded", options={"xatol": 1e-3})
    return float(res.x)


def apply_temperature(logits: np.ndarray, temp: float) -> np.ndarray:
    z = logits / np.float32(temp)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    return e / np.maximum(e.sum(axis=1, keepdims=True), np.float32(1e-12))


def crossfit_stack(
    proba_base: np.ndarray,
    proba_ret: np.ndarray,
    expert_scores: dict[int, np.ndarray],
    y: np.ndarray,
    fold_ids: np.ndarray,
    class_count: int,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    Z = stack_features(proba_base, proba_ret, expert_scores)
    out = np.zeros((len(y), class_count), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for fold in sorted(int(v) for v in np.unique(fold_ids) if v >= 0):
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        fit_idx = sample_meta_rows(train_idx, y, max_rows, rng)
        print(f"[GeoRACS][stack] fold={fold} fit_rows={len(fit_idx)} valid={len(valid_idx)}", flush=True)
        clf = LogisticRegression(
            C=0.35,
            class_weight="balanced",
            max_iter=220,
            multi_class="multinomial",
            solver="lbfgs",
            verbose=0,
        )
        clf.fit(Z[fit_idx], y[fit_idx])
        pred = clf.predict_proba(Z[valid_idx])
        if pred.shape[1] != class_count:
            full = np.zeros((len(valid_idx), class_count), dtype=np.float32)
            for local_idx, cls in enumerate(clf.classes_):
                full[:, int(cls)] = pred[:, local_idx]
            pred = full
        out[valid_idx] = pred.astype(np.float32)
    return normalize_proba(out)


def stack_features(proba_base: np.ndarray, proba_ret: np.ndarray, expert_scores: dict[int, np.ndarray]) -> np.ndarray:
    parts = [np.log(normalize_proba(proba_base) + 1e-12), np.log(normalize_proba(proba_ret) + 1e-12)]
    if expert_scores:
        parts.append(np.column_stack([expert_scores[tid] for tid in sorted(expert_scores)]).astype(np.float32))
    return np.hstack(parts).astype(np.float32)


def sample_meta_rows(train_idx: np.ndarray, y: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if max_rows <= 0 or len(train_idx) <= max_rows:
        return train_idx
    tail_mask = np.isin(y[train_idx], TAIL_IDS)
    keep_tail = train_idx[tail_mask]
    rest = train_idx[~tail_mask]
    take_rest = max(0, max_rows - len(keep_tail))
    if len(rest) > take_rest:
        rest = rng.choice(rest, size=take_rest, replace=False)
    return np.sort(np.concatenate([keep_tail, rest]))


def choose_retrieval_columns(numeric: list[str], max_dim: int) -> list[str]:
    core_order = [
        "GR",
        "SGR",
        "RHOB",
        "DRHO",
        "NPHI",
        "PEF",
        "DTC",
        "DTS",
        "RDEP",
        "RSHA",
        "RMED",
        "RXO",
        "SP",
        "CALI",
        "BS",
        "DCAL",
        "ROP",
        "MUDWEIGHT",
    ]
    selected: list[str] = []
    for stem in core_order:
        for suffix in ("_well_robust_z", "_is_missing"):
            name = f"{stem}{suffix}"
            if name in numeric and name not in selected:
                selected.append(name)
            if len(selected) >= max_dim:
                return selected
    for name in ("missing_rate", "missing_count", "outlier_rate", "rhob_nphi_sep"):
        if name in numeric and name not in selected:
            selected.append(name)
        if len(selected) >= max_dim:
            return selected
    for name in numeric:
        if name.endswith("_well_robust_z") and name not in selected:
            selected.append(name)
        if len(selected) >= max_dim:
            return selected
    return selected


def load_expert_scores(run_dir: Path) -> dict[int, np.ndarray]:
    scores: dict[int, np.ndarray] = {}
    for tid in TAIL_IDS:
        path = run_dir / f"oof_expert_{tid}.npy"
        if path.exists():
            scores[tid] = np.load(path).astype(np.float32)
    return scores


def geometric_blend(base: np.ndarray, ret: np.ndarray, eta: float) -> np.ndarray:
    logp = np.log(normalize_proba(base) + 1e-12) + np.float32(eta) * np.log(normalize_proba(ret) + 1e-12)
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp, dtype=np.float32)
    return p / np.maximum(p.sum(axis=1, keepdims=True), np.float32(1e-12))


def logit_adjust_fast(proba: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    logits = np.log(normalize_proba(proba) + 1e-12) - np.float32(tau) * np.log(prior.astype(np.float32) + 1e-12)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits, dtype=np.float32)
    return p / np.maximum(p.sum(axis=1, keepdims=True), np.float32(1e-12))


def tail_gate(proba: np.ndarray, expert_scores: dict[int, np.ndarray], gamma: float, theta: float) -> np.ndarray:
    p = normalize_proba(proba).copy()
    for tid, score in expert_scores.items():
        fire = score > theta
        if np.any(fire):
            p[fire, tid] *= np.float32(1.0 + gamma)
    return normalize_proba(p)


def bayes_decode(proba: np.ndarray, A: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    out = np.zeros(len(proba), dtype=np.int16)
    for start in range(0, len(proba), chunk):
        end = min(start + chunk, len(proba))
        out[start:end] = (normalize_proba(proba[start:end]) @ A).argmin(axis=1).astype(np.int16)
    return out


def add_candidate(
    candidates: list[dict[str, object]],
    name: str,
    pred: np.ndarray,
    y: np.ndarray,
    well: np.ndarray,
    A: np.ndarray,
    boundary_mask: np.ndarray,
    params: dict[str, float] | None = None,
) -> None:
    metrics = evaluate_pred(y, pred, A, boundary_mask)
    row: dict[str, object] = {"name": name, "params": params or {}, **metrics, "_pred": pred.astype(np.int16, copy=True)}
    candidates.append(row)
    print(
        "[GeoRACS][candidate] "
        f"{name} weighted={metrics['weighted_f1']:.4f} macro={metrics['macro_f1']:.4f} "
        f"boundary={metrics['boundary_f1']:.4f} penalty={metrics['penalty']:.4f} tail={metrics['tail_mean_f1']:.4f}",
        flush=True,
    )


def evaluate_pred(y: np.ndarray, pred: np.ndarray, A: np.ndarray, boundary_mask: np.ndarray) -> dict[str, float]:
    tail_scores = f1_score(y, pred, labels=TAIL_IDS, average=None, zero_division=0)
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(A.shape[0])), average="macro", zero_division=0)),
        "boundary_f1": float(f1_score(y[boundary_mask], pred[boundary_mask], average="weighted", zero_division=0)),
        "penalty": float(-np.mean(A[y.astype(int), pred.astype(int)])),
        "tail_mean_f1": float(np.mean(tail_scores)),
    }


def per_class_report(y: np.ndarray, pred: np.ndarray, class_count: int) -> dict[str, dict[str, float | int | str]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        pred,
        labels=list(range(class_count)),
        zero_division=0,
    )
    return {
        str(i): {
            "name": LABEL_NAMES[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(class_count)
    }


def objective_key(row: dict[str, object]) -> float:
    return (
        3.0 * float(row["penalty"])
        + 0.8 * float(row["macro_f1"])
        + 0.5 * float(row["boundary_f1"])
        + 0.4 * float(row["tail_mean_f1"])
        + 0.2 * float(row["weighted_f1"])
    )


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def resolve_path(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


if __name__ == "__main__":
    main()
