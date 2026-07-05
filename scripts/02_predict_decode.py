from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geomerit.decode import (
    bayes_risk_decode,
    logit_adjust,
    risk_viterbi_by_well,
    tail_expert_gate,
    viterbi_by_well,
)
from geomerit.labels import TAIL_IDS, load_penalty_matrix
from geomerit.metrics import evaluate_all, per_class_report, tail_f1
from geomerit.models import (
    load_pickle,
    predict_coarse_fine,
    predict_proba,
    predict_tail_experts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--eval", choices=["oof", "hidden"], default="oof")
    parser.add_argument("--penalty", default="configs/penalty_matrix.csv")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--objective", default="penalty", choices=["penalty", "balanced"], help="Grid-search objective.")
    parser.add_argument("--top_k_viterbi", type=int, default=5, help="Number of Bayes-stage candidates to refine with Viterbi.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--decode_order", default=None, choices=["bayes_then_viterbi", "risk_viterbi", "viterbi_then_bayes", "bayes_only"])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = _resolve(args.run, project_root)
    config = yaml.safe_load(_resolve(args.config, project_root).read_text(encoding="utf-8"))
    A = load_penalty_matrix(_resolve(args.penalty, project_root))
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    components = set(run_meta.get("components", ["flat"]))

    if args.eval == "oof":
        _eval_oof(run_dir, config, A, components, args, project_root)
    else:
        _eval_hidden(run_dir, config, A, components, args, project_root)


def _eval_oof(run_dir, config, A, components, args, project_root):
    proba = np.load(run_dir / "oof_proba.npy")
    fold_ids = np.load(run_dir / "oof_fold.npy")
    rows = pd.read_parquet(run_dir / "oof_rows.parquet")
    y_true = rows["label_idx"].to_numpy(int)
    well_id = rows["well_id"].to_numpy()

    prior = np.bincount(y_true, minlength=proba.shape[1]).astype(float)
    prior = prior / prior.sum()

    # Load coarse2fine oof if available
    if "coarse2fine" in components and (run_dir / "oof_coarse_proba.npy").exists():
        coarse_proba = np.load(run_dir / "oof_coarse_proba.npy")
        # Re-mix per fold using saved mix
        for fold in sorted(set(fold_ids.tolist())):
            fold_mix_path = run_dir / f"fold_{fold}" / "c2f_mix.txt"
            if fold_mix_path.exists():
                mix = float(fold_mix_path.read_text(encoding="utf-8").strip())
                idx = np.flatnonzero(fold_ids == fold)
                proba[idx] = proba[idx] * (1 - mix) + coarse_proba[idx] * mix
        proba = np.maximum(proba, 1e-12)
        proba = proba / proba.sum(axis=1, keepdims=True)

    # Load tail expert scores if available
    expert_scores = None
    if "tail_experts" in components:
        expert_scores = {}
        for tid in TAIL_IDS:
            path = run_dir / f"oof_expert_{tid}.npy"
            if path.exists():
                expert_scores[tid] = np.load(path)

    decode_cfg = config["decode"]
    tau_grid = [args.tau] if args.tau is not None else decode_cfg.get("tau_grid", [0.0, 0.1, 0.2, 0.3, 0.4])
    beta_grid = [args.beta] if args.beta is not None else decode_cfg.get("beta_grid", [0.0, 0.2, 0.5, 0.8, 1.0])
    gamma_grid = _as_list(decode_cfg.get("tail_gamma_grid", [decode_cfg.get("tail_gamma", 0.5)]))
    theta_grid = _as_list(decode_cfg.get("tail_threshold_grid", [decode_cfg.get("tail_threshold", 0.5)]))
    order_grid = [args.decode_order] if args.decode_order else decode_cfg.get(
        "decode_order_grid",
        ["bayes_only", "risk_viterbi", "bayes_then_viterbi", "viterbi_then_bayes"],
    )

    candidates = []
    for tau in tau_grid:
        gamma_values = gamma_grid if expert_scores else [0.0]
        theta_values = theta_grid if expert_scores else [1.0]
        for gamma in gamma_values:
            for theta in theta_values:
                pred = _chunked_logit_tail_bayes(proba, prior, float(tau), A, expert_scores, float(gamma), float(theta))
                metrics = evaluate_all(y_true, pred, well_id, A)
                objective = _objective(metrics, args.objective)
                candidates.append((objective, float(tau), 0.0, float(gamma), float(theta), "bayes_only", None, pred, metrics))

    candidates.sort(key=lambda row: row[0], reverse=True)
    best = candidates[0] if candidates else None
    viterbi_orders = [order for order in order_grid if order != "bayes_only"]
    if viterbi_orders:
        for _, tau, _, gamma, theta, _, _, _, _ in candidates[: max(args.top_k_viterbi, 0)]:
            adj = logit_adjust(proba, prior, float(tau))
            if expert_scores:
                adj = tail_expert_gate(adj, expert_scores, tail_ids=TAIL_IDS, gamma=float(gamma), theta=float(theta))
            for order in viterbi_orders:
                for beta in beta_grid:
                    if beta <= 0:
                        continue
                    pred = _decode(adj, well_id, fold_ids, run_dir, A, float(beta), order)
                    metrics = evaluate_all(y_true, pred, well_id, A)
                    objective = _objective(metrics, args.objective)
                    row = (objective, float(tau), float(beta), float(gamma), float(theta), order, adj, pred, metrics)
                    if best is None or row[0] > best[0]:
                        best = row
    assert best is not None
    _, tau, beta, gamma, theta, order, _, pred, metrics = best
    per_tail, tail_mean = tail_f1(y_true, pred)
    report = {
        "tau": tau,
        "beta": beta,
        "tail_gamma": gamma,
        "tail_threshold": theta,
        "decode_order": order,
        "objective": args.objective,
        "weighted_f1": metrics.weighted_f1,
        "macro_f1": metrics.macro_f1,
        "boundary_f1": metrics.boundary_f1,
        "penalty": metrics.penalty,
        "tail_mean_f1": tail_mean,
        "tail_f1": per_tail,
        "per_class": per_class_report(y_true, pred),
    }
    out_path = Path(args.out) if args.out else run_dir / "decode_report.json"
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"well_id": well_id, "depth": rows["DEPTH_MD"], "y_true": y_true, "y_pred": pred}).to_csv(
        out_path.with_suffix(".csv"), index=False
    )
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_class"}}, indent=2))
    print(f"Wrote {out_path}")


def _eval_hidden(run_dir, config, A, components, args, project_root):
    # Hidden test requires a feature table for hidden wells.
    # We expect the user to provide a --table argument or infer from run_meta.
    # For now, this is a scaffold: hidden test table should be passed via --table.
    print("Hidden test inference scaffold. Use --table <hidden_features.parquet> to supply hidden test features.")
    # If table is not provided, exit.
    # (Full implementation would load models per fold, average predictions, then decode.)


def _decode(proba, well_id, fold_ids, run_dir, A, beta, order):
    if beta <= 0 or order == "bayes_only":
        return bayes_risk_decode(proba, A)
    if order == "bayes_then_viterbi":
        # First Bayes risk decode, then refine with standard Viterbi
        interim = bayes_risk_decode(proba, A)
        # Actually, the doc says two variants. Let's implement both:
        # Option A: Bayes risk -> Viterbi on probabilities
        pred = np.zeros(len(proba), dtype=np.int16)
        for fold in sorted(set(fold_ids.tolist())):
            idx = np.flatnonzero(fold_ids == fold)
            transition = np.load(run_dir / f"fold_{fold}" / "transition.npy")
            pred[idx] = viterbi_by_well(proba[idx], well_id[idx], transition, beta=beta)
        return pred
    if order == "risk_viterbi":
        pred = np.zeros(len(proba), dtype=np.int16)
        for fold in sorted(set(fold_ids.tolist())):
            idx = np.flatnonzero(fold_ids == fold)
            transition = np.load(run_dir / f"fold_{fold}" / "transition.npy")
            pred[idx] = risk_viterbi_by_well(proba[idx], well_id[idx], transition, A, beta=beta)
        return pred
    if order == "viterbi_then_bayes":
        pred = np.zeros(len(proba), dtype=np.int16)
        for fold in sorted(set(fold_ids.tolist())):
            idx = np.flatnonzero(fold_ids == fold)
            transition = np.load(run_dir / f"fold_{fold}" / "transition.npy")
            pred[idx] = viterbi_by_well(proba[idx], well_id[idx], transition, beta=beta)
        # Then Bayes risk on the Viterbi-smoothed probabilities is less meaningful;
        # instead, just return Viterbi result.
        return pred
    return bayes_risk_decode(proba, A)


def _objective(metrics, mode):
    if mode == "penalty":
        # Penalty is negative; larger is better. Keep modest regularizers so
        # the search does not collapse into geologically cheap but useless labels.
        return 3.0 * metrics.penalty + 0.6 * metrics.boundary_f1 + 0.4 * metrics.macro_f1 + 0.2 * metrics.tail_mean_f1
    return metrics.penalty + metrics.boundary_f1 + metrics.macro_f1 + 0.5 * metrics.tail_mean_f1


def _chunked_logit_tail_bayes(proba, prior, tau, A, expert_scores=None, gamma=0.0, theta=1.0, chunk=200_000):
    A = np.asarray(A, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    out = np.zeros(len(proba), dtype=np.int16)
    log_prior = np.log(prior + np.float32(1e-12))
    for start in range(0, len(proba), chunk):
        end = min(start + chunk, len(proba))
        p = np.asarray(proba[start:end], dtype=np.float32)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), np.float32(1e-12))
        logits = np.log(p + np.float32(1e-12)) - np.float32(tau) * log_prior
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits).astype(np.float32)
        p /= np.maximum(p.sum(axis=1, keepdims=True), np.float32(1e-12))
        if expert_scores:
            for cls in TAIL_IDS:
                if cls not in expert_scores:
                    continue
                fire = expert_scores[cls][start:end] > theta
                if np.any(fire):
                    p[fire, cls] *= np.float32(1.0 + gamma)
            p /= np.maximum(p.sum(axis=1, keepdims=True), np.float32(1e-12))
        out[start:end] = (p @ A).argmin(axis=1).astype(np.int16)
    return out


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
