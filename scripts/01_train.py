from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geomerit import cv
from geomerit.decode import apply_tail_transition_floor, estimate_transition
from geomerit.labels import COARSE_GROUPS, TAIL_IDS, coarse_labels
from geomerit.metrics import evaluate_all
from geomerit.models import (
    ModelUnavailable,
    geometric_ensemble,
    predict_coarse_fine,
    predict_proba,
    predict_tail_experts,
    prepare_matrix,
    save_pickle,
    train_catboost,
    train_coarse_fine,
    train_lightgbm,
    train_tail_experts,
    train_xgboost,
)
from geomerit.weights import boundary_flags, sample_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--split", choices=["groupkfold", "official"], default="groupkfold")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--group", default="well_id")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--penalty", default="configs/penalty_matrix.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--components", default="flat,coarse2fine,tail_experts", help="Comma-separated: flat,coarse2fine,tail_experts")
    parser.add_argument("--save_models", action="store_true", help="Persist fitted models. OOF probabilities are always saved.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    table_path = _resolve(args.table, project_root)
    config = yaml.safe_load(_resolve(args.config, project_root).read_text(encoding="utf-8"))
    meta = json.loads(table_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    meta["numeric"] = _select_numeric_features(meta["numeric"], config)
    meta["categorical"] = _select_categorical_features(meta["categorical"], config)
    needed_cols = list(dict.fromkeys(meta["numeric"] + meta["categorical"] + ["well_id", "DEPTH_MD", "label_idx", "confidence"]))
    df = pd.read_parquet(table_path, columns=needed_cols)

    class_count = int(config.get("class_count", 12))
    out_dir = _resolve(args.out, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    components = {item.strip() for item in args.components.split(",") if item.strip()}
    run_meta = {
        "table": str(table_path),
        "feature_meta": str(table_path.with_suffix(".meta.json")),
        "split": args.split,
        "class_count": class_count,
        "components": sorted(components),
        "save_models": bool(args.save_models),
        "folds": [],
    }

    labeled_index = df.index.to_numpy()[df["label_idx"].to_numpy() >= 0]
    oof = np.zeros((len(labeled_index), class_count), dtype=np.float32)
    oof_coarse = np.zeros((len(labeled_index), class_count), dtype=np.float32) if "coarse2fine" in components else None
    oof_experts = {tid: np.zeros(len(labeled_index), dtype=np.float32) for tid in TAIL_IDS} if "tail_experts" in components else None
    fold_ids = np.full(len(labeled_index), -1, dtype=np.int16)
    index_to_pos = {idx: pos for pos, idx in enumerate(labeled_index)}

    if args.split == "groupkfold":
        folds = args.folds or int(config["training"].get("folds", 10))
        iterator = cv.iter_group_folds(df, folds=folds, group_col=args.group)
    else:
        train_idx, valid_idx = cv.train_valid_split(
            df,
            valid_fraction=float(config["training"].get("validation_fraction", 0.15)),
            group_col=args.group,
            seed=int(config.get("seed", 2026)),
        )
        iterator = [(0, train_idx, valid_idx)]

    for fold, train_idx, valid_idx in iterator:
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"Training fold {fold}: train={len(train_idx)} valid={len(valid_idx)}")
        fold_proba, fold_coarse_proba, fold_expert_scores, trained_names = _train_one_fold(
            df, train_idx, valid_idx, meta, config, fold_dir, components, class_count, save_models=args.save_models
        )

        valid_pos = np.asarray([index_to_pos[idx] for idx in valid_idx], dtype=int)
        oof[valid_pos] = fold_proba
        if oof_coarse is not None and fold_coarse_proba is not None:
            oof_coarse[valid_pos] = fold_coarse_proba
        if oof_experts is not None and fold_expert_scores is not None:
            for tid, scores in fold_expert_scores.items():
                oof_experts[tid][valid_pos] = scores
        fold_ids[valid_pos] = fold

        train_labels_by_well = [
            part["label_idx"].to_numpy(int)
            for _, part in df.loc[train_idx].sort_values([args.group, "DEPTH_MD"]).groupby(args.group)
        ]
        transition = estimate_transition(train_labels_by_well, class_count, lam=1.0)
        transition = apply_tail_transition_floor(transition, floor=float(config["decode"].get("tail_min_transition", 1e-4)))
        np.save(fold_dir / "transition.npy", transition)
        run_meta["folds"].append({"fold": fold, "models": trained_names, "transition": str(fold_dir / "transition.npy")})
        _save_checkpoint(out_dir, oof, fold_ids, labeled_index, oof_coarse, oof_experts, run_meta)

    _save_checkpoint(out_dir, oof, fold_ids, labeled_index, oof_coarse, oof_experts, run_meta)
    df.loc[labeled_index, ["well_id", "DEPTH_MD", "label_idx", "confidence"]].reset_index(drop=True).to_parquet(out_dir / "oof_rows.parquet")
    print(f"Wrote run to {out_dir}")


def _train_one_fold(df, train_idx, valid_idx, meta, config, fold_dir, components, class_count, save_models=False):
    numeric = meta["numeric"]
    categorical = meta["categorical"]
    combined_idx = np.concatenate([train_idx, valid_idx])
    X_all = prepare_matrix(df.loc[combined_idx, numeric + categorical], numeric, categorical)
    X_train = X_all.loc[train_idx]
    X_valid = X_all.loc[valid_idx]
    y_train = df.loc[train_idx, "label_idx"].to_numpy(int)
    y_valid = df.loc[valid_idx, "label_idx"].to_numpy(int)

    flags = boundary_flags(y_train, df.loc[train_idx, "well_id"].to_numpy())
    w_cfg = config["weights"]
    weights = sample_weights(
        y_train,
        df.loc[train_idx, "confidence"].to_numpy(float),
        flags,
        class_count,
        rho=float(w_cfg.get("rho", 0.9995)),
        cap=float(w_cfg.get("cap", 8.0)),
        confidence_map={int(k): float(v) for k, v in w_cfg.get("confidence_map", {}).items()},
        boundary_weight=float(w_cfg.get("boundary_weight", 1.2)),
        interior_weight=float(w_cfg.get("interior_weight", 0.7)),
    )

    probas = {}
    trained_names = []
    model_cfg = config["models"]

    # Flat models
    if "flat" in components:
        if model_cfg.get("lightgbm", {}).get("enabled", False):
            try:
                params = model_cfg["lightgbm"]["params"].copy()
                params.setdefault("num_class", class_count)
                model = train_lightgbm(X_train, y_train, X_valid, y_valid, weights, params, categorical)
                if save_models:
                    save_pickle(fold_dir / "lightgbm.pkl", model)
                probas["lightgbm"] = predict_proba(model, X_valid, "lightgbm", class_count)
                trained_names.append("lightgbm")
            except Exception as exc:
                print(exc)
        if model_cfg.get("xgboost", {}).get("enabled", False):
            try:
                params = model_cfg["xgboost"]["params"].copy()
                params.setdefault("num_class", class_count)
                x_idx = _model_train_index(y_train, X_train.index.to_numpy(), model_cfg["xgboost"].get("max_train_rows"), int(config.get("seed", 2026)))
                x_pos = X_train.index.get_indexer(x_idx)
                model = train_xgboost(X_train.loc[x_idx], df.loc[x_idx, "label_idx"].to_numpy(int), X_valid, y_valid, weights[x_pos], params)
                if save_models:
                    save_pickle(fold_dir / "xgboost.pkl", model)
                probas["xgboost"] = predict_proba(model, X_valid, "xgboost", class_count)
                trained_names.append("xgboost")
            except Exception as exc:
                print(exc)
        if model_cfg.get("catboost", {}).get("enabled", False):
            try:
                c_idx = _model_train_index(y_train, X_train.index.to_numpy(), model_cfg["catboost"].get("max_train_rows"), int(config.get("seed", 2026)) + 17)
                cat_params = model_cfg["catboost"]["params"].copy()
                cat_params.setdefault("classes_count", class_count)
                c_pos = X_train.index.get_indexer(c_idx)
                model = train_catboost(X_train.loc[c_idx], df.loc[c_idx, "label_idx"].to_numpy(int), X_valid, y_valid, weights[c_pos], cat_params, categorical)
                if save_models:
                    save_pickle(fold_dir / "catboost.pkl", model)
                probas["catboost"] = predict_proba(model, X_valid, "catboost", class_count)
                trained_names.append("catboost")
            except Exception as exc:
                print(f"CatBoost skipped: {exc}")

    flat_proba = geometric_ensemble(probas, config["ensemble"].get("alpha", {})) if probas else None

    # Coarse-to-fine
    coarse_proba = None
    if "coarse2fine" in components:
        yc_train = coarse_labels(y_train)
        yc_valid = coarse_labels(y_valid)
        coarse_groups = COARSE_GROUPS
        c2f_models = train_coarse_fine(
            X_train, y_train, X_valid, y_valid, weights,
            yc_train, yc_valid,
            config, categorical, coarse_groups, class_count,
        )
        if save_models:
            save_pickle(fold_dir / "coarse2fine.pkl", c2f_models)
        coarse_proba = predict_coarse_fine(c2f_models, X_valid, coarse_groups, class_count, model_name=c2f_models.get("backend", "lightgbm"))
        trained_names.append("coarse2fine")

    # Merge flat + coarse-to-fine on validation using simple grid search for mixing weight
    fold_proba = flat_proba
    if flat_proba is not None and coarse_proba is not None:
        best_obj = -np.inf
        best_mix = 0.5
        for mix in np.linspace(0.0, 1.0, 11):
            mixed = flat_proba * (1 - mix) + coarse_proba * mix
            pred = mixed.argmax(axis=1)
            obj = f1_score(y_valid, pred, average="macro", zero_division=0)
            if obj > best_obj:
                best_obj = obj
                best_mix = mix
        print(f"  Flat/Coarse2Fine best mix={best_mix:.2f} (val macro_f1={best_obj:.4f})")
        fold_proba = flat_proba * (1 - best_mix) + coarse_proba * best_mix
        (fold_dir / "c2f_mix.txt").write_text(str(best_mix), encoding="utf-8")
    elif coarse_proba is not None:
        fold_proba = coarse_proba

    # Tail experts
    expert_scores = None
    if "tail_experts" in components and flat_proba is not None:
        experts = train_tail_experts(X_train, y_train, X_valid, y_valid, weights, TAIL_IDS, config, categorical)
        if save_models:
            save_pickle(fold_dir / "tail_experts.pkl", experts)
        expert_backend = "lightgbm" if config["models"].get("lightgbm", {}).get("enabled", False) else "xgboost"
        expert_scores = predict_tail_experts(experts, X_valid, model_name=expert_backend)
        trained_names.append("tail_experts")

    if fold_proba is None:
        raise RuntimeError("No models were trained. Install at least one of lightgbm, xgboost, catboost.")
    return fold_proba, coarse_proba, expert_scores, trained_names


def _resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def _select_numeric_features(names, config):
    mode = config.get("feature_set", "full")
    if mode != "reduced":
        return names
    keep = []
    for name in names:
        if name.endswith("_enc") or name.endswith("_well_robust_z") or name.endswith("_is_missing"):
            keep.append(name)
        elif name in {
            "missing_count",
            "missing_rate",
            "outlier_count",
            "outlier_rate",
            "rhob_nphi_sep",
            "dtc_rhob_ratio",
            "gr_log_rdep_ratio",
            "pef_rhob_ratio",
            "dist_to_xy_centroid",
            "x_loc",
            "y_loc",
            "z_loc",
            "dist_to_nearest_casing",
            "is_below_deepest_casing",
        }:
            keep.append(name)
        elif name.endswith("_grad"):
            keep.append(name)
        elif "_w5_mean" in name or "_w5_delta" in name:
            keep.append(name)
    return keep


def _select_categorical_features(names, config):
    return names


def _model_train_index(y_train, train_index, max_rows, seed):
    if not max_rows or len(train_index) <= int(max_rows):
        return train_index
    max_rows = int(max_rows)
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)
    train_index = np.asarray(train_index)
    tail_mask = np.isin(y_train, TAIL_IDS)
    keep = train_index[tail_mask]
    budget = max(max_rows - len(keep), 0)
    pool = train_index[~tail_mask]
    if budget > 0 and len(pool) > 0:
        sampled = rng.choice(pool, size=min(budget, len(pool)), replace=False)
        keep = np.concatenate([keep, sampled])
    return np.sort(keep)


def _save_checkpoint(out_dir, oof, fold_ids, labeled_index, oof_coarse, oof_experts, run_meta):
    np.save(out_dir / "oof_proba.npy", oof)
    np.save(out_dir / "oof_fold.npy", fold_ids)
    np.save(out_dir / "oof_index.npy", labeled_index)
    if oof_coarse is not None:
        np.save(out_dir / "oof_coarse_proba.npy", oof_coarse)
    if oof_experts is not None:
        for tid, arr in oof_experts.items():
            np.save(out_dir / f"oof_expert_{tid}.npy", arr)
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
