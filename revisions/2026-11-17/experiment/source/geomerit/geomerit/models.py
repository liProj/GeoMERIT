from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ModelUnavailable(RuntimeError):
    pass


def prepare_matrix(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    cols = numeric + categorical
    X = df[cols].copy()
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
        elif X[col].dtype == "float64":
            X[col] = X[col].astype("float32")
    for col in categorical:
        X[col] = X[col].astype("category")
    return X


def train_lightgbm(X_train, y_train, X_valid, y_valid, sample_weight, params: dict[str, Any], categorical: list[str]):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ModelUnavailable("lightgbm is not installed") from exc
    train_params = params.copy()
    num_boost_round = int(train_params.pop("num_boost_round", 1000))
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight, categorical_feature=categorical, free_raw_data=False)
    valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=50), lgb.early_stopping(100, verbose=False)]
    return lgb.train(train_params, train_data, num_boost_round=num_boost_round, valid_sets=[valid_data], callbacks=callbacks)


def train_xgboost(X_train, y_train, X_valid, y_valid, sample_weight, params: dict[str, Any]):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ModelUnavailable("xgboost is not installed") from exc
    train_params = params.copy()
    num_boost_round = int(train_params.pop("num_boost_round", 1000))
    # Infer num_class for multiclass objectives.
    obj = train_params.get("objective", "")
    if "multi" in str(obj) and "num_class" not in train_params:
        train_params["num_class"] = int(np.max(y_train)) + 1
    X_train = _category_codes(X_train)
    X_valid = _category_codes(X_valid)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight, missing=np.nan)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, missing=np.nan)
    return xgb.train(train_params, dtrain, num_boost_round=num_boost_round, evals=[(dvalid, "valid")], verbose_eval=50)


def train_catboost(X_train, y_train, X_valid, y_valid, sample_weight, params: dict[str, Any], categorical: list[str]):
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise ModelUnavailable("catboost is not installed") from exc
    cat_idx = [X_train.columns.get_loc(col) for col in categorical if col in X_train.columns]
    train_pool = Pool(X_train, label=y_train, weight=sample_weight, cat_features=cat_idx)
    valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_idx)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def predict_proba(model, X: pd.DataFrame, model_name: str, class_count: int) -> np.ndarray:
    if model_name == "xgboost":
        import xgboost as xgb

        pred = model.predict(xgb.DMatrix(_category_codes(X), missing=np.nan))
    elif model_name == "lightgbm":
        pred = model.predict(X)
    elif model_name == "catboost":
        pred = model.predict_proba(X)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, class_count)
    if pred.shape[1] != class_count:
        full = np.zeros((len(pred), class_count), dtype=float)
        full[:, : pred.shape[1]] = pred
        pred = full
    pred = np.maximum(pred, 1e-12)
    return pred / pred.sum(axis=1, keepdims=True)


def geometric_ensemble(probas: dict[str, np.ndarray], alpha: dict[str, float]) -> np.ndarray:
    names = [name for name in probas if alpha.get(name, 0.0) > 0]
    if not names:
        raise ValueError("No enabled model probabilities for ensemble")
    logp = None
    total = 0.0
    for name in names:
        weight = float(alpha.get(name, 1.0))
        arr = np.log(np.maximum(probas[name], 1e-12)) * weight
        logp = arr if logp is None else logp + arr
        total += weight
    logp /= max(total, 1e-12)
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def save_pickle(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def _category_codes(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.select_dtypes(["category", "object"]).columns:
        X[col] = X[col].astype("category").cat.codes
    return X


# ---------------------------------------------------------------------------
# Coarse-to-fine helpers
# ---------------------------------------------------------------------------

def train_coarse_fine(
    X_train, y_train, X_valid, y_valid, sample_weight,
    coarse_y_train, coarse_y_valid,
    config, categorical, coarse_groups, class_count,
):
    """Train a coarse classifier + per-coarse fine classifiers.
    Returns dict of models and metadata."""
    models = {}
    backend = "lightgbm"
    coarse_cfg = config["models"].get("lightgbm", {})
    if not coarse_cfg.get("enabled", False):
        backend = "xgboost"
    coarse_params = {**(coarse_cfg["params"] if backend == "lightgbm" else config["models"]["xgboost"]["params"])}
    if backend == "lightgbm":
        coarse_model = train_lightgbm(
            X_train, coarse_y_train, X_valid, coarse_y_valid, sample_weight,
            {**coarse_params, "num_class": len(coarse_groups)}, categorical
        )
    else:
        coarse_model = train_xgboost(
            X_train, coarse_y_train, X_valid, coarse_y_valid, sample_weight,
            {**coarse_params, "num_class": len(coarse_groups)}
        )
    models["coarse"] = coarse_model
    models["backend"] = backend

    # Fine classifiers per coarse group
    fine_models = {}
    for coarse_name, fine_ids in coarse_groups.items():
        mask_train = np.isin(y_train, fine_ids)
        mask_valid = np.isin(y_valid, fine_ids)
        if mask_train.sum() < 10 or mask_valid.sum() < 5:
            continue
        # Remap fine ids to 0..k within this group
        local_map = {fid: i for i, fid in enumerate(sorted(fine_ids))}
        yf_train = np.array([local_map[int(v)] for v in y_train[mask_train]], dtype=int)
        yf_valid = np.array([local_map[int(v)] for v in y_valid[mask_valid]], dtype=int)
        if len(local_map) <= 1:
            continue
        if backend == "lightgbm":
            fine_model = train_lightgbm(
                X_train.loc[mask_train], yf_train,
                X_valid.loc[mask_valid], yf_valid,
                sample_weight[mask_train],
                {**coarse_params, "num_class": len(local_map)}, categorical
            )
        else:
            fine_model = train_xgboost(
                X_train.loc[mask_train], yf_train,
                X_valid.loc[mask_valid], yf_valid,
                sample_weight[mask_train],
                {**coarse_params, "num_class": len(local_map)}
            )
        fine_models[coarse_name] = {"model": fine_model, "map": local_map}
    models["fine"] = fine_models
    return models


def predict_coarse_fine(models, X, coarse_groups, class_count, model_name="lightgbm") -> np.ndarray:
    """Return flat 12-class probability from coarse-to-fine cascade."""
    coarse_proba = predict_proba(models["coarse"], X, model_name, len(coarse_groups))
    fine_out = np.zeros((len(X), class_count), dtype=float)
    coarse_name_to_idx = {name: i for i, name in enumerate(coarse_groups.keys())}
    for coarse_name, fine_ids in coarse_groups.items():
        ci = coarse_name_to_idx[coarse_name]
        if coarse_name not in models["fine"]:
            # Fallback: spread coarse mass uniformly over fine classes
            for fid in fine_ids:
                fine_out[:, fid] += coarse_proba[:, ci] / max(len(fine_ids), 1)
            continue
        fine_model_info = models["fine"][coarse_name]
        fine_proba = predict_proba(fine_model_info["model"], X, model_name, len(fine_model_info["map"]))
        inv_map = {v: k for k, v in fine_model_info["map"].items()}
        for local_idx, global_idx in inv_map.items():
            fine_out[:, global_idx] = coarse_proba[:, ci] * fine_proba[:, local_idx]
    # Normalize row-wise
    fine_out = np.maximum(fine_out, 1e-12)
    return fine_out / fine_out.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Tail-expert helpers
# ---------------------------------------------------------------------------

def train_tail_experts(
    X_train, y_train, X_valid, y_valid, sample_weight,
    tail_ids, config, categorical,
) -> dict[int, Any]:
    """Train one-vs-rest binary classifiers for each tail class."""
    experts = {}
    coarse_cfg = config["models"].get("lightgbm", {})
    bin_params = {**coarse_cfg["params"], "objective": "binary", "metric": "binary_logloss"}
    bin_params.pop("num_class", None)
    for tid in tail_ids:
        yb_train = (y_train == tid).astype(int)
        yb_valid = (y_valid == tid).astype(int)
        if yb_train.sum() < 20 or yb_valid.sum() < 1:
            print(f"  Skipping tail expert {tid}: train_pos={int(yb_train.sum())} valid_pos={int(yb_valid.sum())}")
            continue
        # Up-weight tail positives heavily
        sw = sample_weight.copy()
        sw[yb_train == 1] *= 5.0
        if coarse_cfg.get("enabled", False):
            model = train_lightgbm(X_train, yb_train, X_valid, yb_valid, sw, bin_params, categorical)
        else:
            xgb_params = {**config["models"]["xgboost"]["params"], "objective": "binary:logistic", "eval_metric": "logloss"}
            xgb_params.pop("num_class", None)
            model = train_xgboost(X_train, yb_train, X_valid, yb_valid, sw, xgb_params)
        experts[tid] = model
    return experts


def predict_tail_experts(experts, X, model_name="lightgbm") -> dict[int, np.ndarray]:
    scores = {}
    for tid, model in experts.items():
        if model_name == "lightgbm":
            pred = model.predict(X)
        elif model_name == "xgboost":
            import xgboost as xgb
            pred = model.predict(xgb.DMatrix(_category_codes(X), missing=np.nan))
        else:
            pred = model.predict_proba(X)[:, 1]
        scores[tid] = np.asarray(pred, dtype=float).ravel()
    return scores
