"""Fixed-budget models: fitting never accepts evaluation labels or examples."""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .labels import COARSE_GROUPS, TAIL_IDS, coarse_labels


def boundary_flags(y, wells, depth, radius=1, max_gap=1.0):
    y, wells, depth = np.asarray(y), np.asarray(wells), np.asarray(depth)
    flags = np.zeros(len(y), dtype=bool)
    for well in np.unique(wells):
        idx = np.flatnonzero(wells == well)
        dd = np.diff(depth[idx])
        cuts = np.r_[0, np.flatnonzero((dd <= 0) | (dd > max_gap) | ~np.isfinite(dd)) + 1, len(idx)]
        for start, stop in zip(cuts[:-1], cuts[1:]):
            sub = idx[start:stop]
            changes = np.flatnonzero(y[sub[:-1]] != y[sub[1:]])
            for j in changes:
                flags[sub[max(0, j-radius):min(len(sub), j+radius+2)]] = True
    return flags


def training_weights(train, variant, config):
    y = train.label_idx.to_numpy(int)
    counts = np.bincount(y, minlength=12)
    present = counts > 0
    family = variant.get('family', 'geo')
    if family == 'raw':
        return np.ones(len(y), dtype=np.float32)
    radius = 0 if family == 'rfe' else 1
    boundary = boundary_flags(y, train.well_id.to_numpy(), train.DEPTH_MD.to_numpy(), radius)
    w_b = np.where(boundary, 1.2, 0.7)
    if family == 'rfe':
        w_c = len(y) / (12 * counts[y])
        w_q = np.where(train.well_missing_rate.to_numpy() > .6, .7, 1.)
        return (w_c * w_b * w_q).astype(np.float32)
    rho = float(config['weights']['rho'])
    by_class = np.zeros(12)
    by_class[present] = (1-rho) / (1-np.power(rho, counts[present]))
    by_class[present] /= by_class[present].mean()
    by_class = np.minimum(by_class, float(config['weights']['cap']))
    w_c = by_class[y]
    confidence = train.confidence.to_numpy(float)
    w_q = np.ones(len(y))
    w_q[confidence == 2] = .8
    w_q[confidence == 3] = .6
    if variant.get('no_class_weight'): w_c = np.ones(len(y))
    if variant.get('no_boundary_weight'): w_b = np.ones(len(y))
    if variant.get('no_confidence_weight'): w_q = np.ones(len(y))
    weight = w_c * w_b * w_q
    return (weight / weight.mean()).astype(np.float32)


def training_subset(y, limit, seed):
    y = np.asarray(y)
    if not limit or len(y) <= limit:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = np.flatnonzero(np.isin(y, TAIL_IDS))
    # Every observed non-tail class also retains support.
    for label in np.unique(y):
        if label not in TAIL_IDS:
            pool = np.flatnonzero(y == label)
            keep = np.r_[keep, rng.choice(pool, size=min(20, len(pool)), replace=False)]
    keep = np.unique(keep)
    budget = max(0, int(limit)-len(keep))
    pool = np.setdiff1d(np.arange(len(y)), keep, assume_unique=True)
    if budget:
        keep = np.r_[keep, rng.choice(pool, size=min(budget, len(pool)), replace=False)]
    return np.sort(keep)


def categorical_frame(X, categorical_indices):
    frame = pd.DataFrame(X)
    for index in categorical_indices:
        frame[index] = frame[index].fillna(-1).astype('int32').astype(str)
    return frame


@dataclass
class FittedModel:
    backend: str
    model: object
    classes: np.ndarray
    class_count: int
    categorical_indices: list

    def predict(self, X, chunk_size=50000):
        out = np.zeros((len(X), self.class_count), dtype=np.float32)
        if len(self.classes) == 1:
            out[:, self.classes[0]] = 1
            return out
        for start in range(0, len(X), chunk_size):
            end = min(start+chunk_size, len(X)); chunk = X[start:end]
            if self.backend == 'lightgbm':
                p = self.model.predict(chunk)
            elif self.backend == 'xgboost':
                import xgboost as xgb
                p = self.model.predict(xgb.DMatrix(chunk, missing=np.nan))
            else:
                p = self.model.predict_proba(categorical_frame(chunk, self.categorical_indices))
            p = np.asarray(p, dtype=np.float32).reshape(-1, len(self.classes))
            # Columns are local labels, explicitly mapped to the fixed global ontology.
            out[start:end, self.classes] = p
        out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-30)
        return out


def fit_backend(backend, X_train, y_train, sample_weight, class_count, categorical_indices, config, seed, rounds_override=None):
    """No eval_set argument by design; iteration budgets are fixed in advance."""
    classes, local_y = np.unique(y_train, return_inverse=True)
    if not len(classes): raise ValueError('Empty training labels')
    if len(classes) == 1:
        return FittedModel(backend, None, classes.astype(int), class_count, categorical_indices)
    item = config['models'][backend]
    params = dict(item['params'])
    if backend == 'lightgbm':
        import lightgbm as lgb
        rounds = int(rounds_override or params.pop('num_boost_round'))
        params.pop('num_boost_round', None)
        params.update(num_class=len(classes), seed=seed, deterministic=True, force_col_wise=True)
        dataset = lgb.Dataset(X_train, label=local_y, weight=sample_weight,
                              categorical_feature=categorical_indices, free_raw_data=True)
        model = lgb.train(params, dataset, num_boost_round=rounds)
        del dataset
    elif backend == 'xgboost':
        import xgboost as xgb
        rounds = int(rounds_override or params.pop('num_boost_round'))
        params.pop('num_boost_round', None)
        params.update(num_class=len(classes), seed=seed)
        matrix = xgb.DMatrix(X_train, label=local_y, weight=sample_weight, missing=np.nan)
        model = xgb.train(params, matrix, num_boost_round=rounds, verbose_eval=False)
        del matrix
    elif backend == 'catboost':
        from catboost import CatBoostClassifier
        params.pop('classes_count', None)
        params.update(iterations=int(rounds_override or params['iterations']), classes_count=len(classes),
                      random_seed=seed, allow_writing_files=False, verbose=False)
        model = CatBoostClassifier(**params)
        model.fit(categorical_frame(X_train, categorical_indices), local_y,
                  sample_weight=sample_weight, cat_features=categorical_indices, use_best_model=False)
    else:
        raise ValueError(backend)
    return FittedModel(backend, model, classes.astype(int), class_count, categorical_indices)


def train_block(train, valid_inputs, state, variant, config, destination, seed, smoke=False, retain_models=True):
    """Input to prediction has no labels; all model structure follows training support."""
    assert 'label_idx' not in valid_inputs and 'confidence' not in valid_inputs
    assert not set(train.well_id) & set(valid_inputs.well_id)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    y = train.label_idx.to_numpy(int)
    X_train, X_valid = state.transform(train), state.transform(valid_inputs)
    cat_indices = list(range(len(state.numeric), len(state.numeric)+len(state.categorical)))
    weights = training_weights(train, variant, config)
    summary = {'training_rows': len(train), 'prediction_rows': len(valid_inputs), 'models': {},
               'support': np.bincount(y, minlength=12).tolist(), 'seed': seed,
               'test_labels_accepted': False, 'iteration_selection': 'prespecified fixed budgets'}
    outputs = {}

    def one(name, backend, X, labels, w, count):
        path = destination / (name + '.joblib')
        started = time.monotonic()
        print(f'FIT {destination.name}/{name} rows={len(labels)} features={X.shape[1]}', flush=True)
        model = fit_backend(backend, X, labels, w, count, cat_indices, config, seed, 3 if smoke else None)
        pred = model.predict(X_valid)
        if retain_models:
            joblib.dump(model, path, compress=1)
        summary['models'][name] = {'backend': backend, 'observed_classes': model.classes.tolist(),
                                  'seconds': time.monotonic()-started, 'rows': len(labels), 'model_retained': retain_models}
        del model
        gc.collect()
        print(f'DONE {name} seconds={summary["models"][name]["seconds"]:.1f}', flush=True)
        return pred

    backends = ['xgboost'] if variant.get('family', 'geo') != 'geo' else list(variant.get('backends', ['lightgbm', 'xgboost', 'catboost']))
    for backend in backends:
        rows = training_subset(y, config['models'][backend].get('max_train_rows'), seed)
        outputs[backend] = one(backend, backend, X_train[rows], y[rows], weights[rows], 12)
    if variant.get('hierarchy', variant.get('family', 'geo') == 'geo'):
        cy = coarse_labels(y)
        coarse = one('coarse', 'lightgbm', X_train, cy, weights, len(COARSE_GROUPS))
        hierarchy = np.zeros((len(X_valid), 12), dtype=np.float32)
        hard = np.zeros_like(hierarchy)
        selected_group = coarse.argmax(axis=1)
        for ci, (group, ids) in enumerate(COARSE_GROUPS.items()):
            rows = np.flatnonzero(np.isin(y, ids))
            if len(rows) == 0: continue
            if len(rows) < 10:
                local = np.bincount(y[rows], minlength=12).astype(float)
                local /= local.sum()
                fine = np.broadcast_to(local, hierarchy.shape)
                summary['models']['fine_'+group] = {'fallback': 'empirical training support', 'rows': len(rows)}
            else:
                fine = one('fine_'+group, 'lightgbm', X_train[rows], y[rows], weights[rows], 12)
            hierarchy += coarse[:, ci, None] * fine
            hard += (selected_group == ci)[:, None] * fine
        hierarchy /= np.maximum(hierarchy.sum(axis=1, keepdims=True), 1e-30)
        hard /= np.maximum(hard.sum(axis=1, keepdims=True), 1e-30)
        outputs['hierarchy'], outputs['hard_hierarchy'] = hierarchy, hard
    experts = np.zeros((len(X_valid), 12), dtype=np.float32)
    if variant.get('experts', variant.get('family', 'geo') == 'geo'):
        for cls in TAIL_IDS:
            binary = (y == cls).astype(int)
            if binary.sum() < 20 or (1-binary).sum() < 20:
                summary['models'][f'expert_{cls}'] = {'skipped': 'training support only', 'positives': int(binary.sum())}
                continue
            ew = weights.copy(); ew[binary == 1] *= 5
            experts[:, cls] = one(f'expert_{cls}', 'lightgbm', X_train, binary, ew, 2)[:, 1]
    outputs['experts'] = experts
    for name, value in outputs.items():
        assert np.isfinite(value).all(), name
        np.save(destination / (name + '.npy'), value)
    state.save(destination / 'feature_state.json')
    (destination / 'fit_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    del X_train, X_valid
    gc.collect()
    return outputs, summary
