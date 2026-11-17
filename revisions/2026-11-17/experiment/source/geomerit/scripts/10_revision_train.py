#!/usr/bin/env python3
"""Leakage-controlled revision runner. See REVISION_PROTOCOL.md before interpreting results."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from geomerit.revision_data import build_dataset, feature_columns, TrainingFeatureState, digest
from geomerit.revision_models import train_block, boundary_flags
from geomerit.revision_decision import (prior, choose_blend, posterior, choose_decision,
                                       evaluated_predictions, confusion, scores_from_cm)
from geomerit.labels import load_penalty_matrix
from geomerit.io_las import resolve_path


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def source_manifest():
    files = list((ROOT/'geomerit').glob('*.py')) + list((ROOT/'scripts').glob('*revision*.py')) + [ROOT/'configs/data.yaml']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def assert_disjoint(a, b):
    assert not set(a.well_id) & set(b.well_id), 'A well crosses training and evaluation partitions'
    assert not set(a.sample_id) & set(b.sample_id), 'A sample crosses partitions'


def block(train, valid, numeric, categorical, variant, cfg, destination, seed, smoke, signature):
    destination.mkdir(parents=True, exist_ok=True)
    complete = destination/'complete.json'
    ids = hashlib.sha256(('\n'.join(train.sample_id)+'\0'+'\n'.join(valid.sample_id)).encode()).hexdigest()
    token = dict(signature=signature, partitions_sha256=ids, seed=seed)
    if complete.exists():
        old = json.loads(complete.read_text())
        assert old['token'] == token, 'Checkpoint signature mismatch; use a new run directory'
        for name, expected in old['files'].items():
            assert digest(destination/name) == expected, f'Corrupt checkpoint: {name}'
        print('REUSE', destination, flush=True)
        outputs = {name: np.load(destination/(name+'.npy')) for name in old['outputs']}
        return outputs
    assert_disjoint(train, valid)
    write_json(destination/'split.json', dict(training_wells=sorted(train.well_id.unique().tolist()),
                                             evaluation_wells=sorted(valid.well_id.unique().tolist()), **token))
    state = TrainingFeatureState.fit(train, numeric, categorical)
    # No evaluation labels or confidence values are accepted by the fitting routine.
    valid_inputs = valid.drop(columns=['label_idx', 'confidence'])
    outputs, _ = train_block(train, valid_inputs, state, variant, cfg, destination, seed, smoke,
                             retain_models=destination.name == 'outer_fit')
    files = {p.name: digest(p) for p in destination.iterdir() if p.suffix in ['.npy', '.json', '.joblib']}
    write_json(complete, dict(token=token, outputs=list(outputs), files=files))
    return outputs


def run_fold(frame, variant_name, manifest, cfg, run_dir, fold, smoke, signature, cost):
    if shutil.disk_usage(run_dir).free < 15*1024**3:
        raise RuntimeError('Less than 15 GiB remains; stop before training to preserve existing artifacts')
    variant = cfg['variants'][variant_name]
    destination = run_dir/variant_name/f'outer_{fold:02d}'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'fold_complete.json').exists():
        old = json.loads((destination/'fold_complete.json').read_text())
        assert old['signature'] == signature
        assert digest(destination/'predictions.parquet') == old['prediction_sha256']
        print('FOLD ALREADY COMPLETE', variant_name, fold, flush=True)
        return
    numeric, categorical = feature_columns(manifest, variant)
    needed = list(dict.fromkeys(numeric+categorical+['well_id', 'sample_id', 'label_idx', 'confidence',
                       'DEPTH_MD', 'outer_fold', 'well_missing_rate', 'x_loc', 'y_loc']))
    outer_train = frame.loc[frame.outer_fold != fold, needed].reset_index(drop=True)
    outer_test = frame.loc[frame.outer_fold == fold, needed].reset_index(drop=True)
    assert_disjoint(outer_train, outer_test)
    inner_folds = 2 if smoke else int(cfg['inner_folds'])
    splits = GroupKFold(n_splits=inner_folds, shuffle=True, random_state=int(cfg['seed'])+fold)
    oof, assignment = {}, np.full(len(outer_train), -1, dtype=np.int16)
    priors = np.zeros((len(outer_train), 12), dtype=np.float32)
    for inner, (train_idx, valid_idx) in enumerate(splits.split(outer_train, groups=outer_train.well_id)):
        train, valid = outer_train.iloc[train_idx], outer_train.iloc[valid_idx]
        assert not set(train.well_id) & set(outer_test.well_id)
        assert not set(valid.well_id) & set(outer_test.well_id)
        output = block(train, valid, numeric, categorical, variant, cfg, destination/f'inner_{inner:02d}',
                       int(cfg['seed']) + 100*fold + inner, smoke, signature)
        for name, value in output.items():
            if name not in oof: oof[name] = np.zeros((len(outer_train), value.shape[1]), dtype=np.float32)
            oof[name][valid_idx] = value
        priors[valid_idx] = prior(train.label_idx, cfg['selection']['prior_smoothing'])
        assert np.all(assignment[valid_idx] == -1)
        assignment[valid_idx] = inner
        del output, train, valid
        gc.collect()
    assert (assignment >= 0).all()
    for name, value in oof.items(): np.save(destination/('inner_oof_'+name+'.npy'), value)
    np.save(destination/'inner_priors.npy', priors)
    pd.DataFrame({'sample_id': outer_train.sample_id, 'well_id': outer_train.well_id,
                  'inner_fold': assignment}).to_parquet(destination/'inner_assignment.parquet', index=False)
    y_inner = outer_train.label_idx.to_numpy(int)
    print('SELECT INNER ONLY', variant_name, fold, flush=True)
    with threadpool_limits(limits=2, user_api='blas'):
        blend = choose_blend(oof, y_inner, cfg, cost)
        decision, grid = choose_decision(posterior(oof, blend), oof['experts'], priors, y_inner, cfg, cost,
                                        allow_experts=variant.get('family', 'geo') == 'geo')
    selection = dict(blend=blend, decision=decision, candidates=len(grid), signature=signature,
                     selected_before_outer_fit=True, evaluation_labels_used=False)
    write_json(destination/'selected_parameters.json', selection)
    write_json(destination/'inner_decision_grid.json', grid)
    del oof, priors, y_inner, grid
    gc.collect()
    output = block(outer_train, outer_test, numeric, categorical, variant, cfg, destination/'outer_fit',
                   int(cfg['seed'])+100*fold+99, smoke, signature)
    outer_prior = prior(outer_train.label_idx, cfg['selection']['prior_smoothing'])
    np.save(destination/'outer_training_prior.npy', outer_prior)
    with threadpool_limits(limits=2, user_api='blas'):
        predictions, prob = evaluated_predictions(output, selection, outer_prior, cost)
    np.save(destination/'final_probabilities.npy', prob)
    record = outer_test[['sample_id', 'well_id', 'DEPTH_MD', 'label_idx', 'outer_fold', 'well_missing_rate']].copy()
    y = record.label_idx.to_numpy(int)
    boundary = boundary_flags(y, record.well_id.to_numpy(), record.DEPTH_MD.to_numpy(), radius=1)
    record['boundary'] = boundary
    metrics = {}
    for name, pred in predictions.items():
        record['pred_'+name] = pred
        scores = scores_from_cm(confusion(y, pred), cost)
        scores['boundary_f1'] = scores_from_cm(confusion(y[boundary], pred[boundary]), cost)['weighted_f1'] if boundary.any() else None
        metrics[name] = scores
    record.to_parquet(destination/'predictions.parquet', index=False)
    write_json(destination/'metrics.json', metrics)
    write_json(destination/'fold_complete.json', dict(signature=signature,
               prediction_sha256=digest(destination/'predictions.parquet'), rows=len(record),
               completed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), smoke_only=smoke))
    print('OUTER COMPLETE', variant_name, fold, json.dumps(metrics['full']), flush=True)
    del outer_train, outer_test, record, output, prob
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'configs/revision_20261103.yaml'))
    parser.add_argument('--work', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--folds', type=int, nargs='+')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    work = Path(args.work).resolve(); work.mkdir(parents=True, exist_ok=True)
    run_dir = work/('smoke' if args.smoke else 'runs')
    run_dir.mkdir(exist_ok=True)
    provenance = dict(config=cfg, code=source_manifest(), smoke_only=args.smoke)
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    lock = run_dir/'protocol.json'
    if lock.exists():
        assert json.loads(lock.read_text())['signature'] == signature, 'Frozen protocol differs: choose a new work directory'
    else:
        packages = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','xgboost','catboost','pyarrow']}
        write_json(lock, dict(signature=signature, **provenance, packages=packages,
                             host=platform.node(), platform=platform.platform(), python=sys.version))
    print('PROTOCOL', signature, 'SMOKE' if args.smoke else 'FORMAL', flush=True)
    manifest = build_dataset(ROOT, work/'dataset', cfg)
    frame = pd.read_parquet(work/'dataset/features.parquet')
    assert not frame.sample_id.duplicated().any()
    if args.smoke:
        # Preserve row order, sample equally by well; this run is never reportable performance.
        indices = frame.groupby('well_id', sort=False).sample(n=40, random_state=cfg['seed']).index
        frame = frame.loc[np.sort(indices)].reset_index(drop=True)
        cfg['selection']['tau'] = [0., .1]
        cfg['selection']['tail_gamma'] = [0., .5]
        cfg['selection']['tail_threshold'] = [.35]
        cfg['selection']['dolomite_gamma'] = [0., 2.]
        cfg['selection']['dolomite_threshold'] = [.05]
        cfg['models']['lightgbm']['params']['min_data_in_leaf'] = 5
        cfg['models']['lightgbm']['params']['num_leaves'] = 7
    data_cfg = yaml.safe_load((ROOT/'configs/data.yaml').read_text(encoding='utf-8'))
    cost = load_penalty_matrix(resolve_path(data_cfg['paths']['scoring_matrix'], ROOT))
    assert np.isfinite(cost).all() and (cost >= 0).all() and np.diag(cost).sum() == 0
    np.save(run_dir/'penalty_matrix.npy', cost)
    variants = args.variants or (['full', 'rfe_xgb', 'raw_xgb'] if args.smoke else cfg['queue'])
    folds = args.folds or ([0] if args.smoke else list(range(10)))
    write_json(run_dir/'status.json', dict(state='running', pid=os.getpid(), variants=variants, folds=folds, signature=signature))
    # Finish the three main comparisons before the train-again ablations.
    phases = [variants[:3], variants[3:]] if variants[:3] == ['full','rfe_xgb','raw_xgb'] else [variants]
    try:
        for phase in phases:
            for fold in folds:
                for variant in phase:
                    write_json(run_dir/'status.json', dict(state='running', pid=os.getpid(), variant=variant, fold=fold, signature=signature))
                    run_fold(frame, variant, manifest, cfg, run_dir, fold, args.smoke, signature, cost)
        write_json(run_dir/'status.json', dict(state='complete', pid=os.getpid(), smoke_only=args.smoke, variants=variants, folds=folds, signature=signature))
    except BaseException as exc:
        write_json(run_dir/'status.json', dict(state='failed', pid=os.getpid(), error=repr(exc), signature=signature))
        raise
    print('RUN COMPLETE', flush=True)


if __name__ == '__main__': main()
