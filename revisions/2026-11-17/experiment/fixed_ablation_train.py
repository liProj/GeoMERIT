#!/usr/bin/env python3
"""Five fixed-parameter retrained ablations; frozen full architecture, outer folds unchanged."""
from pathlib import Path
import os, sys, json, hashlib, importlib.util, argparse, gc, shutil, time
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT/'source/geomerit'
sys.path.insert(0,str(PROJECT))
spec=importlib.util.spec_from_file_location('frozen_revision_runner',PROJECT/'scripts/10_revision_train.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from geomerit.revision_data import feature_columns, digest
from geomerit.revision_decision import prior, evaluated_predictions, confusion, scores_from_cm
from geomerit.revision_models import boundary_flags
write_json,block,assert_disjoint=old.write_json,old.block,old.assert_disjoint
PARENT_SIGNATURE='2b07090b2d6a520ad45fb241d0500633faac32a08cbd5fe8ec0e0b86c20f49c2'
FIXED_VARIANTS=['no_missing','no_normalization','no_class_weight','no_boundary_weight','no_confidence_weight']

def protocol():
    parent=json.loads((ROOT/'experiment_v1/runs/protocol.json').read_text())
    cfg=yaml.safe_load((PROJECT/'configs/revision_20261103.yaml').read_text())
    assert parent['signature']==PARENT_SIGNATURE and parent['smoke_only'] is False
    assert cfg==parent['config'] and old.source_manifest()==parent['code'], 'Frozen source/config changed'
    definition=dict(protocol_id='fixed_parameters_outer_refit_20261104',parent_signature=PARENT_SIGNATURE,
        runner_sha256=digest(Path(__file__)),variants=FIXED_VARIANTS,outer_folds=10,
        inner_fits_per_ablation_fold=0,outer_fits_per_ablation_fold=1,
        inherited_parameters=['blend','decision'],selection_source='same_outer_fold/full/inner_only',
        full_architecture_retained=True,training_state_and_priors_refitted=True,smoke_only=False)
    signature=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
    return cfg,dict(signature=signature,**definition)

def inherit_selection(source_full, fold, train=None, test=None):
    source=Path(source_full)/f'outer_{fold:02d}'
    marker=json.loads((source/'fold_complete.json').read_text())
    assert marker['signature']==PARENT_SIGNATURE and marker['smoke_only'] is False
    assert digest(source/'predictions.parquet')==marker['prediction_sha256']
    path=source/'selected_parameters.json'
    selection=json.loads(path.read_text())
    assert selection['signature']==PARENT_SIGNATURE
    assert selection['selected_before_outer_fit'] is True and selection['evaluation_labels_used'] is False
    checkpoint=json.loads((source/'outer_fit/complete.json').read_text())
    assert checkpoint['token']['signature']==PARENT_SIGNATURE
    assert digest(source/'outer_fit/split.json')==checkpoint['files']['split.json']
    split=json.loads((source/'outer_fit/split.json').read_text())
    assert split['seed']==20261103+100*fold+99
    assert not set(split['training_wells']) & set(split['evaluation_wells'])
    if train is not None:
        assert sorted(train.well_id.unique().tolist())==split['training_wells']
        assert sorted(test.well_id.unique().tolist())==split['evaluation_wells']
        ids=hashlib.sha256(('\n'.join(train.sample_id)+'\0'+'\n'.join(test.sample_id)).encode()).hexdigest()
        assert ids==split['partitions_sha256'], 'Fold/sample partition mismatch'
    return dict(blend=selection['blend'],decision=selection['decision'],candidates=0,
        source_candidate_count=selection['candidates'],selected_before_outer_fit=True,
        evaluation_labels_used=False,independent_inner_reselection=False,
        selection_source=str(path),source_selection_sha256=digest(path),source_full_fold=fold,
        parent_signature=PARENT_SIGNATURE,ablation_protocol='fixed_parameters_outer_refit')

def run_fold(frame, variant_name, manifest, cfg, run_dir, fold, smoke, signature, cost, source_full):
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
    selection = inherit_selection(source_full, fold, outer_train, outer_test)
    selection['signature'] = signature
    write_json(destination/'selected_parameters.json', selection)
    print('INHERIT SAME-FOLD INNER SELECTION', variant_name, fold, selection['source_selection_sha256'], flush=True)
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
               selection_source=selection['selection_source'], source_selection_sha256=selection['source_selection_sha256'],
               source_full_fold=fold, parent_signature=PARENT_SIGNATURE, ablation_protocol='fixed_parameters_outer_refit',
               completed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), smoke_only=smoke))
    print('OUTER COMPLETE', variant_name, fold, json.dumps(metrics['full']), flush=True)
    del outer_train, outer_test, record, output, prob
    gc.collect()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',required=True)
    parser.add_argument('--variants',nargs='+',choices=FIXED_VARIANTS,required=True)
    parser.add_argument('--folds',type=int,nargs='+',required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    assert all(0<=f<10 for f in args.folds)
    cfg,prov=protocol();signature=prov['signature']
    source_full=ROOT/'experiment_v1/runs/full'
    for fold in args.folds:inherit_selection(source_full,fold)
    if args.check:
        print(json.dumps(dict(state='validation_passed',**prov)),flush=True);return
    work=Path(args.work).resolve();run_dir=work/'runs';run_dir.mkdir(parents=True,exist_ok=True)
    protocol_path=run_dir/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==prov
    else:write_json(protocol_path,prov)
    assert (work/'dataset').resolve()==(ROOT/'experiment_v1/dataset').resolve()
    manifest=old.build_dataset(PROJECT,work/'dataset',cfg)
    frame=pd.read_parquet(work/'dataset/features.parquet')
    assert not frame.sample_id.duplicated().any()
    cost=np.load(ROOT/'experiment_v1/runs/penalty_matrix.npy')
    np.save(run_dir/'penalty_matrix.npy',cost)
    try:
        for f in args.folds:
            for v in args.variants:
                write_json(run_dir/'status.json',dict(state='running',pid=os.getpid(),variant=v,fold=f,signature=signature))
                run_fold(frame,v,manifest,cfg,run_dir,f,False,signature,cost,source_full)
        write_json(run_dir/'status.json',dict(state='complete',pid=os.getpid(),variants=args.variants,folds=args.folds,signature=signature))
    except BaseException as exc:
        write_json(run_dir/'status.json',dict(state='failed',pid=os.getpid(),error=repr(exc),signature=signature));raise
    print('FIXED PARAMETER RUN COMPLETE',flush=True)

if __name__=='__main__':main()
