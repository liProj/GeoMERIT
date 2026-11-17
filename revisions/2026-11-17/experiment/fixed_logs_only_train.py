#!/usr/bin/env python3
"""logs_only as a fixed-parameter outer refit (added 2026-11-10 to meet the one-week reviewer deadline).

Same procedure as fixed_ablation_train.py (imported unchanged): the same-outer-fold full model's inner-selected
blend and decision parameters are inherited, the full architecture is refitted once on the outer training wells
with the logs_only feature set. No independent inner selection for logs_only. Separate result root and signature.
"""
from pathlib import Path
import os, sys, json, hashlib, importlib.util, argparse
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixed_ablation_train',ROOT/'fixed_ablation_train.py')
fa=importlib.util.module_from_spec(spec);spec.loader.exec_module(fa)
import numpy as np, pandas as pd
VARIANTS=['logs_only']
RESULT_ROOT=ROOT/'experiment_fixed_parameters_logs_only_20261110'

def protocol():
    cfg,base=fa.protocol()
    definition=dict(protocol_id='fixed_parameters_outer_refit_logs_only_20261110',parent_signature=fa.PARENT_SIGNATURE,
        runner_sha256=fa.digest(Path(__file__)),base_runner_sha256=base['runner_sha256'],base_fixed_signature=base['signature'],
        variants=VARIANTS,outer_folds=10,inner_fits_per_ablation_fold=0,outer_fits_per_ablation_fold=1,
        inherited_parameters=['blend','decision'],selection_source='same_outer_fold/full/inner_only',
        full_architecture_retained=True,training_state_and_priors_refitted=True,smoke_only=False)
    signature=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
    return cfg,dict(signature=signature,**definition)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work');parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    parser.add_argument('--folds',type=int,nargs='+',default=[]);parser.add_argument('--init',action='store_true')
    args=parser.parse_args()
    cfg,prov=protocol();signature=prov['signature']
    if args.init:
        run=RESULT_ROOT/'runs';run.mkdir(parents=True,exist_ok=True);p=run/'protocol.json'
        if p.exists():assert json.loads(p.read_text())==prov
        else:fa.write_json(p,prov)
        print(json.dumps(prov));return
    assert args.work and args.folds and all(0<=f<10 for f in args.folds)
    source_full=ROOT/'experiment_v1/runs/full'
    for fold in args.folds:fa.inherit_selection(source_full,fold)
    work=Path(args.work).resolve();run_dir=work/'runs';run_dir.mkdir(parents=True,exist_ok=True)
    protocol_path=run_dir/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==prov
    else:fa.write_json(protocol_path,prov)
    assert (work/'dataset').resolve()==(ROOT/'experiment_v1/dataset').resolve()
    manifest=fa.old.build_dataset(fa.PROJECT,work/'dataset',cfg)
    frame=pd.read_parquet(work/'dataset/features.parquet')
    assert not frame.sample_id.duplicated().any()
    cost=np.load(ROOT/'experiment_v1/runs/penalty_matrix.npy')
    np.save(run_dir/'penalty_matrix.npy',cost)
    try:
        for f in args.folds:
            for v in args.variants:
                fa.write_json(run_dir/'status.json',dict(state='running',pid=os.getpid(),variant=v,fold=f,signature=signature))
                fa.run_fold(frame,v,manifest,cfg,run_dir,f,False,signature,cost,source_full)
        fa.write_json(run_dir/'status.json',dict(state='complete',pid=os.getpid(),variants=args.variants,folds=args.folds,signature=signature))
    except BaseException as exc:
        fa.write_json(run_dir/'status.json',dict(state='failed',pid=os.getpid(),error=repr(exc),signature=signature));raise
    print('FIXED PARAMETER RUN COMPLETE',flush=True)

if __name__=='__main__':main()
