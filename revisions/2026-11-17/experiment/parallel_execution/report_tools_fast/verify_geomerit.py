from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.metadata, json, os, fcntl
import numpy as np
import pandas as pd
import psutil
import yaml

ROOT = Path(os.environ.get('GEOMERIT_REVISION_ROOT', str(Path(__file__).resolve().parents[2])))
BASE = ROOT/'parallel_execution'
NAMES = {'parallel_controller_fast.py', 'fixed_logs_only_train.py', 'parallel_controller_fixed.py', 'parallel_controller_essential.py', 'parallel_controller.py',
         'formal_supervisor.py', '10_revision_train.py', 'fixed_ablation_train.py',
         '11_revision_report.py', 'fixed_main_report.py', 'essential_report.py'}

def processes():
    found = []
    for p in psutil.process_iter(['pid', 'ppid', 'cmdline', 'status', 'create_time']):
        try:
            args = p.info['cmdline'] or []
            if p.info['status'] == psutil.STATUS_ZOMBIE:
                continue
            if any(Path(a).name in NAMES for a in args):
                found.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found

def verify(require_stopped=False):
    live = processes()
    lock = None
    if require_stopped:
        assert not live, live
        lock = (BASE/'controller.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = {}
    checked_bytes = 0
    def sha(path):
        nonlocal checked_bytes
        path = Path(path)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(8*1024**2), b''):
                h.update(b); checked_bytes += len(b)
        value = h.hexdigest()
        inventory[str(path.relative_to(ROOT))] = value
        return value
    def check(path, expected):
        actual = sha(path)
        assert actual == expected, f'SHA256 mismatch: {path}'
    scope = json.loads((BASE/'fixed_scope.json').read_text())
    variants = ['full','rfe_xgb','no_missing','no_normalization','no_class_weight',
                'no_boundary_weight','no_confidence_weight','logs_only']
    assert scope['variants'] == variants and scope['expected_folds'] == 80
    assert scope['folds_per_variant'] == 10 and scope['nested_variants'] == ['full','rfe_xgb','logs_only']
    assert scope['main'] == ['full','rfe_xgb']
    assert scope['fixed_parameter_variants'] == variants[2:7]
    normal = ROOT/'experiment_v1/runs'
    fixed = ROOT/'experiment_fixed_parameters_20261104/runs'
    proto = json.loads((normal/'protocol.json').read_text())
    fp = json.loads((fixed/'protocol.json').read_text())
    # 2026-11-10 fast scope: logs_only is a fixed-parameter outer refit with its own root/signature.
    logs = ROOT/'experiment_fixed_parameters_logs_only_20261110/runs'
    lf = json.loads((logs/'protocol.json').read_text())
    assert hashlib.sha256(json.dumps({k:v for k,v in lf.items() if k!='signature'},sort_keys=True).encode()).hexdigest() == lf['signature']
    assert lf['parent_signature'] == json.loads((normal/'protocol.json').read_text())['signature'] and lf['variants'] == ['logs_only']
    assert lf['inner_fits_per_ablation_fold'] == 0 and lf['outer_fits_per_ablation_fold'] == 1 and lf['outer_folds'] == 10
    assert lf['base_fixed_signature'] == fp['signature'] and lf['base_runner_sha256'] == fp['runner_sha256']
    fixedset = fp['variants'] + lf['variants']
    cfg = yaml.safe_load((ROOT/'source/geomerit/configs/revision_20261103.yaml').read_text())
    assert cfg == proto['config']
    assert cfg['outer_folds'] == 10 and cfg['inner_folds'] == 3 and cfg['threads'] == 14
    assert cfg['models']['xgboost']['params']['device'] == 'cpu'
    assert cfg['models']['catboost']['params']['task_type'] == 'CPU'
    assert proto['smoke_only'] is False
    assert hashlib.sha256(json.dumps({k:proto[k] for k in ['config','code','smoke_only']},sort_keys=True).encode()).hexdigest() == proto['signature']
    assert hashlib.sha256(json.dumps({k:v for k,v in fp.items() if k!='signature'},sort_keys=True).encode()).hexdigest() == fp['signature']
    assert fp['parent_signature'] == proto['signature'] and fp['outer_folds'] == 10
    assert fp['inner_fits_per_ablation_fold'] == 0 and fp['outer_fits_per_ablation_fold'] == 1
    assert fp['full_architecture_retained'] and fp['variants'] == variants[2:7]
    for rel, value in proto['code'].items(): check(ROOT/'source/geomerit'/rel, value)
    check(ROOT/'fixed_ablation_train.py', fp['runner_sha256'])
    check(ROOT/'fixed_logs_only_train.py', lf['runner_sha256'])
    for package, version in proto['packages'].items():
        assert importlib.metadata.version(package) == version, package
    data_dir = ROOT/'experiment_v1/dataset'
    manifest = json.loads((data_dir/'dataset.json').read_text())
    assert manifest['status'] == 'complete'
    check(data_dir/'features.parquet', manifest['parquet_sha256'])
    for item in manifest['inputs']: check(Path(item['path']), item['sha256'])
    frame = pd.read_parquet(data_dir/'features.parquet', columns=['sample_id','well_id','label_idx','outer_fold'])
    assert len(frame) == manifest['rows'] == 1431383 and frame.well_id.nunique() == manifest['wells'] == 118
    assert not frame.sample_id.duplicated().any()
    assert frame.groupby('well_id').outer_fold.nunique().eq(1).all()
    outer_map = json.loads((data_dir/'outer_wells.json').read_text())
    assert frame.groupby('well_id').outer_fold.first().to_dict() == outer_map
    completed = {v:[] for v in variants}
    blocks, incomplete, predictions = [], [], []
    for variant in variants:
        run = logs if variant in lf['variants'] else fixed if variant in fp['variants'] else normal
        signature = lf['signature'] if variant in lf['variants'] else fp['signature'] if variant in fp['variants'] else proto['signature']
        for fold in range(10):
            dest = run/variant/f'outer_{fold:02d}'
            marker = dest/'fold_complete.json'
            if marker.exists():
                item = json.loads(marker.read_text())
                assert item['signature'] == signature and item['smoke_only'] is False
                check(dest/'predictions.parquet', item['prediction_sha256'])
                pred = pd.read_parquet(dest/'predictions.parquet')
                ref = frame.loc[frame.outer_fold.eq(fold)].reset_index(drop=True)
                assert len(pred) == item['rows'] == len(ref)
                assert not pred.sample_id.duplicated().any()
                for col in ref.columns: assert np.array_equal(pred[col].to_numpy(),ref[col].to_numpy()), (variant,fold,col)
                for col in pred.columns:
                    if col.startswith('pred_'): assert pred[col].isin(range(12)).all(), col
                selection = json.loads((dest/'selected_parameters.json').read_text())
                assert selection['signature'] == signature and selection['selected_before_outer_fit'] is True
                assert selection['evaluation_labels_used'] is False
                if variant in fixedset:
                    assert item['ablation_protocol'] == 'fixed_parameters_outer_refit'
                    assert item['source_full_fold'] == fold and item['parent_signature'] == proto['signature']
                    check(normal/'full'/f'outer_{fold:02d}'/'selected_parameters.json',item['source_selection_sha256'])
                    assert selection['independent_inner_reselection'] is False
                    assert not list(dest.glob('inner_*/complete.json'))
                completed[variant].append(fold)
                predictions.append({'variant':variant,'fold':fold,'rows':len(pred),'sha256':item['prediction_sha256']})
            if dest.exists():
                cps = sorted(dest.glob('*/complete.json'))
                if marker.exists():
                    required = {'outer_fit'} if variant in fixedset else {'inner_00','inner_01','inner_02','outer_fit'}
                    assert {p.parent.name for p in cps} == required, (variant,fold,'incomplete training blocks')
                for cp in cps:
                    item = json.loads(cp.read_text())
                    assert item['token']['signature'] == signature
                    for name, value in item['files'].items():
                        path = (cp.parent/name).resolve()
                        assert path.is_relative_to(cp.parent.resolve())
                        check(path,value)
                    split = json.loads((cp.parent/'split.json').read_text())
                    assert all(split[k] == v for k,v in item['token'].items())
                    assert not set(split['training_wells']) & set(split['evaluation_wells'])
                    blocks.append(str(cp.relative_to(ROOT)))
                if not marker.exists():
                    incomplete.append({'variant':variant,'fold':fold,'verified_blocks':[p.parent.name for p in cps],
                        'incomplete_blocks':[p.name for p in sorted(dest.iterdir()) if p.is_dir() and (p.name.startswith('inner_') or p.name=='outer_fit') and not (p/'complete.json').exists()]})
    old = json.loads((BASE/'user_pause_20261104T232649Z/verification.json').read_text())
    assert all(set(fs) <= set(completed[v]) for v,fs in old['per_variant_completed'].items())
    assert set(old['verified_training_blocks']) <= set(blocks)
    for p in [BASE/'fixed_scope.json',BASE/'logs_only_fixed_scope.json',ROOT/'parallel_controller_fast.py',logs/'protocol.json',ROOT/'parallel_controller_fixed.py',BASE/'fixed_main_report.py',normal/'protocol.json',fixed/'protocol.json',data_dir/'outer_wells.json',data_dir/'dataset.json']:
        sha(p)
    if require_stopped: assert not processes()
    result = {'verified_at_utc':datetime.now(timezone.utc).isoformat(),'state':'pre_resume_verified' if require_stopped else 'artifacts_verified',
        'scope_id':scope['id'],'expected_folds':80,'completed_folds':sum(map(len,completed.values())),
        'per_variant_completed':completed,'verified_training_block_count':len(blocks),'verified_training_blocks':blocks,
        'incomplete_folds':incomplete,'prediction_records':predictions,'completed_prediction_sha256_verified':True,
        'frozen_source_verified':True,'dataset_and_input_sha256_verified':True,'packages_match_protocol':True,
        'protocol_signatures':{'nested':proto['signature'],'fixed':fp['signature'],'fixed_logs_only':lf['signature']},'sha256_bytes_checked':checked_bytes,
        'sha256_inventory':inventory,'processes':live,'cpus':sorted(os.sched_getaffinity(0)),
        'memory_available_gib':psutil.virtual_memory().available/1024**3,'disk_free_gib':psutil.disk_usage(ROOT).free/1024**3}
    if lock: lock.close()
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-stopped',action='store_true');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=verify(args.require_stopped)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['sha256_inventory','verified_training_blocks','prediction_records']},indent=2))
