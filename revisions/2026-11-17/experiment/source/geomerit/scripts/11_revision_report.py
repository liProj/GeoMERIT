#!/usr/bin/env python3
"""Paired, well-clustered reporting; refuses incomplete ten-fold comparisons."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits
from geomerit.revision_decision import confusion, scores_from_cm
from geomerit.labels import LABEL_NAMES


def load_predictions(run, name):
    paths = []
    for fold in range(10):
        directory = run/name/f'outer_{fold:02d}'
        if not (directory/'fold_complete.json').exists():
            raise RuntimeError(f'Incomplete comparison: {name} fold {fold}')
        if json.loads((directory/'fold_complete.json').read_text()).get('smoke_only'):
            raise RuntimeError('Smoke results cannot enter publication reporting')
        paths.append(directory/'predictions.parquet')
    df = pd.concat([pd.read_parquet(p) for p in paths]).sort_values('sample_id').reset_index(drop=True)
    assert len(df) == 1431383 and df.well_id.nunique() == 118
    assert not df.sample_id.duplicated().any()
    return df


def vector_scores(cm, cost):
    cm = np.asarray(cm).reshape(-1,12,12)
    sup = cm.sum(axis=2); pred = cm.sum(axis=1)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    f1 = np.divide(2*tp, sup+pred, out=np.zeros_like(tp, dtype=float), where=sup+pred>0)
    n = np.maximum(cm.sum(axis=(1,2)), 1)
    return {'weighted_f1': (f1*sup).sum(axis=1)/n, 'penalty': -(cm*cost).sum(axis=(1,2))/n}


def holm(values):
    values = np.asarray(values); order = np.argsort(values); m = len(values)
    adjusted = np.maximum.accumulate(values[order]*(m-np.arange(m)))
    result = np.empty(m); result[order] = np.minimum(adjusted, 1)
    return result.tolist()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--work', required=True)
    args = parser.parse_args(); run = Path(args.work)/'runs'
    protocol = json.loads((run/'protocol.json').read_text())
    cfg = protocol['config']; cost = np.load(run/'penalty_matrix.npy')
    out = run/'report'; out.mkdir(exist_ok=True)
    datasets = {name: load_predictions(run, name) for name in ['full','rfe_xgb','raw_xgb']}
    first = datasets['full']; y = first.label_idx.to_numpy(int); boundary = first.boundary.to_numpy(bool)
    for df in datasets.values():
        for col in ['sample_id','well_id','label_idx','outer_fold','boundary']:
            assert np.array_equal(first[col].to_numpy(), df[col].to_numpy()), col
    wells = sorted(first.well_id.unique()); idx = [np.flatnonzero(first.well_id.to_numpy() == w) for w in wells]
    summaries, matrices, folds = {}, {}, {}
    for name, df in datasets.items():
        for column in [c for c in df if c.startswith('pred_')]:
            key = name+'/'+column[5:]; pred = df[column].to_numpy(int)
            cm = np.stack([confusion(y[i], pred[i]) for i in idx])
            bc = np.stack([confusion(y[i][boundary[i]], pred[i][boundary[i]]) for i in idx])
            matrices[key] = (cm, bc)
            summaries[key] = scores_from_cm(cm.sum(axis=0), cost)
            summaries[key]['boundary_f1'] = scores_from_cm(bc.sum(axis=0), cost)['weighted_f1']
            fm = []
            for f in range(10):
                keep = first.outer_fold.to_numpy() == f
                scores = scores_from_cm(confusion(y[keep], pred[keep]), cost)
                scores['boundary_f1'] = scores_from_cm(confusion(y[keep & boundary], pred[keep & boundary]), cost)['weighted_f1']
                fm.append(scores)
            folds[key] = pd.DataFrame(fm)
            summaries[key]['fold_mean'] = folds[key].mean().to_dict()
            summaries[key]['fold_sd'] = folds[key].std(ddof=1).to_dict()
    rng = np.random.default_rng(cfg['inference']['bootstrap_seed'])
    b = int(cfg['inference']['bootstrap_replicates'])
    counts = rng.multinomial(len(wells), np.full(len(wells), 1/len(wells)), size=b)
    boot = {}
    main_key = 'full/full'; controls = ['rfe_xgb/posterior_argmax','rfe_xgb/full']
    with threadpool_limits(limits=2, user_api='blas'):
        for name in [main_key]+controls:
            cm, bc = matrices[name]
            boot[name] = vector_scores(counts@cm.reshape(len(wells),-1), cost)
            boot[name]['boundary_f1'] = vector_scores(counts@bc.reshape(len(wells),-1), cost)['weighted_f1']
    family_size = 6; comparisons = []
    for control in controls:
        for metric in ['weighted_f1','boundary_f1','penalty']:
            delta = boot[main_key][metric]-boot[control][metric]
            fd = (folds[main_key][metric]-folds[control][metric]).to_numpy()
            pvalue = 1. if np.all(fd == 0) else float(wilcoxon(fd, alternative='two-sided', zero_method='wilcox', method='auto').pvalue)
            comparisons.append(dict(control=control, metric=metric,
                  delta=summaries[main_key][metric]-summaries[control][metric],
                  paired_well_ci95=np.quantile(delta,[.025,.975]).tolist(),
                  familywise_bonferroni_ci=np.quantile(delta,[.05/(2*family_size),1-.05/(2*family_size)]).tolist(),
                  secondary_fold_p=pvalue))
    for item, p in zip(comparisons, holm([r['secondary_fold_p'] for r in comparisons])):
        item['secondary_fold_p_holm'] = p
    # Natural missingness: disjoint bins plus cumulative thresholds; report composition.
    strata = []
    for lo, hi in [(0,.4),(.4,.5),(.5,.6),(.6,1.000001),(.4,1.000001),(.5,1.000001)]:
        keep = (first.well_missing_rate >= lo) & (first.well_missing_rate < hi)
        sub = first.loc[keep]; n = len(sub)
        row = dict(low=lo, high=hi, wells=int(sub.well_id.nunique()), rows=n,
                   class_counts=np.bincount(sub.label_idx, minlength=12).tolist(),
                   uncertainty='descriptive; one-well strata do not support between-well intervals')
        row['scores'] = {key: scores_from_cm(confusion(y[keep], datasets[key.split('/')[0]]['pred_'+key.split('/')[1]].to_numpy()[keep]), cost)
                         for key in [main_key]+controls} if n else {}
        strata.append(row)
    per_class = []
    cm = matrices[main_key][0].sum(axis=0)
    for c, name in enumerate(LABEL_NAMES):
        support = int(cm[c].sum()); predicted = int(cm[:,c].sum())
        per_class.append(dict(class_index=c, name=name, rows=support, wells=int(first.loc[first.label_idx == c,'well_id'].nunique()),
                              precision=float(cm[c,c]/max(predicted,1)), recall=float(cm[c,c]/max(support,1)),
                              f1=float(2*cm[c,c]/max(support+predicted,1))))
    result = dict(status='complete_main_comparisons', protocol_signature=protocol['signature'],
                  summaries=summaries, paired_comparisons=comparisons, per_class=per_class, missingness=strata,
                  uncertainty_scope='Resampling observed wells conditional on fitted models; does not include retraining variability. Fold tests are secondary because training sets overlap.',
                  pending=['spatial region holdout','raw-input controlled masking','remaining retrained ablations as registered'])
    (out/'main_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame(per_class).to_csv(out/'per_class.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    print('REPORT COMPLETE', out, flush=True)


if __name__ == '__main__': main()
