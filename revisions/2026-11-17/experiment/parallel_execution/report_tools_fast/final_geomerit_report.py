"""Postprocessing only: require all 80 real folds; keep protocol result roots separate."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.util, json, os, shutil, sys

os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path(os.environ.get('GEOMERIT_REVISION_ROOT', str(Path(__file__).resolve().parents[2])))
BASE=ROOT/'parallel_execution'
sys.path.insert(0,str(ROOT/'source/geomerit'))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits
from geomerit.revision_decision import confusion,scores_from_cm
from verify_geomerit import verify

spec=importlib.util.spec_from_file_location('existing_main_report',BASE/'fixed_main_report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    scope=json.loads((BASE/'fixed_scope.json').read_text())
    normal=ROOT/'experiment_v1/runs';fixed=ROOT/'experiment_fixed_parameters_20261104/runs'
    logs=ROOT/'experiment_fixed_parameters_logs_only_20261110/runs';fixed_all=scope['fixed_parameter_variants']+['logs_only']
    roots={v:logs if v=='logs_only' else fixed if v in scope['fixed_parameter_variants'] else normal for v in scope['variants']}
    missing=[f'{v}/outer_{f:02d}' for v,run in roots.items() for f in range(10) if not (run/v/f'outer_{f:02d}'/'fold_complete.json').is_file()]
    if missing:
        print(json.dumps({'status':'waiting_for_all_80_folds','completed_folds':80-len(missing),'missing':missing},indent=2));return 2
    main_report=json.loads((normal/'report/main_report.json').read_text())
    assert main_report['status']=='complete_main_comparisons' and main_report['scope_id']==scope['id']
    verification=verify()
    assert verification['completed_folds']==80 and verification['verified_training_block_count']==140
    proto=json.loads((normal/'protocol.json').read_text());cfg=proto['config']
    assert main_report['protocol_signature']==proto['signature']
    cost=np.load(normal/'penalty_matrix.npy')
    reference=report.load_predictions(normal,'full')
    y=reference.label_idx.to_numpy(int);boundary=reference.boundary.to_numpy(bool)
    fold_id=reference.outer_fold.to_numpy(int);well_id=reference.well_id.to_numpy()
    wells=sorted(reference.well_id.unique());indices=[np.flatnonzero(well_id==w) for w in wells]
    rng=np.random.default_rng(cfg['inference']['bootstrap_seed'])
    counts=rng.multinomial(len(wells),np.full(len(wells),1/len(wells)),size=cfg['inference']['bootstrap_replicates'])
    summaries={};fold_scores={};boot={};provenance=[]
    for variant,run in roots.items():
        df=reference if variant=='full' else report.load_predictions(run,variant)
        for col in ['sample_id','well_id','label_idx','outer_fold','boundary']:
            assert np.array_equal(reference[col].to_numpy(),df[col].to_numpy()), (variant,col)
        if variant in fixed_all:
            for f in range(10):
                dest=run/variant/f'outer_{f:02d}'
                selection=json.loads((dest/'selected_parameters.json').read_text())
                source=json.loads((normal/'full'/f'outer_{f:02d}'/'selected_parameters.json').read_text())
                assert selection['candidates']==0 and selection['independent_inner_reselection'] is False
                assert selection['blend']==source['blend'] and selection['decision']==source['decision']
        columns=['pred_full']+(['pred_posterior_argmax'] if variant=='rfe_xgb' else [])
        for col in columns:
            key=f'{variant}/{col[5:]}';pred=df[col].to_numpy(int)
            cm=np.stack([confusion(y[i],pred[i]) for i in indices])
            bc=np.stack([confusion(y[i][boundary[i]],pred[i][boundary[i]]) for i in indices])
            summary=scores_from_cm(cm.sum(axis=0),cost)
            summary['boundary_f1']=scores_from_cm(bc.sum(axis=0),cost)['weighted_f1']
            fs=[]
            for f in range(10):
                keep=fold_id==f;score=scores_from_cm(confusion(y[keep],pred[keep]),cost)
                score['boundary_f1']=scores_from_cm(confusion(y[keep & boundary],pred[keep & boundary]),cost)['weighted_f1']
                fs.append(score)
            fold_scores[key]=pd.DataFrame(fs)
            summary['fold_mean']=fold_scores[key].mean().to_dict();summary['fold_sd']=fold_scores[key].std(ddof=1).to_dict()
            summaries[key]=summary
            with threadpool_limits(limits=2,user_api='blas'):
                boot[key]=report.vector_scores(counts@cm.reshape(len(wells),-1),cost)
                boot[key]['boundary_f1']=report.vector_scores(counts@bc.reshape(len(wells),-1),cost)['weighted_f1']
            if key in main_report['summaries']:
                for metric in ['weighted_f1','macro_f1','penalty','boundary_f1']:
                    assert np.isclose(summary[metric],main_report['summaries'][key][metric],rtol=0,atol=1e-12),(key,metric)
        for f in range(10):
            p=run/variant/f'outer_{f:02d}'/'predictions.parquet'
            provenance.append({'variant':variant,'fold':f,'path':str(p),'sha256':verification['sha256_inventory'][str(p.relative_to(ROOT))]})
        if variant!='full': del df
    comparisons=[];controls=scope['fixed_parameter_variants']+['logs_only'];family_size=len(controls)*3
    for control in controls:
        key=control+'/full'
        for metric in ['weighted_f1','boundary_f1','penalty']:
            delta=boot['full/full'][metric]-boot[key][metric]
            fd=(fold_scores['full/full'][metric]-fold_scores[key][metric]).to_numpy()
            p=1.0 if np.all(fd==0) else float(wilcoxon(fd,alternative='two-sided',zero_method='wilcox',method='auto').pvalue)
            comparisons.append({'control':control,'metric':metric,'delta_full_minus_control':summaries['full/full'][metric]-summaries[key][metric],
                'paired_well_ci95':np.quantile(delta,[.025,.975]).tolist(),
                'familywise_bonferroni_ci':np.quantile(delta,[.05/(2*family_size),1-.05/(2*family_size)]).tolist(),
                'secondary_fold_p':p})
    for item,p in zip(comparisons,report.holm([r['secondary_fold_p'] for r in comparisons])):item['secondary_fold_p_holm']=p
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result={'status':'complete_80_folds_and_reports_verified','verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope_id':scope['id'],'completed_folds':80,'verified_training_blocks':140,'logs_only_protocol':'fixed_parameters_outer_refit (changed 2026-11-10, see parallel_execution/logs_only_fixed_scope.json)','protocol_signatures':verification['protocol_signatures'],
        'result_roots':{v:str(r) for v,r in roots.items()},'rows_per_variant':len(reference),'wells':len(wells),
        'summaries':summaries,'ablation_comparisons':comparisons,'prediction_provenance':provenance,
        'main_report_path':str(normal/'report/main_report.json'),
        'uncertainty_scope':main_report['uncertainty_scope'],
        'multiplicity':'Main comparisons retain their original family of 6; ablation family has 18 outcomes. Paired well bootstrap uses the frozen seed and 5000 replicates.',
        'interpretation':'Five no_* variants and logs_only inherit same-fold full inner selections (blend and decision parameters) and refit the full outer architecture once; logs_only was not independently tuned. Penalty is the existing negative cost score; higher is better.'}
    rows=[]
    for key,s in summaries.items():
        rows.append({'model':key,**{m:s[m] for m in ['weighted_f1','macro_f1','boundary_f1','penalty']}})
    pd.DataFrame(rows).to_csv(out/'summary_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'ablation_comparisons.csv',index=False)
    pd.concat([df.assign(model=k,fold=np.arange(10)) for k,df in fold_scores.items()],ignore_index=True).to_csv(out/'fold_metrics.csv',index=False)
    write(out/'artifact_verification.json',verification)
    write(out/'final_report.json',result)
    shutil.copy2(normal/'report/main_report.json',out/'main_report.json')
    lines=['# GeoMERIT 实验结果核验','',f'80/80 折、140 个训练块均通过核验；每个配置覆盖 {len(reference):,} 个样本、{len(wells)} 口井。','',
        '| 配置/输出 | Weighted F1 | Macro F1 | Boundary F1 | 负代价评分 |','|---|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+row['model']+' | '+' | '.join(f'{row[m]:.6f}' for m in ['weighted_f1','macro_f1','boundary_f1','penalty'])+' |')
    lines += ['','五项 no_* 消融与 logs_only 均使用同外层折 full 的内层选参（融合与决策参数），仅重训外层；logs_only 未做独立内层调参（2026-11-10 为赶审稿期限调整）。三套结果目录相互独立。','',
        '配对井 bootstrap 区间以已拟合模型为条件，不包含重训变异。外层训练集相互重叠，折级检验仅作为次要描述。主要比较与消融比较分别控制各自检验族的多重比较。','',
        '所有数值由已核验的外层预测计算；逐折指标、消融区间及 SHA256 来源见同目录 CSV/JSON。']
    (out/'实验结果汇总.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'output':str(out)},ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
