from pathlib import Path
import json,hashlib,datetime,argparse
parser=argparse.ArgumentParser(description='Recompute S7 evidence from the ten full-run outer-fold predictions.')
parser.add_argument('--root',type=Path,required=True,help='Experiment root containing experiment_v1 and source')
parser.add_argument('--report-dir',type=Path,required=True,help='Final 80-fold report directory')
parser.add_argument('--out',type=Path,required=True,help='New output directory for the evidence JSON and summary')
args=parser.parse_args()
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

root=args.root
rp=args.report_dir
main=json.loads((rp/'main_report.json').read_text('utf-8'))
final=json.loads((rp/'final_report.json').read_text('utf-8'))
art=json.loads((rp/'artifact_verification.json').read_text('utf-8'))
names=[r['name'] for r in main['per_class']]
out=args.out;out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inventory=[];parts=[]
for fold in range(10):
 p=root/f'experiment_v1/runs/full/outer_{fold:02d}/predictions.parquet'
 expected=next(d['sha256'] for d in final['prediction_provenance'] if d['variant']=='full' and d['fold']==fold)
 actual=sha(p);assert actual==expected
 frame=pq.read_table(p,columns=['sample_id','well_id','outer_fold','label_idx','boundary','pred_full','pred_xgboost_argmax']).to_pandas()
 assert frame.outer_fold.eq(fold).all()
 inventory.append({'fold':fold,'rows':len(frame),'sha256':actual,'matches_final80':True})
 parts.append(frame)
frame=pd.concat(parts,ignore_index=True)
assert len(frame)==1431383 and frame.sample_id.is_unique and frame.well_id.nunique()==118
y=frame.label_idx.to_numpy(int);pred=frame.pred_full.to_numpy(int);xgb=frame.pred_xgboost_argmax.to_numpy(int)
cm=lambda p:np.bincount(y*12+p,minlength=144).reshape(12,12)
fullcm=cm(pred);xgbcm=cm(xgb)

costp=root/'source/lithology scoring matrix cost function.xlsx'
costsha=sha(costp);assert costsha==art['sha256_inventory']['source/lithology scoring matrix cost function.xlsx']
raw=pd.read_excel(costp,header=None)
excel_names=['Sandstone','Sandstone/Shale','Shale','Marl','Dolomite','Limestone','Chalk','Halite','Anhydrite','Tuff','Coal','Crystalline Basement']
lookup={}
for i,row in raw.iterrows():
 for value in row:
  if str(value).strip() in excel_names:lookup[str(value).strip()]=i;break
excel=np.stack([raw.iloc[lookup[n],3:15].to_numpy(float) for n in excel_names])
order=[excel_names.index('Crystalline Basement' if n=='Basement' else n) for n in names]
cost=excel[np.ix_(order,order)]

def metrics(confusion):
 support=confusion.sum(axis=1);proposed=confusion.sum(axis=0);n=support.sum()
 f1=np.divide(2*np.diag(confusion),support+proposed,out=np.zeros(12),where=support+proposed>0)
 return {'weighted_f1':float(f1@support/n),'macro_f1':float(f1.mean()),'tail_f1':float(f1[[5,6,8,9,10,11]].mean()),'accuracy':float(np.trace(confusion)/n),'penalty':float(-(confusion*cost).sum()/n)}
scores={'full':metrics(fullcm),'xgboost_argmax':metrics(xgbcm)}
for key,values in scores.items():
 expected=main['summaries']['full/'+key]
 for metric,value in values.items():assert abs(value-expected[metric])<1e-14,(key,metric,value,expected[metric])

dolo=names.index('Dolomite');dolosub=frame[frame.label_idx.eq(dolo)];dolo_n=int(fullcm[dolo].sum());dolo_errors=dolo_n-int(fullcm[dolo,dolo])
dolo_distribution=[{'predicted_class':n,'count':int(fullcm[dolo,i]),'percent_of_true_dolomite':float(fullcm[dolo,i]/dolo_n*100),'percent_of_dolomite_errors':None if i==dolo else float(fullcm[dolo,i]/dolo_errors*100)} for i,n in enumerate(names)]
dolo_rank=sorted([r for r in dolo_distribution if r['predicted_class']!='Dolomite'],key=lambda r:r['count'],reverse=True)
shale=0;n=len(frame)
contrib=[]
for i,name in enumerate(names):
 support=int(fullcm[i].sum());fc=int(fullcm[i,i]);xc=int(xgbcm[i,i]);pc_full=float(-(fullcm[i]*cost[i]).sum()/n);pc_xgb=float(-(xgbcm[i]*cost[i]).sum()/n)
 contrib.append({'true_class':name,'support':support,'full_correct':fc,'xgb_correct':xc,'correct_difference_xgb_minus_full':xc-fc,'accuracy_difference_contribution':(xc-fc)/n,'full_predicted_shale':int(fullcm[i,0]),'xgb_predicted_shale':int(xgbcm[i,0]),'shale_prediction_difference':int(xgbcm[i,0]-fullcm[i,0]),'penalty_difference_contribution_xgb_minus_full':pc_xgb-pc_full})
shale_switch=(pred!=shale)&(xgb==shale);away_switch=(pred==shale)&(xgb!=shale)
def group_diff(mask):return {'samples':int(mask.sum()),'correct_difference_xgb_minus_full':int((xgb[mask]==y[mask]).sum()-(pred[mask]==y[mask]).sum()),'accuracy_difference_contribution':float(((xgb[mask]==y[mask]).sum()-(pred[mask]==y[mask]).sum())/n),'penalty_difference_contribution_xgb_minus_full':float((cost[y[mask],pred[mask]]-cost[y[mask],xgb[mask]]).sum()/n)}
groups={'full_non_shale_to_xgb_shale':group_diff(shale_switch),'full_shale_to_xgb_non_shale':group_diff(away_switch),'both_non_shale':group_diff((pred!=shale)&(xgb!=shale)),'both_shale':group_diff((pred==shale)&(xgb==shale))}
result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Fresh read of 10 full-run outer-fold predictions; both full and single-backend XGBoost conditional argmax are saved on identical samples. No retraining.','prediction_sha256_match_final80':inventory,'rows':n,'wells':int(frame.well_id.nunique()),'score_recomputation_matches_main_report':True,'cost_matrix_sha256_matches_artifact_inventory':costsha,'labels':names,'dolomite':{'samples':dolo_n,'wells':int(dolosub.well_id.nunique()),'correct':int(fullcm[dolo,dolo]),'errors':dolo_errors,'prediction_distribution':dolo_distribution,'ranked_error_destinations':dolo_rank,'Limestone_Marl_Shale_combined_errors':sum(r['count'] for r in dolo_rank if r['predicted_class'] in ['Limestone','Marl','Shale']),'interpretation':'Observed confusion destinations describe classification failures; they do not establish physical log overlap, a causal mechanism, or a reason that any post-hoc remedy will succeed.'},'xgboost_vs_full':{'metrics':scores,'full_shale_prediction_count':int(fullcm[:,0].sum()),'xgb_shale_prediction_count':int(xgbcm[:,0].sum()),'full_shale_prediction_fraction':float(fullcm[:,0].sum()/n),'xgb_shale_prediction_fraction':float(xgbcm[:,0].sum()/n),'true_shale_samples':int(fullcm[0].sum()),'overall_correct_difference_xgb_minus_full':int(np.trace(xgbcm)-np.trace(fullcm)),'by_true_class':contrib,'prediction_switch_partition':groups,'full_confusion_matrix':fullcm.tolist(),'xgb_confusion_matrix':xgbcm.tolist(),'causal_boundary':'The paired output comparison supports a descriptive allocation of correct predictions and cost differences, not a causal claim that a propensity for Shale alone causes higher Accuracy or Penalty. Full and XGBoost differ in architecture and decision pipeline, and other-class effects also contribute.'}}
(out/'error_mechanism_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
top=dolo_rank[:5];true_shale=contrib[0];rem_correct=result['xgboost_vs_full']['overall_correct_difference_xgb_minus_full']-true_shale['correct_difference_xgb_minus_full']
lines=['# 最新80折回复：误分类与页岩预测核查','',
'本次直接读取完整模型十折预测；十份SHA均与最终80折报告相同。共1,431,383行、118口井，full与单后端XGBoost argmax在相同样本上比较。5项指标及惩罚矩阵均重新核对吻合。没有训练或bootstrap重算。','',
'## 白云岩误分类','',
f'真实白云岩 {dolo_n} 个样本、{dolosub.well_id.nunique()} 口井，正确 {fullcm[dolo,dolo]} 个，错误 {dolo_errors} 个。错误去向依次为：','']
for row in dolo_rank:
 if row['count']:lines.append(f"- {row['predicted_class']}: {row['count']}，占全部真实白云岩 {row['percent_of_true_dolomite']:.2f}%，占白云岩错误 {row['percent_of_dolomite_errors']:.2f}%。")
lines+=['',f"Limestone/Marl/Shale 合计 {result['dolomite']['Limestone_Marl_Shale_combined_errors']} 个。可据实际排序描述主要错误去向；不能将此直接解释为物理测井响应重叠的定因。",'',
'## XGBoost与完整模型','',
f"XGBoost预测为Shale的样本 {xgbcm[:,0].sum():,}（{xgbcm[:,0].sum()/n:.2%}），完整模型 {fullcm[:,0].sum():,}（{fullcm[:,0].sum()/n:.2%}）。真实Shale共 {fullcm[0].sum():,}。",'',
f"XGBoost的正确预测总数比完整模型多 {np.trace(xgbcm)-np.trace(fullcm):,}；其中真实Shale的正确数多 {true_shale['correct_difference_xgb_minus_full']:,}，其他类别净变化 {rem_correct:+,}。",'',
f"Accuracy: XGBoost {scores['xgboost_argmax']['accuracy']:.8f}，full {scores['full']['accuracy']:.8f}；Penalty: XGBoost {scores['xgboost_argmax']['penalty']:.8f}，full {scores['full']['penalty']:.8f}。真实Shale对总Penalty差值贡献 {true_shale['penalty_difference_contribution_xgb_minus_full']:+.8f}，其他类别合计贡献 {sum(c['penalty_difference_contribution_xgb_minus_full'] for c in contrib[1:]):+.8f}，净差 {scores['xgboost_argmax']['penalty']-scores['full']['penalty']:+.8f}。",'',
'最强可支持表述：XGBoost的预测分布更集中于Shale，其较高Accuracy和Penalty主要由Shale类收益驱动，同时伴随其他类别的净损失。这里“驱动”指实际计数与成本的分解，不能扩展为已经识别出模型的唯一因果机制。完整模型与单后端条件输出有多处差异，不能仅凭更多Shale预测推出更高Penalty；本次Penalty结论另有逐类真实成本分解支持。','']
(out/'error_mechanism_check.md').write_text('\n'.join(lines),'utf-8')
print(json.dumps({'dolomite_top_errors':top,'dolomite_limestone_marl_shale_total':result['dolomite']['Limestone_Marl_Shale_combined_errors'],'metrics':scores,'shale_prediction_counts':[int(fullcm[:,0].sum()),int(xgbcm[:,0].sum())],'true_shale_delta':true_shale,'other_class_correct_delta':rem_correct,'switches':groups,'outputs':[str(out/'error_mechanism_check.json'),str(out/'error_mechanism_check.md')]},ensure_ascii=False,indent=2))
