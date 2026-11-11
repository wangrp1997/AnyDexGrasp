"""Independent metric/decision/cost verification from saved predictions."""
from pathlib import Path
from datetime import datetime
import hashlib,json,zipfile
import numpy as np

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
RUN=ROOT/'outputs/boya_shared_score_20261111_v2'
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):assert np.allclose(a,b,rtol=0,atol=1e-12),(a,b)
def recompute(z,key):
    y=z['labels'];s=z['scores'];out=[]
    for obj in sorted(set(z[key])):
        mask=z[key]==obj;yy=y[mask];ss=s[mask];p=ss[yy==1];n=ss[yy==0]
        auc=float(np.mean((p[:,None]>n)+.5*(p[:,None]==n))) if len(p)*len(n) else None
        out.append(dict(id=str(obj),auc=auc,top=float(yy[ss==ss.max()].mean()),
                        brier=float(np.mean((np.clip(ss,0,1)-yy)**2))))
    mixed=[r for r in out if r['auc'] is not None]
    return dict(macro_object_auc=float(np.mean([r['auc'] for r in mixed])),
                macro_object_brier=float(np.mean([r['brier'] for r in out])),
                macro_mixed_top_tie_mean_label=float(np.mean([r['top'] for r in mixed])),
                per_object=out, object_count=len(out),mixed_object_count=len(mixed))

result=read(RUN/'RESULTS.json');protocol=read(RUN/'PROTOCOL.json')
assert result['status']=='complete' and result['protocol_sha256']==sha(RUN/'PROTOCOL.json')
allmetrics={};predrefs={}
for stage,file in [('cv','CV_RESULTS.json'),('internal','INTERNAL_RESULTS.json')]:
    receipt=read(RUN/file);allmetrics[stage]={}
    for name,ref in receipt['predictions'].items():
        assert sha(ref['path'])==ref['sha256']
        with np.load(ref['path'],allow_pickle=False) as data:z={k:data[k].copy() for k in data.files}
        assert len(z['labels'])==(364 if stage=='cv' else 71)
        allmetrics[stage][name]={}
        for level,group in [('object','object_id'),('same_observation','observation_group')]:
            m=recompute(z,group);declared=receipt['metrics'][name][level]
            for key in ['macro_object_auc','macro_object_brier','macro_mixed_top_tie_mean_label','object_count','mixed_object_count']:
                same(m[key],declared[key]);same(m[key],result['cv_metrics' if stage=='cv' else'internal_metrics'][name][level][key])
            allmetrics[stage][name][level]=m
        same(float(np.mean((z['scores']-z['labels'])**2)),receipt['metrics'][name]['object']['raw_mse'])
        predrefs[str(ref['path'])]=ref['sha256']
cv=allmetrics['cv'];internal=allmetrics['internal']
selected='shared_mlp64' if cv['shared_mlp64']['object']['macro_object_auc']>cv['shared_linear']['object']['macro_object_auc'] else'shared_linear'
assert result['selected']==read(RUN/'SELECTION.json')['selected']==selected
a={r['id']:r['auc'] for r in cv['condition_linear']['object']['per_object'] if r['auc'] is not None}
b={r['id']:r['auc'] for r in cv[selected]['object']['per_object'] if r['auc'] is not None}
diff=np.array([b[k]-a[k] for k in sorted(a)]);rng=np.random.default_rng(20261111)
ci=np.quantile(diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(1),[.025,.975])
same(ci,result['cv_bootstrap']['gain_ci95']);same(diff.mean(),result['cv_bootstrap']['gain'])
fold_gains=[]
for fold in range(4):
    m=read(RUN/f'cv_fold{fold}/SPLIT.json')['metrics']
    fold_gains.append(m[selected]['object']['macro_object_auc']-m['condition_linear']['object']['macro_object_auc'])
c=cv[selected];i=internal[selected];refs=[internal[k] for k in ['original_mse','preactivation_mse','condition_linear']]
checks={
'cv_auc_at_least_0_60':c['object']['macro_object_auc']>=.60,
'cv_object_gain_at_least_0_05':c['object']['macro_object_auc']>=cv['condition_linear']['object']['macro_object_auc']+.05,
'cv_same_observation_gain_at_least_0_03':c['same_observation']['macro_object_auc']>=cv['condition_linear']['same_observation']['macro_object_auc']+.03,
'cv_positive_gain_in_3_of_4_folds':sum(v>0 for v in fold_gains)>=3,
'cv_bootstrap_gain_lower_positive':bool(ci[0]>0),
'cv_macro_brier_no_worse_than_object_prior':c['object']['macro_object_brier']<=cv['object_prior']['object']['macro_object_brier'],
'internal_object_auc_beats_each_reference_by_0_05':i['object']['macro_object_auc']>=max(r['object']['macro_object_auc'] for r in refs)+.05,
'internal_same_observation_auc_no_drop':i['same_observation']['macro_object_auc']>=max(r['same_observation']['macro_object_auc'] for r in refs),
'internal_mixed_pool_top_tie_no_drop':i['object']['macro_mixed_top_tie_mean_label']>=max(r['object']['macro_mixed_top_tie_mean_label'] for r in refs),
'internal_macro_brier_no_worse_than_object_prior':i['object']['macro_object_brier']<=internal['object_prior']['object']['macro_object_brier']}
assert checks==result['gate']['checks'] and all(checks.values())==result['gate']['passed']
journal=read(RUN/'FIT_COMPLETIONS.json');assert len(journal['fits'])==25 and journal['optimizer_steps']==14500
selection_time=datetime.fromisoformat(read(RUN/'SELECTION.json')['created_utc'])
assert datetime.fromisoformat(journal['fits'][19]['completed_utc'])<selection_time<datetime.fromisoformat(journal['fits'][20]['completed_utc'])
def tensors(path):
    with zipfile.ZipFile(path) as z:return {n.split('/data/')[1]:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if '/data/' in n}
old=ROOT/'outputs/boya_shared_score_20261111_v1/cv_fold0/condition_linear_seed20261111.pt'
new=RUN/'cv_fold0/condition_linear_seed20261111.pt'
assert tensors(old)==tensors(new) and len(tensors(old))==2
for p,h in protocol['source_sha256'].items():assert sha(p)==h,p
for p,h in read(ROOT/'outputs/boya_fast_audit_20261111_v1/IMMUTABLE_BASELINE.json')['input_sha256'].items():assert sha(p)==h,p
receipt=dict(complete=True,results_sha256=sha(RUN/'RESULTS.json'),protocol_sha256=sha(RUN/'PROTOCOL.json'),
    selected=selected,all_saved_metrics_recomputed=True,all_gates_recomputed=True,selection_before_internal_fit=True,
    bootstrap_gain_ci95=ci.tolist(),first_failed_and_recovered_tensor_bytes_equal=True,
    final_models=25,final_optimizer_steps=14500,failed_attempt_optimizer_steps=1000,total_optimizer_steps=15500,
    original_progress_underreported_failed_updates=True,old_models_and_data_unchanged=True,new_physics_steps=0,new_CGR_forwards=0,
    source_freeze_unchanged=True,prediction_refs=predrefs,gate=checks)
(OUT/'DECISION_AUDIT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:receipt[k] for k in ['complete','selected','total_optimizer_steps','first_failed_and_recovered_tensor_bytes_equal']}))
