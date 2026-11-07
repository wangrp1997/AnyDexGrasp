"""Summarize frozen S40 training and post-ranking diagnostics; no model selection."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize(root):
 root=Path(root).resolve();dpath=root/'dataset/manifest.json';tpath=root/'training/TRAINING_RESULTS.json';rpath=root/'raw_internal_diagnostic/RESULTS.json'
 d=json.loads(dpath.read_text());t=json.loads(tpath.read_text());r=json.loads(rpath.read_text())
 assert t['status']==r['status']=='complete' and t['inputs_and_sources_unchanged']
 assert t['dataset_manifest']['sha256']==sha(dpath)
 inputs={str(p):sha(p)for p in [Path(__file__).resolve(),dpath,tpath,rpath]};metrics={};support={};weighted=constant=0.;total=0
 for g,entry in d['gesture_files'].items():
  p=Path(entry['path']);assert sha(p)==entry['sha256'];inputs[str(p)]=sha(p)
  with np.load(p,allow_pickle=False)as a:
   fit=a['split']=='train';check=a['split']=='internal_check';prior=float(a['labels'][fit].mean());n=int(check.sum());baseline=float(np.mean((a['labels'][check]-prior)**2));coverage=Counter(map(int,a['target_index'][fit]))
   support[int(g)]=coverage
  head=t['internal_heads'][g];mse=head['internal_check_metric']['mse'];assert head['internal_check_metric']['count']==n
  weighted+=n*mse;constant+=n*baseline;total+=n
  metrics[g]=dict(count=n,network_MSE=mse,fit_prior_constant=prior,constant_MSE=baseline,network_better_than_constant=mse<baseline,fit_observed_targets=sorted(coverage),fit_unobserved_targets=[i for i in range(20)if not coverage[i]])
 reasons=Counter();gestures=Counter();missing=0;targets=Counter();object_rows=[]
 for obj in r['objects']:
  p=root/'raw_internal_diagnostic'/obj['object_id']/'geometry.json';inputs[str(p)]=sha(p);rows=json.loads(p.read_text())['checked'];localmissing=0
  for row in rows:
   g=int(row['gesture_id']);target=int(row['target_index']);reasons[row['geometry'].get('reason','admitted')]+=1;gestures[g]+=1;targets[(g,target)]+=1
   if not support[g][target]:missing+=1;localmissing+=1
  object_rows.append(dict(**obj,shortlist_without_direct_fit_target_supervision=localmissing))
 out=dict(schema='anydex_s40_decision_summary_v1',status='complete',collection=dict(slots=640,actual=290,eligible=240,positive=165,negative=75,terminal_null=50,preparation_null=350,physical_steps=509766,labelled_objects=31,target_labels=400,shortfall=160),
  training=dict(config=t['config'],optimizer_updates=t['new_optimizer_steps'],training_wall_s=t['wall_s'],heads={g:{k:x[k]for k in ['path','sha256','fit_records','fit_positive','fit_negative','fit_mesh_groups','strict_state_dict_reload']}for g,x in t['heads'].items()},internal_metrics=metrics,internal_record_count=total,weighted_internal_MSE=weighted/total,weighted_constant_MSE=constant/total),
  raw_internal=dict(object_count=r['object_count'],head_queries=r['head_queries'],expanded_combinations=sum(x['expanded_combinations']for x in r['objects']),geometry_checked=sum(x['geometry_checked']for x in r['objects']),geometry_admitted=sum(x['geometry_admitted']for x in r['objects']),rejection_reasons=dict(reasons),shortlist_gesture_counts=dict(gestures),shortlist_without_direct_fit_target_supervision=missing,shortlist_target_counts=[dict(gesture_id=g,target_index=i,count=c,fit_supervision=support[g][i])for(g,i),c in sorted(targets.items())],objects=object_rows,profile=r['profile'],no_threshold_tuning=True,labels_used_for_ranking=False),
  conclusion='Training completes, but internal raw score-first deployment has no geometry-admitted candidates. Not accepted for deployment; no physical success or external-generalization claim.',
  boundaries=['Internal44 records are conditioned on original preparation and full execution; not full8-object success rate.','Final all40 heads include internal records; metrics above use only32-fit heads.','Sparse per-target supervision and preparation-conditioned training distribution are observed; individual causal contribution is not isolated.','EGAD remains unused; no extra training or physical attempts are performed.'],
  new_physics_steps=0,new_CGR_forwards=0,EGAD_objects_accessed=0,input_sha256=inputs)
 p=root/'SUMMARY.json';p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(path=str(p),weighted_internal_MSE=weighted/total,weighted_constant_MSE=constant/total,reasons=dict(reasons),without_target_supervision=missing,total_checked=out['raw_internal']['geometry_checked']),indent=2));return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);summarize(p.parse_args().run_dir)
