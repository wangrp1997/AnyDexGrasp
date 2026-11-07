import sys,json,hashlib,ast
from pathlib import Path
import numpy as np,torch
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path.insert(0,str(ROOT/'reproduction/boya'))
import train_boya_decision as b
read=lambda p:json.loads(Path(p).read_text())
old_path=ROOT/'outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json';new_path=ROOT/'outputs/s40_preactivation_control_20261107_v1/TRAINING_RESULTS.json'
old=read(old_path);new=read(new_path);manifest=read(old['dataset_manifest']['path']);cls=b.official_components()[0];torch.set_num_threads(4)
result={'schema':'independent_s40_preactivation_single_difference_audit_v1','old_report':{'path':str(old_path),'sha256':b.sha(old_path)},'new_report':{'path':str(new_path),'sha256':b.sha(new_path)},'same_config':new['config']==old['config'],'same_dataset_manifest':new['dataset_manifest']==old['dataset_manifest'],'same_stage_record_counts':new['stage_record_counts']==old['stage_record_counts'],'same_mesh_groups':new['mesh_groups']==old['mesh_groups'],'new_training_variant':new['training_variant'],'per_head':{},'scope':'Read-only independent original Inference forward; no new training/physics/EGAD; seeded construction is independently repeated, initial snapshots were not saved during original training.'}
def modelhash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()
fitnums=[];checknums=[];allpassed=True
for gs in ['1','6','7','8']:
 g=int(gs);e=manifest['gesture_files'][gs]
 with np.load(e['path'],allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
 fit=np.flatnonzero(a['split']=='train');check=np.flatnonzero(a['split']=='internal_check');prior=float(a['labels'][fit].mean())
 oc=torch.load(old['internal_heads'][gs]['path'],map_location='cpu',weights_only=True);nc=torch.load(new['internal_heads'][gs]['path'],map_location='cpu',weights_only=True)
 oh=old['internal_heads'][gs]['history'];nh=new['internal_heads'][gs]['history'];keys=['epoch','learning_rate','samples','distinct_samples','batch_sizes','ordered_trial_ids_sha256'];matches={k:all(x[k]==y[k]for x,y in zip(oh,nh))for k in keys}
 seed_old=oc['training_config']['seed']+g;seed_new=nc['training_config']['seed']+g
 torch.manual_seed(seed_old);m0=cls(**oc['architecture_kwargs']);init_old=modelhash(m0)
 torch.manual_seed(seed_new);m1=cls(**nc['architecture_kwargs']);init_new=modelhash(m1)
 # Independent replay of each epoch's exact ID shuffle under same original sampler.
 rng=np.random.default_rng(seed_new);replay=[]
 for epoch in range(100):
  batches=b.epoch_batches(fit,128,rng);seen=np.concatenate(batches);ids=[str(a['trial_id'][i])for i in seen];replay.append(hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
 replay_ok=all(h==row['ordered_trial_ids_sha256']for h,row in zip(replay,nh))
 model=cls(**nc['architecture_kwargs']);strict=model.load_state_dict(nc['model_state_dict'],strict=True);model.eval();stats={}
 for split,ids in [('fit',fit),('internal',check)]:
  with torch.inference_mode():p=model(torch.from_numpy(a['features'][ids]))[0][:,0].numpy()
  y=a['labels'][ids];t=a['target_index'][ids];s=p[np.arange(len(ids)),t];pos=y==1
  stats[split]={'count':len(ids),'positive':int(pos.sum()),'negative':int((~pos).sum()),'independent_original_forward_MSE':float(np.mean((s-y)**2)),'reported_original_forward_MSE':new['internal_heads'][gs]['fit_metric'if split=='fit'else'internal_check_metric']['mse'],'positive_selected_exact_zero':int(np.sum(pos&(s==0))),'positive_selected_exact_zero_fraction':float(np.mean((s==0)[pos])),'negative_selected_exact_zero':int(np.sum((~pos)&(s==0))),'selected_score_max':float(s.max()),'selected_score_min':float(s.min()),'fit_prior_constant_MSE':float(np.mean((y-prior)**2))}
 stats['fit']['old_positive_selected_exact_zero']=read('/tmp/CODE100_SCORE_SUPPORT_AUDIT_20261107.json')['per_head'][gs]['fit_positive_selected_exact_zero']
 r={'seed_old':seed_old,'seed_new':seed_new,'seeded_initial_state_sha256_old_recipe':init_old,'seeded_initial_state_sha256_new_recipe':init_new,'seeded_initial_state_equal':init_old==init_new,'initial_snapshots_saved_by_training':False,'epoch_count_old':len(oh),'epoch_count_new':len(nh),'history_exact_match_excluding_MSE':matches,'independent_epoch_shuffle_replay_matches':replay_ok,'optimizer_updates_old':old['internal_heads'][gs]['optimizer_steps'],'optimizer_updates_new':new['internal_heads'][gs]['optimizer_steps'],'same_fit_trial_ids':oc['fit_trial_ids']==nc['fit_trial_ids'],'same_fit_mesh_groups':oc['fit_mesh_sha256']==nc['fit_mesh_sha256'],'same_architecture_kwargs':oc['architecture_kwargs']==nc['architecture_kwargs'],'same_state_dict_shapes':{k:list(v.shape)for k,v in oc['model_state_dict'].items()}=={k:list(v.shape)for k,v in nc['model_state_dict'].items()},'BN_batches_old':{str(i):int(oc['model_state_dict'][f'bn{i}.num_batches_tracked'])for i in range(1,8)},'BN_batches_new':{str(i):int(nc['model_state_dict'][f'bn{i}.num_batches_tracked'])for i in range(1,8)},'fit_positive_prior':prior,**stats}
 result['per_head'][gs]=r;fitnums.append(stats['fit']);checknums.append(stats['internal'])
 allpassed&=all(matches.values())and replay_ok and init_old==init_new and r['same_fit_trial_ids']and r['same_fit_mesh_groups']and r['same_architecture_kwargs']and r['same_state_dict_shapes']and len(oh)==len(nh)==100 and r['optimizer_updates_old']==r['optimizer_updates_new']==100
for split,rows in [('fit',fitnums),('internal',checknums)]:
 n=sum(r['count']for r in rows);result[split+'_weighted']={'count':n,'new_original_forward_MSE':sum(r['count']*r['independent_original_forward_MSE']for r in rows)/n,'fit_prior_constant_MSE':sum(r['count']*r['fit_prior_constant_MSE']for r in rows)/n,'positive_count':sum(r['positive']for r in rows),'positive_selected_exact_zero':sum(r['positive_selected_exact_zero']for r in rows)}
result['single_difference_runtime_records_pass']=bool(allpassed and result['same_config']and result['same_dataset_manifest']and result['same_mesh_groups']and result['same_stage_record_counts'])
json.dump(result,open('/tmp/S40_PREACTIVATION_CONTROL_INDEPENDENT_AUDIT_20261107.json','w'),indent=2,allow_nan=False)
print(json.dumps({'pass':result['single_difference_runtime_records_pass'],'fit_weighted':result['fit_weighted'],'internal_weighted':result['internal_weighted'],'per_head':{g:{k:v for k,v in r.items()if k in ['seed_old','seed_new','history_exact_match_excluding_MSE','independent_epoch_shuffle_replay_matches','fit','internal']}for g,r in result['per_head'].items()}},indent=2))
