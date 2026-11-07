import sys,json,copy,hashlib
from pathlib import Path
import numpy as np, torch
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path.insert(0,str(ROOT/'reproduction/boya'))
import train_boya_decision as b
report=json.loads((ROOT/'outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json').read_text())
import train_s40_decision as s40
m,data,files=s40.load_dataset(report['dataset_manifest']['path'])
cls=b.official_components()[0];result={}
torch.set_num_threads(4)
for g in b.GESTURES:
 a=data[g];fit=np.flatnonzero(a['split']=='train');internal=np.flatnonzero(a['split']=='internal_check')
 ck=torch.load(report['internal_heads'][str(g)]['path'],map_location='cpu',weights_only=True)
 model=cls(**ck['architecture_kwargs']);model.load_state_dict(ck['model_state_dict']);model.eval()
 x=torch.from_numpy(a['features'][fit]);y=torch.from_numpy(a['labels'][fit]).float();t=torch.from_numpy(a['target_index'][fit])
 with torch.inference_mode():p=model(x)[0][:,0];selected=p.gather(1,t[:,None]).flatten();eval_mse=((selected-y)**2).mean().item()
 stats={};hooks=[]
 for i in range(1,8):
  bn=getattr(model,f'bn{i}')
  def hook(mod,args,i=i):
   v=args[0].detach().squeeze(-1);stats[str(i)]={'actual_mean_abs':v.mean(0).abs().mean().item(),'actual_var_median':v.var(0,unbiased=True).median().item(),'saved_var_median':mod.running_var.median().item(),'mean_drift_abs_mean':(v.mean(0)-mod.running_mean).abs().mean().item(),'mean_drift_scaled_rms':torch.mean(((v.mean(0)-mod.running_mean)/torch.sqrt(mod.running_var+mod.eps))**2).sqrt().item(),'n_batches':mod.num_batches_tracked.item()}
  hooks.append(bn.register_forward_pre_hook(hook))
 # clone ensures no mutation persisted; training computes each layer current full fit stats.
 model.train()
 with torch.inference_mode():q=model(x)[0][:,0];trainselected=q.gather(1,t[:,None]).flatten();batch_mse=((trainselected-y)**2).mean().item()
 for h in hooks:h.remove()
 coverage={str(i):{'fit':int(np.sum(a['target_index'][fit]==i)),'fit_positive':int(np.sum((a['target_index'][fit]==i)&(a['labels'][fit]==1))),'fit_negative':int(np.sum((a['target_index'][fit]==i)&(a['labels'][fit]==0))),'internal':int(np.sum(a['target_index'][internal]==i))}for i in range(20)}
 result[str(g)]={'fit_count':len(fit),'internal_count':len(internal),'eval_fit_mse':eval_mse,'current_weight_full_fit_batch_statistics_mse':batch_mse,'last_preupdate_training_mse':report['internal_heads'][str(g)]['history'][-1]['mse'],'fit_eval_output_min':float(p.min()),'fit_eval_output_max':float(p.max()),'fit_selected_score_mean':float(selected.mean()),'fit_selected_score_gt1':int((selected>1).sum()),'fit_output_gt1':int((p>1).sum()),'BN_train_fullfit_statistics':stats,'target_coverage':coverage,'total_feature_l2_norm_fit_median':float(np.median(np.linalg.norm(a['features'][fit],axis=1)))}
json.dump(result,open('/tmp/CODE100_SCORE_BN_AUDIT_20261107.json','w'),indent=2)
print(json.dumps(result,indent=2))
