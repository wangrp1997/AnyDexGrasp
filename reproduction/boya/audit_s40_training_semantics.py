"""Read-only fixed-weight training semantics audit; no optimization or BN updates."""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya';sys.path.insert(0,str(BOYA))
import train_boya_decision as train
RUN=ROOT/'outputs/s40_stable_supplement_20261107_v2/training_delivery_v3/training'
report=json.loads((RUN/'TRAINING_RESULTS.json').read_text());dataset_path=Path(report['dataset_manifest']['path']);dataset=json.loads(dataset_path.read_text())
started=time.monotonic();torch.set_num_threads(4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(v,msg):
 if not v:raise AssertionError(msg)
def selected(pred,idx):return pred.reshape(len(idx),20).gather(1,torch.as_tensor(idx,dtype=torch.int64)[:,None]).flatten()
def fixed_forward(model,x,batch_stats=False):
 """Functional BN has no running buffers when batch_stats=True; no state writes."""
 y=x.reshape(len(x),480,1);stats={}
 def block(i,value):
  conv=getattr(model,f'conv{i}');bn=getattr(model,f'bn{i}');pre=F.conv1d(value,conv.weight,conv.bias)
  dims=(0,2);var=pre.var(dims,unbiased=True);mean=pre.mean(dims)
  stats[str(i)]=dict(actual_batch_var_median=float(var.median()),saved_running_var_median=float(bn.running_var.median()),saved_running_var_min=float(bn.running_var.min()),mean_abs_difference_mean=float((mean-bn.running_mean).abs().mean()))
  norm=F.batch_norm(pre,None if batch_stats else bn.running_mean,None if batch_stats else bn.running_var,bn.weight,bn.bias,training=batch_stats,momentum=0.,eps=bn.eps)
  return F.relu(norm)
 y=block(1,y);res=block(2,y);y=block(3,res);y=block(4,y);y=block(5,y+res);y=block(6,y);y=block(7,y)
 return y.transpose(1,2).contiguous(),stats

sources=[BOYA/'train_boya_decision.py',BOYA/'train_s40_decision.py',ROOT/'train_multifinger.py',train.MODEL_SOURCE,ROOT/'models/minkowski_graspnet.py',train.LOSS_SOURCE,train.SOLVER_SOURCE,ROOT/'dataset/multifinger_hand.py']
for path,h in report['source_sha256'].items():check(sha(path)==h,'Training source changed: '+path)
check(sha(dataset_path)==report['dataset_manifest']['sha256'],'Dataset changed')
cls,loss_cls,_=train.official_components()
training_cls=train.extract(ROOT/'models/minkowski_graspnet.py',['MinkowskiGraspNetMultifingerType1'],dict(torch=torch,nn=nn,F=F))['MinkowskiGraspNetMultifingerType1']
output=dict(schema='anydex_s40_readonly_training_semantics_audit_v1',sources_sha256={str(p):sha(p)for p in sources},training_results_sha256=sha(RUN/'TRAINING_RESULTS.json'),dataset_manifest_sha256=sha(dataset_path),config=report['config'],gesture_reports={},new_optimizer_steps=0,new_training_updates=0,new_physics_steps=0,new_CGR_forwards=0,BN_buffers_mutated=False,internal_batch_statistics_not_used=True)
all_check_errors=[];all_prior_errors=[];all_fit_errors=[];all_check_y=[];all_check_scores=[];all_batch_fit_errors=[];bound_records=0
for gesture in(1,6,7,8):
 entry=dataset['gesture_files'][str(gesture)];check(sha(entry['path'])==entry['sha256'],'Gesture NPZ changed')
 with np.load(entry['path'],allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
 fit=np.flatnonzero(a['split']=='train');internal=np.flatnonzero(a['split']=='internal_check');prior=float(a['labels'][fit].mean())
 # No source raw-CGR/camera model is recomputed. Verify every exported feature
 # and target against its original saved bound record.
 metas={s['trial_id']:s for s in dataset['samples']if s['gesture_id']==gesture}
 for i,tid in enumerate(a['trial_id']):
  meta=metas[str(tid)];record=json.loads(Path(meta['record_json']).read_text());b=record['binding'];feature=np.asarray(b['canonical_CGR480'],dtype=np.float32)
  check(feature.tobytes()==a['features'][i].tobytes(),'Bound canonical480 bytes differ')
  check(b['two_fingers_pose_depth_type']*4+b['multifinger_pose_depth_type']==int(a['target_index'][i]),'Output index miswire')
  check(record['label_eligible']is True and record['result']==int(a['labels'][i])and b['multifinger_pose_finger_type']==gesture,'Label/type miswire');bound_records+=1
 per={}
 for stage_key in('internal_heads','heads'):
  receipt=report[stage_key][str(gesture)];path=Path(receipt['path']);check(sha(path)==receipt['sha256'],'Checkpoint changed')
  checkpoint=torch.load(path,map_location='cpu',weights_only=True);state=checkpoint['model_state_dict'];model=cls(**checkpoint['architecture_kwargs']);model.load_state_dict(state,strict=True);model.eval()
  original_state={k:v.detach().clone()for k,v in model.state_dict().items()}
  x=torch.from_numpy(a['features']);target=a['target_index'];labels=a['labels'].astype(float)
  with torch.no_grad():
   pred=model(x)[0];values=selected(pred,target).double().numpy()
   frozen_math,layer_stats=fixed_forward(model,x[fit],False);check(torch.allclose(frozen_math,model(x[fit])[0],atol=2e-6,rtol=1e-5),'Functional eval differs from original')
   batch_pred,_=fixed_forward(model,x[fit],True);batch_values=selected(batch_pred,target[fit]).double().numpy()
   direct=float(np.mean((values[fit]-labels[fit])**2))
   criterion=loss_cls(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5,train_type=gesture)
   loss=float(train.selected_loss(criterion,pred[fit],torch.from_numpy(a['labels'][fit]),torch.from_numpy(target[fit]),gesture))
   check(abs(loss-direct)<1e-6,'Original loss differs from direct selected MSE')
   # Original dictionary training class, configured to published20 outputs,
   # must agree numerically with the direct-input Inference class.
   original=training_cls(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5);original.load_state_dict(state,strict=True);original.eval()
   endpoints=dict(grasp_preds_features=x,two_fingers_pose_angle_type=torch.zeros((len(x),1),dtype=torch.int64),two_fingers_pose_depth_type=torch.from_numpy(target//4)[:,None],multifinger_pose_finger_type=torch.full((len(x),1),gesture),multifinger_pose_depth_type=torch.from_numpy(target%4)[:,None],if_flip=torch.zeros((len(x),1),dtype=torch.int64))
   equivalent=original(endpoints)[0];eq_error=float((equivalent-pred).abs().max());check(eq_error==0,'Original training/Inference classes differ')
  check(all(torch.equal(model.state_dict()[k],v)for k,v in original_state.items()),'Diagnostic mutated saved BN/parameters')
  seed=report['config']['seed']+gesture+(0 if stage_key=='internal_heads'else 1000);torch.manual_seed(seed);initial=cls(**checkpoint['architecture_kwargs'])
  conv_delta=float((initial.conv1.weight-state['conv1.weight']).norm());bn_counters={str(i):int(state[f'bn{i}.num_batches_tracked'])for i in range(1,8)}
  errors=(values-labels)**2;support=set(map(int,a['target_index'][fit]));unseen=np.array([int(t)not in support for t in target[internal]])
  per[stage_key]=dict(checkpoint_sha256=receipt['sha256'],bn_batches_tracked=bn_counters,optimizer_steps=receipt['optimizer_steps'],initial_conv1_weight_delta_L2=conv_delta,
   fit_eval_mse=float(errors[fit].mean()),fit_current_weight_batch_statistics_mse=float(np.mean((batch_values-labels[fit])**2)),last_history_train_mode_pre_update_mse=receipt['history'][-1]['mse'],
   internal_eval_mse=float(errors[internal].mean()),training_class_equivalence_max_error=eq_error,original_loss_direct_MSE_error=abs(loss-direct),fit_layer_BN_statistics=layer_stats,
   internal_selected_score_mean=float(values[internal].mean()),internal_label_positive_rate=float(labels[internal].mean()),internal_positive_zero_scores=int(np.sum((labels[internal]==1)&(values[internal]==0))),
   internal_positive_count=int(np.sum(labels[internal]==1)),internal_negative_zero_scores=int(np.sum((labels[internal]==0)&(values[internal]==0))),
   internal_positive_mse=float(errors[internal][labels[internal]==1].mean()),internal_negative_mse=float(errors[internal][labels[internal]==0].mean()),
   internal_targets_absent_from_fit=int(unseen.sum()),internal_unseen_target_mse=float(errors[internal][unseen].mean())if unseen.any()else None,
   internal_seen_target_mse=float(errors[internal][~unseen].mean())if (~unseen).any()else None)
  if stage_key=='internal_heads':
   all_check_errors.extend(errors[internal]);all_prior_errors.extend((prior-labels[internal])**2);all_fit_errors.extend(errors[fit]);all_check_y.extend(labels[internal]);all_check_scores.extend(values[internal]);all_batch_fit_errors.extend((batch_values-labels[fit])**2)
 output['gesture_reports'][str(gesture)]=dict(fit_count=len(fit),internal_count=len(internal),fit_prior=prior,fit_prior_constant_internal_mse=float(np.mean((prior-labels[internal])**2)),fit_target_counts=np.bincount(target[fit],minlength=20).tolist(),internal_target_counts=np.bincount(target[internal],minlength=20).tolist(),**per)

# Original loss's20 observed supervision positions, independently exhaust all.
synthetic=torch.arange(400,dtype=torch.float32).reshape(20,1,20)/400.;idx=torch.arange(20);ys=torch.arange(20)%2
for gesture in(1,6,7,8):
 criterion=loss_cls(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5,train_type=gesture)
 got=train.selected_loss(criterion,synthetic,ys,idx,gesture);wanted=((synthetic.reshape(20,20)[torch.arange(20),idx]-ys.float())**2).mean();check(torch.equal(got,wanted),'20-index loss wiring failed')
y=np.array(all_check_y);v=np.array(all_check_scores)
output.update(status='passed_readonly_semantic_checks',bound_records_verified=bound_records,internal_count=len(y),internal_weighted_MSE=float(np.mean(all_check_errors)),fit_prior_constant_weighted_internal_MSE=float(np.mean(all_prior_errors)),fit_weighted_eval_MSE=float(np.mean(all_fit_errors)),fit_fixed_weight_batch_statistics_MSE=float(np.mean(all_batch_fit_errors)),internal_positive_count=int(y.sum()),internal_positive_zero_scores=int(np.sum((y==1)&(v==0))),internal_selected_score_mean=float(v.mean()),BN_initial_running_variance_retention_at20updates=.9**20,elapsed_wall_s=time.monotonic()-started)
out=Path('/tmp/s40_training_semantics_audit_20261107.json');out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in output.items()if k not in('gesture_reports','sources_sha256')},indent=2))
