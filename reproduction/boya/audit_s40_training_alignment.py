#!/usr/bin/env python3
"""Compare the unchanged released training epoch with the S40 adapter on real rows.

No trained checkpoint is written or deployed. Numeric CPU pairs use the released
20-output configuration present in published models; the public training
script's default 240-output experiment remains a separately documented boundary.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import random
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
sys.path.insert(0,str(ROOT/'reproduction/boya'))
import train_boya_decision as adapter
import train_s40_code_recipe as recipe

DATASET=ROOT/'outputs/s40_stable_supplement_20261107_v2/training_delivery_v3/merged_dataset/manifest.json'
PINNED={
 ROOT/'dataset/multifinger_hand.py':'29e57cb152b65e2c4f84148a8a2e6e2a2cb9247b95387a52b87e3c18d8528e7e',
 ROOT/'models/minkowski_graspnet.py':'8b92ae7f48be0926d992d209f97345466fd365ed9d63ad4d6a0a3dab0af8e74d',
 ROOT/'robot_dh3.py':'',
}

def definitions(path,names,scope):
 tree=ast.parse(Path(path).read_text())
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Missing source definition')
 ns=dict(scope,__name__='isolated_official_training_alignment')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns

def require(value,message):
 if not value:raise ValueError(message)

def difference(a,b):
 require(a.keys()==b.keys(),'Tensor key mismatch')
 return max([0.,*[float(torch.max(torch.abs(a[k].detach().double()-b[k].detach().double()))) for k in a]])

def source_components():
 official_model=definitions(ROOT/'models/minkowski_graspnet.py',{'MinkowskiGraspNetMultifingerType1'},dict(torch=torch,nn=nn,F=torch.nn.functional))['MinkowskiGraspNetMultifingerType1']
 dataset=definitions(ROOT/'dataset/multifinger_hand.py',{'MultifingerDataset'},dict(np=np,torch=torch,Dataset=torch.utils.data.Dataset,os=__import__('os'),json=json,random=random,copy=copy))['MultifingerDataset']
 canonical=definitions(ROOT/'robot_dh3.py',{'get_graspgroup_features'},dict(np=np,copy=copy))['get_graspgroup_features']
 return official_model,dataset,canonical

def raw_dataset_information(raw480,gesture=1,angle=0,depth=2,extra=1,flip=False,result=0):
 return dict(grasp_preds_features=np.asarray(raw480,dtype=np.float32).tolist(),two_fingers_pose_angle_type=angle,two_fingers_pose_depth_type=depth,
  Allegro_pose_finger_type=gesture,Allegro_pose_depth_type=extra,if_flip=int(flip),result=result)

def loader_contract(dataset_cls,canonical,raw480):
 """Execute actual original loader selection/canonicalization, no copied formula."""
 with tempfile.TemporaryDirectory(prefix='anydex_training_loader_')as tmp:
  tmp=Path(tmp);information={f'row_{i:03d}':raw_dataset_information(raw480,result=int(i%7!=0))for i in range(101)}
  information['other_type']=raw_dataset_information(raw480,gesture=6,result=0)
  (tmp/'obj140.json').write_text(json.dumps(information));(tmp/'test_single_point.json').write_text(json.dumps(information))
  with contextlib.redirect_stdout(io.StringIO()):
   train=dataset_cls(str(tmp/'train_sr'),multifinger_type='Allegro',dataset_type='train',train_type=1)
   test=dataset_cls(str(tmp/'test_sr'),multifinger_type='Allegro',dataset_type='test',train_type=1)
  expected=set(list(information)[:100]);require(set(train.data)==expected,'Original first100 filtering changed')
  require(len(test)==101 and 'other_type'not in test.data,'Original test filtering changed')
  # No padding, oversampling, balancing, feature jitter, or extra fake records.
  positives=sum(information[k]['result']for k in train.data)
  probe=dataset_cls.__new__(dataset_cls);probe.multifinger_type='Allegro'
  max_error=0.;cases=0
  for angle in range(12):
   for flip in [False,True]:
    probe.data=['probe'];probe.informations={'probe':raw_dataset_information(raw480,angle=angle,flip=flip)}
    actual=probe[0]['grasp_preds_features'];raw=np.zeros((1,2262),np.float32)
    raw[0,240:720]=raw480;raw[0,-3]=12+2*angle;raw[0,-2]=.02;raw[0,-1]=float(flip)
    expected=canonical(raw,None)['grasp_preds_features'][0]
    max_error=max(max_error,float(np.max(np.abs(actual-expected))));cases+=1
  require(max_error==0.,'Original train/deployment feature convention no longer agrees')
  return dict(train_candidates=102,matching_gesture_candidates=101,train_retained=100,test_retained=101,
   train_positive=positives,train_negative=100-positives,first100_before_shuffle_exact=True,
   class_balancing=False,label_dependent_sampling=False,legacy_angle_flip_equivalence_cases=cases,
   legacy_angle_flip_max_abs_error=max_error,
   legacy_angle_boundary='Original loader uses12+2*legacy_angle; actual Boya queries include odd modern bins, so stored canonical480 must not be recanonicalized through this legacy loader.')

class NullWriter:
 def add_scalar(self,*args,**kwargs):pass

class RecordingAdam(torch.optim.Adam):
 def __init__(self,named_parameters,**kwargs):
  self.named=dict(named_parameters);self.last_gradients={};super().__init__(self.named.values(),**kwargs)
 def step(self,*args,**kwargs):
  self.last_gradients={k:v.grad.detach().clone()for k,v in self.named.items()if v.grad is not None}
  return super().step(*args,**kwargs)

def step_pairs(gesture,a,original_cls,steps=2):
 inference_cls,loss_cls,scheduler_cls=adapter.official_components();seed=recipe.CONFIG['seed']+gesture
 torch.manual_seed(seed);original=original_cls(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5)
 torch.manual_seed(seed);port=inference_cls(input_num=480,num_multifinger_type=1,num_multifinger_depth=4,two_finger_num_angle=1,two_finger_num_depth=5)
 initial_error=difference(original.state_dict(),port.state_dict());require(initial_error==0.,'Different original/default initialization')
 optimizers=[RecordingAdam(m.named_parameters(),lr=5e-4,weight_decay=5e-4)for m in [original,port]]
 schedulers=[scheduler_cls(o,max_iter=100,power=.99,last_step=-1)for o in optimizers]
 criterions=[loss_cls(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5,train_type=gesture)for _ in [0,1]]
 captures=[{},{}]
 handles=[m.register_forward_hook(lambda mod,args,result,i=i:captures[i].update(preds=result[0].detach().clone()))for i,m in enumerate([original,port])]
 ids=np.flatnonzero(a['split']=='train');rng=np.random.default_rng(seed);rows=[]
 train_source=ROOT/'train_multifinger.py'
 scope=dict(np=np,time=time,log_string=lambda *x:None,TRAIN_WRITER=NullWriter(),NUM_MULTIFINGER_TYPE=1,GRIPPER_TYPE='Boya',MULTIFINGER_TYPE=gesture,
  multifinger_net=original,criterion=criterions[0],optimizer=optimizers[0],lr_scheduler=schedulers[0],convert_data_to_gpu=lambda x:x)
 official_epoch=definitions(train_source,{'train_one_epoch'},scope)
 for epoch in range(steps):
  batch=adapter.epoch_batches(ids,128,rng);require(len(batch)==1,'Current frozen fit should be one original batch')
  ix=batch[0];x=torch.from_numpy(a['features'][ix]);y=torch.from_numpy(a['labels'][ix]);target=torch.from_numpy(a['target_index'][ix]);n=len(ix)
  endpoints=dict(grasp_preds_features=x,result=y[:,None],two_fingers_pose_angle_type=torch.zeros((n,1),dtype=torch.long),
   two_fingers_pose_depth_type=(target//4)[:,None],multifinger_pose_finger_type=torch.full((n,1),gesture,dtype=torch.long),multifinger_pose_depth_type=(target%4)[:,None],if_flip=torch.zeros((n,1),dtype=torch.long))
  official_epoch.update(TRAIN_DATALOADER=[endpoints],EPOCH_CNT=epoch)
  official_epoch['train_one_epoch']();schedulers[0].step()
  port.train();optimizers[1].zero_grad(set_to_none=True);preds,_=port(x)
  loss=adapter.selected_loss(criterions[1],preds,y,target,gesture);loss.backward();optimizers[1].step();schedulers[1].step()
  error=dict(epoch=epoch+1,original_epoch_function_executed=True,batch_records=n,
   forward_max_abs_error=difference(captures[0],captures[1]),
   gradient_max_abs_error=difference(optimizers[0].last_gradients,optimizers[1].last_gradients),
   state_including_all_BN_buffers_max_abs_error=difference(original.state_dict(),port.state_dict()),
   lr_original=schedulers[0].get_last_lr()[0],lr_port=schedulers[1].get_last_lr()[0],
   BN_batches={str(i):int(port.state_dict()[f'bn{i}.num_batches_tracked'])for i in range(1,8)})
  require(all(error[k]==0. for k in ['forward_max_abs_error','gradient_max_abs_error','state_including_all_BN_buffers_max_abs_error'])and error['lr_original']==error['lr_port'],'Actual original epoch and adapter update differ')
  rows.append(error)
 for handle in handles:handle.remove()
 return dict(seed=seed,initial_state_max_abs_error=initial_error,fit_records=len(ids),fit_count_below_original_cap=True,paired_epochs=rows,
  boundary='Same explicit record order; original shuffle RNG and startup seed are not published. CPU numerical equality is not a convergence/generalization/physics result.')

def run(output):
 output=Path(output).resolve();require(not output.exists(),'Fresh audit output required')
 sources={p:adapter.sha(p)for p in [*recipe.PINNED,*PINNED,Path(__file__).resolve()]}
 for p,h in PINNED.items():
  if h:require(sources[p]==h,'Original source changed: '+str(p))
 for p,h in recipe.PINNED.items():require(sources[p]==h,'Frozen original adapter source changed')
 require(adapter.sha(DATASET)==recipe.DATASET_SHA256,'Frozen435 dataset changed')
 m=json.loads(DATASET.read_text());inputs={DATASET:recipe.DATASET_SHA256};data={};actual_canonical=0;odd=0;raw0=None
 for g in adapter.GESTURES:
  e=m['gesture_files'][str(g)];p=Path(e['path']);require(adapter.sha(p)==e['sha256'],'NPZ changed');inputs[p]=e['sha256']
  with np.load(p,allow_pickle=False)as z:data[g]={k:z[k].copy()for k in z.files}
 original_cls,dataset_cls,canonical=source_components()
 for sample in m['samples']:
  p=Path(sample['record_json']);require(adapter.sha(p)==sample['record_sha256'],'Actual trial binding changed');inputs[p]=sample['record_sha256']
  b=json.loads(p.read_text())['binding'];raw=np.asarray(b['candidate_features_raw2261'],np.float32)
  if raw0 is None:raw0=raw[240:720].copy()
  candidate=np.c_[raw[None].copy(),[b['if_flip']]];c=canonical(candidate,None)['grasp_preds_features'][0]
  a=data[sample['gesture_id']];i=sample['npz_row'];require(c.tobytes()==a['features'][i].tobytes(),'Real canonical435 differs from original deployment')
  require(b['two_fingers_pose_depth_type']*4+b['multifinger_pose_depth_type']==int(a['target_index'][i]),'Output index mismatch')
  require(int(json.loads(p.read_text())['result'])==int(a['labels'][i]),'Actual outcome mismatch')
  actual_canonical+=1;odd+=int(b['two_fingers_pose_angle_type']%2==1)
 torch.set_num_threads(4);started=time.monotonic()
 result=dict(schema='anydex_s40_original_training_alignment_audit_v1',status='running',
  published_recipe=recipe.recipe_from_ast(ROOT/'train_multifinger.py'),loader_contract=loader_contract(dataset_cls,canonical,raw0),
  actual_trial_rows_canonical_exact=actual_canonical,actual_trial_rows_with_odd_modern_angle_bin=odd,
  source_sha256={str(p):h for p,h in sources.items()},input_sha256={str(p):h for p,h in inputs.items()},per_head={},
  simulation_steps=0,CGR_forwards=0,trained_checkpoints_written=0,models_deployed=0,EGAD_accessed=False,
  prior_models_unchanged=True,internal_metrics_used_for_selection=False,
  remaining_non_equivalences=['public train script default240 outputs versus published checkpoints/Boya20 outputs',
   'original script saves best test P/R checkpoints each epoch; Boya fixedfinal100 avoids selection on repeatedly debugged internal8',
   'original global RNG unspecified versus fixed Boya seed and explicit identical-order numeric comparison',
   'real-robot labels/hand calibration/transport task versus simulated Boya lift labels',
   'preactivation control changed objective and remains a separately marked non-author ablation, not a replacement original model'])
 for g,a in data.items():result['per_head'][str(g)]=step_pairs(g,a,original_cls)
 result['support']={str(g):dict(fit_records=int(np.sum(a['split']=='train')),internal_records=int(np.sum(a['split']=='internal_check')),
  fit_target_counts=np.bincount(a['target_index'][a['split']=='train'],minlength=20).tolist(),
  fit_target_positive_counts=np.bincount(a['target_index'][(a['split']=='train')&(a['labels']==1)],minlength=20).tolist(),
  fit_target_negative_counts=np.bincount(a['target_index'][(a['split']=='train')&(a['labels']==0)],minlength=20).tolist())for g,a in data.items()}
 allcounts=np.concatenate([np.asarray(x['fit_target_counts'])for x in result['support'].values()])
 result['support_summary']=dict(gesture_output_cells=len(allcounts),zero_label_cells=int(np.sum(allcounts==0)),one_or_two_label_cells=int(np.sum((allcounts>=1)&(allcounts<=2))),
  both_outcomes_cells=sum(int(np.sum((np.asarray(x['fit_target_positive_counts'])>0)&(np.asarray(x['fit_target_negative_counts'])>0)))for x in result['support'].values()),
  interpretation='Output support is a diagnostic, not proof every output needs both labels; frozen deployment generalization is still unvalidated.')
 result.update(status='complete',wall_s=time.monotonic()-started,ephemeral_optimizer_steps_original=8,ephemeral_optimizer_steps_adapter=8,
  all_bound_sources_unchanged=all(adapter.sha(p)==h for p,h in sources.items()),all_bound_inputs_unchanged=all(adapter.sha(p)==h for p,h in inputs.items()),
  demonstrated_adapter_mismatch=False,training_or_model_changes_made=False,
  conclusion='Code100 core training arithmetic matches unchanged original epoch exactly for current364 fit rows; no justified training patch. Physical mapping/control and training-label/candidate distribution remain unresolved.')
 require(result['all_bound_sources_unchanged']and result['all_bound_inputs_unchanged'],'Audit input changed')
 output.mkdir(parents=True);adapter.save(output/'SUMMARY.json',result)
 print(json.dumps({k:v for k,v in result.items()if k not in ['source_sha256','input_sha256','published_recipe','support']},indent=2));return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
 run(p.parse_args().output_dir)
if __name__=='__main__':main()
