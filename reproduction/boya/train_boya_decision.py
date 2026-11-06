#!/usr/bin/env python3
"""Train four original AnyDex 480->20 decision heads on bound Boya records.

CGR is never constructed or trained. External validation meshes are never used
by this trainer. Final checkpoints use the fixed final epoch, without selection.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
GESTURES=(1,6,7,8)
MODEL_SOURCE=ROOT/'models/minkowski_graspnet_single_point.py'
LOSS_SOURCE=ROOT/'models/loss.py'
SOLVER_SOURCE=ROOT/'utils/solvers.py'
CONFIG=dict(epochs=20,batch_size=128,learning_rate=1e-4,weight_decay=5e-4,optimizer='Adam',scheduler='original PolyLR',poly_power=.99,seed=20261106,input_dim=480,output_dim=20)


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()

def load(path):return json.loads(Path(path).read_text())
def save(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def extract(path,names,scope):
 nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Official definition missing: '+str(names))
 namespace=dict(scope,__name__='official_boya_decision_overlay');exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace

def official_components():
 scope=dict(torch=torch,nn=nn,F=F,np=np,LambdaLR=LambdaLR)
 cls=extract(MODEL_SOURCE,['MinkowskiGraspNetMultifingerType1Inference'],scope)['MinkowskiGraspNetMultifingerType1Inference']
 loss=extract(LOSS_SOURCE,['MultifingerType1Loss'],scope)['MultifingerType1Loss']
 scheduler=extract(SOLVER_SOURCE,['LambdaStepLR','PolyLR'],scope)['PolyLR']
 return cls,loss,scheduler

def epoch_batches(indices,batch_size,rng):
 """Merge a singleton tail into the preceding batch; never drop or repeat IDs."""
 indices=np.asarray(indices,dtype=np.int64)
 if len(indices)<2 or batch_size<2:raise ValueError('Original BatchNorm training needs at least two distinct records')
 order=rng.permutation(indices);batches=[order[start:start+batch_size]for start in range(0,len(order),batch_size)]
 if len(batches[-1])==1:
  if len(batches)==1:raise ValueError('Cannot invent a second BatchNorm training example')
  batches[-2]=np.concatenate((batches[-2],batches[-1]));batches.pop()
 merged=np.concatenate(batches)
 if len(merged)!=len(indices)or len(set(map(int,merged)))!=len(indices)or set(merged)!=set(indices):raise ValueError('Batch sampler dropped or repeated records')
 return batches

def verify_groups(groups):
 expected={'train':32,'internal_check':8,'external_validation':10}
 if not isinstance(groups,dict)or set(groups)!=set(expected):raise ValueError('Require explicit frozen 32/8/10 mesh groups')
 flat=[]
 for split,count in expected.items():
  if len(groups[split])!=count or len(set(groups[split]))!=count:raise ValueError('Wrong frozen mesh group count: '+split)
  flat.extend(groups[split])
 if len(set(flat))!=50:raise ValueError('Mesh leakage across frozen splits')
 return {mesh:split for split,meshes in groups.items()for mesh in meshes}

def read_dataset(path,development_smoke=False):
 manifest=load(path)
 if manifest.get('complete')is not True:raise ValueError('Dataset preparation must be complete')
 if manifest.get('benchmark_only')and not development_smoke:raise ValueError('Development snapshot requires --development-smoke')
 if development_smoke is False and manifest.get('benchmark_only')is not False:raise ValueError('Formal training requires an explicitly non-benchmark final dataset')
 groups=manifest.get('mesh_groups');authority=verify_groups(groups)
 files=manifest['gesture_files']
 if set(files)!=set(map(str,GESTURES)):raise ValueError('Exactly four declared gesture datasets required')
 data={};inputs={str(Path(path).resolve()):sha(path)};all_ids=set()
 for gesture in GESTURES:
  entry=files[str(gesture)];file=Path(entry['path']);file=file if file.is_absolute()else Path(path).parent/file
  if sha(file)!=entry['sha256']:raise ValueError('Dataset SHA mismatch: '+str(file))
  inputs[str(file.resolve())]=entry['sha256']
  with np.load(file,allow_pickle=False)as z:a={key:z[key].copy()for key in z.files}
  required={'features','labels','target_index','gesture_ids','mesh_sha256','trial_id','object_id','split'}
  if not required.issubset(a):raise ValueError('Required bound dataset columns missing')
  n=len(a['labels'])
  if a['features'].shape!=(n,480)or a['features'].dtype!=np.float32 or not np.isfinite(a['features']).all():raise ValueError('Finite original canonical float32 features[N,480] required')
  for key in required-{'features'}:
   if a[key].shape!=(n,)or a[key].dtype.hasobject:raise ValueError('Unpickled equal-length dataset columns required: '+key)
  if any(a[key].dtype!=np.int64 for key in('labels','target_index','gesture_ids')):raise ValueError('Labels/index/gesture must be int64')
  if not np.isin(a['labels'],[0,1]).all()or not ((a['target_index']>=0)&(a['target_index']<20)).all()or not (a['gesture_ids']==gesture).all():raise ValueError('Invalid actual label, supervision index or gesture')
  for tid,mesh,split in zip(a['trial_id'],a['mesh_sha256'],a['split']):
   if str(tid)in all_ids:raise ValueError('Repeated bound actual trial identity')
   all_ids.add(str(tid))
   if authority.get(str(mesh))!=str(split):raise ValueError('Dataset row conflicts with frozen mesh split')
  if entry.get('count',n)!=n:raise ValueError('Dataset row count receipt mismatch')
  data[gesture]=a
 return manifest,data,inputs

def selected_loss(criterion,preds,labels,index,gesture):
 n=len(labels)
 endpoints=dict(stage4_grasp_preds_five_hand=preds,result=labels,
  multifinger_pose_finger_type=torch.full((n,1),gesture,dtype=torch.long,device=preds.device),
  multifinger_pose_depth_type=(index%4).reshape(n,1),two_fingers_pose_depth_type=(index//4).reshape(n,1),
  two_fingers_pose_angle_type=torch.zeros((n,1),dtype=torch.long,device=preds.device))
 return criterion(endpoints)[0]

def evaluate(model,a,indices,device):
 if not len(indices):return dict(count=0,mse=None,status='no_internal_check_records_yet')
 model.eval();scores=[]
 with torch.no_grad():
  for block in range(0,len(indices),128):
   ids=indices[block:block+128];x=torch.from_numpy(a['features'][ids]).to(device);y=model(x)[0].reshape(-1,20)
   target=torch.from_numpy(a['target_index'][ids]).to(device);scores.extend(y.gather(1,target[:,None]).flatten().cpu().tolist())
 values=np.asarray(scores,dtype=np.float64);labels=a['labels'][indices]
 return dict(count=len(indices),mse=float(np.mean((values-labels)**2)),score_min=float(values.min()),score_max=float(values.max()),scores_are_uncalibrated_ReLU_MSE=True)

def train_head(gesture,stage,a,fit,check,output,device,config,components,provenance,benchmark):
 cls,loss_cls,scheduler_cls=components;seed=config['seed']+gesture+(0 if stage=='internal_check'else 1000)
 torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
 if device.type=='cuda':torch.cuda.manual_seed_all(seed)
 model=cls(input_num=480,num_multifinger_type=1,num_multifinger_depth=4,two_finger_num_angle=1,two_finger_num_depth=5).to(device)
 criterion=loss_cls(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5,train_type=gesture).to(device)
 optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
 scheduler=scheduler_cls(optimizer,max_iter=config['epochs'],power=config['poly_power'],last_step=-1)
 start=time.monotonic();history=[];steps=0
 for epoch in range(1,config['epochs']+1):
  model.train();batches=epoch_batches(fit,config['batch_size'],rng);squared_sum=0.;seen=[];lr=optimizer.param_groups[0]['lr']
  for indices in batches:
   x=torch.from_numpy(a['features'][indices]).to(device);labels=torch.from_numpy(a['labels'][indices]).to(device);target=torch.from_numpy(a['target_index'][indices]).to(device)
   optimizer.zero_grad(set_to_none=True);preds,_=model(x);loss=selected_loss(criterion,preds,labels,target,gesture)
   if not torch.isfinite(loss):raise ValueError('Nonfinite original MSE loss')
   loss.backward();optimizer.step();squared_sum+=float(loss.detach().cpu())*len(indices);seen.extend(indices.tolist());steps+=1
  scheduler.step();ids=[str(a['trial_id'][i])for i in seen]
  history.append(dict(epoch=epoch,learning_rate=lr,mse=squared_sum/len(fit),samples=len(seen),distinct_samples=len(set(ids)),batch_sizes=list(map(len,batches)),ordered_trial_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
  print('BOYA_HEAD_EPOCH '+json.dumps(dict(stage=stage,gesture_id=gesture,**history[-1])),flush=True)
 model.eval();fit_metric=evaluate(model,a,fit,device);check_metric=evaluate(model,a,check,device)if stage=='internal_check'else None
 path=output/f'g{gesture:02d}.pt';state={key:value.detach().cpu().clone()for key,value in model.state_dict().items()}
 checkpoint=dict(schema='anydex_boya_decision_state_dict_v1',model_state_dict=state,gesture_id=gesture,input_dim=480,output_dim=20,epoch=config['epochs'],stage=stage,benchmark_only=benchmark,model_class='MinkowskiGraspNetMultifingerType1Inference',architecture_kwargs=dict(input_num=480,num_multifinger_type=1,num_multifinger_depth=4,two_finger_num_angle=1,two_finger_num_depth=5),training_config=config,dataset_manifest_sha256=provenance['dataset_manifest_sha256'],source_sha256=provenance['source_sha256'],fit_mesh_sha256=sorted(set(map(str,a['mesh_sha256'][fit]))),fit_trial_ids=list(map(str,a['trial_id'][fit])),supervision='raw_depth_type*4 + extra_depth_index; one executed output MSE only',cgr_trained=False)
 torch.save(checkpoint,path)
 # A newly constructed original deployment class must safely load every tensor.
 reloaded=cls(**checkpoint['architecture_kwargs']);loaded=torch.load(path,map_location='cpu',weights_only=True);reloaded.load_state_dict(loaded['model_state_dict'],strict=True);reloaded.eval()
 report=dict(path=str(path),sha256=sha(path),gesture_id=gesture,input_dim=480,output_dim=20,epoch=config['epochs'],stage=stage,benchmark_only=benchmark,fit_records=len(fit),fit_positive=int(a['labels'][fit].sum()),fit_negative=int((a['labels'][fit]==0).sum()),fit_mesh_groups=len(set(a['mesh_sha256'][fit])),optimizer_steps=steps,history=history,fit_metric=fit_metric,internal_check_metric=check_metric,wall_s=time.monotonic()-start,strict_state_dict_reload=True,checkpoint_selection='fixed final epoch; no external validation access')
 save(path.with_suffix('.json'),report);del model,optimizer,scheduler,criterion,reloaded;return report

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset-manifest',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--development-smoke','--benchmark-only',dest='development_smoke',action='store_true');parser.add_argument('--epochs',type=int);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 if args.epochs is not None and(not args.development_smoke or not 1<=args.epochs<=2):parser.error('Formal training is fixed20epochs; smoke override must be1..2')
 output=args.output_dir.resolve()
 if output.exists():raise ValueError('Training output already exists; never overwrite or resume silently')
 manifest,data,inputs=read_dataset(args.dataset_manifest.resolve(),args.development_smoke)
 config=dict(CONFIG,epochs=(args.epochs or 2)if args.development_smoke else 20)
 sources={str(path):sha(path)for path in[Path(__file__).resolve(),MODEL_SOURCE,LOSS_SOURCE,SOLVER_SOURCE,ROOT/'train_multifinger.py']}
 stages={}
 for gesture,a in data.items():
  stages[str(gesture)]={stage:int(np.sum(np.isin(a['split'],splits)))for stage,splits in [('internal_train',['train']),('internal_check',['internal_check']),('all_train_meshes',['train','internal_check'])]}
  if stages[str(gesture)]['internal_train']<2 or stages[str(gesture)]['all_train_meshes']<2:raise ValueError('Insufficient real unique training records for original BatchNorm head '+str(gesture))
 output.mkdir(parents=True);(output/'heads').mkdir();(output/'internal_check').mkdir();provenance=dict(dataset_manifest_sha256=sha(args.dataset_manifest),source_sha256=sources)
 report=dict(schema='anydex_boya_decision_training_v1',status='prepared'if args.prepare_only else'training_running',benchmark_only=args.development_smoke,dataset_manifest=dict(path=str(args.dataset_manifest.resolve()),sha256=provenance['dataset_manifest_sha256']),input_sha256=inputs,source_sha256=sources,config=config,device=args.device,argv=sys.argv,software=dict(python=sys.version,torch=str(torch.__version__),numpy=str(np.__version__)),split_policy='Fixed32 train /8 internal check; then all40 refit. External10 excluded from all fits/metrics/selection.',stage_record_counts=stages,heads={},internal_heads={},cgr_model_loaded=False,cgr_trained=False,external_validation_evaluated=False,new_physics_steps=0,official_code_adaptations=dict(paper='20epochs, Adam initial1e-4,batch128',released_code_defaults='100epochs,initial5e-4,weight_decay5e-4,PolyLRpower.99',chosen='Paper epochs/initialLR/batch; released-code WD5e-4 and PolyLR.99 explicitly retained',tail='A singleton final batch joins its preceding batch; no skipped/duplicated records',model_selection='Fixed final epoch; no threshold/scheduler/epoch selection from external10',data='Direct stored canonical480; legacy dataset canonicalization is not repeated'))
 save(output/'TRAINING_RESULTS.json',report)
 if args.prepare_only:print(json.dumps(dict(status=report['status'],config=config,stage_record_counts=stages,new_optimizer_steps=0),indent=2));return
 started=time.monotonic()
 try:
  device=torch.device(args.device)
  if device.type=='cuda'and not torch.cuda.is_available():raise ValueError('Requested GPU unavailable')
  torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  report['runtime']=dict(device=str(device),torch_threads=4,cudnn_benchmark=False,cudnn_deterministic=True)
  if device.type=='cuda':report['runtime'].update(gpu_name=torch.cuda.get_device_name(device),gpu_capability=list(torch.cuda.get_device_capability(device)))
  components=official_components()
  for stage,splits in [('internal_check',['train']),('all_train_meshes',['train','internal_check'])]:
   for gesture in GESTURES:
    a=data[gesture];fit=np.flatnonzero(np.isin(a['split'],splits));check=np.flatnonzero(a['split']=='internal_check')if stage=='internal_check'else np.empty(0,dtype=np.int64)
    result=train_head(gesture,stage,a,fit,check,output/('internal_check'if stage=='internal_check'else'heads'),device,config,components,provenance,args.development_smoke)
    report['internal_heads'if stage=='internal_check'else'heads'][str(gesture)]=result;report['wall_s']=time.monotonic()-started;save(output/'TRAINING_RESULTS.json',report)
  report.update(status='benchmark_smoke_complete'if args.development_smoke else'complete',new_optimizer_steps=sum(row['optimizer_steps']for group in('heads','internal_heads')for row in report[group].values()),wall_s=time.monotonic()-started)
 except BaseException:
  report.update(status='training_infrastructure_error',error=traceback.format_exc(),wall_s=time.monotonic()-started);raise
 finally:
  unchanged=all(Path(path).is_file()and sha(path)==value for path,value in dict(inputs,**sources).items());report['inputs_and_sources_unchanged']=unchanged
  if not unchanged:report['status']='frozen_input_changed'
  save(output/'TRAINING_RESULTS.json',report)
 print(json.dumps(dict(status=report['status'],heads={g:r['path']for g,r in report['heads'].items()},wall_s=report['wall_s'],new_optimizer_steps=report['new_optimizer_steps']),indent=2))

if __name__=='__main__':main()
