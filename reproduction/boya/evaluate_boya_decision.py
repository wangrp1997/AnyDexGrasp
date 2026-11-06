"""Freeze label-blind CGR-vs-learned top1 recommendations; never run physics.

This compares the same previously geometry/arm-admitted pool, not unfiltered
end-to-end detection. Existing outcomes are opened only AFTER ranking files
are frozen and are descriptive replay evidence, not new physical validation.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent.parent
GESTURES=(1,6,7,8)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
class Reader:
 def __init__(self):self.files={}
 def freeze(self,path,expected=None):
  path=str(Path(path).resolve());actual=sha(path)
  if expected is not None and actual!=expected:raise ValueError('SHA mismatch: '+path)
  if path in self.files and self.files[path]!=actual:raise ValueError('Input changed: '+path)
  self.files[path]=actual;return actual
 def read(self,path,expected=None):
  self.freeze(path,expected);return json.loads(Path(path).read_text())
 def verify(self):
  for p,h in self.files.items():
   if sha(p)!=h:raise ValueError('Input changed during ranking: '+p)

def choose_top1(rows,score_key):
 """Pure rank; outcomes/labels are deliberately absent from this interface."""
 if score_key not in ('cgr_score','learned_score'):raise ValueError('Unknown ranking policy')
 if not rows:return None
 if any(not math.isfinite(float(r[score_key])) for r in rows):raise ValueError('Nonfinite score')
 return min(rows,key=lambda r:(-float(r[score_key]),r['queue_index']))

def features_for(binding,row):
 identity=binding['canonical_CGR480_identity'];a=np.asarray(binding['canonical_CGR480'],dtype=identity['dtype'])
 if a.shape!=(480,)or identity['shape']!=[480]or not np.isfinite(a).all()or hashlib.sha256(a.tobytes()).hexdigest()!=identity['c_order_bytes_sha256']:raise ValueError('Canonical480 shape/identity mismatch')
 raw=binding['two_fingers_pose_depth_type'];extra=binding['multifinger_pose_depth_type']
 if type(raw)is not int or not 0<=raw<5 or type(extra)is not int or not 0<=extra<4 or extra!=row['extra_depth_index']:raise ValueError('Original5rawdepth/4extra index mismatch')
 if binding['multifinger_pose_finger_type']!=row['gesture_id']or binding.get('canonicalization_applied_once')is not True:raise ValueError('Wrong gesture or missing canonical contract')
 source=np.asarray(binding['source_row15'],dtype=float)
 if source.shape!=(15,)or not np.isfinite(source).all():raise ValueError('Original raw candidate row15 required')
 return a.astype(np.float32),raw*4+extra,float(source[0])

def original_inference_class(source,torch):
 nodes=[n for n in ast.parse(Path(source).read_text()).body if isinstance(n,ast.ClassDef)and n.name=='MinkowskiGraspNetMultifingerType1Inference']
 if len(nodes)!=1:raise ValueError('Original inference class missing')
 scope={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'__name__':'official_boya_head_only_overlay'}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
 return scope[nodes[0].name]

def score_candidates(rows,features,training,training_path,dataset_sha,reader,device,allow_benchmark,train_meshes,validation_meshes):
 import torch
 if device=='cpu':torch.set_num_threads(2)
 if training.get('status')not in (('complete','benchmark_smoke_complete')if allow_benchmark else ('complete',)):raise ValueError('Completed fixed-final-epoch training required')
 if training.get('external_validation_evaluated')is not False or training.get('inputs_and_sources_unchanged')is not True:raise ValueError('External validation must be unused and training inputs unchanged')
 if training.get('benchmark_only')is True and not allow_benchmark:raise ValueError('Benchmark-only heads require --allow-benchmark-only')
 ref=training['dataset_manifest'];reader.freeze(ref['path'],ref['sha256'])
 if ref['sha256']!=dataset_sha:raise ValueError('Heads refer to another dataset manifest')
 heads=training['heads']
 if set(heads)!={str(g)for g in GESTURES}:raise ValueError('Exactly four final heads required; internal heads are never used')
 source=ROOT/'models/minkowski_graspnet_single_point.py';reader.freeze(source);cls=original_inference_class(source,torch);meta={}
 for g in GESTURES:
  entry=heads[str(g)];path=Path(entry['path']);path=path if path.is_absolute()else Path(training_path).parent/path;reader.freeze(path,entry['sha256']);checkpoint=torch.load(path,map_location='cpu',weights_only=True)
  if not isinstance(checkpoint,dict)or 'model_state_dict'not in checkpoint:raise ValueError('Pure dictionary model_state_dict checkpoint required')
  if checkpoint.get('gesture_id')!=g or checkpoint.get('input_dim')!=480 or checkpoint.get('output_dim')!=20 or checkpoint.get('dataset_manifest_sha256')!=dataset_sha:raise ValueError('Head metadata/shape/dataset mismatch')
  if not allow_benchmark and checkpoint.get('epoch')!=20:raise ValueError('Expected frozen final20epoch head, never validation-selected epoch')
  fit=set(checkpoint.get('fit_mesh_sha256',[]))
  if checkpoint.get('stage')!='all_train_meshes'or not fit or not fit<=train_meshes or fit&validation_meshes:raise ValueError('Final head fit groups include heldout or wrong stage')
  state=checkpoint['model_state_dict']
  if not isinstance(state,dict)or not all(isinstance(v,torch.Tensor)for v in state.values())or tuple(state['conv1.weight'].shape)!=(512,480,1)or tuple(state['conv7.weight'].shape)!=(20,512,1):raise ValueError('Original480to20 tensor state shape mismatch')
  model=cls(input_num=480,num_multifinger_type=1,num_multifinger_depth=4,two_finger_num_angle=1,two_finger_num_depth=5);model.load_state_dict(state,strict=True);model.to(device).eval();indices=[i for i,r in enumerate(rows)if r['gesture_id']==g]
  with torch.inference_mode():
   for start in range(0,len(indices),128):
    ids=indices[start:start+128];batch=torch.from_numpy(np.stack([features[i]for i in ids])).to(device);pred,_=model(batch)
    if tuple(pred.shape)!=(len(ids),1,20)or not torch.isfinite(pred).all():raise ValueError('Nonfinite/wrong original head output')
    scores=pred[:,0].cpu().numpy()
    for i,s in zip(ids,scores):rows[i]['learned_score']=float(s[rows[i]['target_index']]);rows[i]['all_head_outputs20']=s.tolist()
  meta[str(g)]=dict(path=str(path.resolve()),sha256=entry['sha256'],strict_load=True,input_dim=480,output_dim=20,epoch=checkpoint.get('epoch'));del model
 return meta

def existing_evidence(manifest,selection,reader):
 """Only called after selection/policy files have been written and hashed."""
 index={}
 for sample in manifest.get('samples',[]):
  tid=sample['trial_id']
  if tid in index:raise ValueError('Duplicate dataset sample identity')
  index[tid]=sample
 cache={}
 def outcome(selected):
  if selected is None:return dict(label=None,status='no_admitted_candidate')
  tid=selected['trial_id']
  if tid in cache:return cache[tid]
  s=index.get(tid)
  if not s:return dict(label=None,status='no_existing_eligible_record',trial_id=tid)
  p=Path(s['record_json']);record=reader.read(p,s['record_sha256']);b=record['binding'];art=record['execution_artifacts']
  if b['trial_binding_sha256']!=selected['trial_binding_sha256']or art['commands_sha256']!=selected['commands_sha256']:raise ValueError('Existing outcome belongs to different commands/binding')
  ev=reader.read(art['evaluation_json'],art['evaluation_sha256']);reader.freeze(art['commands_npz'],art['commands_sha256'])
  if record.get('label_eligible')is not True or type(record.get('result'))is not int or record['result']not in(0,1)or ev.get('complete')is not True or 'lift'not in ev.get('completed_phases',[]):raise ValueError('Dataset index contains an ineligible outcome')
  value=dict(trial_id=tid,label=record['result'],status='existing_bound_execution_only',record_json=str(p),record_sha256=s['record_sha256']);cache[tid]=value;return value
 pairs=[]
 for item in selection['objects']:
  base=outcome(item['baseline']);learned=outcome(item['learned']);pairs.append(dict(object_id=item['object_id'],same_plan=item['same_plan'],baseline=base,learned=learned,paired_labels_available=base['label']is not None and learned['label']is not None))
 paired=[x for x in pairs if x['paired_labels_available']]
 return dict(schema='anydex_boya_existing_top1_paired_evidence_v1',scope='Post-selection replay of already executed bound plans; not new physical top1 validation, not an oracle selector or full unfiltered deployment rate.',objects=pairs,paired_object_count=len(paired),baseline_positive=sum(x['baseline']['label']for x in paired),learned_positive=sum(x['learned']['label']for x in paired),new_physics_steps=0)

def run(args):
 began=time.monotonic();reader=Reader();dataset_sha=reader.freeze(args.dataset_manifest);training=reader.read(args.training_results);original=args.original_run.resolve();protocol=reader.read(original/'protocol.json');objects=reader.read(original/'objects.json')['objects'];qpath=original/'planning/queue_manifest.json';queue=reader.read(qpath);queue_sha=reader.files[str(qpath.resolve())]
 if protocol['gesture_ids']!=list(GESTURES)or len(queue['trials'])!=431 or queue.get('complete')is not True or queue.get('inputs_unchanged')is not True:raise ValueError('Frozen four-gesture431admitted source queue required')
 val=[o for o in objects if o['split']=='validation'];oids=[o['object_id']for o in val];valset=set(oids)
 if len(valset)!=10:raise ValueError('Original external10 object identities required')
 train_meshes={o['mesh_sha256']for o in objects if o['split']=='train'};validation_meshes={o['mesh_sha256']for o in val}
 if len(train_meshes)!=40 or len(validation_meshes)!=10 or train_meshes&validation_meshes:raise ValueError('Original source mesh groups overlap')
 for p,h in queue.get('source_sha256',{}).items():reader.freeze(p,h)
 rows=[];features=[];queue_rows={}
 for qi,row in enumerate(queue['trials']):
  if row['object_id']not in valset:continue
  if row.get('split')!='validation'or row['gesture_id']not in GESTURES:raise ValueError('Heldout queue split/type mismatch')
  binding=reader.read(row['binding_json'],row['binding_sha256']);body={k:v for k,v in binding.items()if k!='trial_binding_sha256'}
  if digest(body)!=binding['trial_binding_sha256']or binding['trial_binding_sha256']!=row['trial_binding_sha256']:raise ValueError('Frozen trial binding changed')
  for pk,hk in [('source_npz','source_npz_sha256'),('official_canonical_source','official_canonical_source_sha256')]:
   if str(Path(binding[pk]).resolve())not in reader.files:reader.freeze(binding[pk],binding[hk])
   elif reader.files[str(Path(binding[pk]).resolve())]!=binding[hk]:raise ValueError('Conflicting source input digest')
  feat,target,cgr=features_for(binding,row);planpath=Path(row['plan_dir'])/'plan.json';plan=reader.read(planpath)
  if not plan.get('complete')or not plan.get('admitted')or plan['commands_sha256']!=row['commands_sha256']:raise ValueError('Admitted original plan changed')
  clean=dict(queue_index=qi,trial_id=row['trial_id'],object_id=row['object_id'],gesture_id=row['gesture_id'],extra_depth_index=row['extra_depth_index'],raw_depth_index=binding['two_fingers_pose_depth_type'],target_index=target,source_CGR_row=binding['source_row_index'],if_flip=binding['if_flip'],cgr_score=cgr,plan_dir=row['plan_dir'],binding_json=row['binding_json'],binding_sha256=row['binding_sha256'],trial_binding_sha256=row['trial_binding_sha256'],commands_sha256=row['commands_sha256'],expected_command_steps=plan['total_command_steps']);rows.append(clean);features.append(feat);queue_rows[row['trial_id']]=row
 models=score_candidates(rows,features,training,args.training_results,dataset_sha,reader,args.device,args.allow_benchmark_only,train_meshes,validation_meshes)
 selection=dict(schema='anydex_boya_label_blind_top1_selection_v1',scope='Same original geometry/arm-admitted pool; external10; no outcome/label input to ranking.',tie_break='Lowest original ready-queue index, exact score ties',labels_accessed_during_ranking=False,dataset_sample_npz_files_read=0,score_threshold=None,objects=[])
 for oid in oids:
  candidates=[r for r in rows if r['object_id']==oid];b=choose_top1(candidates,'cgr_score');l=choose_top1(candidates,'learned_score');selection['objects'].append(dict(object_id=oid,admitted_count=len(candidates),baseline=copy.deepcopy(b),learned=copy.deepcopy(l),same_plan=(b['trial_id']==l['trial_id'])if b and l else None))
 out=args.output_dir.resolve()
 if out.exists():raise ValueError('Ranking output must be new, never overwritten')
 policies={}
 for method in ['baseline','learned']:
  chosen=[];nulls=[]
  for item in selection['objects']:
   r=item[method]
   if r is None:nulls.append(dict(trial_id=item['object_id']+'__'+method+'__no_admitted',object_id=item['object_id'],status='no_original_admitted_candidate',training_label=None,actual_physics_steps=0));continue
   reader.freeze(Path(r['plan_dir'])/'commands.npz',r['commands_sha256']);originalrow=copy.deepcopy(queue_rows[r['trial_id']]);originalrow['ranking_policy']=dict(name=method,score=r['cgr_score']if method=='baseline'else r['learned_score'],queue_index=r['queue_index'],target_index=r['target_index'],labels_accessed=False);chosen.append(originalrow)
  policy=copy.deepcopy(queue);policy.update(complete=True,trials=chosen,nulls=nulls,errors=[],inputs_unchanged=True,not_dispatched_trial_ids=[],benchmark_only=True,training_benchmark_only=bool(training.get('benchmark_only')),policy_name=method,learned_Boya_head_loaded=method=='learned',parent_queue_manifest=str(qpath),parent_queue_manifest_sha256=queue_sha,external_validation_object_ids=oids,labels_accessed_during_selection=False,requires_executor_benchmark_flag=True,scope='One original admitted top1 per external object; future policy trials must reset independently and stay outside training; this file itself executes nothing.');policies[method]=policy
 reader.verify();out.mkdir(parents=True);save(out/'scores.json',dict(schema='anydex_boya_four_head_scores_v1',candidates=rows,models=models,labels_accessed=False));save(out/'selection.json',selection)
 for method,policy in policies.items():save(out/'policy_manifests'/(method+'.json'),policy)
 frozen=[out/'scores.json',out/'selection.json',*(out/'policy_manifests'/(m+'.json')for m in policies)];freeze=dict(schema='anydex_boya_ranking_freeze_v1',outputs_sha256={str(p):sha(p)for p in frozen},ranking_input_sha256=copy.deepcopy(reader.files),ranking_source_sha256=sha(__file__),labels_accessed_during_ranking=False,record_or_label_files_opened_before_freeze=0,dataset_manifest_deserialized_before_freeze=False,new_physics_steps=0);save(out/'RANKING_FREEZE.json',freeze)
 # The manifest contains per-sample labels: deserialize it only after rank freeze.
 dataset=reader.read(args.dataset_manifest,dataset_sha);dsval={o['object_id']for o in dataset['objects']if o['split']=='external_validation'};dsm={o['object_id']:o['mesh_sha256']for o in dataset['objects']}
 if dsval!=valset or any(dsm[o['object_id']]!=o['mesh_sha256']for o in objects):raise ValueError('Dataset source split/mesh groups differ')
 if dataset['source_collection']['ready_queue_sha256']!=queue_sha or Path(dataset['source_collection']['original_run']).resolve()!=original:raise ValueError('Dataset belongs to another source queue/run')
 if not args.allow_benchmark_only and (dataset.get('benchmark_only')is not False or dataset.get('collection_complete')is not True):raise ValueError('Formal ranking requires complete nonbenchmark dataset')
 evidence=None
 if not args.skip_paired_evidence:evidence=existing_evidence(dataset,selection,reader);save(out/'paired_evidence.json',evidence)
 reader.verify();save(out/'RESULTS.json',dict(schema='anydex_boya_decision_ranking_results_v1',status='complete',benchmark_only=bool(training.get('benchmark_only')),dataset_manifest=dict(path=str(args.dataset_manifest.resolve()),sha256=dataset_sha),training_results=str(args.training_results.resolve()),original_run=str(original),models=models,admitted_validation_candidates=len(rows),external_objects=10,objects_with_admitted_candidates=sum(bool(x['admitted_count'])for x in selection['objects']),policy_manifests={m:str(out/'policy_manifests'/(m+'.json'))for m in policies},recommendation_count={m:len(policies[m]['trials'])for m in policies},same_plan_objects=sum(x['same_plan']is True for x in selection['objects']),selection_json=str(out/'selection.json'),scores_json=str(out/'scores.json'),ranking_freeze_sha256=sha(out/'RANKING_FREEZE.json'),paired_evidence=str(out/'paired_evidence.json')if evidence is not None else None,training_benchmark_only=bool(training.get('benchmark_only')),policy_trials_benchmark_only=True,labels_accessed_during_ranking=False,inputs_sha256=reader.files,cost=dict(new_physics_steps=0,new_CGR_forwards=0,new_training_steps=0,head_inference_only=True,wall_s=time.monotonic()-began)))
 print(json.dumps(dict(status='complete',output_dir=str(out),candidate_count=len(rows),policy_trial_count={m:len(policies[m]['trials'])for m in policies},new_physics_steps=0),indent=2))

def self_test():
 a=[dict(trial_id='positive',queue_index=1,cgr_score=.1,learned_score=.9,label=1),dict(trial_id='negative',queue_index=0,cgr_score=.8,learned_score=.2,label=0)]
 assert choose_top1(a,'cgr_score')['trial_id']=='negative'and choose_top1(a,'learned_score')['trial_id']=='positive'
 b=[dict(r,label=1-r['label'])for r in a];assert choose_top1(b,'cgr_score')['trial_id']=='negative'and choose_top1(b,'learned_score')['trial_id']=='positive'
 tied=[dict(r,cgr_score=.5,learned_score=.5)for r in a];assert choose_top1(tied,'cgr_score')['queue_index']==0 and choose_top1(list(reversed(tied)),'learned_score')['queue_index']==0
 assert choose_top1([],'cgr_score')is None
 try:choose_top1([dict(queue_index=0,cgr_score=float('nan'))],'cgr_score')
 except ValueError:pass
 else:raise AssertionError('NaN accepted')
 print(json.dumps(dict(status='passed',checks=['CGR rank independent of outcome','learned rank independent of outcome','label permutation leaves both selections unchanged','ties use original queue index independent of input order','empty pool null','nonfinite score rejected'],new_physics_steps=0,new_training_steps=0,torch_imported='torch'in sys.modules),indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-manifest',type=Path);p.add_argument('--training-results',type=Path);p.add_argument('--original-run',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0');p.add_argument('--skip-paired-evidence',action='store_true');p.add_argument('--allow-benchmark-only',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:self_test();return
 if any(getattr(a,k)is None for k in ['dataset_manifest','training_results','original_run','output_dir']):p.error('dataset-manifest, training-results, original-run, output-dir are required')
 run(a)
if __name__=='__main__':main()
