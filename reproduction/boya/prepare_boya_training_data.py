#!/usr/bin/env python3
"""Prepare source-bound canonical480/observed-index Boya training arrays.

Default: completed collection only. Explicit development snapshots are immutable,
benchmark-only engineering inputs. No training, physics or recanonicalization.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

GESTURES=(1,6,7,8)
SEED=20261106
SELECTION_SHA='487017083e79eb1323f243946ddf2b931dac6bd87d86f52632a254ba5226f60d'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def json_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write_json(path,value):
 path=Path(path)
 with path.open('x')as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def split_plan(objects):
 if len(objects)!=50 or Counter(o['split']for o in objects)!={'train':40,'validation':10}:raise ValueError('Expected existing40train/10validation')
 if len({o['object_id']for o in objects})!=50 or len({o['mesh_sha256']for o in objects})!=50:raise ValueError('Repeated source object or mesh group')
 train=sorted(o['mesh_sha256']for o in objects if o['split']=='train')
 # Fixed hash ordering avoids dependence on input record availability/outcomes.
 ranked=sorted(train,key=lambda group:(hashlib.sha256((str(SEED)+':'+group).encode()).hexdigest(),group))
 internal=set(ranked[:8]);mapping={};groups={s:[]for s in ['train','internal_check','external_validation']}
 for obj in objects:
  split='external_validation'if obj['split']=='validation'else'internal_check'if obj['mesh_sha256']in internal else'train'
  mapping[obj['object_id']]=dict(object_id=obj['object_id'],mesh_sha256=obj['mesh_sha256'],source_split=obj['split'],split=split)
  groups[split].append(obj['mesh_sha256'])
 for value in groups.values():value.sort()
 assert [len(groups[s])for s in ['train','internal_check','external_validation']]==[32,8,10]
 assert not(set(groups['train'])&set(groups['internal_check'])or set(groups['train'])&set(groups['external_validation'])or set(groups['internal_check'])&set(groups['external_validation']))
 return mapping,groups


def prepare(args):
 began=time.monotonic();run=args.collection_run.resolve();output=args.output_dir.resolve();repo=Path(__file__).resolve().parents[2]
 if(output/'manifest.json').exists():
  m=read(output/'manifest.json')
  if m['source_collection']['run']!=str(run)or m['benchmark_only']!=args.development_snapshot:raise ValueError('Existing immutable dataset belongs to another mode/collection')
  for path,digest in m['input_sha256'].items():
   if sha(path)!=digest:raise ValueError('Existing dataset input changed: '+path)
  for d in m['gesture_files'].values():
   if sha(d['path'])!=d['sha256']:raise ValueError('Existing dataset NPZ changed')
  print(json.dumps(dict(status='verified_reused',manifest=str(output/'manifest.json'),samples=m['sample_count'],benchmark_only=m['benchmark_only'])),flush=True);return m
 if output.exists()and any(output.iterdir()):raise ValueError('Partial/foreign dataset directory retained; use a new output')
 protocol_path=run/'PROTOCOL.json';resume=read(protocol_path);original=Path(resume['original_run']);parent_path=original/'protocol.json';protocol=read(parent_path)
 if resume['schema']!='anydex_boya_collection_execution_resume_v3'or protocol['schema']!='anydex_boya_collection_pipeline_v2':raise ValueError('Expected source-bound collection resume_v3')
 if list(resume['gesture_ids'])!=list(GESTURES)or list(protocol['gesture_ids'])!=list(GESTURES):raise ValueError('Expected declared four gesture families')
 files={};cache={}
 def bind(path,expected=None):
  path=str(Path(path).resolve());stat=Path(path).stat();stamp=(stat.st_size,stat.st_mtime_ns)
  if path not in cache or cache[path][0]!=stamp:cache[path]=(stamp,sha(path))
  digest=cache[path][1]
  if expected is not None and digest!=expected:raise ValueError('Bound source/artifact SHA changed: '+path)
  if path in files and files[path]!=digest:raise ValueError('Input changed during dataset preparation: '+path)
  files[path]=digest;return digest
 bind(__file__);bind(protocol_path);bind(parent_path,resume['original_protocol_sha256'])
 for path,digest in resume['original_refs_sha256'].items():bind(path,digest)
 for path,digest in resume['source_sha256'].items():bind(path,digest)
 selection_path=Path(protocol['config']['selection']);bind(selection_path,SELECTION_SHA);selection=read(selection_path)
 objects=read(original/'objects.json')['objects'];mapping,groups=split_plan(objects)
 if {o['object_id']for o in objects}!={o['object_id']for o in selection['objects']}:raise ValueError('Run changed original frozen selection')
 for obj in selection['objects']:
  if obj['mesh_sha256']!=mapping[obj['object_id']]['mesh_sha256']or obj['split']!=mapping[obj['object_id']]['source_split']:raise ValueError('Mesh/split differs from frozen selection')
 queue_path=original/'planning/queue_manifest.json';queue=read(queue_path);bind(queue_path)
 planning=read(original/'stages/planning.json');bind(original/'stages/planning.json')
 bind(queue_path,planning['artifacts_sha256'][str(queue_path)])
 slots={s['trial_id']:s for s in protocol['slots']};ready={r['trial_id']:r for r in queue['trials']}
 if len(slots)!=800 or len(ready)!=len(queue['trials'])or len(ready)+len(queue['nulls'])!=800:raise ValueError('Original finite slot/queue contract differs')
 final_path=run/'RESULTS.json';progress_bytes=None;exports={};source_export_files={};excluded=[]
 if not args.development_snapshot:
  if not final_path.exists():raise ValueError('Final collection RESULTS.json is not ready; use explicit --development-snapshot only for benchmark engineering')
  final=read(final_path)
  if final.get('complete')is not True or final.get('status')not in ['complete','complete_with_null']or final.get('unknown_execution_trial_ids')or final.get('unexecuted_ready_trial_ids'):raise ValueError('Final collection must be complete, with no unknown/unexecuted ready trials')
  bind(final_path);progress=read(run/'PROGRESS.json');bind(run/'PROGRESS.json')
  if final['protocol_sha256']!=bind(protocol_path):raise ValueError('Final protocol differs')
  actual_rows=final['trials']
  for split in ['train','validation']:
   path=run/(split+'_records.jsonl');source_export_files[split]=dict(path=str(path),sha256=bind(path))
   for line in path.read_text().splitlines():
    if not line.strip():continue
    record=json.loads(line);context=record['collection_context'];tid=context['trial_id']
    if tid in exports:raise ValueError('Duplicate original export trial')
    if context['split']!=split:raise ValueError('Export file contains wrong frozen split')
    exports[tid]=record
  if len(actual_rows)!=len(ready)or set(exports)!=set(ready):raise ValueError('Final exports do not cover exactly the ready trial IDs')
 else:
  progress_bytes=(run/'PROGRESS.json').read_bytes();progress=json.loads(progress_bytes)
  if progress['protocol_sha256']!=bind(protocol_path):raise ValueError('Snapshot continuation protocol differs')
  actual_rows=[row for obj in progress['objects'].values()if obj['status']in ['complete','inherited_complete']for row in obj.get('result',{}).get('trials',[])]
 final_complete=not args.development_snapshot
 locations={}
 for oid,obj in progress['objects'].items():
  if obj['status']in ['complete','inherited_complete']:
   if obj.get('result',{}).get('input_sha256_unchanged')is not True or obj.get('result',{}).get('source_sha256_unchanged')is not True:raise ValueError('Completed object source binding not confirmed')
   locations[oid]=obj
 samples=[];vectors={g:[]for g in GESTURES};seen=set();collection_ids=set()
 helper_path=repo/'reproduction/boya/trial_record.py';spec=importlib.util.spec_from_file_location('boya_training_bound_record',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 # Preserve original collect_record validation while hashing each shared raw
 # source only once; all bound files are rehashed again before publication.
 helper.file_sha=bind;bind(helper_path)
 for row in actual_rows:
  tid=row['trial_id']
  if tid in seen:raise ValueError('Repeated actual trial ID')
  seen.add(tid)
  if tid not in ready or tid not in slots:raise ValueError('Actual record not in this collection ready queue')
  if row.get('label')not in [0,1]or row.get('label_eligible')is not True or row.get('physics_steps',0)<=0:
   if final_complete:raise ValueError('Unknown/null actual record cannot enter final dataset')
   excluded.append(dict(trial_id=tid,reason='unknown_or_ineligible_actual_not_a_negative_label'));continue
  slot=slots[tid];obj=mapping[slot['object_id']]
  if row['object_id']!=slot['object_id']or row['gesture_id']!=slot['gesture_id']or row['extra_depth_index']!=slot['extra_depth_index']:raise ValueError('Actual row differs from frozen slot')
  if row['object_id']not in locations:raise ValueError('Actual object lacks proven complete location')
  location=locations[row['object_id']];record_path=Path(location['output_dir'])/'trials'/row.get('output_identity',tid)/'training_record.json';record=read(record_path);record_sha=bind(record_path)
  if record.get('benchmark_only')is True:raise ValueError('Earlier benchmark/development record cannot enter this collection')
  binding=record['binding'];artifact=record['execution_artifacts'];evaluation_path=Path(artifact['evaluation_json']);commands_path=Path(artifact['commands_npz'])
  bind(evaluation_path,artifact['evaluation_sha256']);bind(commands_path,artifact['commands_sha256'])
  expected=helper.collect_record(binding,evaluation_path,commands_path)
  if any(record.get(k)!=value for k,value in expected.items()):raise ValueError('Source record differs from original collect_record: '+tid)
  evaluation=read(evaluation_path);plan_path=Path(row['source_plan_dir'])/'plan.json';plan=read(plan_path);bind(plan_path)
  if binding['trial_binding_sha256']!=row['trial_binding_sha256']or record['result']!=row['label']or not record['label_eligible']:raise ValueError('Actual outcome/binding mismatch')
  if evaluation.get('commands_sha256')!=row['commands_sha256']or bind(commands_path)!=row['commands_sha256']or bind(Path(row['source_plan_dir'])/'commands.npz')!=row['commands_sha256']:raise ValueError('Original command bytes differ')
  steps=plan['total_command_steps']
  if record['total_physics_steps']!=steps or row['physics_steps']!=steps or evaluation['total_physics_steps']!=steps:raise ValueError('Incomplete native adaptive command sequence')
  with np.load(commands_path,allow_pickle=False)as command:
   if len(command['phase'])!=steps or not{'close','hold','lift'}.issubset(set(map(str,command['phase']))):raise ValueError('Incomplete command phases')
  if row.get('guards')and not all(row['guards'].values()):raise ValueError('Actual execution guard failed')
  gesture=binding['multifinger_pose_finger_type'];raw_depth=binding['two_fingers_pose_depth_type'];extra=binding['multifinger_pose_depth_type']
  if type(raw_depth)is not int or not 0<=raw_depth<5 or type(extra)is not int or not 0<=extra<4 or gesture!=slot['gesture_id']or extra!=slot['extra_depth_index']:raise ValueError('Invalid original raw/extra depth indices')
  feature=np.asarray(binding['canonical_CGR480'],dtype=np.float32)
  if feature.shape!=(480,)or not np.isfinite(feature).all()or helper.array_ref(feature)!=binding['canonical_CGR480_identity']or binding['canonicalization_applied_once']is not True:raise ValueError('Canonical480 must reuse exact original float32 bytes once')
  if tid in exports:
   exported=copy.deepcopy(exports[tid]);context=exported.pop('collection_context')
   if exported!=record or any(context[k]!=slot[k]for k in ['trial_id','object_id','split','gesture_id','extra_depth_index']):raise ValueError('Final JSONL export differs from actual source record/frozen split')
   if context['protocol_sha256']!=resume['original_protocol_sha256']:raise ValueError('Export belongs to another original collection')
  meta=dict(trial_id=tid,object_id=slot['object_id'],gesture_id=gesture,mesh_sha256=obj['mesh_sha256'],source_split=slot['split'],split=obj['split'],
   target_index=raw_depth*4+extra,raw_depth_type=raw_depth,extra_depth_index=extra,cgr_score=float(binding['source_row15'][0]),
   record_json=str(record_path),record_sha256=record_sha,evaluation_json=str(evaluation_path),evaluation_sha256=artifact['evaluation_sha256'],
   commands_npz=str(commands_path),commands_sha256=artifact['commands_sha256'],trial_binding_sha256=binding['trial_binding_sha256'],
   source_candidate_npz=binding['source_npz'],source_candidate_sha256=binding['source_npz_sha256'],source_row_index=binding['source_row_index'],if_flip=binding['if_flip'],label=record['result'],physics_steps=steps)
  samples.append(meta);vectors[gesture].append((feature,meta));collection_ids.add(tid)
 if not samples:raise ValueError('No proven eligible actual records for this snapshot/final dataset')
 if final_complete and collection_ids!=set(ready):raise ValueError('Final eligible records do not reconcile with all ready slots')
 if args.development_snapshot:
  pending=sorted(set(ready)-seen)
 else:pending=[]
 # Recheck bound artifacts without rereading unrelated public meshes or weights.
 for path,digest in files.items():
  if sha(path)!=digest:raise ValueError('Bound dataset input changed before publication: '+path)
 output.mkdir(parents=True,exist_ok=True)
 if progress_bytes is not None:
  snapshot=output/'source_snapshots';snapshot.mkdir();f=snapshot/'PROGRESS.json';f.write_bytes(progress_bytes);files[str(f)]=sha(f)
 gesture_files={};allcounts=Counter();heads=[]
 for g in GESTURES:
  values=sorted(vectors[g],key=lambda item:item[1]['trial_id']);meta=[x[1]for x in values];features=np.stack([x[0]for x in values])if values else np.empty((0,480),dtype=np.float32)
  path=output/('gesture_'+str(g)+'.npz')
  with path.open('xb')as f:np.savez_compressed(f,features=features,labels=np.asarray([x['label']for x in meta],dtype=np.int64),
   target_index=np.asarray([x['target_index']for x in meta],dtype=np.int64),gesture_ids=np.full(len(meta),g,dtype=np.int64),
   mesh_sha256=np.asarray([x['mesh_sha256']for x in meta],dtype='<U64'),trial_id=np.asarray([x['trial_id']for x in meta],dtype='<U160'),
   object_id=np.asarray([x['object_id']for x in meta],dtype='<U128'),split=np.asarray([x['split']for x in meta],dtype='<U24'),cgr_scores=np.asarray([x['cgr_score']for x in meta],dtype=np.float32))
  counts={s:dict(count=sum(x['split']==s for x in meta),positive=sum(x['split']==s and x['label']==1 for x in meta),negative=sum(x['split']==s and x['label']==0 for x in meta))for s in groups}
  gesture_files[str(g)]=dict(path=str(path),sha256=sha(path),count=len(meta),split_counts=counts,feature_dim=480,output_dim=20)
  for i,x in enumerate(meta):x['npz']=str(path);x['npz_row']=i
  for x in meta:allcounts[x['split']]+=1
 source_collection=dict(run=str(run),protocol=str(protocol_path),protocol_sha256=sha(protocol_path),original_run=str(original),original_protocol=str(parent_path),original_protocol_sha256=sha(parent_path),
  selection=str(selection_path),selection_sha256=SELECTION_SHA,ready_queue=str(queue_path),ready_queue_sha256=sha(queue_path),
  final_results=str(final_path)if final_complete else None,final_results_sha256=sha(final_path)if final_complete else None,
  source_export_files=source_export_files,progress_snapshot=str(output/'source_snapshots/PROGRESS.json')if progress_bytes is not None else None)
 manifest=dict(schema='anydex_boya_training_dataset_v1',collection_run=str(run),all_records_jsonl=str(run/'all_records.jsonl')if final_complete else None,status='prepared',complete=True,collection_complete=final_complete,benchmark_only=bool(args.development_snapshot),development_snapshot=bool(args.development_snapshot),
  gesture_ids=list(GESTURES),feature_dim=480,output_dim=20,sample_count=len(samples),split_counts=dict(allcounts),gesture_files=gesture_files,
  supervision=dict(feature_source='Exact source record.binding.canonical_CGR480 float32; original collect_record replay validates source commands/evaluation only. No second canonicalization.',
   target_index='two_fingers_pose_depth_type*4+multifinger_pose_depth_type',one_observed_index_per_actual_trial=True,unobserved_19_outputs_are_unlabelled=True,no_first100_truncation=True),
  internal_split=dict(seed=SEED,method='Sort the 40 frozen train mesh SHA groups by SHA256(str(seed)+colon+mesh_sha256), first8 internal_check. Independent of available records, labels and outcomes.',train_groups=32,internal_check_groups=8,external_validation_groups=10),
  mesh_groups=groups,objects=[mapping[k]for k in sorted(mapping)],source_collection=source_collection,input_sha256=files,
  excluded=dict(planning_zero_step_null_count=len(queue['nulls']),unknown_or_ineligible_actuals=excluded,pending_ready_trial_count=len(pending),pending_ready_trial_ids=pending,
   earlier_development105_included=False,source_records_outside_this_frozen_collection_allowed=False),
  samples=sorted(samples,key=lambda s:(s['gesture_id'],s['trial_id'])),
  dataset_content_sha256=json_sha(dict(files={k:v['sha256']for k,v in gesture_files.items()},mesh_groups=groups,trial_ids=sorted(collection_ids))),
  source_sha256={str(Path(__file__).resolve()):sha(__file__),str(helper_path):sha(helper_path)},
  cost=dict(new_physics_steps=0,new_network_forwards=0,new_training_steps=0,preparation_wall_s=time.monotonic()-began),
  boundaries=['Observed actual trials only: null, incomplete, unknown and repeated identities never become negative labels.',
   'Fixed mesh-disjoint32/8/10 groups include groups without labels. Original external10 validation is not used for hyperparameter, epoch or gesture choice.',
   'Internal-check data may be used only under a separately frozen training recipe. Final full40 refit must remain separate from internal-check evidence.',
   'Development snapshots are engineering smoke inputs, not final training data or a final model. Original old105 development outcomes are excluded.',
   'No materialized 20-label expansion, observation/CGR recomputation, physical trial or source-asset modification.'])
 write_json(output/'manifest.json',manifest)
 print(json.dumps(dict(status='prepared',manifest=str(output/'manifest.json'),manifest_sha256=sha(output/'manifest.json'),sample_count=len(samples),benchmark_only=manifest['benchmark_only'],split_counts=manifest['split_counts'],gesture_counts={k:v['count']for k,v in gesture_files.items()}),indent=2),flush=True)
 return manifest


def main():
 repo=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--collection-run',type=Path,default=repo/'outputs/collection40_20261106_v2/execution_resume_v3')
 p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--development-snapshot',action='store_true')
 args=p.parse_args()
 lock_path=Path('/tmp')/('anydex_training_data_'+hashlib.sha256(str(args.output_dir.resolve()).encode()).hexdigest()+'.lock')
 with lock_path.open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prepare(args)


if __name__=='__main__':main()
