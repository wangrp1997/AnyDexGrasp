#!/usr/bin/env python3
"""Prepare source-bound canonical480/observed-index Boya training arrays.

Default: completed v4 collection only, with every ready slot either eligible or
independently certified terminal-null. Nulls never become negative labels. Explicit
development snapshots are immutable engineering inputs. No new physics/training.
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
import sys

import numpy as np

GESTURES=(1,6,7,8)
SEED=20261106
SELECTION_SHA='487017083e79eb1323f243946ddf2b931dac6bd87d86f52632a254ba5226f60d'
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'reproduction/boya/trial_record.py').is_file()),Path('/home/wangrenpeng/AnyDexGrasp'))


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


def eligible_row(row):
 return type(row.get('label'))is int and row['label']in (0,1)and row.get('label_eligible')is True and type(row.get('physics_steps'))is int and row['physics_steps']>0


def terminal_certificate_index(final):
 """Index only an explicit per-trial certificate mapping; never infer null proof."""
 raw=final.get('terminal_certificates')
 if raw is None:raise ValueError('Completed v4 must declare terminal_certificates, even when empty')
 if not isinstance(raw,dict):raise ValueError('terminal_certificates must be an explicit trial-ID mapping')
 out={}
 for tid,certificate in raw.items():
  if not isinstance(tid,str)or not isinstance(certificate,dict):raise ValueError('Malformed terminal-null certificate map')
  if certificate.get('trial_id')!=tid:raise ValueError('Certificate trial-ID differs from map key')
  out[tid]=certificate
 if final.get('terminal_null_count')!=len(out):raise ValueError('Declared terminal-null count differs from certificates')
 summary=final.get('terminal_nulls')
 if not isinstance(summary,list)or len(summary)!=len(out):raise ValueError('Terminal-null summary must account exactly for certificate map')
 byid={}
 for row in summary:
  tid=row['trial_id']
  if tid in byid or tid not in out:raise ValueError('Repeated/foreign terminal-null summary')
  expected={key:out[tid][key]for key in ('trial_id','reason','physics_steps','label','label_eligible')}
  if row!=expected:raise ValueError('Terminal-null summary differs from actual certificate')
  byid[tid]=row
 return out


def reconcile_final_ids(actual_rows,ready,exports,certificates):
 """A terminal run accounts for slots, not necessarily one label per slot."""
 if not isinstance(actual_rows,list):raise ValueError('Final raw trials must be an explicit list')
 actual={}
 for row in actual_rows:
  tid=row['trial_id']
  if tid in actual or tid not in ready:raise ValueError('Repeated/foreign final actual trial')
  for key in ('object_id','gesture_id','extra_depth_index'):
   if row[key]!=ready[tid][key]:raise ValueError('Final trial identity differs from original ready queue')
  actual[tid]=row
 if set(actual)!=set(ready):raise ValueError('Final raw executed trials do not cover all original ready IDs')
 eligible={tid for tid,row in actual.items()if eligible_row(row)}
 nulls=set(actual)-eligible
 if set(certificates)!=nulls:raise ValueError('Every ineligible actual needs exactly one explicit terminal-null certificate; no eligible trial may be certified null')
 if set(exports)!=eligible:raise ValueError('Final exports must cover exactly eligible original ready IDs, excluding every certified null')
 for tid in nulls:
  row=actual[tid]
  if row.get('label')is not None or row.get('label_eligible')is not False or type(row.get('physics_steps'))is not int or row['physics_steps']<=0:raise ValueError('Only explicitly null, positive-step outcomes can be certified terminal here')
 if eligible&nulls or eligible|nulls!=set(ready):raise ValueError('Final eligible/null slot partition differs')
 return eligible,nulls


_TERMINAL_HELPERS={}


def validate_terminal_null(row,certificate,ready_row,slot,resume,repo,bind):
 """Recompute the runtime certificate from original saved evidence, byte-bound."""
 for key in ('trial_id','object_id','gesture_id','extra_depth_index'):
  if row[key]!=ready_row[key]or row[key]!=slot[key]:raise ValueError('Terminal-null row differs from frozen queue/slot')
 if row.get('benchmark_only')is True:raise ValueError('Development task-stop outcome cannot enter final collection ledger')
 if certificate.get('schema')!='anydex_native_task_stop_certificate_v1'or certificate.get('certified')is not True or certificate.get('outcome')!='known_task_terminated_null':raise ValueError('Unrecognised terminal-null certificate')
 helper_path=repo/'reproduction/boya/terminal_collection_outcome.py';digest=bind(helper_path)
 key=(str(helper_path),digest)
 if key not in _TERMINAL_HELPERS:
  # Loading the NumPy-only helper imports no physics backend or model.
  spec=importlib.util.spec_from_file_location('boya_training_terminal_verifier',helper_path)
  helper=importlib.util.module_from_spec(spec)
  previous=list(sys.path)
  try:
   sys.path.insert(0,str(helper_path.parent));spec.loader.exec_module(helper)
  finally:sys.path[:]=previous
  _TERMINAL_HELPERS[key]=helper
 helper=_TERMINAL_HELPERS[key]
 helper.sha=bind
 reproduced=helper.certify_known_task_stop(row)
 if reproduced!=certificate:raise ValueError('Saved terminal-null certificate differs from independent runtime helper replay: '+row['trial_id'])
 for section in ('evidence_sha256','native_source_sha256'):
  if not isinstance(certificate.get(section),dict)or not certificate[section]:raise ValueError('Certificate lacks bound evidence/source hashes')
  for path,digest in certificate[section].items():bind(path,digest)
 return dict(trial_id=row['trial_id'],reason=certificate['reason'],physics_steps=certificate['physics_steps'],label=None,label_eligible=False,
             certificate_sha256=json_sha(certificate),helper_sha256=bind(helper_path),original_evidence_recomputed=True)


def prepare(args):
 began=time.monotonic();run=args.collection_run.resolve();output=args.output_dir.resolve();repo=ROOT
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
 if resume['schema']not in ['anydex_boya_collection_execution_resume_v3','anydex_boya_collection_execution_resume_v4']or protocol['schema']!='anydex_boya_collection_pipeline_v2':raise ValueError('Expected source-bound collection resume_v3/v4')
 if not args.development_snapshot and resume['schema']!='anydex_boya_collection_execution_resume_v4':raise ValueError('Formal v2 preparation requires terminal-certified resume_v4; stopped v3 is development-snapshot only')
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
 final_path=run/'RESULTS.json';progress_bytes=None;exports={};source_export_files={};excluded=[];terminal_certificates={};terminal_checks={}
 if not args.development_snapshot:
  if not final_path.exists():raise ValueError('Final collection RESULTS.json is not ready; use explicit --development-snapshot only for benchmark engineering')
  final=read(final_path)
  if final.get('complete')is not True or final.get('status')not in ['complete','complete_with_null']or final.get('unknown_execution_trial_ids')or final.get('unexecuted_ready_trial_ids')or final.get('uncertified_task_trial_ids'):raise ValueError('Final collection must be complete, with no unknown/unexecuted ready trials')
  if final.get('source_sha256_unchanged')is not True or final.get('input_sha256_unchanged')is not True:raise ValueError('Final v4 source/input binding not confirmed')
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
  terminal_certificates=terminal_certificate_index(final)
  eligible_ids,null_ids=reconcile_final_ids(actual_rows,ready,exports,terminal_certificates)
  if final.get('eligible_labels')!=len(eligible_ids)or final.get('physical_steps')!=sum(row['physics_steps']for row in actual_rows)or final.get('preparation_nulls')!=len(queue['nulls']):raise ValueError('Final counts/costs do not reconcile with actual ready-slot ledger')
  all_path=run/'all_records.jsonl';source_export_files['all_records']=dict(path=str(all_path),sha256=bind(all_path))
  all_exports={}
  for line in all_path.read_text().splitlines():
   if not line.strip():continue
   value=json.loads(line);tid=value['collection_context']['trial_id']
   if tid in all_exports:raise ValueError('Duplicate actual in mixed export')
   all_exports[tid]=value
  if all_exports!=exports:raise ValueError('Mixed all_records export differs from train/validation partition')
 else:
  progress_bytes=(run/'PROGRESS.json').read_bytes();progress=json.loads(progress_bytes)
  if progress['protocol_sha256']!=bind(protocol_path):raise ValueError('Snapshot continuation protocol differs')
  actual_rows=[row for obj in progress['objects'].values()if obj['status']in ['complete','inherited_complete']for row in obj.get('result',{}).get('trials',[])]
 final_complete=not args.development_snapshot
 locations={}
 for oid,obj in progress['objects'].items():
  if obj['status']in ['complete','inherited_complete']or (final_complete and obj.get('result',{}).get('input_sha256_unchanged')is True and obj.get('result',{}).get('source_sha256_unchanged')is True):
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
  if not eligible_row(row):
   if final_complete:
    terminal_checks[tid]=validate_terminal_null(row,terminal_certificates[tid],ready[tid],slots[tid],resume,repo,bind)
    excluded.append(dict(trial_id=tid,reason='certified_terminal_null_not_a_negative_label',certificate=terminal_checks[tid]));continue
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
 if final_complete and (collection_ids!=set(exports)or collection_ids&set(terminal_certificates)or collection_ids|set(terminal_certificates)!=set(ready)):raise ValueError('Final ready slots must reconcile exactly as eligible disjoint-union certified terminal-null')
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
   certified_terminal_null_count=len(terminal_checks),certified_terminal_null_trial_ids=sorted(terminal_checks),terminal_nulls_never_imputed_as_zero=True,
   earlier_development105_included=False,source_records_outside_this_frozen_collection_allowed=False),
  samples=sorted(samples,key=lambda s:(s['gesture_id'],s['trial_id'])),
  dataset_content_sha256=json_sha(dict(files={k:v['sha256']for k,v in gesture_files.items()},mesh_groups=groups,trial_ids=sorted(collection_ids))),
  source_sha256={str(Path(__file__).resolve()):sha(__file__),str(helper_path):sha(helper_path),**({str(repo/'reproduction/boya/terminal_collection_outcome.py'):bind(repo/'reproduction/boya/terminal_collection_outcome.py')}if terminal_checks else{})},
  final_ledger=dict(ready=len(ready),eligible=len(collection_ids),certified_terminal_null=len(terminal_checks),planning_null=len(queue['nulls']),pending=len(pending),equation='ready = eligible + certified terminal null; disjoint exact identity sets',verified=final_complete),
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
 repo=ROOT;p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--collection-run',type=Path,default=repo/'outputs/collection40_20261106_v2/execution_resume_v4')
 p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--development-snapshot',action='store_true')
 args=p.parse_args()
 lock_path=Path('/tmp')/('anydex_training_data_'+hashlib.sha256(str(args.output_dir.resolve()).encode()).hexdigest()+'.lock')
 with lock_path.open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prepare(args)


if __name__=='__main__':main()
