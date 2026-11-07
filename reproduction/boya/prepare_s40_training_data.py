"""Export completed source-bound S40 Boya labels without training or physics.

The original40 objects are outer training objects. The protocol's predeclared
32/8 mesh split is retained; nulls never become negative examples. Canonical480
bytes are reused once and independently checked against original paired CGR.
"""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import numpy as np

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'reproduction/boya/trial_record.py').is_file()),Path('/home/wangrenpeng/AnyDexGrasp'))
GESTURES=(1,6,7,8)
S40_IDS=frozenset((0,2,5,7,8,9,11,14,15,17,18,20,21,22,26,27,29,30,34,36,37,38,40,41,43,44,46,48,51,52,56,57,58,60,61,62,63,66,69,70))

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def json_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def digest(value):
 require(isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None,'Invalid source mesh SHA256')
 return value

def split_plan(objects,declared):
 require(isinstance(objects,list)and len(objects)==40,'Complete S40 original40 object manifest required')
 require({o.get('graspnet_object_id')for o in objects}==S40_IDS,'Original source S40 identity union differs')
 require(all(o.get('split')=='train'and o.get('scale')==1 for o in objects),'Only native-scale original40 outer training objects allowed')
 byid={o['object_id']:o for o in objects};require(len(byid)==40,'Repeated source object identity')
 require(set(declared)=={'fit','internal_check'},'Predeclared protocol 32/8 mesh groups required')
 mapping={};groups={'train':[],'internal_check':[]}
 for source,split,count in [('fit','train',32),('internal_check','internal_check',8)]:
  rows=declared[source];require(isinstance(rows,list)and len(rows)==count,'Wrong frozen mesh group size')
  for row in rows:
   oid=row['object_id'];mesh=digest(row['mesh_sha256'])
   require(oid in byid and oid not in mapping,'Repeated or foreign declared object')
   require(mesh==byid[oid]['source_visual_mesh_sha256'],'Protocol split differs from original source visual mesh SHA')
   mapping[oid]=dict(object_id=oid,mesh_sha256=mesh,outer_split='train',source_split='train',split=split,
                    source_visual_mesh_sha256=mesh,graspnet_object_id=byid[oid]['graspnet_object_id'])
   groups[split].append(mesh)
 require(set(mapping)==set(byid),'Frozen mesh split must account for all40 objects')
 require(len(set(groups['train']+groups['internal_check']))==40,'Repeated mesh or mesh leakage across internal split')
 for group in groups.values():group.sort()
 return mapping,groups

def eligible_row(row):
 return type(row.get('label'))is int and row['label']in(0,1)and row.get('label_eligible')is True and type(row.get('physics_steps'))is int and row['physics_steps']>0

def unique_rows(rows,label):
 result={}
 for row in rows:
  tid=row['trial_id'];require(tid not in result,'Repeated '+label+' trial identity');result[tid]=row
 return result

def reconcile(protocol,queue,final,exports):
 require(protocol.get('schema')=='anydex_boya_s40_collection_v1','S40 protocol required')
 require(protocol.get('slot_count')==640 and list(protocol.get('gesture_ids',[]))==list(GESTURES)and protocol.get('extra_depth_indices')==[0,1,2,3],'Frozen 40x4x4 slot budget required')
 require(final.get('schema')=='anydex_boya_s40_collection_results_v1'and final.get('complete')is True and final.get('status')in('complete','complete_with_null'),'Complete final S40 collection required')
 require(final.get('input_sha256_unchanged')is True and final.get('source_sha256_unchanged')is True and final.get('physical_cost_complete')is True,'Final input/source/cost evidence required')
 require(not final.get('unexecuted_ready_trial_ids')and not final.get('unknown_execution_trial_ids'),'Unknown/unexecuted slots cannot enter formal dataset')
 require(queue.get('complete')is True and not queue.get('errors')and queue.get('inputs_unchanged')is True,'Complete error-free frozen ready queue required')
 slots=unique_rows(protocol['slots'],'slot');ready=unique_rows(queue['trials'],'ready');null=unique_rows(queue['nulls'],'preparation null');actual=unique_rows(final['trials'],'actual')
 require(len(slots)==640 and set(ready).isdisjoint(null)and set(ready)|set(null)==set(slots),'Ready/null slots do not partition frozen640')
 require(set(actual)==set(ready),'Final actual rows must cover exactly all ready slots')
 for tid,row in {**ready,**null}.items():
  require(all(row[k]==slots[tid][k]for k in('object_id','gesture_id')),'Queue identity differs from frozen slot')
 for tid,row in actual.items():
  require(all(row[k]==slots[tid][k]for k in('object_id','gesture_id','extra_depth_index')),'Actual identity differs from frozen slot')
 eligible={tid for tid,row in actual.items()if eligible_row(row)};terminal=set(actual)-eligible;cert=final.get('terminal_certificates')
 require(isinstance(cert,dict)and set(cert)==terminal and final.get('terminal_nulls')==len(terminal),'Terminal null certificate identity partition differs')
 require(set(exports)==eligible,'Exports must cover exactly eligible rows; no null imputation')
 require(final.get('eligible_labels')==len(eligible)and final.get('actual_grasps')==len(actual),'Final label/actual count mismatch')
 require(final.get('positives')==sum(actual[t]['label']==1 for t in eligible)and final.get('negatives')==sum(actual[t]['label']==0 for t in eligible),'Final positive/negative count mismatch')
 require(final.get('physical_steps')==sum(r['physics_steps']for r in actual.values()),'Final physical cost mismatch')
 require(unique_rows(final['preparation_nulls'],'final preparation null')==null,'Final preparation-null ledger differs from frozen queue')
 for tid in terminal:
  row=actual[tid];c=cert[tid]
  require(row.get('label')is None and row.get('label_eligible')is False and row.get('physics_steps',0)>0,'Only explicit positive-step terminal null allowed')
  require(c.get('schema')=='anydex_native_task_stop_certificate_v1'and c.get('certified')is True and c.get('outcome')=='known_task_terminated_null'and c.get('trial_id')==tid,'Invalid terminal-null certificate')
 return slots,ready,actual,eligible,terminal

def import_helper(path,name):
 spec=importlib.util.spec_from_file_location(name,path);helper=importlib.util.module_from_spec(spec)
 previous=list(sys.path)
 try:sys.path.insert(0,str(Path(path).parent));spec.loader.exec_module(helper)
 finally:sys.path[:]=previous
 return helper

class FrozenFiles:
 def __init__(self):self.files={};self.cache={}
 def bind(self,path,expected=None):
  path=str(Path(path).resolve());s=Path(path).stat();stamp=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  if path not in self.cache or self.cache[path][0]!=stamp:self.cache[path]=(stamp,sha(path))
  value=self.cache[path][1]
  require(expected is None or value==expected,'Frozen artifact/source SHA mismatch: '+path)
  require(path not in self.files or self.files[path]==value,'Input changed during export: '+path)
  self.files[path]=value;return value
 def verify(self):
  for path,expected in self.files.items():require(sha(path)==expected,'Frozen input changed before publication: '+path)

class CanonicalValidator:
 def __init__(self,helper,bind):self.helper=helper;self.bind=bind;self.sources=OrderedDict();self.functions={}
 def feature(self,binding):
  helper=self.helper;source=str(Path(binding['source_npz']).resolve());self.bind(source,binding['source_npz_sha256'])
  official=str(Path(binding['official_canonical_source']).resolve());self.bind(official,binding['official_canonical_source_sha256'])
  if source not in self.sources:
   with np.load(source,allow_pickle=False)as z:self.sources[source]=(z['candidates'].copy(),z['candidate_features'].copy())
   if len(self.sources)>2:self.sources.popitem(last=False)
  self.sources.move_to_end(source);rows,raw=self.sources[source];index=binding['source_row_index']
  require(type(index)is int and 0<=index<len(rows)and rows.shape[1:]==(15,)and raw.shape==(len(rows),2261),'Invalid original paired CGR source row')
  row,features=rows[index],raw[index]
  require(helper.array_ref(row)==binding['source_row15_identity']and helper.array_ref(features)==binding['candidate_features_raw2261_identity'],'Original raw source identity differs')
  require(np.array_equal(row,np.asarray(binding['source_row15'],dtype=row.dtype))and np.array_equal(features,np.asarray(binding['candidate_features_raw2261'],dtype=features.dtype)),'Raw candidate feature values differ')
  if official not in self.functions:self.functions[official]=helper.official_functions(official)
  info=helper.canonical_candidate(row,features,self.functions[official],if_flip=binding['if_flip'])
  feature=np.asarray(binding['canonical_CGR480'],dtype=np.float32)
  require(feature.shape==(480,)and np.isfinite(feature).all()and helper.array_ref(feature)==binding['canonical_CGR480_identity'],'Canonical480 original bytes differ')
  require(helper.array_ref(info['canonical_CGR480'])==binding['canonical_CGR480_identity']and np.array_equal(info['canonical_CGR480'],feature)and binding['canonicalization_applied_once']is True,'Canonical480 differs from original AST function output bytes')
  for key in('two_fingers_pose_angle_type','two_fingers_pose_depth_type','if_flip','official_policy_if_flip'):
   require(binding[key]==info[key],'Original canonicalization index/flip mismatch')
  require(np.array_equal(info['candidate_gg17_after_flip'][4:16],np.asarray(binding['executed_camera_cgr_after_flip'])),'Executed flipped grasp differs from original pairing')
  return feature

def observed_target(binding,slot):
 g=binding['multifinger_pose_finger_type'];raw=binding['two_fingers_pose_depth_type'];extra=binding['multifinger_pose_depth_type']
 require(type(g)is int and g in GESTURES and g==slot['gesture_id']and extra==slot['extra_depth_index']and type(raw)is int and 0<=raw<5 and type(extra)is int and 0<=extra<4,'Gesture/depth target differs from frozen actual slot')
 return g,raw,extra,raw*4+extra

def coverage(samples,gesture):
 rows=[r for r in samples if r['gesture_id']==gesture]
 out=dict(count=len(rows),positive=sum(r['label']==1 for r in rows),negative=sum(r['label']==0 for r in rows),
  labelled_mesh_groups=len({r['mesh_sha256']for r in rows}),labelled_objects=len({r['object_id']for r in rows}),
  target_coverage={str(i):dict(count=sum(r['target_index']==i for r in rows),positive=sum(r['target_index']==i and r['label']==1 for r in rows),negative=sum(r['target_index']==i and r['label']==0 for r in rows))for i in range(20)},
  split_counts={s:dict(count=sum(r['split']==s for r in rows),positive=sum(r['split']==s and r['label']==1 for r in rows),negative=sum(r['split']==s and r['label']==0 for r in rows),labelled_mesh_groups=len({r['mesh_sha256']for r in rows if r['split']==s}))for s in('train','internal_check')})
 out['unobserved_targets']=[int(i)for i,v in out['target_coverage'].items()if v['count']==0]
 return out

def prepare(args):
 started=time.monotonic();run=Path(args.collection_run).resolve();out=Path(args.output_dir).resolve()
 require(not out.exists(),'Refusing to overwrite existing dataset output')
 freeze=FrozenFiles();bind=freeze.bind;bind(__file__)
 protocol_path=run/'protocol.json';protocol=read(protocol_path);protocol_sha=bind(protocol_path)
 for section in('input_sha256','source_sha256','manifest_sha256'):
  require(isinstance(protocol.get(section),dict)and protocol[section],'Missing frozen collection '+section)
  for path,value in protocol[section].items():bind(path,value)
 objects_path=run/'objects.json';objects_doc=read(objects_path);bind(objects_path,protocol['selection_sha256'])
 mapping,groups=split_plan(objects_doc['objects'],protocol['internal_mesh_groups'])
 inventory=objects_doc['source_inventory'];bind(inventory,objects_doc['source_inventory_sha256'])
 require(objects_doc['source_active_list_sha256']==objects_doc['source_inventory_sha256'],'Source inventory compatibility SHA differs')
 for obj in objects_doc['objects']:
  bind(obj['source_visual_mesh'],obj['source_visual_mesh_sha256'])
  for path,value in obj.get('source_input_sha256',{}).items():bind(path,value)
 final_path=run/'execution/RESULTS.json';final=read(final_path);bind(final_path)
 queue_path=run/'planning/queue_manifest.json';queue=read(queue_path);bind(queue_path)
 for path,value in queue.get('source_sha256',{}).items():bind(path,value)
 if queue.get('jobs_json'):bind(queue['jobs_json'],queue['jobs_sha256'])
 exports={};export_files={}
 for name in('train_records.jsonl','all_records.jsonl'):
  path=run/'execution'/name;export_files[name]=dict(path=str(path),sha256=bind(path));found={}
  for line in path.read_text().splitlines():
   if not line.strip():continue
   record=json.loads(line);tid=record['collection_context']['trial_id'];require(tid not in found,'Repeated export trial identity');found[tid]=record
  if exports:require(exports==found,'Train/all export files differ')
  else:exports=found
 slots,ready,actual,eligible,terminal=reconcile(protocol,queue,final,exports)
 helper_path=ROOT/'reproduction/boya/trial_record.py';bind(helper_path);helper=import_helper(helper_path,'_s40_export_trial_record')
 helper.file_sha=bind  # Pure hash caching; original validation code is unchanged.
 canonical=CanonicalValidator(helper,bind);samples=[];vectors={g:[]for g in GESTURES}
 for tid in slots:
  if tid not in eligible:continue
  exported=copy.deepcopy(exports[tid]);context=exported.pop('collection_context');row=actual[tid];slot=slots[tid]
  require(all(context[k]==slot[k]for k in('trial_id','object_id','gesture_id','extra_depth_index'))and context['split']=='train'and context.get('benchmark_only')is False,'Export context differs from frozen slot')
  require(context['protocol_sha256']==protocol_sha,'Export protocol SHA differs')
  require(exported.get('benchmark_only')is False and row.get('benchmark_only')is False,'Benchmark records cannot enter formal dataset')
  artifact=exported['execution_artifacts'];evaluation=Path(artifact['evaluation_json']);commands=Path(artifact['commands_npz'])
  bind(evaluation,artifact['evaluation_sha256']);bind(commands,artifact['commands_sha256'])
  source_record=evaluation.parent/'training_record.json';bind(source_record)
  require(read(source_record)==exported,'JSONL export differs from original source record')
  binding=exported['binding'];require(read(row['binding_json'])==binding,'Original saved trial binding differs');bind(row['binding_json'])
  expected=helper.collect_record(binding,evaluation,commands)
  require(all(exported.get(k)==value for k,value in expected.items()),'Original collect_record replay differs: '+tid)
  require(exported['label_eligible']is True and exported['result']==row['label']and binding['trial_binding_sha256']==row['trial_binding_sha256'],'Actual label or binding differs')
  require(artifact['commands_sha256']==row['commands_sha256'],'Actual command bytes differ')
  plan_path=Path(row['source_plan_dir'])/'plan.json';plan=read(plan_path);bind(plan_path)
  bind(Path(row['source_plan_dir'])/'commands.npz',row['commands_sha256'])
  require(exported['total_physics_steps']==row['physics_steps']==plan['total_command_steps'],'Actual full command count differs')
  require(row.get('guards')and all(row['guards'].values()),'Actual queue binding guard failed')
  with np.load(commands,allow_pickle=False)as z:
   require(len(z['phase'])==row['physics_steps']and {'close','hold','lift'}.issubset(set(map(str,z['phase']))),'Incomplete actual command phases')
  for section in('source_sha256','extra_input_sha256'):
   for path,value in plan.get(section,{}).items():bind(path,value)
  g,raw,extra,target=observed_target(binding,slot)
  feature=canonical.feature(binding);obj=mapping[row['object_id']]
  meta=dict(trial_id=tid,object_id=row['object_id'],gesture_id=g,mesh_sha256=obj['mesh_sha256'],outer_split='train',split=obj['split'],label=row['label'],target_index=target,raw_depth_type=raw,extra_depth_index=extra,
    record_json=str(source_record),record_sha256=bind(source_record),evaluation_json=str(evaluation),evaluation_sha256=artifact['evaluation_sha256'],commands_npz=str(commands),commands_sha256=artifact['commands_sha256'],canonical_CGR480_identity=binding['canonical_CGR480_identity'])
  vectors[g].append(feature);samples.append(meta)
 # Recompute every null certificate from original saved evidence, never label it.
 terminal_checks={}
 if terminal:
  path=ROOT/'reproduction/boya/terminal_collection_outcome.py';bind(path);terminal_helper=import_helper(path,'_s40_export_terminal_helper');terminal_helper.sha=bind
  for tid in sorted(terminal):
   saved=final['terminal_certificates'][tid];reproduced=terminal_helper.certify_known_task_stop(actual[tid]);require(saved==reproduced,'Terminal-null certificate replay differs: '+tid)
   for section in('evidence_sha256','native_source_sha256'):
    for path,value in saved[section].items():bind(path,value)
   terminal_checks[tid]=dict(trial_id=tid,certificate_sha256=json_sha(saved),reason=saved['reason'],physics_steps=saved['physics_steps'])
 require(len(samples)==final['eligible_labels'],'Sample count differs from final eligible ledger')
 freeze.verify()
 out.mkdir(parents=True,exist_ok=False);gesture_files={}
 for g in GESTURES:
  meta=[r for r in samples if r['gesture_id']==g];path=out/f'gesture_{g}.npz';features=np.asarray(vectors[g],dtype=np.float32).reshape(-1,480)
  with path.open('xb')as stream:np.savez_compressed(stream,features=features,labels=np.asarray([r['label']for r in meta],dtype=np.int64),target_index=np.asarray([r['target_index']for r in meta],dtype=np.int64),gesture_ids=np.full(len(meta),g,dtype=np.int64),mesh_sha256=np.asarray([r['mesh_sha256']for r in meta],dtype='<U64'),trial_id=np.asarray([r['trial_id']for r in meta],dtype='<U160'),object_id=np.asarray([r['object_id']for r in meta],dtype='<U128'),split=np.asarray([r['split']for r in meta],dtype='<U24'))
  stats=coverage(samples,g);gesture_files[str(g)]=dict(path=str(path),sha256=sha(path),**stats,feature_dim=480,output_dim=20)
  for i,row in enumerate(meta):row.update(npz=str(path),npz_row=i)
 freeze.verify()
 for value in gesture_files.values():require(sha(value['path'])==value['sha256'],'New NPZ changed before manifest publication')
 manifest=dict(schema='anydex_s40_decision_dataset_v1',status='prepared',complete=True,collection_complete=True,benchmark_only=False,gesture_ids=list(GESTURES),feature_dim=480,output_dim=20,sample_count=len(samples),positive=sum(s['label']==1 for s in samples),negative=sum(s['label']==0 for s in samples),
  mesh_groups=groups,objects=[mapping[k]for k in sorted(mapping)],split_counts=dict(Counter(s['split']for s in samples)),labelled_mesh_groups=len({s['mesh_sha256']for s in samples}),gesture_files=gesture_files,samples=samples,
  source_collection=dict(run=str(run),protocol=str(protocol_path),protocol_sha256=protocol_sha,final_results=str(final_path),final_results_sha256=bind(final_path),ready_queue=str(queue_path),ready_queue_sha256=bind(queue_path),objects_manifest=str(objects_path),objects_manifest_sha256=bind(objects_path),source_export_files=export_files),
  internal_split=dict(source='Original collection protocol.internal_mesh_groups, frozen before outcomes',train_groups=32,internal_check_groups=8,external_test_groups=0),
  supervision=dict(feature_source='Original source-bound canonical480 float32 bytes, independently replayed original AST functions; no second canonicalization for training',target_index='two_fingers_pose_depth_type*4+multifinger_pose_depth_type',unobserved_19_outputs_are_unlabelled=True),
  excluded=dict(preparation_null_count=len(queue['nulls']),certified_terminal_null_count=len(terminal),terminal_nulls=terminal_checks,nulls_never_imputed_as_zero=True,EGAD_objects=0),
  final_ledger=dict(slots=640,ready=len(ready),eligible=len(eligible),certified_terminal_null=len(terminal),preparation_null=len(queue['nulls']),verified=True),input_sha256=freeze.files,
  dataset_content_sha256=json_sha(dict(files={k:v['sha256']for k,v in gesture_files.items()},mesh_groups=groups,trial_ids=sorted(eligible))),
  cost=dict(new_physics_steps=0,new_network_forwards=0,new_training_steps=0,preparation_wall_s=time.monotonic()-started),
  boundary='Original S40 GraspNet-derived identity training objects with a synthetic Boya collection protocol; original real-object parameters and full paper reproduction remain unaligned. No EGAD/test object is used.')
 with(out/'manifest.json').open('x')as stream:stream.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status='prepared',manifest=str(out/'manifest.json'),manifest_sha256=sha(out/'manifest.json'),sample_count=len(samples),split_counts=manifest['split_counts'],gesture_counts={k:v['count']for k,v in gesture_files.items()}),indent=2),flush=True)
 return manifest

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection-run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);prepare(p.parse_args())
if __name__=='__main__':main()
