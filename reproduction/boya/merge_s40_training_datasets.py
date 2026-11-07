"""Merge two completed source-bound S40 datasets without relabelling/retraining.

Original per-run trial IDs are namespaced by their frozen protocol SHA. Actual
trace identities must remain unique. Features, observed-output indices and real
binary labels retain their source bytes/values; no null is converted to zero.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np

GESTURES=(1,6,7,8)
S40_IDS=frozenset((0,2,5,7,8,9,11,14,15,17,18,20,21,22,26,27,29,30,34,36,37,38,40,41,43,44,46,48,51,52,56,57,58,60,61,62,63,66,69,70))
REQUIRED={'features','labels','target_index','gesture_ids','mesh_sha256','trial_id','object_id','split'}

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def digest(value):
 require(isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None,'Expected SHA256 identity')
 return value
def resolve(path,parent):
 p=Path(path);return(p if p.is_absolute()else parent/p).resolve()
def uarray(values):return np.asarray(values,dtype=f'<U{max([1,*map(len,values)])}')
def array_identity(array):
 a=np.ascontiguousarray(array);return dict(shape=list(a.shape),dtype=a.dtype.str,c_order_bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest())

class Frozen:
 def __init__(self):self.files={};self.cache={}
 def bind(self,path,expected=None):
  p=str(Path(path).resolve());s=Path(p).stat();stamp=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  if p not in self.cache or self.cache[p][0]!=stamp:self.cache[p]=(stamp,sha(p))
  h=self.cache[p][1];require(expected is None or h==expected,'Frozen source SHA mismatch: '+p)
  require(p not in self.files or self.files[p]==h,'Source changed while merging: '+p);self.files[p]=h;return h
 def verify(self):
  for path,value in self.files.items():require(sha(path)==value,'Frozen source changed before publication: '+path)


def authority(manifest):
 require(manifest.get('schema')=='anydex_s40_decision_dataset_v1'and manifest.get('complete')is True and manifest.get('collection_complete')is True and manifest.get('benchmark_only')is False,'Two complete formal S40 datasets required')
 require(manifest.get('gesture_ids')==list(GESTURES),'Four original hand types required')
 groups=manifest.get('mesh_groups');require(isinstance(groups,dict)and set(groups)=={'train','internal_check'},'Predeclared32/8 authority required')
 require(len(groups['train'])==32 and len(groups['internal_check'])==8 and len(set(groups['train']+groups['internal_check']))==40,'Wrong or leaking source mesh groups')
 for mesh in groups['train']+groups['internal_check']:digest(mesh)
 objects=manifest.get('objects');require(isinstance(objects,list)and len(objects)==40,'Exactly40 original S40 outer training objects required')
 byid={o['object_id']:o for o in objects};require(len(byid)==40 and all(o.get('outer_split')=='train'for o in objects),'Original outer-training identity differs')
 require({o.get('graspnet_object_id')for o in objects}==S40_IDS,'Original S40 source-ID union differs')
 mesh_split={mesh:split for split,meshes in groups.items()for mesh in meshes}
 require(len({o['mesh_sha256']for o in objects})==40 and {o['mesh_sha256']for o in objects}==set(mesh_split),'Original source mesh identity set differs')
 for obj in objects:
  require(obj['split']==mesh_split[obj['mesh_sha256']],'Object crosses frozen internal split')
  if 'source_visual_mesh_sha256'in obj:require(obj['source_visual_mesh_sha256']==obj['mesh_sha256'],'Source visual mesh SHA differs')
 return byid,mesh_split

def namespaced(protocol_sha,local_trial_id):
 digest(protocol_sha);require(isinstance(local_trial_id,str)and bool(local_trial_id),'Original local trial identity required')
 return 's40_'+protocol_sha[:16]+'__'+local_trial_id

def read_source(path,frozen):
 path=Path(path).resolve();m=read(path);byid,splits=authority(m);frozen.bind(path)
 require(isinstance(m.get('input_sha256'),dict)and m['input_sha256'],'Source dataset freeze map required')
 for p,h in m['input_sha256'].items():frozen.bind(p,h)
 c=m['source_collection'];protocol=resolve(c['protocol'],path.parent);protocol_sha=digest(c['protocol_sha256']);frozen.bind(protocol,protocol_sha)
 final_path=resolve(c['final_results'],path.parent);frozen.bind(final_path,c['final_results_sha256']);final=read(final_path)
 require(final.get('complete')is True and final.get('input_sha256_unchanged')is True and final.get('source_sha256_unchanged')is True,'Source actual collection closeout is incomplete')
 actual={r['trial_id']:r for r in final['trials']};require(len(actual)==len(final['trials']),'Repeated source actual trial ID')
 eligible={tid for tid,r in actual.items()if r.get('label_eligible')is True and type(r.get('label'))is int and r['label']in(0,1)}
 samplemeta={r['trial_id']:r for r in m['samples']};require(len(samplemeta)==len(m['samples'])==m['sample_count']and set(samplemeta)==eligible,'Source dataset does not cover exactly original eligible actuals')
 files=m['gesture_files'];require(set(files)==set(map(str,GESTURES)),'Four gesture NPZ files required')
 samples=[];seen=set()
 for gesture in GESTURES:
  entry=files[str(gesture)];file=resolve(entry['path'],path.parent);frozen.bind(file,entry['sha256'])
  with np.load(file,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
  require(REQUIRED<=set(arrays),'Missing source training columns');n=len(arrays['labels'])
  require(arrays['features'].shape==(n,480)and arrays['features'].dtype==np.float32 and np.isfinite(arrays['features']).all(),'Finite original float32 canonical480 required')
  require(all(arrays[k].shape==(n,)and not arrays[k].dtype.hasobject for k in REQUIRED-{'features'}),'Invalid parallel source columns')
  require(all(arrays[k].dtype==np.int64 for k in('labels','target_index','gesture_ids')),'Exact int64 label/index/gesture arrays required')
  require(all(arrays[k].dtype.kind=='U'for k in('mesh_sha256','trial_id','object_id','split')),'Source identity columns must be Unicode')
  require(np.isin(arrays['labels'],[0,1]).all()and ((arrays['target_index']>=0)&(arrays['target_index']<20)).all()and (arrays['gesture_ids']==gesture).all()and entry['count']==n,'Invalid original observed label/target/type/count')
  for i in range(n):
   local=str(arrays['trial_id'][i]);require(local not in seen and local in samplemeta,'Repeated/foreign source NPZ trial');seen.add(local)
   meta=samplemeta[local];row=actual[local];oid=str(arrays['object_id'][i]);mesh=str(arrays['mesh_sha256'][i]);split=str(arrays['split'][i]);label=int(arrays['labels'][i]);target=int(arrays['target_index'][i])
   require(oid in byid and byid[oid]['mesh_sha256']==mesh and splits.get(mesh)==split,'Source row crosses original object/internal mesh authority')
   require(all(meta[k]==v for k,v in dict(object_id=oid,mesh_sha256=mesh,split=split,label=label,target_index=target,gesture_id=gesture).items()),'Source NPZ differs from source sample metadata')
   require(row['object_id']==oid and row['gesture_id']==gesture and row['label']==label and row['physics_steps']>0,'Source NPZ label differs from actual physical trial')
   record_path=resolve(meta['record_json'],path.parent);frozen.bind(record_path,meta['record_sha256']);record=read(record_path);b=record['binding'];a=record['execution_artifacts']
   require(record.get('label_eligible')is True and type(record.get('result'))is int and record['result']==label,'Null or synthetic label cannot enter merge')
   require(b['multifinger_pose_finger_type']==gesture and b['two_fingers_pose_depth_type']*4+b['multifinger_pose_depth_type']==target,'Observed original output index changed')
   feature=arrays['features'][i].copy();require(array_identity(feature)==b['canonical_CGR480_identity']and feature.tobytes()==np.asarray(b['canonical_CGR480'],dtype=np.float32).tobytes(),'Canonical480 source bytes changed')
   evaluation=resolve(a['evaluation_json'],record_path.parent);commands=resolve(a['commands_npz'],record_path.parent)
   frozen.bind(evaluation,a['evaluation_sha256']);frozen.bind(commands,a['commands_sha256']);ev=read(evaluation)
   require(ev.get('complete')is True and ev.get('label_eligible')is True and type(ev.get('task_success'))is bool and int(ev['task_success'])==label and ev.get('total_physics_steps')==row['physics_steps'],'Actual evaluation does not certify this label')
   require(ev.get('trial_binding_sha256')==b['trial_binding_sha256']==row['trial_binding_sha256']and ev.get('commands_sha256')==a['commands_sha256']==row['commands_sha256'],'Actual command/evaluation binding differs')
   native=Path(row['execution_dir']).resolve();trace=native/'trace.npz';trace_sha=digest(ev['trace_sha256']);frozen.bind(trace,trace_sha)
   native_eval=native/'evaluation.json';frozen.bind(native_eval,ev['original_native_evaluation_sha256'])
   merged=copy.deepcopy(meta);merged.update(trial_id=namespaced(protocol_sha,local),local_trial_id=local,source_protocol_sha=protocol_sha,
    source_collection_manifest=str(path),source_collection_manifest_sha256=frozen.bind(path),source_trace=str(trace),source_trace_sha256=trace_sha,
    source_native_evaluation=str(native_eval),source_native_evaluation_sha256=frozen.bind(native_eval),source_record_json=str(record_path),source_record_sha256=meta['record_sha256'])
   merged.pop('npz',None);merged.pop('npz_row',None);samples.append((merged,feature))
 require(seen==set(samplemeta),'Source dataset metadata/NPZ coverage differs')
 return m,byid,samples,dict(manifest=str(path),manifest_sha256=frozen.bind(path),protocol=str(protocol),protocol_sha256=protocol_sha,final_results=str(final_path),final_results_sha256=frozen.bind(final_path),sample_count=len(samples))

def reject_duplicate_actuals(samples):
 traces={};origins={};names={}
 for meta,_ in samples:
  tid=meta['trial_id'];trace=meta['source_trace_sha256'];origin=str(Path(meta['source_trace']).resolve())
  require(trace not in traces,'Repeated actual physical trace SHA: '+tid+' duplicates '+str(traces.get(trace)))
  require(origin not in origins,'Repeated actual execution source path: '+origin)
  require(tid not in names,'Repeated namespaced trial ID or protocol-prefix collision: '+tid)
  traces[trace]=tid;origins[origin]=tid;names[tid]=True

def stats(rows):
 return dict(count=len(rows),positive=sum(r['label']==1 for r in rows),negative=sum(r['label']==0 for r in rows),
  labelled_mesh_groups=len({r['mesh_sha256']for r in rows}),labelled_objects=len({r['object_id']for r in rows}),
  target_coverage={str(i):dict(count=sum(r['target_index']==i for r in rows),positive=sum(r['target_index']==i and r['label']==1 for r in rows),negative=sum(r['target_index']==i and r['label']==0 for r in rows))for i in range(20)},
  split_counts={s:dict(count=sum(r['split']==s for r in rows),positive=sum(r['split']==s and r['label']==1 for r in rows),negative=sum(r['split']==s and r['label']==0 for r in rows))for s in('train','internal_check')})

def merge(paths,output_dir):
 started=time.monotonic();out=Path(output_dir).resolve();require(not out.exists(),'Fresh merged dataset output required; refusing overwrite')
 require(len(paths)==2 and len({str(Path(p).resolve())for p in paths})==2,'Exactly two different source dataset manifests required')
 frozen=Frozen();frozen.bind(__file__);loaded=[read_source(p,frozen)for p in paths]
 first=loaded[0];second=loaded[1]
 require(first[0]['mesh_groups']==second[0]['mesh_groups'],'Source32/8 mesh authority differs')
 require({k:(o['mesh_sha256'],o['split'],o['outer_split'])for k,o in first[1].items()}=={k:(o['mesh_sha256'],o['split'],o['outer_split'])for k,o in second[1].items()},'Original40 source object authority differs')
 combined=[sample for item in loaded for sample in item[2]];reject_duplicate_actuals(combined)
 prefixes={}
 for descriptor in(item[3]for item in loaded):
  full=descriptor['protocol_sha256'];prefix=full[:16];require(prefix not in prefixes or prefixes[prefix]==full,'Distinct protocols have same namespace prefix');prefixes[prefix]=full
 frozen.verify();out.mkdir(parents=True,exist_ok=False);files={};samples=[]
 for gesture in GESTURES:
  selected=[(m,f)for m,f in combined if m['gesture_id']==gesture];meta=[m for m,_ in selected];path=out/f'gesture_{gesture}.npz'
  arrays=dict(features=np.asarray([f for _,f in selected],dtype=np.float32).reshape(-1,480),labels=np.asarray([m['label']for m in meta],dtype=np.int64),target_index=np.asarray([m['target_index']for m in meta],dtype=np.int64),gesture_ids=np.full(len(meta),gesture,dtype=np.int64))
  for key in('mesh_sha256','trial_id','object_id','split','local_trial_id','source_protocol_sha','source_collection_manifest'):
   arrays[key]=uarray([m[key]for m in meta])
  with path.open('xb')as stream:np.savez_compressed(stream,**arrays)
  coverage=stats(meta);coverage['unobserved_targets']=[int(i)for i,d in coverage['target_coverage'].items()if not d['count']]
  files[str(gesture)]=dict(path=str(path),sha256=sha(path),feature_dim=480,output_dim=20,**coverage)
  for index,row in enumerate(meta):row.update(npz=str(path),npz_row=index)
  samples+=meta
 frozen.verify()
 for item in files.values():require(sha(item['path'])==item['sha256'],'Merged NPZ changed before publication')
 summary=stats(samples)
 manifest=dict(schema='anydex_s40_decision_dataset_v1',dataset_variant='two_collection_merge_v1',status='prepared',complete=True,collection_complete=True,benchmark_only=False,
  gesture_ids=list(GESTURES),feature_dim=480,output_dim=20,sample_count=len(samples),positive=summary['positive'],negative=summary['negative'],mesh_groups=copy.deepcopy(first[0]['mesh_groups']),
  objects=copy.deepcopy(first[0]['objects']),gesture_files=files,samples=samples,split_counts=dict(Counter(r['split']for r in samples)),labelled_mesh_groups=summary['labelled_mesh_groups'],target_coverage=summary['target_coverage'],
  source_datasets=[item[3]for item in loaded],source_collection=dict(kind='two_complete_source_bound_s40_datasets',datasets=[item[3]for item in loaded]),
  internal_split=copy.deepcopy(first[0].get('internal_split')),supervision=copy.deepcopy(first[0].get('supervision')),input_sha256=frozen.files,
  namespace=dict(method='s40_ + first16 source protocol SHA256 + double underscore + original local_trial_id',source_fields=['local_trial_id','source_protocol_sha','source_collection_manifest'],protocol_prefix_collisions_rejected=True),
  duplicate_policy='Repeated source physical trace SHA, execution trace path or namespaced actual trial identity rejected, never silently dropped or counted twice.',
  excluded=dict(nulls_included=0,labels_synthesized=0,EGAD_objects=0),cost=dict(new_physics_steps=0,new_network_forwards=0,new_training_steps=0,merge_wall_s=time.monotonic()-started),
  boundary='All40 original S40 identities remain outer training; same frozen32/8 internal split. Retained actual source-bound labels only. Synthetic Boya port is not full original real-robot paper reproduction.')
 with(out/'manifest.json').open('x')as stream:stream.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status='merged',manifest=str(out/'manifest.json'),manifest_sha256=sha(out/'manifest.json'),sample_count=len(samples),split_counts=manifest['split_counts']),indent=2),flush=True)
 return manifest

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-manifests',type=Path,nargs=2,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();merge(a.dataset_manifests,a.output_dir)
if __name__=='__main__':main()
