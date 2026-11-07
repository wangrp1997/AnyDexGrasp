"""Fresh S40 collection at independently validated settled poses.

All40 training identities receive new RGB-D/CGR and four gestures x four depths.
The previous32/8 source-mesh split is copied; no prior physical trial is adopted.
Each new attempt cold-resets to the settled measured pose with declared zero
velocity, after a separate stable-pose recreation check. This is not continuous
scene handoff. The original execution/physical gates remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
PINS={'run_s40_collection.py':'6e33bf735a1119bfa62d6155be6fa3d485e8d1e0d81edf718bb11f032b98fd0b',
      'run_collection_pipeline_v2.py':'82cb5deadbe82d6cbcded446a241d3244f1b3114ed11694a830d163ad3aab6d9'}
for name,expected in PINS.items():
 if hashlib.sha256((BOYA/name).read_bytes()).hexdigest()!=expected:raise ValueError('Read-only collection dependency changed: '+name)
from run_s40_collection import (OriginalPipeline,bounded_jobs,sha,read,save,immutable,hashes,verify,
    native_profile_identity,NATIVE,GESTURES,DEPTHS,job_id,validate_scene,validate_null,verify_frame_binding)
from prepare_s40_training_data import split_plan

def require(value,message):
 if not value:raise ValueError(message)

def settling_binding(scene):
 rows=validate_scene(scene);p=scene.get('settling_provenance')
 require(isinstance(p,dict)and p.get('complete')is True and p.get('ready_objects')==40,'Complete all40 settling provenance required')
 require(isinstance(p.get('results_json'),str)and isinstance(p.get('results_sha256'),str),'Saved settling result path/SHA required')
 verify({p['results_json']:p['results_sha256']});result=read(p['results_json'])
 require(result.get('complete')is True and result.get('ready_objects')==40 and not result.get('errors'),'All40 settling results must be ready before rendering')
 measured=result.get('objects');require(isinstance(measured,list)and len(measured)==40,'Exactly40 measured settling objects required')
 byid={r['object_id']:r for r in measured};require(len(byid)==40 and set(byid)=={r['object_id']for r in rows},'Settling result/manifest identity sets differ')
 refs={p['results_json']:p['results_sha256']}
 for section in('input_sha256','source_sha256'):
  require(isinstance(p.get(section),dict)and bool(p[section]),'Settling source/input freeze map required')
  verify(p[section]);refs.update(p[section])
 for row in rows:
  record=byid[row['object_id']]
  require(row.get('settling_ready')is True and record.get('ready')is True and row.get('settling')==record,'Manifest settling record differs from measured result')
  require(record.get('cold_recreation_verified')is True and record.get('settle_window',{}).get('stable')is True and record.get('final_window',{}).get('stable')is True,'Stable settling and cold recreation gates required')
  T=np.asarray(record['world_T_object'],dtype=float);declared=np.asarray(row['world_T_object'],dtype=float)
  pose=np.asarray(record['measured_settled_pose_xyzw'],dtype=float)
  require(T.shape==(4,4)and pose.shape==(7,)and np.isfinite(T).all()and np.isfinite(pose).all()and np.array_equal(T,declared),'Manifest/root/measured settling T differ')
  reconstructed=np.eye(4);reconstructed[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();reconstructed[:3,3]=pose[:3]
  require(np.allclose(reconstructed,T,rtol=0,atol=1e-10)and np.array_equal(np.asarray(row['world_pose_xyzw'],float),pose),'Measured XYZW pose does not match stable world transform')
  trace=Path(record['trace']).resolve();verify({str(trace):record['trace_sha256']});refs[str(trace)]=record['trace_sha256']
  with np.load(trace,allow_pickle=False)as z:
   step=record['settle_steps'];require(type(step)is int and 0<=step<len(z['pose_xyzw']),'Invalid settling trace index')
   raw=np.asarray(z['pose_xyzw'][step],float);raw[3:]/=np.linalg.norm(raw[3:])
   require(np.allclose(raw,pose,rtol=0,atol=1e-10)and np.array_equal(np.asarray(z['velocity'][step],float),np.asarray(record['measured_settled_velocity'],float)),'Settled pose/velocity differs from saved source trace')
   require(len(z['pose_xyzw'])==record['physics_steps']+1 and record['recreation_steps']>=750,'Complete3s cold recreation trace required')
  posefile=Path(row['pose_path']).resolve();verify({str(posefile):row['pose_sha256']});refs[str(posefile)]=row['pose_sha256']
  entries=np.asarray(read(posefile),float);index=row['selected_source_pose_index']
  require(entries.ndim==2 and entries.shape[1]==7 and type(index)is int and 0<=index<len(entries),'Invalid frozen WXYZ pose file')
  saved=entries[index];saved_T=np.eye(4);saved_T[:3,3]=saved[:3];saved_T[:3,:3]=Rotation.from_quat(saved[[4,5,6,3]]).as_matrix()
  require(np.allclose(saved_T,T,rtol=0,atol=1e-10),'WXYZ pose file does not reproduce measured stable T')
 return refs

def candidate_argv(cfg,job,row):
 return [sys.executable,'-B',cfg['candidate_builder'],'--observation',job['observation_json'],
  '--frame-dir',row['representation_dir'],'--calibration',cfg['calibration'],'--gesture-id',str(job['gesture_id']),
  '--extra-depth-index',str(job['extra_depth_index']),'--candidate-limit','1000','--keep','8',
  '--width-expansion','1.7','--opponent','ff','--coverage-history',cfg['coverage_history'],
  '--geometry-checks-per-depth','64','--coverage-target-per-output','4',
  '--flip-policy','both','--output-dir',job['candidate_dir']]

def freeze(args):
 run=args.run_dir.resolve();pp=run/'protocol.json'
 cfg=dict(scene_manifest=str(args.scene_manifest.resolve()),previous_protocol=str(args.previous_protocol.resolve()),
  previous_dataset_manifest=str(args.previous_dataset_manifest.resolve()),previous_collection_results=str(args.previous_collection_results.resolve()),
  native_assets_run=str(args.native_assets_run.resolve()),candidate_builder=str(args.candidate_builder.resolve()),
  calibration=str(args.calibration.resolve()),backend_config=str(args.backend_config.resolve()),checkpoint=str(args.checkpoint.resolve()),native_deps=str(args.native_deps.resolve()),
  representation_device=args.device,geometry_workers=args.geometry_workers,plan_workers=args.plan_workers,execution_workers=args.execution_workers,native_workers=args.native_workers,
  physics_device='cpu',execution_mode='serial',batch_size=16,candidate_limit=1000,keep=8,max_arm_plans=4,seed=20261108,gesture_ids=GESTURES,
  geometry_checks_per_depth=64,coverage_target_per_output=4,coverage_history=str(run/'coverage_history/history.json'))
 if pp.exists():
  p=read(pp);require(p.get('collection_variant')=='settled_pose_fresh_v2'and p.get('config')==cfg,'Frozen settled-collection option changed')
  verify(p['input_sha256']);verify(p['source_sha256']);verify(p['manifest_sha256']);return p
 require(not run.exists()or not any(run.iterdir()),'Fresh empty collection run required')
 scene=read(cfg['scene_manifest']);settle_refs=settling_binding(scene);previous=read(cfg['previous_protocol'])
 require(previous.get('schema')=='anydex_boya_s40_collection_v1','Original S40 split protocol required')
 mapping,groups=split_plan(scene['objects'],previous['internal_mesh_groups'])
 dataset=read(cfg['previous_dataset_manifest'])
 require(dataset.get('schema')=='anydex_s40_decision_dataset_v1'and dataset.get('complete')is True and dataset.get('benchmark_only')is False,'Completed previous S40 dataset needed for output-coverage policy')
 require(dataset['mesh_groups']==groups and dataset['source_collection']['protocol_sha256']==sha(cfg['previous_protocol']),'Previous dataset source/split differs from frozen protocol')
 previous_results=read(cfg['previous_collection_results']);require(previous_results.get('complete')is True,'Previous actual canonical exclusions require complete results')
 require(sha(cfg['previous_collection_results'])==dataset['source_collection']['final_results_sha256'],'Previous actual outcome ledger differs from dataset authority')
 run.mkdir(parents=True,exist_ok=True);rebased=copy.deepcopy(scene)
 rebased.update(source_scene_manifest=cfg['scene_manifest'],source_scene_manifest_sha256=sha(cfg['scene_manifest']))
 for row in rebased['objects']:
  oid=row['object_id'];row.update(observation_json=str(run/'observations'/oid/'observation.json'),observation=str(run/'observations'/oid/'observation.json'),
    representation_dir=str(run/'representations'/oid),observation_inherited_from_pilot=False)
 save(run/'objects.json',rebased)
 save(run/'representation_manifest.json',dict(objects=[{k:r[k]for k in('object_id','split','observation_json','representation_dir')}for r in rebased['objects']]))
 slots=[dict(trial_id=job_id(r['object_id'],g,d),object_id=r['object_id'],split='train',gesture_id=g,extra_depth_index=d,
   candidate_dir=str(run/'pools'/r['object_id']/f'g{g}'/f'd{d}'),observation_json=r['observation_json'],source_npz=str(Path(r['representation_dir'])/'raw_camera_candidates.npz'),calibration=cfg['calibration'],max_arm_plans=4)
   for r in rebased['objects']for g in GESTURES for d in DEPTHS]
 files=dict(settle_refs)
 def add(path,expected=None):
  path=str(Path(path).resolve());actual=sha(path);require(expected is None or expected==actual,'Source SHA changed: '+path)
  require(path not in files or files[path]==actual,'Conflicting source SHA: '+path);files[path]=actual
 for key in('scene_manifest','previous_protocol','previous_dataset_manifest','previous_collection_results','calibration','backend_config','checkpoint'):add(cfg[key])
 for item in dataset['gesture_files'].values():add(item['path'],item['sha256'])
 add(scene['source_inventory'],scene['source_inventory_sha256'])
 backend=read(cfg['backend_config']);cal=read(cfg['calibration'])
 for value in backend.values():
  if isinstance(value,str)and value.startswith('/')and Path(value).is_file():add(value)
 for path in Path(backend['reference_root']).rglob('*.json'):add(path)
 for path,item in cal['source_lock']['source_files'].items():add(path,item['sha256'])
 for path,value in cal['source_lock']['readonly_fk_dependencies'].items():add(path,value)
 for row in rebased['objects']:
  for pk,hk in [('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256'),('metadata_path','metadata_sha256')]:add(row[pk],row[hk])
  for part in row['convex_parts']:add(part['path'],part['sha256'])
  for path,value in row['source_input_sha256'].items():add(path,value)
 prior_native=Path(cfg['native_assets_run'])/'native_preprocessing/SUMMARY.json';add(prior_native)
 source_paths=[Path(__file__).resolve(),Path(cfg['candidate_builder']),*BOYA.glob('*.py'),NATIVE,ROOT/'robot_dh3.py',ROOT/'models/minkowski_graspnet_single_point.py',ROOT/'reproduction/runtime/run_official_rgbd_frame.py']
 canon=NATIVE.parents[2];source_paths += [*(canon/'canondex').rglob('*.py'),*(canon/'scripts/research').glob('*.py')]
 p=dict(schema='anydex_boya_s40_collection_v1',collection_variant='settled_pose_fresh_v2',internal_mesh_groups=copy.deepcopy(previous['internal_mesh_groups']),config=cfg,
   slots=slots,slot_count=640,split_slot_counts={'train':640,'validation':0},gesture_ids=GESTURES,extra_depth_indices=DEPTHS,per_object_slot_limit=16,
   valid_label_targets_per_gesture={'train':100,'validation':0},valid_label_target_totals={'train':400,'validation':0},selection_sha256=sha(run/'objects.json'),
   input_sha256=files,source_sha256=hashes(source_paths),manifest_sha256=hashes([run/'objects.json',run/'representation_manifest.json']),
   inherited_slot_count=0,new_slot_count=640,inherited_actual_trials=0,inherited_physics_steps=0,inherited_protocol_sha256=None,
   source_internal_split_protocol=cfg['previous_protocol'],source_internal_split_protocol_sha256=sha(cfg['previous_protocol']),settling_provenance=copy.deepcopy(scene['settling_provenance']),
   independent_reset_boundary='Each trial independently recreates the settled measured pose with declared zero initial velocity, after validated 3s zero-recreation hold. Not continuous scene or nonzero-velocity handoff.',
   candidate_policy='Exclude prior all-actual canonical480 x gesture x extra including terminal nulls; output coverage and feature diversity only, no learned or success-value ranking.',
   target_boundary='100 usable labels per gesture is a target, not guaranteed; no result-conditioned replacement beyond frozen640 slots.',
   scope='Fresh40 GraspNet-derived S40 train views at measured stable poses; four-depth/four-gesture Boya simulation collection port, not original real hardware protocol.',
   no_test_objects=True,no_decision_head_loaded=True,new_training_updates=0,created_unix_s=time.time())
 verify(files);save(pp,p);return p

class Pipeline(OriginalPipeline):
 def publish(self,status):
  super().publish(status)
  print('S40_SETTLED_COLLECTION '+json.dumps(dict(status=status,time_unix=time.time(),details=self.progress.get('detail'))),flush=True)

 def native(self):
  # Static source/profile-compatible native import assets may be reused. No
  # prior observation, candidate, physical command, trace or label is adopted.
  output=self.run/'native_preprocessing';output.mkdir(exist_ok=False)
  old=read(Path(self.c['native_assets_run'])/'native_preprocessing/SUMMARY.json')
  require(old.get('complete')is True and old.get('new_physics_steps')==0,'Completed zero-step native import source required')
  byid={r['object_id']:r for r in old['results']};require(set(byid)=={r['object_id']for r in self.objects},'All40 native asset entries required')
  profile=native_profile_identity(read(self.c['backend_config'])['physics_config']);results=[];assets=[]
  for row in self.objects:
   source=byid[row['object_id']];rp=Path(source['result_json']);doc=read(rp);bound=doc['binding']['inputs']
   require(doc.get('complete')and doc.get('trial_physics_steps')==0 and doc.get('original_or_refined_within_20um'),'Original20um native import gate not passed')
   require(bound['object_id']==row['object_id']and bound['scale']==1 and bound['physics_profile_sha256']==profile and bound['tolerance_m']==20e-6 and bound['max_failed_partitions']==32,'Native source/profile mismatch')
   verify({str(Path(row['object_root'])/k):v for k,v in bound['source_files'].items()});verify({str(NATIVE.parents[2]/k):v for k,v in bound['implementations'].items()})
   cache=Path(doc['cache_directory']);refs=[rp]
   for name,value in doc.get('artifact_sha256',{}).items():verify({str(cache/name):value});refs.append(cache/name)
   repair=doc.get('repair_path')
   if repair:refs.append(Path(repair))
   results.append(dict(object_id=row['object_id'],result_json=str(rp),adopted_static_native_preflight=True,repair_path=repair,repair_sha256=sha(repair)if repair else None,artifacts=list(map(str,refs))));assets+=refs
  path=output/'SUMMARY.json';save(path,dict(complete=True,results=results,new_physics_steps=0,physics_profile_sha256=profile))
  return dict(objects=40,static_native_assets_reused=40,inherited_actual_grasps=0,new_physics_steps=0),[path,*assets]

 def observations(self):
  settling_binding(read(self.run/'objects.json'))
  output=self.run/'observations';require(not output.exists(),'Partial observation output retained; no automatic rerender')
  self.command('render',[sys.executable,'-B',str(BOYA/'render_s40_collection_observations.py'),'--objects-json',str(self.run/'objects.json'),'--output-dir',str(output)])
  index=read(output/'OBSERVATIONS.json');require(index['status']=='complete'and len(index['observations'])==40 and index['source_objects_sha256']==sha(self.run/'objects.json'),'Fresh40 render index required')
  assets=[output/'OBSERVATIONS.json'];entries={r['object_id']:r for r in index['observations']}
  require(set(entries)=={r['object_id']for r in self.objects},'Render identity union differs')
  for row in self.objects:
   path=Path(row['observation_json']);verify({str(path):entries[row['object_id']]['sha256']});doc=read(path);assets.append(path)
   require(doc['object_id']==row['object_id']and doc['object_scale']==1 and doc['world_T_object']==row['world_T_object']and doc.get('explicit_physics_steps')==0 and not doc.get('robot_present'),'Fresh observation does not bind measured stable pose')
   for item in doc['files'].values():verify({item['path']:item['sha256']});assets.append(Path(item['path']))
  return dict(objects=40,inherited=0,new_renders=40,new_physics_steps=0),assets

 def representation(self):
  output=self.run/'representation_stage'
  self.command('representation',[sys.executable,'-B',str(BOYA/'run_collection_representation.py'),'--manifest',str(self.run/'representation_manifest.json'),'--run-dir',str(output),'--device',self.c['representation_device'],'--checkpoint',self.c['checkpoint'],'--native-deps',self.c['native_deps'],'--seed',str(self.c['seed'])])
  batch=output/'BATCH_REPRESENTATION.json';doc=read(batch)
  require(doc.get('manifest_sha256')==sha(self.run/'representation_manifest.json')and doc['status']=='all_representation_forwards_complete'and doc['network_forward_count']==40 and doc['model_load_count']==1 and doc['manifest_unchanged_at_end'],'One shared CGR load and40 fresh forwards required')
  assets=[batch]
  for row in self.objects:
   path=Path(row['representation_dir'])/'RESULTS.json';receipt=read(path);verify_frame_binding(row,receipt);assets.append(path)
   for item in receipt['output_files'].values():verify({item['path']:item['sha256']});assets.append(Path(item['path']))
  return dict(objects=40,model_load_count=1,new_network_forwards=40,inherited_network_forwards=0),assets

 def coverage_history(self):
  output=self.run/'coverage_history';require(not output.exists(),'Partial coverage-history output retained')
  self.command('coverage_history',[sys.executable,'-B',self.c['candidate_builder'],'--prepare-history',
    '--previous-dataset-manifest',self.c['previous_dataset_manifest'],'--previous-collection-results',self.c['previous_collection_results'],
    '--output-dir',str(output)])
  receipt=output/'history.json';doc=read(receipt);require(doc.get('complete')is True,'Complete source-bound coverage history required')
  require(isinstance(doc.get('input_sha256'),dict)and doc['input_sha256'],'Coverage history frozen inputs required');verify(doc['input_sha256'])
  artifacts=doc.get('artifacts_sha256')
  if artifacts is None:
   npz=doc['history_npz'];artifacts={npz['path']:npz['sha256']}
  require(isinstance(artifacts,dict)and artifacts,'Coverage history artifacts required');verify(artifacts)
  return dict(complete=True,new_physics_steps=0,new_network_forwards=0),[receipt,*map(Path,artifacts)]

 def geometry(self):
  byid={r['object_id']:r for r in self.objects}
  def one(job):
   output=Path(job['candidate_dir']);require(not output.exists(),'Partial candidate output retained; no automatic retry')
   output.parent.mkdir(parents=True,exist_ok=True);proc=self.command('pool_'+job['trial_id'],candidate_argv(self.c,job,byid[job['object_id']]),allowed_failure=True)
   gen=output/'generation.json';null=output/'no_candidate.json';coverage=output/'coverage_selection.json'
   require(coverage.exists(),'Source-bound canonical exclusion/output-coverage receipt required')
   coverage_doc=read(coverage);require(isinstance(coverage_doc.get('input_sha256'),dict)and coverage_doc['input_sha256'],'Coverage selection frozen inputs required');verify(coverage_doc['input_sha256'])
   if gen.exists():
    require(proc['returncode']==0 and not null.exists(),'Conflicting candidate generation outcome');doc=read(gen);verify(doc['input_sha256']);verify({str(output/'candidates.npz'):doc['candidates_sha256']})
    require(doc['complete']and doc['gesture_id']==job['gesture_id']and doc['additional_depth_index']==job['extra_depth_index']and Path(doc['source_observation'])==Path(job['observation_json'])and Path(doc['source_frame'])==Path(byid[job['object_id']]['representation_dir']),'New pool source/slot identity differs')
    return dict(trial_id=job['trial_id'],status='ready'),[gen,output/'candidates.npz',output/'world_object_pose.json',coverage]
   if null.exists():
    result=validate_null(read(null),job,self.c['calibration']);binding=output/'NULL_BINDING.json'
    save(binding,dict(**result,geometry_command=proc,no_candidate_sha256=sha(null),declared_source_sha256={job['source_npz']:sha(job['source_npz']),job['observation_json']:sha(job['observation_json'])}))
    return result,[null,binding,coverage]
   raise RuntimeError('Candidate builder failed without ordinary source-bound null')
  values=self.parallel(self.p['slots'],self.c['geometry_workers'],one);path=self.run/'POOL_PREPARATION.json'
  save(path,dict(complete=True,jobs=[r for r,_ in values],inherited_terminal_slots=[],new_physics_steps=0,new_network_forwards=0))
  return dict(new_slots=640,inherited_slots=0,ready=sum(r['status']=='ready'for r,_ in values),no_candidate=sum(r['status']=='no_candidate'for r,_ in values)),[path,*[p for _,files in values for p in files]]

 # The generic source-bound arm preparation already handles arbitrary declared
 # slots and ordinary nulls and contains no pilot-result/count offsets.
 planning=OriginalPipeline.planning
 def execution(self):
     from s40_execution_outcomes import inspect_execution
     queue=read(self.run/'planning/queue_manifest.json');output=self.run/'execution'
     output.mkdir(exist_ok=False);(output/'object_queues').mkdir();(output/'terminal_certificates').mkdir()
     started=time.monotonic();jobs=[];certificates={};assets=[]
     progress=dict(schema='anydex_boya_s40_execution_v1',status='running',objects={},dispatch_events=[],peak_inflight_object_jobs=0,
         inherited_actual_grasps=0,inherited_physics_steps=0)
     for obj in self.objects:
         oid=obj['object_id'];trials=[r for r in queue['trials']if r['object_id']==oid]
         nulls=[r for r in queue['nulls']if r['object_id']==oid]
         sub=dict(queue,trials=trials,nulls=nulls,parent_queue_manifest=str(self.run/'planning/queue_manifest.json'),
             parent_queue_manifest_sha256=sha(self.run/'planning/queue_manifest.json'))
         path=output/'object_queues'/(oid+'.json');save(path,sub);assets.append(path)
         jobs.append(dict(object_id=oid,manifest=str(path),ready=len(trials)))
     save(output/'PROGRESS.json',progress)
     def one(job):
         oid=job['object_id'];sub=read(job['manifest']);target=output/oid
         if not job['ready']:return dict(object_id=oid,status='no_ready_plans',result=dict(trials=[]),terminal_certificates={})
         save(output/'object_queues'/(oid+'.SUBMITTING.json'),dict(object_id=oid,trial_ids=[r['trial_id']for r in sub['trials']],manifest=job['manifest'],manifest_sha256=sha(job['manifest']),time_unix=time.time(),state='submission_intent_not_proof_of_unexecuted'))
         proc=self.command('execution_'+oid,[sys.executable,'-B',str(BOYA/'execute_collection_queue_v2.py'),
             '--manifest',job['manifest'],'--output-dir',str(target),'--device','cpu','--mode','serial','--batch-size','16'],allowed_failure=True)
         doc=read(target/'RESULTS.json')if(target/'RESULTS.json').exists()else{}
         checked=inspect_execution(doc,[r['trial_id']for r in sub['trials']],proc['returncode'])
         return dict(object_id=oid,status='complete',output_dir=str(target),result=doc,terminal_certificates=checked['terminal_certificates'])
     def completed(job,value,bad):
         progress['objects'][job['object_id']]=dict(value,infrastructure_error=bad)
         for tid,cert in value.get('terminal_certificates',{}).items():
             certificates[tid]=cert;save(output/'terminal_certificates'/(tid+'.json'),cert)
         progress['wall_s']=time.monotonic()-started
         save(output/'PROGRESS.json',progress)
         vals=[r for v in progress['objects'].values()for r in v.get('result',{}).get('trials',[])]
         self.progress['detail']=dict(objects_finished=len(progress['objects']),objects=40,inherited_actual=0,
             new_actual=sum(r.get('physics_steps',0)>0 for r in vals),new_physics_steps=sum(r.get('physics_steps',0)for r in vals),
             eligible=sum(r.get('label_eligible')is True for r in vals),terminal_nulls=len(certificates),last=job['object_id'],error=bad)
         self.publish('execution_running')
     def dispatched(job,active):
         progress['peak_inflight_object_jobs']=max(progress['peak_inflight_object_jobs'],sum(bool(j['ready'])for j in active))
         progress['dispatch_events'].append(dict(object_id=job['object_id'],ready=job['ready'],elapsed_s=time.monotonic()-started));save(output/'PROGRESS.json',progress)
     work=bounded_jobs(jobs,self.c['execution_workers'],one,completed,dispatched)
     failed=any(r['failed']for r in work)
     # Read every dispatched executor's durable result even if validation failed.
     for job in jobs:
         oid=job['object_id'];item=progress['objects'].setdefault(oid,dict(status='not_dispatched',result={}))
         rp=output/oid/'RESULTS.json'
         if rp.exists():item['result']=read(rp);item['output_dir']=str(output/oid)
     rows=[r for value in progress['objects'].values()for r in value.get('result',{}).get('trials',[])]
     order={j['trial_id']:i for i,j in enumerate(self.p['slots'])};byid={j['trial_id']:j for j in self.p['slots']}
     if len({r['trial_id']for r in rows})!=len(rows):raise ValueError('Duplicate physical trial identity')
     rows.sort(key=lambda r:order[r['trial_id']]);records=[]
     def add_record(value,tid,inherited=False):
         if value.get('label_eligible')is True and value.get('result')in (0,1):
             row=next(r for r in rows if r['trial_id']==tid)
             if (type(value['result'])is not int or value['result']!=row['label']
                 or value['binding']['trial_binding_sha256']!=row['trial_binding_sha256']
                 or value['total_physics_steps']!=row['physics_steps']):
                 raise ValueError('Training record differs from actual trial binding/outcome/count')
             artifact=value['execution_artifacts']
             verify({artifact['evaluation_json']:artifact['evaluation_sha256'],artifact['commands_npz']:artifact['commands_sha256']})
             if artifact['commands_sha256']!=row['commands_sha256']:
                 raise ValueError('Training record command bytes differ from actual trace')
             from trial_record import collect_record
             expected=collect_record(value['binding'],artifact['evaluation_json'],artifact['commands_npz'])
             if any(value.get(k)!=v for k,v in expected.items()):
                 raise ValueError('Training export differs from original artifact-derived record')
             slot=byid[tid];value=copy.deepcopy(value)
             if (value['binding']['multifinger_pose_finger_type']!=slot['gesture_id']
                 or value['binding']['multifinger_pose_depth_type']!=slot['extra_depth_index']):
                 raise ValueError('Training record finger type/depth differs from declared slot')
             value['collection_context']=dict(trial_id=tid,object_id=slot['object_id'],split='train',gesture_id=slot['gesture_id'],
                 extra_depth_index=slot['extra_depth_index'],benchmark_only=False,protocol_sha256=sha(self.run/'protocol.json'),
                 inherited_from_pilot=False,independent_reset_from_settled_pose=True,original_protocol_sha256=None)
             records.append(value)
     for item in progress['objects'].values():
         if not item.get('output_dir'):continue
         dest=Path(item['output_dir'])
         for row in item.get('result',{}).get('trials',[]):
             rp=dest/'trials'/row.get('output_identity',row['trial_id'])/'training_record.json'
             if rp.exists():add_record(read(rp),row['trial_id']);assets.append(rp)
         for name in ['RESULTS.json','PREPARED.json','training_records.jsonl']:
             if(dest/name).exists():assets.append(dest/name)
     records.sort(key=lambda r:order[r['collection_context']['trial_id']])
     eligible={r['trial_id']for r in rows if r.get('label_eligible')is True and r.get('label')in (0,1)}
     if {r['collection_context']['trial_id']for r in records}!=eligible or len(records)!=len(eligible):raise ValueError('Training exports differ from actual eligible labels')
     actual={r['trial_id']for r in rows if r.get('physics_steps',0)>0}
     ready={r['trial_id']for r in queue['trials']}
     submitted=set()
     for marker in (output/'object_queues').glob('*.SUBMITTING.json'):
         submitted.update(read(marker)['trial_ids'])
     unknown=sorted((actual-eligible-set(certificates)) | ((submitted & ready)-actual))
     unexecuted=sorted(ready-actual-submitted)
     if unknown or unexecuted:failed=True
     counts={str(g):dict(target=100,labels=sum(r['gesture_id']==g and r.get('label_eligible')is True for r in rows),
         positive=sum(r['gesture_id']==g and r.get('label')==1 for r in rows),negative=sum(r['gesture_id']==g and r.get('label')==0 for r in rows))for g in GESTURES}
     for gesture,value in counts.items():
         value.update(shortfall=max(0,100-value['labels']),extra=max(0,value['labels']-100),depths={str(d):dict(target=25,labels=sum(r['gesture_id']==int(gesture) and r['extra_depth_index']==d and r.get('label_eligible')is True for r in rows))for d in DEPTHS})
         for depth in value['depths'].values():depth['shortfall']=max(0,25-depth['labels'])
     physical_steps=sum(r.get('physics_steps',0)for r in rows)
     result=dict(schema='anydex_boya_s40_collection_results_v1',status='infrastructure_stopped'if failed else'complete_with_null',complete=not failed,
         slot_count=640,objects=40,all_objects_training=True,collection_variant='settled_pose_fresh_v2',actual_grasps=len(actual),physical_steps=physical_steps,
         inherited_actual_grasps=0,inherited_physics_steps=0,new_actual_grasps=len(actual),new_physical_steps=physical_steps,
         eligible_labels=len(records),positives=sum(r.get('label')==1 for r in rows),negatives=sum(r.get('label')==0 for r in rows),
         preparation_nulls=queue['nulls'],terminal_certificates=certificates,terminal_nulls=len(certificates),
         unexecuted_ready_trial_ids=unexecuted,unknown_execution_trial_ids=unknown,physical_cost_complete=not unknown,counts=counts,trials=rows,
         new_CGR_forwards=40,inherited_CGR_forwards=0,new_training_updates=0,no_decision_head_loaded=True,
         wall_s=time.monotonic()-started,source_sha256_unchanged=True,input_sha256_unchanged=True)
     self.check()
     for fn in ['train_records.jsonl','all_records.jsonl']:
         with(output/fn).open('x')as f:
             for value in records:f.write(json.dumps(value,allow_nan=False)+'\n')
         assets.append(output/fn)
     save(output/'RESULTS.json',result);save(self.run/'collection_counts.json',{k:v for k,v in result.items()if k not in ['trials','terminal_certificates','preparation_nulls']})
     progress.update(status=result['status'],wall_s=result['wall_s']);save(output/'PROGRESS.json',progress)
     assets += [output/'RESULTS.json',output/'PROGRESS.json',self.run/'collection_counts.json']
     if failed:raise RuntimeError('Unknown execution/infrastructure evidence retained; no automatic replay')
     return dict(eligible_labels=len(records),physical_steps=physical_steps,new_physical_steps=physical_steps,counts=counts),assets

def parse_args():
 p=argparse.ArgumentParser(description=__doc__)
 for name in('run-dir','scene-manifest','previous-protocol','previous-dataset-manifest','previous-collection-results','native-assets-run','candidate-builder'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--calibration',type=Path,default=ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json')
 p.add_argument('--backend-config',type=Path,default=ROOT/'outputs/boya_pilot8_20261105_v1/backend_pilot.json')
 p.add_argument('--checkpoint',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18'));p.add_argument('--native-deps',type=Path,default=ROOT/'outputs/native_deps')
 p.add_argument('--device',default='cuda:0');p.add_argument('--geometry-workers',type=int,default=4);p.add_argument('--plan-workers',type=int,default=8);p.add_argument('--execution-workers',type=int,default=4);p.add_argument('--native-workers',type=int,default=4)
 p.add_argument('--phase',choices=['prepare','execute','all'],default='prepare');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 for name,maximum in [('geometry_workers',8),('plan_workers',16),('execution_workers',4),('native_workers',4)]:
  if not 1<=getattr(a,name)<=maximum:p.error('Invalid '+name)
 return a

def main():
 a=parse_args();run=a.run_dir.resolve();run.parent.mkdir(parents=True,exist_ok=True)
 with(run.parent/('.'+run.name+'.s40_settled.lock')).open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=freeze(a);pipeline=Pipeline(run,p);pipeline.check()
  if a.dry_run:
   immutable(run/'DRY_PLAN.json',dict(protocol_sha256=sha(run/'protocol.json'),slots=640,inherited_slots=0,new_slots=640,inherited_actual_grasps=0,new_CGR_forwards_planned=40,valid_label_target_per_gesture=100,gesture_ids=GESTURES,extra_depth_indices=DEPTHS,no_test_objects=True,new_physics_steps=0,new_training_updates=0))
   pipeline.publish('frozen_dry_plan_only');print(json.dumps(read(run/'DRY_PLAN.json'),indent=2));return
  try:
   phases=[('native_assets',pipeline.native),('observations',pipeline.observations),('representation',pipeline.representation),('coverage_history',pipeline.coverage_history),('geometry',pipeline.geometry),('planning',pipeline.planning)]
   if a.phase=='execute':
    for name,_ in phases:
     r=read(run/'stages'/(name+'.json'));require(r.get('complete')is True and r['protocol_sha256']==sha(run/'protocol.json'),'Execute requires all frozen preparation');verify(r['artifacts_sha256'])
   else:
    for name,fn in phases:pipeline.stage(name,fn)
   if a.phase in('execute','all'):
    r=pipeline.stage('execution',pipeline.execution);pipeline.progress.update(physical_steps=r['physical_steps'],new_physical_steps=r['new_physical_steps'],eligible_labels=r['eligible_labels']);pipeline.publish('collection_complete')
   else:pipeline.publish('preparation_complete')
  except BaseException:
   pipeline.progress['error']=traceback.format_exc();pipeline.publish('infrastructure_stopped');raise
if __name__=='__main__':main()
