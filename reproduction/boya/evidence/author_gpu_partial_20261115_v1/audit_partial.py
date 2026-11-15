"""One-pass read-only partial-run integrity/cost audit, never label validation."""
from pathlib import Path
import json
import time
import numpy as np
import audit_author_collection_runtime as A
import author_gpu_vector_audit as G

ROOT=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/gpu_vector_control_20261115_v2')
OUT=Path('/home/wangrenpeng/AnyDexGrasp/outputs/boya_gpu_vector_audit_20261115_v1/single_g6/AUDIT.json')
start=time.monotonic(); b=A.Bindings()
b.add(__file__); b.add(G.__file__,'7c508445f7905845b952b622d15a7767ad61eabcae038ed1f4b372796ae5b7ec'); b.add(A.__file__)
path=ROOT/'single_g6/result/GPU_VECTOR_EXECUTION.json';b.add(path);group=A.read(path)
assert group['status']=='infrastructure_failure' and group['complete'] is False
assert group['label'] is None and group['exclude_training'] is True and group['formal_training_eligible'] is False
assert group['retry_count']==0 and group['automatic_training'] is False
assert 'KeyboardInterrupt' in group['error'] and 'fetch_results()' in group['error']
b.mapping(group['source_sha256']);b.tree(group['reference_assets']);b.tree(group['protocol'])
protocol=A.read(group['protocol']['path']);b.tree(protocol['inputs']);b.tree(protocol['sources'])
b.add(ROOT/'GPU_CONTROL_RESULTS.json');outer=A.read(ROOT/'GPU_CONTROL_RESULTS.json');b.tree(outer['stages'])
assert outer['status']=='stopped_after_failure' and outer['label'] is None and outer['exclude_training'] is True
assert outer['formal_training_eligible'] is False and [s['name'] for s in outer['stages']]==['single_g6']
assert not (ROOT/'batch_g1_g6/STAGE.json').exists() and not (ROOT/'batch_g1_g6/result/GPU_VECTOR_STARTED.json').exists()
assert not (ROOT/'batch_g1_g6/result/GPU_VECTOR_EXECUTION.json').exists()
assert len(group['jobs'])==1 and group['environment_roots']==['/World/envs/env_0'] and group['origins_m']==[[0.,0.,0.]]
job=group['jobs'][0];b.tree(job['input_bindings']);b.tree(job['evaluation'])
source=Path(job['source_prepared']);ready=A.read(source/'READY.json')
for key in ('protocol','scene','candidate','plan','commands','command_binding'):
 assert job['input_bindings'][key]==ready[key]
source_protocol=A.read(ready['protocol']['path'])
for key in ('inputs','sources','capture_source_bindings'):b.tree(source_protocol.get(key,{}))
scene=A.read(ready['scene']['path']);plan=A.read(ready['plan']['path']);candidate=A.read(ready['candidate']['path']);command=A.read(ready['command_binding']['path'])
for name,digest in scene.get('source_sha256',{}).items():
 b.add(source_protocol.get('scene_archive_sources',{}).get(str(Path(name).resolve()),name),digest)
b.mapping(scene.get('input_sha256',{}))
for key in ('source_sha256','input_sha256','extra_input_sha256'):b.mapping(plan.get(key,{}))
for obj in scene['objects']:
 b.add(obj['source_mesh'],obj['source_mesh_sha256']);b.add(obj['source_visual_mesh'],obj['source_visual_mesh_sha256'])
arrays,_,_,counts=A._commands(source,ready,plan,candidate,command,b)
execution=ROOT/'single_g6/result/env_0/execution';evaluation=A.read(job['evaluation']['path'])
assert evaluation['complete'] is False and evaluation['label_eligible'] is False
assert job['label'] is None and job['raw_label'] is None and job['formal_training_eligible'] is False and job['exclude_training'] is True
assert not (execution.parent/'DIAGNOSTIC_RECORD.json').exists() and not (execution.parent/'COLLECTION_RECORD.json').exists()
for p in (ROOT/'single_g6/result').rglob('COLLECTION_RECORD.json'):raise AssertionError('Unexpected training record '+str(p))
trace_path=execution/'MULTI_OBJECT_TRACE.json';b.add(trace_path,evaluation['trace_sha256']);trace=A.read(trace_path)['rows']
n=len(trace);assert n==937 and group['physical_clock_steps']==group['recorded_command_environment_steps']==group['actual_integrated_environment_steps']==n
assert group['lane_recorded_physical_steps']==[n] and job['observations_completed_steps']==job['recorded_physical_steps']==n
assert evaluation['total_physics_steps']==n and evaluation['dynamic_object_steps']==n*3 and job['command_steps']==len(arrays['phase'])==3024
ids=[o['object_id'] for o in scene['objects']];assert len(ids)==3 and evaluation['original_object_ids']==ids
for i,row in enumerate(trace):
 assert row['step']==i+1 and row['phase']==str(arrays['phase'][i]) and abs(row['time_s']-(i+1)*.004)<=1e-12 and set(row['objects'])==set(ids)
 for state in row['objects'].values():
  A._array(state['pose_xyzw'],(7,));A._array(state['velocity'],(6,));A._array(state['finger_loads_n'],(5,))
contact_path=execution/'MULTI_OBJECT_CONTACTS.json';b.add(contact_path,evaluation['contacts_sha256'])
pointer=A.read(contact_path);assert pointer['record_count']==n and pointer['lossless'] is True and pointer['contact_records_omitted']==0
contact_rows=contact_points=0
for row in A.iter_contact_rows(contact_path,b):
 contact_rows+=1;assert row['step']==contact_rows
 for key in ('object_contacts','robot_contacts'):
  assert isinstance(row[key],list)
  for c in row[key]:
   contact_points+=1
   assert len(c['bodies'])==2
   for p in c['bodies']:G.normalize_path(p,'/World/envs/env_0')
   A._array(c['position_world_m'],(3,));assert np.isfinite(c['normal_force_n']) and np.isfinite(c['separation_m'])
   if key=='object_contacts':A._array(c['raw_normal_world'],(3,))
assert contact_rows==n
native_path=execution/'evaluation.json';b.add(native_path);native=A.read(native_path)
b.add(execution/'trace.npz',native['trace_sha256']);b.add(execution/'native_structure.json');b.add(execution/'MULTI_OBJECT_SETUP.json')
setup=A.read(execution/'MULTI_OBJECT_SETUP.json');structure=A.read(execution/'native_structure.json')
assert setup['native']['within_20um'] is True and structure['is_fixed_base'] is True and structure['num_base_dofs']==0
assert setup['physical_state_writes_during_execution']==0 and native['complete'] is False and native.get('gates') is None
with np.load(execution/'trace.npz',allow_pickle=False) as z:actual={k:z[k].copy() for k in z.files}
joints,legal=A._robot_xml(plan,b);names=list(map(str,actual['joint_names']))
assert len(names)==len(set(names))==28 and set(names)==set(joints)
expected=A._full_target(list(map(str,arrays['joint_names'])),arrays['joint_target'],joints,names)
assert np.array_equal(actual['joint_target'],expected[:n]) and np.array_equal(actual['desired_palm'],arrays['desired_palm'][:n])
assert np.array_equal(actual['time_s'],arrays['time_s'][:n]) and np.array_equal(actual['phase'],arrays['phase'][:n])
for key in ('joint_pos','joint_target','joint_velocity','joint_velocity_target'):A._array(actual[key],(n,28))
primary=scene['target_object_id'];assert np.max(abs(actual['object_pose_xyzw']-np.asarray([r['objects'][primary]['pose_xyzw']for r in trace])))<=1e-10
probe=G._probe_evidence(group,b,group['environment_roots']);assert group['native_scene']['actual_gpu_validated'] is True
assert group['native_scene']['native_scene']['physxScene:enableGPUDynamics'] is True
assert group['native_scene']['native_scene']['physxScene:broadphaseType']=='GPU'
assert not group.get('journal_finalize_errors') and not group.get('tail_finalize_errors') and not group.get('tail_observations')
b.verify()
result=dict(schema='anydex_boya_gpu_partial_integrity_audit_v1',status='partial_integrity_pass',
 full_execution_audit_run=False,full_gates_certified=False,label=None,raw_label=None,formal_training_eligible=False,exclude_training=True,
 known_completed_shared_ticks=n,integrated_shared_tick_interval=[n,n+1],known_observation_environment_steps=n,
 integrated_environment_step_interval=[n,n+1],integrated_dynamic_object_step_interval=[3*n,3*(n+1)],
 inflight_scope='SIGINT in physx_sim.fetch_results for the next single synchronous sim.step; its integration completion is unconfirmed. Completed callback counters are not an exact total integration count.',
 command_steps=len(arrays['phase']),unexecuted_command_suffix=len(arrays['phase'])-n,phase_prefix=list(dict.fromkeys(map(str,actual['phase']))),
 full_command_phase_counts=counts,observations=n,contact_rows=contact_rows,contact_points=contact_points,
 candidate_sha256=candidate['candidate_sha256'],commands_sha256=ready['commands']['sha256'],original_object_ids=ids,
 source_mesh_sha256=[o['source_mesh_sha256']for o in scene['objects']],original_source_mesh_sha256=[o['source_visual_mesh_sha256']for o in scene['objects']],
 actual_24_to_28_prefix_targets_exact=True,actual_palm_phase_clock_prefix_exact=True,source_native_20um_receipt=True,
 native_hulls=setup['native'],gpu_probes=probe,full_native_gates_present=False,
 source_nul_failure_or_old_labels_modified=False,batch_dispatched=False,new_labels=0,training_updates=0,
 execution_wall_before_close_s=group['wall_before_close_s'],outer_wall_s=outer['wall_s'],
 audit_physics_steps=0,audit_Kit_launches=0,audit_model_forwards=0,audit_wall_s=time.monotonic()-start,
 audit_inputs_sha256=b.files,audit_source=dict(path=str(Path(__file__).resolve()),sha256=A.sha(__file__)))
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:result[k]for k in ('status','observations','contact_rows','contact_points','integrated_shared_tick_interval','audit_wall_s','label')},indent=2))
