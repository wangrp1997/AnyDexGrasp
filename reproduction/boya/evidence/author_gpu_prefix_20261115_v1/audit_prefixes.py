"""One read-only audit of the frozen two-group, three-lane 20-tick profile."""
from pathlib import Path
import json,time
import numpy as np
import audit_author_collection_runtime as A
import author_gpu_vector_audit as G
ROOT=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/gpu_partition_profile_20261115_v1')
OUT=Path('/home/wangrenpeng/AnyDexGrasp/outputs/boya_gpu_prefix_audit_20261115_v1/AUDIT.json')
class Bindings(A.Bindings):
 def add(self,path,expected=None):
  path=str(Path(path).resolve())
  if path in self.files:
   assert expected is None or expected==self.files[path], 'Conflicting source binding '+path
   return dict(path=path,sha256=self.files[path])
  return super().add(path,expected)
start=time.monotonic();b=Bindings();b.add(__file__);b.add(A.__file__);b.add(G.__file__,'7c508445f7905845b952b622d15a7767ad61eabcae038ed1f4b372796ae5b7ec')
outer_path=ROOT/'GPU_PROFILE_CONTROL_RESULTS.json';protocol_path=ROOT/'GPU_PROFILE_CONTROL_PROTOCOL.json'
b.add(outer_path);b.add(protocol_path);outer=A.read(outer_path);protocol=A.read(protocol_path)
b.tree(outer);b.tree(protocol);assert outer['status']=='bounded_prefixes_complete' and outer['label'] is None and outer['exclude_training'] is True
assert outer['formal_training_eligible'] is False and outer['training_updates']==0
reports=[];all_contact_rows=all_contact_points=0
for group_name,lanes in [('single_g6',1),('batch_g1_g6',2)]:
 root=ROOT/group_name/'result';gp=root/'GPU_VECTOR_EXECUTION.json';pp=root/'GPU_VECTOR_PROFILE.json'
 b.add(gp);b.add(pp);group=A.read(gp);profile=A.read(pp)
 b.mapping(group['source_sha256']);b.mapping(profile['source_sha256']);b.tree(group.get('reference_assets',{}));b.tree(group['protocol']);b.tree(profile['group_receipt']);b.tree(profile['native_execution_profiles'])
 assert group['status']==profile['status']=='partial_profile' and group['complete'] is False and profile['complete'] is False
 assert group['label'] is None and profile['label'] is None and profile['raw_label'] is None
 assert group['exclude_training'] is True and group['formal_training_eligible'] is False and profile['formal_training_eligible'] is False
 assert group['physical_clock_steps']==profile['actual_physics_steps']==profile['prefix_ticks']==20
 assert group['actual_integrated_environment_steps']==group['recorded_command_environment_steps']==profile['actual_integrated_environment_steps']==20*lanes
 assert group['clock']['prefix_limited'] is True and group['clock']['requested_prefix_ticks']==20 and group['clock']['commands_added']==0
 assert group['clock']['lane_steps']==[20]*lanes and group['clock']['completed_lane_tail_integrated_steps']==[0]*lanes
 assert len(group['jobs'])==lanes and not group.get('tail_observations') and not group.get('journal_finalize_errors')
 assert profile['error'] is None and group.get('retry_count')==0
 assert group['native_scene']['actual_gpu_validated'] is True and group['native_scene']['cross_environment_contacts']==0
 attrs=group['native_scene']['native_scene'];assert attrs['physxScene:enableGPUDynamics'] is True and attrs['physxScene:broadphaseType']=='GPU' and attrs['physxScene:timeStepsPerSecond']==250
 probes=G._probe_evidence(group,b,group['environment_roots']);lane_reports=[]
 for i,job in enumerate(group['jobs']):
  assert job['environment']==i and job['label'] is None and job['raw_label'] is None and job['complete'] is False
  assert job['formal_training_eligible'] is False and job['exclude_training'] is True and job['completion_tick']==20 and job['tail_integrated_steps']==0
  b.tree(job['input_bindings']);b.tree(job['evaluation']);b.tree(job['diagnostic_record'])
  source=Path(job['source_prepared']);ready=A.read(source/'READY.json')
  for key in ('protocol','scene','candidate','plan','commands','command_binding'):assert ready[key]==job['input_bindings'][key]
  sp=A.read(ready['protocol']['path'])
  for key in ('inputs','sources','capture_source_bindings'):b.tree(sp.get(key,{}))
  scene=A.read(ready['scene']['path']);plan=A.read(ready['plan']['path']);candidate=A.read(ready['candidate']['path']);command=A.read(ready['command_binding']['path'])
  for key in ('source_sha256','input_sha256','extra_input_sha256'):b.mapping(plan.get(key,{}))
  for name,digest in scene.get('source_sha256',{}).items():b.add(sp.get('scene_archive_sources',{}).get(str(Path(name).resolve()),name),digest)
  b.mapping(scene.get('input_sha256',{}))
  for obj in scene['objects']:
   b.add(obj['source_mesh'],obj['source_mesh_sha256']);b.add(obj['source_visual_mesh'],obj['source_visual_mesh_sha256'])
  arrays,_,_,_=A._commands(source,ready,plan,candidate,command,b)
  execution=root/f'env_{i}/execution';evaluation=A.read(job['evaluation']['path']);record=A.read(job['diagnostic_record']['path'])
  assert evaluation['total_physics_steps']==20 and evaluation['dynamic_object_steps']==60 and evaluation['complete'] is False and evaluation['label_eligible'] is False
  assert record['label'] is None and record['diagnostic_raw_label'] is None and record['formal_training_eligible'] is False and record['exclude_training'] is True
  assert record['candidate']==candidate and record['command_binding']==command and record['evaluation_binding']==job['evaluation']
  assert not (execution.parent/'COLLECTION_RECORD.json').exists()
  trace_path=execution/'MULTI_OBJECT_TRACE.json';contact_path=execution/'MULTI_OBJECT_CONTACTS.json'
  b.add(trace_path,evaluation['trace_sha256']);b.add(contact_path,evaluation['contacts_sha256'])
  rows=A.read(trace_path)['rows'];pointer=A.read(contact_path);ids=[o['object_id']for o in scene['objects']]
  assert len(rows)==pointer['record_count']==20 and len(ids)==3 and evaluation['original_object_ids']==ids
  assert pointer['lossless'] is True and pointer['contact_records_omitted']==0
  for j,row in enumerate(rows):
   assert row['step']==j+1 and row['phase']==str(arrays['phase'][j]) and abs(row['time_s']-(j+1)*.004)<=1e-12 and set(row['objects'])==set(ids)
   for state in row['objects'].values():A._array(state['pose_xyzw'],(7,));A._array(state['velocity'],(6,));A._array(state['finger_loads_n'],(5,))
  count=points=0
  for c_row in A.iter_contact_rows(contact_path,b):
   count+=1;assert c_row['step']==count
   for category in ('object_contacts','robot_contacts'):
    assert isinstance(c_row[category],list)
    for c in c_row[category]:
     points+=1;assert len(c['bodies'])==2
     for path in c['bodies']:G.normalize_path(path,job['environment_root'])
     A._array(c['position_world_m'],(3,));assert np.isfinite(c['normal_force_n']) and np.isfinite(c['separation_m'])
     if category=='object_contacts':A._array(c['raw_normal_world'],(3,))
  assert count==20
  native_path=execution/'evaluation.json';b.add(native_path);native=A.read(native_path)
  for key in ('source_sha256','input_sha256'):b.mapping(native.get(key,{}))
  b.add(execution/'trace.npz',native['trace_sha256']);b.add(execution/'MULTI_OBJECT_SETUP.json');b.add(execution/'native_structure.json')
  setup=A.read(execution/'MULTI_OBJECT_SETUP.json');structure=A.read(execution/'native_structure.json')
  assert setup['native']['within_20um'] is True and structure['is_fixed_base'] is True and structure['num_base_dofs']==0
  assert setup['physical_state_writes_during_execution']==0 and native['complete'] is False
  origin=np.asarray(job['origin_m']);normalized=G._normalized_setup(setup,job['environment_root'],origin)
  for obj in scene['objects']:
   entry=normalized['entry'][obj['object_id']];wanted=np.asarray(obj['world_T_object'])[:3,3]
   assert np.max(abs(np.asarray(entry['pose_xyzw'])[:3]-wanted))<=2e-5
  with np.load(execution/'trace.npz',allow_pickle=False)as z:actual={k:z[k].copy()for k in z.files}
  joints,_=A._robot_xml(plan,b);names=list(map(str,actual['joint_names']));assert len(names)==len(set(names))==28 and set(names)==set(joints)
  targets=A._full_target(list(map(str,arrays['joint_names'])),arrays['joint_target'],joints,names)
  assert np.array_equal(actual['joint_target'],targets[:20]) and np.array_equal(actual['desired_palm'],arrays['desired_palm'][:20])
  assert np.array_equal(actual['time_s'],arrays['time_s'][:20])and np.array_equal(actual['phase'],arrays['phase'][:20])
  for key in ('joint_pos','joint_target','joint_velocity','joint_velocity_target'):A._array(actual[key],(20,28))
  assert np.max(abs(actual['object_pose_xyzw']-np.asarray([r['objects'][scene['target_object_id']]['pose_xyzw']for r in rows])))<=1e-10
  lane_reports.append(dict(environment=i,origin_m=origin.tolist(),original_object_ids=ids,candidate_sha256=candidate['candidate_sha256'],commands_sha256=ready['commands']['sha256'],trace_rows=20,contact_rows=count,contact_points=points,targets24_to_28_prefix_exact=True,palm_phase_clock_prefix_exact=True,entry_origin_projection_ok=True,native_hulls=setup['native']['hulls'],native_20um_receipt=True,label=None,raw_label=None,formal_training_eligible=False))
  all_contact_rows+=count;all_contact_points+=points
 timing=profile['timing'];prefix=timing['execution_prefix'];wall=group['clock']['prefix_clock_including_generator_close_wall_s']
 for phase,metrics in timing.items():
  assert phase in ('initialization','execution_prefix','finalization')
  for value in metrics.values():assert type(value['calls'])is int and value['calls']>=0 and 0<=value['exclusive_wall_s']<=value['inclusive_wall_s']+1e-12
 for name in ('physx.simulate','physx.fetch_results','sim.step'):assert prefix[name]['calls']==20
 simulate=prefix['physx.simulate']['inclusive_wall_s'];fetch=prefix['physx.fetch_results']['inclusive_wall_s'];step=prefix['sim.step']['inclusive_wall_s']
 assert abs(step-simulate-fetch-prefix['sim.step']['exclusive_wall_s'])<1e-9
 exclusive=sum(x['exclusive_wall_s']for x in prefix.values());assert exclusive<=wall+1e-6
 reports.append(dict(group=group_name,physical_ticks=20,environment_steps=20*lanes,dynamic_object_steps=60*lanes,cross_environment_contacts=0,gpu_probes=probes,lanes=lane_reports,
  timing=dict(prefix_wall_s=wall,group_wall_before_close_s=group['wall_before_close_s'],profile_wall_before_close_s=profile['wall_before_close_s'],physx_simulate_s=simulate,physx_fetch_results_s=fetch,sim_step_inclusive_s=step,fetch_fraction_of_prefix=fetch/wall,filtered6_readback_inclusive_s=prefix['contact.filtered6_readback']['inclusive_wall_s'],allbody_decode_s=prefix['all_body.contact_decode']['inclusive_wall_s'],journal_inclusive_s=prefix['journal.encode_gzip_append']['inclusive_wall_s'],exclusive_instrumented_sum_s=exclusive,uninstrumented_prefix_residual_s=wall-exclusive,timing_all_sections=timing),label=None,full_lift_certified=False))
b.verify()
result=dict(schema='anydex_boya_gpu_prefix_integrity_audit_v1',status='prefix_integrity_pass',groups=reports,shared_ticks=40,environment_steps=60,dynamic_object_steps=180,trace_rows=60,contact_rows=all_contact_rows,contact_points=all_contact_points,
 full_gates_certified=False,full_lift_certified=False,full_collection_ETA=None,label=None,raw_label=None,formal_training_eligible=False,exclude_training=True,new_labels=0,training_updates=0,
 timing_scope='Host wall and wait attribution only, no added GPU sync. fetch includes blocked wait for submitted async work; sim.step includes fetch and simulate. Only disjoint exclusive buckets may be summed. Initialization/finalization kept separate. Native per-lane physics buckets overlap across lanes.',
 outer_wall_s=outer['wall_s'],audit_wall_s=time.monotonic()-start,audit_physics_steps=0,audit_Kit_launches=0,audit_inputs_sha256=b.files,audit_source=dict(path=str(Path(__file__).resolve()),sha256=A.sha(__file__)))
with OUT.open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:result[k]for k in ('status','shared_ticks','environment_steps','dynamic_object_steps','trace_rows','contact_rows','contact_points','audit_wall_s','label')},indent=2))
