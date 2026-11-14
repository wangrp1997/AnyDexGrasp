"""Read-only audit of the frozen four-type pure preparation, no simulator."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,time
import numpy as np
ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
OUT=ROOT/'outputs/boya_author_pipeline_fourtype_20261114_v2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
queue=read(OUT/'QUEUE_PROTOCOL.json');launch=read(OUT/'LAUNCH_SOURCE_SHA256.json')
start=time.monotonic();json_checks=[]
for path in OUT.rglob('*.json'):
 raw=path.read_bytes();assert b'\x00' not in raw,str(path);json.loads(raw);json_checks.append(str(path))
for row in queue['inputs'].values():assert sha(row['path'])==row['sha256'],row['path']
for p,d in launch['sources'].items():
 resolved=Path(p)
 if sha(resolved)!=d:
  archive={'author_pipeline.py':'AUTHOR_PIPELINE_SOURCE.py','author_transport.py':'AUTHOR_TRANSPORT_SOURCE.py'}.get(resolved.name)
  assert archive and sha(OUT/archive)==d,p
  resolved=OUT/archive
 assert sha(resolved)==d,p
with np.load(queue['inputs']['raw_candidates']['path'],allow_pickle=False) as z:
 raw=z['candidates'].copy();features=z['candidate_features'].copy()
scene=read(queue['inputs']['scene']['path']);obs=read(queue['inputs']['observation']['path'])
W=np.asarray(obs['T_WC']);raw_world=raw[:,12:15]@W[:3,:3].T+W[:3,3]
R=raw[:,3:12].reshape(-1,3,3);world_axes=R[:,:,0]@W[:3,:3].T
width=(raw[:,1]>.01)&(raw[:,1]<.099)
workspace=((raw_world>[-1.,-.04,-.02])&(raw_world<[-.57,.24,.30])).all(axis=1)
approach=world_axes[:,2]<-.9
keep=np.flatnonzero(width&workspace&approach);top=keep[np.argsort(raw[keep,0])[::-1][:500]]
assert len(top)==97
rng=np.random.RandomState(20261114);extra=rng.randint(0,4,len(top))*.01
results=[];locked={};counts_total=0
for job in queue['jobs']:
 g=job['gesture_id'];out=Path(job['output']);result=read(out/'PIPELINE_RESULTS.json');prep=read(out/'collection/PREPARED.json');c=prep['candidates'][0]
 assert result['status']=='no_plan' and result['stage']=='fixed_candidate_arm' and result['complete']
 assert result['physics_steps']==result['new_labels']==result['training_updates']==result['new_CGR_forwards']==0
 assert c['source_row_index']==879 and c['source_top500_rank']==73 and c['source_if_flip'] is False
 assert c['slot_id']==f'g{g}:distal_000:ff' and c['extra_depth_m']==.01
 assert c['label'] is None and c['label_state']=='unexecuted'
 assert np.array_equal(np.asarray(c['source_row15_camera']),raw[879])
 assert np.array_equal(np.asarray(c['source_feature2261']),features[879])
 prior=read(ROOT/f'outputs/boya_author_collector_snapshot_three_20261114_v1/prepare_g{g}/PREPARED.json')['candidates'][0]
 for k in ('actual_row15_camera','actual_row17_camera','source_row15_camera','canonical_cgr480','camera_T_palm','world_T_palm','source_feature2261','slot_id','extra_depth_m','source_if_flip'):
  assert c[k]==prior[k],(g,k)
 with np.load(out/'collection/PREPARATION_ARRAYS.npz',allow_pickle=False) as z:
  assert np.array_equal(z['domain_keep_indices'],keep)
  assert np.array_equal(z['source_indices'],top)
  assert np.array_equal(z['extra_depth_m'],extra)
  source_count=len(z['source_indices'])
  counts_total+=prep['counts']['mapping_available']
  for i,index in enumerate(top):
   expected=raw[index].copy();flip=bool(expected[7]>0)
   if flip:expected[3:12].reshape(3,3)[:,:2]*=-1
   row17=np.r_[expected[:2],np.float32(.03),expected[2:],np.float32(-1)]
   assert np.array_equal(z['actual_rows17_camera'][i],row17),(g,i)
   assert bool(z['source_if_flip'][i])==flip
 arm=read(out/'arm/ARM_PREPARATION.json');plan=read(out/'arm/native/plan.json');geometry=read(out/'arm/native/path_geometry.json');failure=read(out/'arm/native/admission_failure.json')
 assert arm['status']=='no_plan' and arm['complete'] and arm['returncode']==1 and not arm['timed_out']
 assert failure['complete'] and failure['kind']=='arm_plan_rejected' and failure['physical_executed'] is False
 assert plan['approach_only'] and not plan['admitted']
 assert plan['arguments']['pregrasp_distance_m']==.04
 assert read(out/'BACKEND.json')['pregrasp_distance_m']==.04
 assert read(out/'BACKEND.json')['original_backend_pregrasp_distance_m']==.1
 assert sha(out/'arm/native/commands.npz')==plan['commands_sha256']
 assert sha(out/'arm/native/path_geometry.json')==plan['path_geometry_sha256']
 assert not (out/'READY.json').exists() and not (out/'transport_raw').exists() and not (out/'execution').exists()
 with np.load(out/'arm/native/commands.npz',allow_pickle=False) as z:commands={k:z[k].copy() for k in z.files}
 phases=commands['phase'];assert Counter(phases.tolist())=={'settle':50,'home_to_pregrasp':750,'approach':500}
 assert commands['joint_target'].shape==(1300,24)
 names=list(commands['joint_names']);expected_open=np.asarray(plan['open_motors']);mi=[names.index(x) for x in plan['hand_motor_names']]
 assert np.max(abs(commands['joint_target'][:,mi]-expected_open))==0
 palm_error=float(np.max(abs(commands['desired_palm'][-1]-np.asarray(c['world_T_palm']))));assert palm_error<1e-6
 v=geometry['violations'];phase_details={}
 for p in ('home_to_pregrasp','approach'):
  rows=[x for x in v if phases[x['sample']]==p]
  if not rows:
   phase_details[p]=dict(violations=0,endpoint_sample=int(np.flatnonzero(phases==p)[-1]),endpoint_violations=0);continue
  first=min(x['sample'] for x in rows);last=max(x['sample'] for x in rows);phase_start=int(np.flatnonzero(phases==p)[0])
  phase_details[p]=dict(violations=len(rows),first_global_sample=first,last_global_sample=last,first_local_sample=first-phase_start,last_local_sample=last-phase_start,first_time_s=float(commands['time_s'][first]),last_time_s=float(commands['time_s'][last]),max_penetration_m=-min(x['distance_m'] for x in rows),endpoint_sample=int(np.flatnonzero(phases==p)[-1]),endpoint_violations=sum(x['sample']==int(np.flatnonzero(phases==p)[-1]) for x in rows))
 assert all(x['kind']=='object' and x['other'] in (0,1) for x in v)
 assert all(x['link'].startswith(('th','ff','mf','rf','lf')) for x in v)
 protocol=read(out/'PIPELINE_PROTOCOL.json')
 for b in list(protocol['inputs'].values())+list(protocol['sources'].values())+list(protocol['capture_source_bindings'].values()):
  path=Path(b['path'])
  if sha(path)!=b['sha256']:
   archive={'author_pipeline.py':'AUTHOR_PIPELINE_SOURCE.py','author_transport.py':'AUTHOR_TRANSPORT_SOURCE.py'}.get(path.name)
   assert archive and sha(OUT/archive)==b['sha256'],path
  else:assert sha(path)==b['sha256']
 for p in out.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz'):locked[str(p)]=sha(p)
 results.append(dict(gesture_id=g,status=result['status'],candidate_sha256=c['candidate_sha256'],selected_source_row=879,slot_id=c['slot_id'],source_geometry_counts=prep['counts'],source_top500_count=source_count,phase_steps=dict(Counter(phases.tolist())),command_rows=1300,command_columns=24,source_open_hand_max_error_rad=0.,endpoint_candidate_palm_matrix_error=palm_error,violation_count=len(v),violation_kinds=dict(Counter(x['kind'] for x in v)),violation_links=dict(Counter(x['link'] for x in v)),phase_details=phase_details,all_violations_fingers_vs_primary=True,primary_object_part_counts=dict(Counter(str(x['other']) for x in v)),all_other_kinds_clear=True,final_approach_endpoint_violations=sum(x['sample']==1299 for x in v),unused_lift_audit=read(out/'arm/native/lift_arm_geometry.json'),arm_wall_s=arm['wall_s'],pipeline_wall_s=result['wall_s']))
value=dict(schema='anydex_boya_author_fourtype_preparation_independent_audit_v1',status='passed',queue=binding(OUT/'QUEUE_PROTOCOL.json'),launch=binding(OUT/'LAUNCH_SOURCE_SHA256.json'),input_identity_audit=queue['inputs']['input_identity_audit'],json_parse_and_no_nul={'checked':len(json_checks),'all_passed':True},independent_domain_and_source_rows={'raw_candidates':2048,'source_domain_keep':97,'raw_world_selected_translation_m':raw_world[879].tolist(),'selected_raw_camera_translation_m':raw[879,12:15].tolist(),'MT19937_all97_depths_equal':True,'source_single_flip_all388_rows_equal':True},results=results,source_collision_checks=counts_total,ready_count=0,no_plan_count=4,infrastructure_failure_count=0,transport_preparations=0,physics_steps=0,new_labels=0,training_updates=0,new_CGR_forwards=0,kit_launches=0,source_and_inputs_unchanged=True,output_bindings=locked,audit_wall_s=time.monotonic()-start,boundary='Source4cm corrected the previously colliding pregrasp endpoint/approach, but the same first choices still fail middle home-to-pregrasp direct paths. No transport/physics was reached. Known release-GT defect remains explicitly pending; no execution is allowed.')
(OUT/'INDEPENDENT_PREPARATION_AUDIT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:value[k] for k in ['status','json_parse_and_no_nul','independent_domain_and_source_rows','ready_count','no_plan_count','source_collision_checks']},indent=2))
