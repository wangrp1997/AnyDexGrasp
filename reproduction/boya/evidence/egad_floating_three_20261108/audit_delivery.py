from pathlib import Path
import ast, hashlib, json, math, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/home/wangrenpeng/AnyDexGrasp/outputs/egad_floating_three_20261108_v1')
B=Path('/home/wangrenpeng/AnyDexGrasp/outputs/egad_external12_20261107_v1')
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
 return h.hexdigest()
def zread(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def ref(p):return dict(path=str(p),sha256=sha(p))
def phase_stats(t):
 out={}
 for phase in dict.fromkeys(t['phase']):
  ix=np.flatnonzero(t['phase']==phase);l=t['finger_loads_n'][ix];both=(l[:,0]>.001)&(l[:,1:].sum(1)>.001)
  dp=t['object_pose_xyzw'][ix[-1],:3]-t['object_pose_xyzw'][ix[0],:3]
  dr=(Rotation.from_quat(t['object_pose_xyzw'][ix[0],3:]).inv()*Rotation.from_quat(t['object_pose_xyzw'][ix[-1],3:])).magnitude()
  out[str(phase)]=dict(steps=len(ix),start_step=int(ix[0]+1),end_step=int(ix[-1]+1),max_finger_loads_n=l.max(0).tolist(),final_finger_loads_n=l[-1].tolist(),both_loaded_steps=int(both.sum()),last_both_loaded_local_step=int(np.flatnonzero(both)[-1]+1)if both.any()else None,object_displacement_m=dp.tolist(),object_rotation_rad=float(dr))
 return out
p=read(R/'protocol.json');res=read(R/'RESULTS.json');assert res['status']=='complete' and not res['errors'];assert sha(R/'protocol.json')==res['protocol_sha256']
report=dict(schema='anydex_floating_three_readonly_independent_audit_v1',protocol=ref(R/'protocol.json'),results=ref(R/'RESULTS.json'),scope='Saved trace/input/source read-only recomputation; no physical execution or model update',objects=[],limitations=[])
for section in ['source_sha256','input_sha256']:
 for f,h in p[section].items():assert sha(f)==h,(section,f)
report['freeze_hashes_match']=True;report['source_count']=len(p['source_sha256']);report['input_count']=len(p['input_sha256'])
# Source inspection supplements reported runtime counters, not an instrumented independent write log.
source=Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya/run_boya_floating_diagnostic.py');tree=ast.parse(source.read_text())
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
loop=next(n for n in ast.walk(run)if isinstance(n,ast.For)and any(isinstance(a,ast.Call)and isinstance(a.func,ast.Attribute)and a.func.attr=='step'for a in ast.walk(n)))
calls=sorted({n.func.attr for n in ast.walk(loop)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)})
forbidden=[s for s in calls if any(k in s for k in ['write_root','write_joint_state','set_world_pose','set_local_pose','set_kinematic'])]
assert not forbidden
report['physical_loop_no_pose_or_state_setter_calls']=True;report['physical_loop_calls']=calls
oldresults=read(B/'execution/RESULTS.json')
for oid in p['object_ids']:
 job=read(R/'jobs'/f'{oid}.json');D=R/'execution'/oid;e=read(D/'evaluation.json');na=read(D/'native_audit.json');t=zread(D/'trace.npz');cmd=zread(D/'commands.npz');pool=zread(Path(job['candidate_dir'])/'candidates.npz');generation=read(Path(job['candidate_dir'])/'generation.json')
 assert e['complete']and e['total_physics_steps']==2100 and all(e['gates'].values())
 assert not e['arm_present']and not e['outcome_dependent_retry'] and not e['training_writeback']
 assert all(e[k]==0 for k in ['state_writes_during_execution','root_pose_writes_during_execution','object_state_writes_during_execution','new_CGR_forwards','new_training_updates'])
 assert sha(D/'trace.npz')==e['trace_sha256'] and sha(D/'commands.npz')==e['commands_sha256']
 assert e['hand_candidate_sha256']==sha(Path(job['candidate_dir'])/'candidates.npz')==generation['candidates_sha256']
 assert generation['records'][0]==job['candidate_record']
 assert all(np.isfinite(x).all()for x in t.values()if x.dtype.kind in 'fiu')
 for key in ['joint_target','velocity_target','desired_palm','phase','time_s','joint_names']:assert np.array_equal(t[key],cmd[key]),(oid,key)
 assert np.array_equal(t['time_s'],np.arange(1,2101)*.004)
 phases=np.concatenate([np.repeat(ph,n)for ph,n in p['phases_steps'].items()]);assert np.array_equal(t['phase'],phases)
 names=list(t['joint_names']);motors=list(pool['independent_motor_names']);mi=[names.index(x)for x in motors]
 opened=pool['proposal_q'][0];closed=pool['proposal_q_close'][0];palm=pool['proposal_palm'][0];direction=pool['proposal_approach_direction'][0]
 # Reconstruct independent motor path without calling new runner.
 wanted=np.repeat(opened[None],2100,axis=0)
 v=np.arange(1,251,dtype=float)/250.;wanted[1300:1550]=(1-v[:,None])*opened+v[:,None]*closed
 wanted[1550:]=closed
 assert np.array_equal(t['joint_target'][:,mi],wanted.astype(np.float32))
 # Independently reconstruct fixed wrist orientation and original 0.1m approach/lift.
 desired=np.repeat(palm[None],2100,axis=0);pre=palm[:3,3]-.1*direction;desired[:800,:3,3]=pre
 s=np.arange(1,501,dtype=float)/500.;u=s*s*s*(10.+s*(-15.+6.*s))
 desired[800:1300,:3,3]=pre+u[:,None]*.1*direction
 desired[1600:,:3,3]+=u[:,None]*np.array([0,0,.1])
 assert np.allclose(t['desired_palm'],desired,atol=2e-14,rtol=0)
 xyz=(desired[:,:3,3]-pre)@palm[:3,:3];slides=[names.index('diagnostic_slide_'+a)for a in 'xyz'];assert np.array_equal(t['joint_target'][:,slides],xyz.astype(np.float32))
 # All-joint positions, actual limit and mimic errors from generated original subtree.
 urdf=ET.parse(D/'boya_translation_gantry.urdf').getroot();mxlimit=mxmimic=0.
 for j in urdf.findall('joint'):
  name=j.get('name')
  if name not in names:continue
  values=t['joint_pos'][:,names.index(name)]
  lim=j.find('limit')
  if lim is not None and j.get('type')in ['revolute','prismatic']:
   mxlimit=max(mxlimit,float(np.maximum(np.maximum(float(lim.get('lower'))-values,values-float(lim.get('upper'))),0).max()))
  mimic=j.find('mimic')
  if mimic is not None:
   delta=values-t['joint_pos'][:,names.index(mimic.get('joint'))]*float(mimic.get('multiplier','1'))-float(mimic.get('offset','0'))
   mxmimic=max(mxmimic,float(np.abs(delta).max()))
 assert abs(mxlimit-e['maxima']['joint_limit_violation_rad'])<2e-7
 assert abs(mxmimic-e['maxima']['mimic_error_rad'])<2e-7
 pp=np.linalg.norm(t['palm_pose'][:,:3,3]-desired[:,:3,3],axis=1)
 angles=(Rotation.from_matrix(t['palm_pose'][:,:3,:3]).inv()*Rotation.from_matrix(desired[:,:3,:3])).magnitude()
 assert np.isclose(pp.max(),e['maxima']['palm_position_error_m'],atol=1e-12)
 assert np.isclose(angles.max(),e['maxima']['palm_orientation_error_rad'],atol=5e-8)
 maxpenetration=float(max(t['object_penetration_m'].max(),t['robot_penetration_m'].max()))
 assert np.isclose(maxpenetration,e['maxima']['penetration_m'],atol=1e-12)
 assert na['within_20um']and e['native_within_20um']and na['physics_steps']==0
 assert na['max_source_outside_cooked_bound_m']<=2e-5 and na['max_cooked_outside_source_bound_m']<=2e-5
 # Source full triangle vertices, no OBB/reference-only shortcut.
 row=job['object_row'];mesh=Path(row['mesh_path']);assert sha(mesh)==row['mesh_sha256']
 vertices=np.array([[float(x)for x in l.split()[1:4]]for l in mesh.read_text().splitlines()if l.startswith('v ')])*row['scale']
 poses=t['object_pose_xyzw'].astype(float);rot=Rotation.from_quat(poses[:,3:]).as_matrix();bottom=np.empty(2100)
 for i in range(0,2100,64):bottom[i:i+64]=(vertices@rot[i:i+64,2,:].T+poses[i:i+64,2]).min(0)
 hold=np.flatnonzero(t['phase']=='hold');lift=t['phase']=='lift';rise=poses[:,2]-poses[hold[-1],2];both=(t['finger_loads_n'][:,0]>.001)&(t['finger_loads_n'][:,1:].sum(1)>.001)
 picked=lift&(bottom>=.005)&(rise>=.005)&both
 pickup=e['tabletop_pickup'];assert int(picked.sum())==pickup['qualifying_steps']and bool(picked.any())==pickup['observed_pickup']and bool(picked[-1])==pickup['final_state_still_lifted_and_bearing']
 assert np.isclose(bottom[-1],pickup['final_bottom_clearance_m'],atol=2e-12)and np.isclose(rise[-1],pickup['final_reference_rise_m'],atol=2e-12)
 success=bool(all(e['gates'].values())and picked[-1]and rise[-1]>=.05);assert success==e['task_success']
 rec=dict(object_id=oid,candidate_rank=job['candidate_rank'],gesture_id=job['candidate_record']['gesture_id'],candidate=ref(Path(job['candidate_dir'])/'candidates.npz'),evaluation=ref(D/'evaluation.json'),trace=ref(D/'trace.npz'),all_2100_motor_targets_reconstructed_exact_float32=True,gantry_targets_reconstructed_exact_float32=True,desired_palm_reconstructed=True,trace_commands_and_time_arrays_exact=True,reported_state_writes=0,source_loop_state_setters_absent=True,gate_values=e['gates'],native_hulls=na['hulls'],native_20um_pass=True,recomputed_maximum_joint_limit_violation_rad=mxlimit,recomputed_maximum_mimic_error_rad=mxmimic,recomputed_maximum_penetration_m=maxpenetration,recomputed_palm_position_error_m=float(pp.max()),recomputed_palm_angle_rad=float(angles.max()),recomputed_pickup=dict(qualifying_steps=int(picked.sum()),final_bottom_clearance_m=float(bottom[-1]),final_reference_rise_m=float(rise[-1]),lift_peak_bottom_clearance_m=float(bottom[lift].max()),lift_peak_reference_rise_m=float(rise[lift].max()),two_sides_loaded_lift_steps=int((both&lift).sum()),final_finger_loads_n=t['finger_loads_n'][-1].tolist(),task_success=success),phases=phase_stats(t))
 if oid in ['egad_A3','egad_E0']:
  prior=next(x for x in oldresults['results']if x['object_id']==oid);od=Path(prior['trial']['execution_dir']);ot=zread(od/'trace.npz');oe=read(od/'evaluation.json');onames=list(ot['joint_names']);common=[n for n in names if n in onames];columns=[names.index(n)for n in common];ocolumns=[onames.index(n)for n in common]
  # All 20 finger DOFs including mimics, complete command arrays; only prehold phase label differs.
  assert len(common)==20 and np.array_equal(t['joint_target'][:,columns],ot['joint_target'][:,ocolumns])
  for ph in ['approach','close','hold','lift']:assert np.allclose(t['desired_palm'][t['phase']==ph],ot['desired_palm'][ot['phase']==ph],atol=2e-14,rtol=0)
  oldact=oe['audit']['hand_actuation']['joints'];newact=e['audit']['actuation']['joints']
  for n in newact:assert newact[n]==oldact[n],(oid,n)
  rec['old_full_arm_comparison']=dict(old_evaluation=ref(od/'evaluation.json'),old_trace=ref(od/'trace.npz'),same_candidate_identity=True,all20_hand_dof_targets_bitwise_identical_all2100_steps=True,same_approach_close_hold_lift_palm_desired=True,all20_hand_actuation_audit_records_identical=True,actual_object_trajectory_identical=bool(np.array_equal(t['object_pose_xyzw'],ot['object_pose_xyzw'])),old_pickup=oe['pickup'],old_phases=phase_stats(ot),old_max_palm_position_error_m=oe['palm_position_error_m'],object_final_position_difference_m=(poses[-1,:3]-ot['object_pose_xyzw'][-1,:3]).tolist())
 report['objects'].append(rec)
report['success']=sum(x['recomputed_pickup']['task_success']for x in report['objects']);report['physical_steps']=sum(sum(v['steps']for v in x['phases'].values())for x in report['objects']);assert report['success']==res['success']==0 and report['physical_steps']==res['physical_steps']==6300
report['limitations']=['No new physics or counterfactual proxy replacement performed by this audit. Original coarse convex collision model remains.', 'Zero runtime pose/state writes are reported counters plus direct source-loop inspection, not an independently instrumented API interception.', 'Joint/mimic/penetration/palm tracking and pickup are rederived from saved full-step arrays. Non-hand load and fixed-root gate reports are bound to original native output; raw every-step contacts are not retained.', 'Three hand-only selected failed cases are diagnostic, not independent benchmark. Original external full-arm 1/12 retained.', 'Three gantry joints remove the arm and constrain orientation; inertia/constraint topology differs, even though original 20 finger drive settings and target arrays match A3/E0 exactly.']
out=Path('/tmp/EGAD_FLOATING_THREE_AUDIT_20261108.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('WROTE',out, 'bytes',out.stat().st_size,'source',report['source_count'],'inputs',report['input_count'])
for x in report['objects']:
 print(x['object_id'],x['recomputed_pickup']);print('lift',x['phases']['lift']);print('close',x['phases']['close'])
