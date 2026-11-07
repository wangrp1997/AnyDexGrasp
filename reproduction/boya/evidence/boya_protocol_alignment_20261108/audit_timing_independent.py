"""Read-only independent closure-timing audit; no new physical steps."""
from pathlib import Path
import ast,hashlib,json,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
NEW=ROOT/'outputs/boya_position_timing_alignment_20261108_v1'
OLD=ROOT/'outputs/egad_floating_three_20261108_v1'
PROXY=ROOT/'outputs/boya_contact_proxy_alignment_20261108_v2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
p=read(NEW/'protocol.json');res=read(NEW/'RESULTS.json')
assert res['status']=='complete'and not res['errors']and res['protocol_sha256']==sha(NEW/'protocol.json')
for section in ('source_sha256','input_sha256'):
 for path,h in p[section].items():assert sha(path)==h,(section,path)
records=[]
for oid in p['object_ids']:
 oldjob=PROXY/'jobs/egad_A0_coarse.json'if oid=='egad_A0'else OLD/'jobs'/(oid+'.json')
 original=read(oldjob);j=read(NEW/'jobs'/(oid+'.json'))
 expected=dict(original,object_id=oid,source_object_id=oid,output_dir=j['output_dir'],protocol_path=j['protocol_path'],protocol_sha256=j['protocol_sha256'],closure_control='published_allegro_ten_position_steps_v1')
 assert j==expected
 base=PROXY/'execution/egad_A0_coarse'if oid=='egad_A0'else OLD/'execution'/oid
 D=NEW/'execution'/oid;e=read(D/'evaluation.json');e0=read(base/'evaluation.json')
 t=arrays(D/'trace.npz');cmd=arrays(D/'commands.npz');old=arrays(base/'commands.npz');native=read(D/'native_audit.json')
 receipt=next(x for x in res['results']if x['object_id']==oid)
 assert receipt['evaluation_sha256']==sha(D/'evaluation.json')
 assert sha(D/'trace.npz')==e['trace_sha256']and sha(D/'commands.npz')==e['commands_sha256']
 assert e['complete']and e['total_physics_steps']==2100 and all(e['gates'].values())
 assert e['state_writes_during_execution']==e['root_pose_writes_during_execution']==e['object_state_writes_during_execution']==0
 assert e['new_CGR_forwards']==e['new_training_updates']==0 and not e['training_writeback']
 assert all(np.isfinite(v).all()for v in t.values()if v.dtype.kind in'fiu')
 for k in cmd:np.testing.assert_array_equal(cmd[k],t[k])
 np.testing.assert_array_equal(t['time_s'],np.arange(1,2101)*.004)
 close=cmd['phase']=='close'
 assert int(close.sum())==250
 for k in old:
  if k!='joint_target':np.testing.assert_array_equal(cmd[k],old[k])
 np.testing.assert_array_equal(cmd['joint_target'][~close],old['joint_target'][~close])
 changed=np.any(cmd['joint_target']!=old['joint_target'],axis=1)
 assert changed.sum()==240 and not changed[~close].any()
 # Independent exact target reconstruction from unrounded source opening/closing.
 pool=arrays(Path(j['candidate_dir'])/'candidates.npz');names=list(cmd['joint_names'])
 motor_names=list(pool['independent_motor_names']);mi=[names.index(n)for n in motor_names]
 opened,closed=pool['proposal_q'][0],pool['proposal_q_close'][0]
 progress=np.repeat(np.arange(1,11,dtype=float)/10,25)
 target=(1-progress[:,None])*opened+progress[:,None]*closed
 np.testing.assert_array_equal(cmd['joint_target'][close][:,mi],target.astype(np.float32))
 for col in ['diagnostic_slide_x','diagnostic_slide_y','diagnostic_slide_z']:
  np.testing.assert_array_equal(cmd['joint_target'][:,names.index(col)],old['joint_target'][:,names.index(col)])
 for n in e['audit']['actuation']['joints']:assert e['audit']['actuation']['joints'][n]==e0['audit']['actuation']['joints'][n]
 assert e['audit']['mass_kg']==e0['audit']['mass_kg']==float(np.float32(.1))
 assert e['audit']['materials']==e0['audit']['materials']
 assert native['within_20um']and e['native_within_20um']
 row=j['object_row'];mesh=Path(row['mesh_path']);assert sha(mesh)==row['mesh_sha256']
 for part in row['convex_parts']:assert sha(part['path'])==part['sha256']
 # Limits/mimics and original source-mesh bottom independently rederived.
 limit=mimic=0.;urdf=ET.parse(D/'boya_translation_gantry.urdf').getroot()
 for node in urdf.findall('joint'):
  name=node.get('name')
  if name not in names:continue
  q=t['joint_pos'][:,names.index(name)];lim=node.find('limit');mm=node.find('mimic')
  if lim is not None:limit=max(limit,float(np.maximum(np.maximum(float(lim.get('lower'))-q,q-float(lim.get('upper'))),0).max()))
  if mm is not None:
   driver=t['joint_pos'][:,names.index(mm.get('joint'))];mimic=max(mimic,float(np.abs(q-driver*float(mm.get('multiplier','1'))-float(mm.get('offset','0'))).max()))
 assert np.isclose(limit,e['maxima']['joint_limit_violation_rad'],atol=2e-7)
 assert np.isclose(mimic,e['maxima']['mimic_error_rad'],atol=2e-7)
 palm=np.linalg.norm(t['palm_pose'][:,:3,3]-t['desired_palm'][:,:3,3],axis=1)
 assert np.isclose(palm.max(),e['maxima']['palm_position_error_m'],atol=1e-12)
 angle=(Rotation.from_matrix(t['palm_pose'][:,:3,:3]).inv()*Rotation.from_matrix(t['desired_palm'][:,:3,:3])).magnitude()
 assert np.isclose(angle.max(),e['maxima']['palm_orientation_error_rad'],atol=5e-8)
 penetration=float(max(t['object_penetration_m'].max(),t['robot_penetration_m'].max()))
 assert np.isclose(penetration,e['maxima']['penetration_m'],atol=1e-12)
 verts=np.array([[float(x)for x in line.split()[1:4]]for line in mesh.read_text().splitlines()if line.startswith('v ')])*row['scale']
 poses=t['object_pose_xyzw'].astype(float);rot=Rotation.from_quat(poses[:,3:]).as_matrix();bottom=np.empty(2100)
 for k in range(0,2100,64):bottom[k:k+64]=(verts@rot[k:k+64,2,:].T+poses[k:k+64,2]).min(0)
 hold=np.flatnonzero(t['phase']=='hold');lift=t['phase']=='lift';rise=poses[:,2]-poses[hold[-1],2]
 both=(t['finger_loads_n'][:,0]>.001)&(t['finger_loads_n'][:,1:].sum(1)>.001)
 picked=lift&(bottom>=.005)&(rise>=.005)&both
 pickup=e['tabletop_pickup'];assert int(picked.sum())==pickup['qualifying_steps']
 assert np.isclose(bottom[-1],pickup['final_bottom_clearance_m'],atol=2e-12)
 assert np.isclose(rise[-1],pickup['final_reference_rise_m'],atol=2e-12)
 success=bool(all(e['gates'].values())and picked[-1]and rise[-1]>=.05)
 assert success==e['task_success']==receipt['task_success']
 records.append(dict(object_id=oid,task_success=success,physical_steps=2100,changed_close_rows=240,
  all_nonclose_targets_and_all_other_command_arrays_identical=True,exact_ten_level_motor_targets_reconstructed=True,
  palm_and_gantry_targets_identical=True,coarse_object_geometry_and_initial_pose_identical=True,
  hand_drive_audits_and_mass_and_friction_identical=True,all_gates_reported_true=True,native20um=True,
  max_joint_limit_error_rad=limit,max_mimic_error_rad=mimic,max_penetration_m=penetration,max_palm_error_m=float(palm.max()),
  source_bottom_final_m=float(bottom[-1]),reference_final_rise_from_last_hold_m=float(rise[-1]),
  qualifying_pickup_steps=int(picked.sum()),both_loaded_lift_steps=int((both&lift).sum()),
  trace_sha256=sha(D/'trace.npz'),evaluation_sha256=sha(D/'evaluation.json')))
assert sum(x['task_success']for x in records)==res['success']==1
assert sum(x['physical_steps']for x in records)==res['physical_steps']==8400
# Inspect root audit without running or changing its source.
audit=ROOT/'reproduction/boya/audit_boya_alignment_results.py';source=audit.read_text()
review=dict(path=str(audit),sha256=sha(audit),hold_reference_correct=True,source_bottom_formula_correct_for_all_current_scale1=True,
 qualification_metric_issue="source_rise.qualifying_lift_steps currently counts bottom>0.05 only; it is NOT the executor's bottom>=0.005 plus reference-rise>=0.005 plus two-sided load condition. Rename to above50mm_steps or reconstruct the full pickup predicate.",
 additional_verification_limits=['Root script does not assert commands.npz SHA against evaluation; this independent audit does.',
 'Root script checks all7 saved booleans but does not independently reconstruct all7; raw full contact/fixed-base streams are unavailable.',
 'Root script source_rise omits object_row.scale; all current rows have scale1, so current numbers are valid.'])
report=dict(schema='anydex_boya_timing_independent_trace_audit_v1',protocol_sha256=sha(NEW/'protocol.json'),
 all_frozen_hashes_match=True,source_hash_count=len(p['source_sha256']),input_hash_count=len(p['input_sha256']),
 success=1,physical_steps=8400,new_physics_steps_by_audit=0,cases=records,root_audit_review=review,
 limitations=['No new physics or training.', 'Physical success and limit/mimic/penetration/palm tracking rederived. No-nonhand-support and fixed-base retain bound runtime gate evidence; raw streams not all retained.',
 'Counterfactual timing control remains frozen-opening anchored; original Allegro reads measured onset positions.'])
out=Path('/tmp/BOYA_TIMING_INDEPENDENT_AUDIT_20261108.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for x in records:print(x['object_id'],x['task_success'],'rise',x['reference_final_rise_from_last_hold_m'],'bottom',x['source_bottom_final_m'],'pickupsteps',x['qualifying_pickup_steps'])
print('WROTE',out)
