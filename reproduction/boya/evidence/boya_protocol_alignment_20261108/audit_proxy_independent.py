"""Independent read-only protocol, command, trace and pickup audit of proxy v2."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
NEW=ROOT/'outputs/boya_contact_proxy_alignment_20261108_v2'
V1=ROOT/'outputs/boya_contact_proxy_alignment_20261108_v1'
OLD=ROOT/'outputs/egad_floating_three_20261108_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def differences(a,b,p=''):
 if type(a)!=type(b):return[p]
 if isinstance(a,dict):return sum((differences(a.get(k),b.get(k),p+'/'+k)for k in a.keys()|b.keys()),[])
 if isinstance(a,list):return[]if a==b else[p]
 return[]if a==b else[p]
p=read(NEW/'protocol.json');res=read(NEW/'RESULTS.json')
assert res['status']=='complete'and not res['errors']and res['protocol_sha256']==sha(NEW/'protocol.json')
for section in ('source_sha256','input_sha256'):
 for path,h in p[section].items():assert sha(path)==h,(section,path)
previous_protocol=read(V1/'protocol.json')
pdiff=differences(p,previous_protocol)
assert set(pdiff)=={'/parent_zero_step_failure','/environment_correction','/input_sha256/'+str(V1/'RESULTS.json')}
records=[]
for oid in p['object_ids']:
 j=read(NEW/'jobs'/(oid+'.json'));j1=read(V1/'jobs'/(oid+'.json'))
 jd=differences(j,j1);assert set(jd)=={'/output_dir','/protocol_path','/protocol_sha256'}
 sourceid=j['source_object_id'];D=NEW/'execution'/oid
 e=read(D/'evaluation.json');t=arrays(D/'trace.npz');cmd=arrays(D/'commands.npz');native=read(D/'native_audit.json')
 assert e['complete']and e['total_physics_steps']==2100 and all(e['gates'].values())
 assert sha(D/'trace.npz')==e['trace_sha256']and sha(D/'commands.npz')==e['commands_sha256']
 assert e['state_writes_during_execution']==e['root_pose_writes_during_execution']==e['object_state_writes_during_execution']==0
 assert e['new_CGR_forwards']==e['new_training_updates']==0 and not e['training_writeback']
 for k in cmd:np.testing.assert_array_equal(cmd[k],t[k])
 np.testing.assert_array_equal(t['time_s'],np.arange(1,2101)*.004)
 assert all(np.isfinite(v).all()for v in t.values()if v.dtype.kind in'fiu')
 assert native['within_20um']and e['native_within_20um']
 assert e['audit']['mass_kg']==float(np.float32(.1))
 for which in ('robot','object'):
  m=e['audit']['materials'][which];assert m['native_readback_ok']and m['static_friction']==m['dynamic_friction']==float(np.float32(.6))
 row=j['object_row'];mesh=Path(row['mesh_path']);assert sha(mesh)==row['mesh_sha256']
 if sourceid in ('egad_A3','egad_A6','egad_E0'):
  original=read(OLD/'jobs'/(sourceid+'.json'));originalrow=original['object_row'];ot=arrays(OLD/'execution'/sourceid/'trace.npz')
  for k in ('candidate_dir','candidate_record','candidate_rank','hand_urdf','hand_scene','package','physics_config','diagnostic_config'):assert j[k]==original[k]
  allowed={'object_root','mesh_path','urdf_path','urdf_sha256','pose_path','convex_parts'}
  for k in row.keys()|originalrow.keys():
   if k not in allowed:assert row[k]==originalrow[k],(oid,k)
  assert sha(mesh)==sha(originalrow['mesh_path'])and sha(row['pose_path'])==sha(originalrow['pose_path'])
  for k in ('joint_target','velocity_target','desired_palm','phase','time_s','joint_names'):np.testing.assert_array_equal(t[k],ot[k])
 else:
  ot=arrays(NEW/'execution/egad_A0_coarse/trace.npz')
  for k in ('joint_target','velocity_target','desired_palm','phase','time_s','joint_names'):np.testing.assert_array_equal(t[k],ot[k])
 # Actual limit/mimic errors independent of report.
 names=list(t['joint_names']);limit=mimic=0.;urdf=ET.parse(D/'boya_translation_gantry.urdf').getroot()
 for node in urdf.findall('joint'):
  name=node.get('name')
  if name not in names:continue
  q=t['joint_pos'][:,names.index(name)];lim=node.find('limit');mi=node.find('mimic')
  if lim is not None:
   limit=max(limit,float(np.maximum(np.maximum(float(lim.get('lower'))-q,q-float(lim.get('upper'))),0).max()))
  if mi is not None:
   other=t['joint_pos'][:,names.index(mi.get('joint'))];mimic=max(mimic,float(np.abs(q-other*float(mi.get('multiplier','1'))-float(mi.get('offset','0'))).max()))
 assert np.isclose(limit,e['maxima']['joint_limit_violation_rad'],atol=2e-7)
 assert np.isclose(mimic,e['maxima']['mimic_error_rad'],atol=2e-7)
 pe=np.linalg.norm(t['palm_pose'][:,:3,3]-t['desired_palm'][:,:3,3],axis=1)
 assert np.isclose(pe.max(),e['maxima']['palm_position_error_m'],atol=1e-12)
 penetration=float(max(t['object_penetration_m'].max(),t['robot_penetration_m'].max()))
 assert np.isclose(penetration,e['maxima']['penetration_m'],atol=1e-12)
 # Source full vertex support, no convex proxy used in pickup measurement.
 v=np.array([[float(x)for x in line.split()[1:4]]for line in mesh.read_text().splitlines()if line.startswith('v ')])*row['scale']
 poses=t['object_pose_xyzw'].astype(float);rot=Rotation.from_quat(poses[:,3:]).as_matrix();bottom=np.empty(2100)
 for k in range(0,2100,64):bottom[k:k+64]=(v@rot[k:k+64,2,:].T+poses[k:k+64,2]).min(0)
 hold=np.flatnonzero(t['phase']=='hold');lift=t['phase']=='lift';rise=poses[:,2]-poses[hold[-1],2]
 both=(t['finger_loads_n'][:,0]>.001)&(t['finger_loads_n'][:,1:].sum(1)>.001)
 picked=lift&(bottom>=.005)&(rise>=.005)&both
 pickup=e['tabletop_pickup'];assert int(picked.sum())==pickup['qualifying_steps']
 assert np.isclose(bottom[-1],pickup['final_bottom_clearance_m'],atol=2e-12)
 assert np.isclose(rise[-1],pickup['final_reference_rise_m'],atol=2e-12)
 success=bool(all(e['gates'].values())and picked[-1]and rise[-1]>=.05);assert success==e['task_success']
 records.append(dict(object_id=oid,task_success=success,physical_steps=2100,all_command_arrays_match_coarse=True,
  v2_job_changed_only_output_and_protocol=True,visual_source_sha256=sha(mesh),native_20um=True,native_mass_kg=e['audit']['mass_kg'],
  native_friction=float(np.float32(.6)),source_bottom_final_m=float(bottom[-1]),reference_final_rise_m=float(rise[-1]),
  qualifying_pickup_steps=int(picked.sum()),both_loaded_lift_steps=int((both&lift).sum()),
  max_joint_limit_error_rad=limit,max_mimic_error_rad=mimic,max_penetration_m=penetration,max_palm_error_m=float(pe.max()),
  saved_trace_sha256=sha(D/'trace.npz'),saved_evaluation_sha256=sha(D/'evaluation.json')))
assert sum(x['task_success']for x in records)==res['success']==2
assert sum(x['physical_steps']for x in records)==res['physical_steps']==10500
report=dict(schema='anydex_boya_proxy_alignment_v2_independent_trace_audit_v1',protocol_sha256=sha(NEW/'protocol.json'),
 protocol_v1_v2_differences=pdiff,source_hash_count=len(p['source_sha256']),input_hash_count=len(p['input_sha256']),
 all_frozen_hashes_match=True,new_physics_steps_by_this_audit=0,original_external_success_rate_unchanged='1/12',
 cases=records,success=2,physical_steps=10500,
 limitations=['Existing traces only, no new physics.', 'Every-step motor torques and contacts are not stored; exact raw friction impulse audit unavailable.',
 'Nonhand-object-support/fixed-base gate audit binds saved reports; not all raw sensor streams available.',
 'Fine proxy changes collision shape, not original source mesh mass/COM/inertia formula; this remains a finite decomposition approximation.'])
out=Path('/tmp/BOYA_PROXY_ALIGNMENT_V2_INDEPENDENT_AUDIT_20261108.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for rec in records:print(rec['object_id'],rec['task_success'],rec['reference_final_rise_m'],rec['source_bottom_final_m'])
print('WROTE',out)
