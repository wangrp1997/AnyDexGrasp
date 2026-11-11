"""Independent actual trace audit. No simulator, model inference or mutation."""
from pathlib import Path
import json
import hashlib
import datetime
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
RUN=ROOT/'outputs/boya_fast6_20261111_v1'
OUT=ROOT/'outputs/boya_fast_audit_20261111_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def close(x,y,tol=2e-12):assert np.allclose(x,y,atol=tol,rtol=0),(x,y)
def transforms(poses):
    out=np.repeat(np.eye(4)[None],len(poses),axis=0)
    out[:,:3,:3]=Rotation.from_quat(poses[:,3:]).as_matrix()
    out[:,:3,3]=poses[:,:3]
    return out

protocol=read(RUN/'protocol.json')
queue=read(RUN/'planning/queue_manifest.json')
report=read(RUN/'execution/RESULTS.json')
manifest=read(RUN/'objects.json')
assert report['status'] in ('complete','physical_complete_media_partial') and not report['errors']
assert report['objects']==6 and report['final_success_rate_available']
assert report['benchmark_only'] and not report['training_writeback'] and not report['outcome_dependent_retry']
assert report['new_training_updates']==report['new_CGR_forwards']==0
assert queue['complete'] and not queue['errors']
assert sha(RUN/'protocol.json')==read(RUN/'EVALUATION_FREEZE.json')['protocol_sha256']
ids=[x['object_id'] for x in protocol['objects']]
assert ids==[x['object_id'] for x in manifest['objects']]
assert {x['object_id'] for x in queue['objects']}==set(ids)
selected={x['object_id']:x for x in queue['trials']}
assert len(selected)==len(queue['trials'])==report['selected_top1']
assert {x['object_id'] for x in report['results']}==set(selected)
assert report['actual_attempts']==len(selected)<=6
assert set(report['no_plan_objects'])==set(ids)-set(selected)
hashes={}
for doc in (protocol,queue,read(OUT/'IMMUTABLE_BASELINE.json')):
    for key in ('source_sha256','input_sha256'):
        for p,h in doc.get(key,{}).items():
            if p not in hashes:hashes[p]=sha(p)
            assert hashes[p]==h,p
rows=[]
for item in report['results']:
    oid=item['object_id'];trial=item['trial'];planned=selected[oid]
    assert item['actual_grasps']==1 and trial['trial_id']==planned['trial_id']
    binding=read(trial['binding_json'])
    digest=hashlib.sha256(json.dumps({k:v for k,v in binding.items() if k!='trial_binding_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert trial['trial_binding_sha256']==planned['trial_binding_sha256']==binding['trial_binding_sha256']==digest
    d=Path(trial['execution_dir']);pd=Path(trial['source_plan_dir'])
    e=read(d/'evaluation.json');plan=read(pd/'plan.json');t=load(d/'trace.npz');cmd=load(pd/'commands.npz')
    assert sha(d/'trace.npz')==e['trace_sha256']
    assert sha(pd/'commands.npz')==e['commands_sha256']==trial['commands_sha256']==plan['commands_sha256']
    assert sha(pd/'plan.json')==e['plan_sha256']
    n=len(t['time_s']);expected=len(cmd['time_s'])
    assert n==e['total_physics_steps']==trial['physics_steps']==item['physical_steps']
    assert expected==trial['expected_command_steps']==plan['total_command_steps']==e['declared_command_steps']
    assert np.array_equal(t['time_s'],np.arange(1,n+1)*.004)
    assert np.array_equal(t['time_s'],cmd['time_s'][:n])
    assert np.array_equal(t['phase'],cmd['phase'][:n])
    assert np.array_equal(t['desired_palm'],cmd['desired_palm'][:n])
    assert all(np.isfinite(a).all() for a in t.values() if a.dtype.kind in 'fiu')
    assert e['complete'] and e['state_writes_during_execution']==0 and not e['external_palm_wrench_used']
    assert e['native_within_20um'] and e['arm_present'] and e['engine']=='PhysX'
    assert e['object_mass_kg']==float(np.float32(.1)) and e['object_friction']==.6 and e['dt_s']==.004
    native=read(d/'native_audit.json')
    assert native['within_20um']
    names=list(t['joint_names']);cnames=list(cmd['joint_names']);joints=ET.parse(d/'ur16e_boya.urdf').getroot().findall('joint')
    targets={name:cmd['joint_target'][:,i].astype(float) for i,name in enumerate(cnames)}
    for joint in joints:
        m=joint.find('mimic')
        if m is not None:
            targets[joint.get('name')]=targets[m.get('joint')]*float(m.get('multiplier','1'))+float(m.get('offset','0'))
    assert set(names)==set(targets)
    full=np.stack([targets[name] for name in names],axis=1).astype(np.float32)
    assert np.array_equal(full[:n],t['joint_target'])
    velocity=np.zeros_like(full);arm=e['arm_velocity_feedforward']['joints'];columns=[names.index(name) for name in arm]
    velocity[:,columns]=np.gradient(full[:,columns].astype(float),.004,axis=0,edge_order=2)
    assert np.array_equal(velocity[:n],t['joint_velocity_target'])
    for phase,count in [('settle',50),('approach',500),('close',250),('hold',50),('lift',500)]:
        assert int(np.sum(cmd['phase']==phase))==count
    assert int(np.sum(cmd['phase']=='home_to_pregrasp'))>=750
    limmax=mimicmax=0.
    for joint in joints:
        name=joint.get('name')
        if name not in names:continue
        vals=t['joint_pos'][:,names.index(name)];lim=joint.find('limit');m=joint.find('mimic')
        if lim is not None and lim.get('lower') is not None:
            limmax=max(limmax,float(np.maximum(np.maximum(float(lim.get('lower'))-vals,vals-float(lim.get('upper'))),0).max()))
        if m is not None:
            mimicmax=max(mimicmax,float(np.abs(vals-t['joint_pos'][:,names.index(m.get('joint'))]*float(m.get('multiplier','1'))-float(m.get('offset','0'))).max()))
    body=list(t['body_names']);palm=transforms(t['body_pose_xyzw'][:,body.index(e['palm_rigid_owner'])].astype(float))@np.asarray(e['palm_owner_to_frame'])
    posmax=float(np.linalg.norm(palm[:,:3,3]-t['desired_palm'][:,:3,3],axis=1).max())
    anglemax=float(Rotation.from_matrix(palm[:,:3,:3]@t['desired_palm'][:,:3,:3].transpose(0,2,1)).magnitude().max())
    depthmax=max(float(t['object_depth_m'].max()),float(t['robot_depth_m'].max()))
    base=transforms(t['body_pose_xyzw'][:,0].astype(float))
    basepos=float(np.linalg.norm(base[:,:3,3]-base[0,:3,3],axis=1).max())
    baseang=float(Rotation.from_matrix(base[:,:3,:3]@base[0,:3,:3].T).magnitude().max())
    for key,value,tol in [('joint_limit_violation_rad',limmax,2e-7),('mimic_error_rad',mimicmax,2e-7),('penetration_m',depthmax,2e-12),('palm_position_error_m',posmax,2e-12),('palm_orientation_error_rad',anglemax,1e-7)]:close(e[key],value,tol)
    gates=dict(joint_limits=limmax<=.002,mimics=mimicmax<=.002,penetration=depthmax<=.001,palm_tracking=posmax<=.03 and anglemax<=np.pi/12,no_arm_object_support=e['illegal_object_contact_force_n']<=.001,fixed_base=e['base_position_drift_m']<=2e-5 and e['base_orientation_drift_rad']<=2e-5,complete_execution=n==expected)
    assert gates==e['gates']
    assert basepos<=2e-5 and baseang<=2e-5
    obj=next(x for x in manifest['objects'] if x['object_id']==oid)
    assert sha(obj['mesh_path'])==obj['mesh_sha256']
    verts=np.array([[float(x) for x in l.split()[1:4]] for l in Path(obj['mesh_path']).read_text().splitlines() if l.startswith('v ')])*obj['scale']
    poses=t['object_pose_xyzw'].astype(float);rz=Rotation.from_quat(poses[:,3:]).as_matrix()[:,2,:]
    bottom=np.array([np.min(verts@z)+p[2] for z,p in zip(rz,poses)])
    holds=np.flatnonzero(t['phase']=='hold');haslift=bool(np.any(t['phase']=='lift'))
    qualified=np.zeros(n,dtype=bool);rise=None;longest=0
    if len(holds) and haslift:
        rise=poses[:,2]-poses[holds[-1],2]
        bilateral=(t['finger_loads_n'][:,0]>.001)&(t['finger_loads_n'][:,1:].sum(1)>.001)
        qualified=(t['phase']=='lift')&(bottom>=.005)&(rise>=.005)&bilateral
        run=0
        for value in qualified:
            run=run+1 if value else 0;longest=max(longest,run)
        pickup=e['pickup']
        assert int(qualified.sum())==pickup['qualifying_steps'] and bool(qualified[-1])==pickup['final_state_still_lifted_and_bearing']
        close(longest*.004,pickup['longest_observed_pickup_run_s'])
        close(rise[-1],pickup['final_reference_rise_m']);close(bottom[-1],pickup['final_bottom_clearance_m'])
    else:assert e['pickup'] is None
    success=bool(rise is not None and qualified[-1] and rise[-1]>=.05 and all(gates.values()))
    assert success==e['task_success']==(trial.get('label')==1)
    if n<expected:assert e['failure_reason'] and item['terminal_certificate'] is not None
    rows.append(dict(object_id=oid,trial_id=trial['trial_id'],task_success=success,label=trial.get('label'),label_eligible=trial['label_eligible'],steps=n,expected_steps=expected,complete_command_targets_all_28_equal=True,complete_arm_velocity_and_zero_hand_velocity_targets_equal=True,original_phase_and_palm_arrays_equal=True,all_trace_finite=True,native20um=True,gates=gates,source_mesh_vertices=len(verts),final_reference_rise_m=float(rise[-1]) if rise is not None else None,final_source_bottom_clearance_m=float(bottom[-1]),qualifying_steps=int(qualified.sum()),longest_pickup_s=longest*.004,palm_tracking_maximum_m=posmax,base_drift_from_first_recorded_state_m=basepos,failure_reason=e['failure_reason'],source_evaluation_sha256=sha(d/'evaluation.json'),source_trace_sha256=sha(d/'trace.npz'),source_commands_sha256=sha(pd/'commands.npz')))
assert sum(x['steps'] for x in rows)==report['physical_steps']
assert sum(x['task_success'] for x in rows)==report['successes']
assert report['success_fraction_over_external_objects']==report['successes']/6
receipt=dict(schema='anydex_boya_fast6_independent_saved_physics_audit_v1',complete=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(RUN/'protocol.json'),queue_sha256=sha(RUN/'planning/queue_manifest.json'),execution_results_sha256=sha(RUN/'execution/RESULTS.json'),execution_results_status_at_audit=report['status'],denominator=6,successes=report['successes'],actual_attempts=report['actual_attempts'],physical_steps=report['physical_steps'],no_plan_objects=report['no_plan_objects'],rows=rows,unique_source_input_files_verified=len(hashes),original26_baseline_unchanged=True,new_audit_physics_steps=0,new_audit_network_forwards=0,new_audit_training_steps=0,limits=['Source bottom evaluated against original visual mesh, contacts remain declared approximate convex proxies','Nonhand contact and initial-base-reference gates bind original native maxima; all-step contact lists not retained','Full28 command echo checked independently; planner/source trajectory reconstruction covered by separate planning audit','Six new shapes and one placement each are limited validation, not paper hardware success or global generalization'])
(OUT/'FAST6_PHYSICAL_AUDIT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:receipt[k] for k in ('complete','denominator','successes','actual_attempts','physical_steps','no_plan_objects')}))
