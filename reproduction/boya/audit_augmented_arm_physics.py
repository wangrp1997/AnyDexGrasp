"""Independent actual trace audit. No simulator, model inference or mutation."""
from pathlib import Path
import json
import hashlib
import datetime
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
RUN=ROOT/'outputs/boya_augmented_arm_external6_20261114_v1'
OUT=ROOT/'outputs/boya_augmented_arm_audit_20261114_v1'
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
planning=read(RUN/'planning/RESULTS.json')
report=read(RUN/'execution/RESULTS.json')
manifest=read(RUN/'objects.json')
choices=read(RUN/'execution/ALL_CHOICES_FREEZE.json')
assert report['status']=='complete' and not report['errors'] and report['final_success_rates_available']
assert planning['status']=='complete' and not planning['errors']
assert report['benchmark_only'] and not report['training_writeback'] and not report['outcome_dependent_retry']
assert sha(RUN/'protocol.json')==read(RUN/'EVALUATION_FREEZE.json')['protocol_sha256']==choices['protocol_sha256']
assert sha(RUN/'planning/RESULTS.json')==choices['planning_sha256']
assert sha(RUN/'execution/ALL_CHOICES_FREEZE.json')==report['physical_choices_sha256']
ids=[r['object_id'] for r in protocol['objects']]
recipes=('original_mse','preactivation')
assert ids==['egad_'+r+str(c) for r in 'ACE' for c in (2,5)]
assert ids==[r['object_id'] for r in manifest['objects']]
assert {(r['object_id'],r['recipe']) for r in choices['conditions']}=={(o,r) for o in ids for r in recipes}
selected={(v['object_id'],v['recipe']):v['selected'] for v in choices['conditions'] if v['selected'] is not None}
preflight=read(OUT/'PREPHYSICAL_AUDIT.json')
assert preflight['status']=='complete' and preflight['planning_sha256']==sha(RUN/'planning/RESULTS.json')
for case in preflight['cases']:
    v=selected.get((case['object_id'],case['recipe']))
    assert case['selected_trial_id']==(v['trial_id'] if v else None)
    assert case['selected_binding_sha256']==(v['trial_binding_sha256'] if v else None)
    assert case['selected_commands_sha256']==(v['commands_sha256'] if v else None)
assert (OUT/'PREPHYSICAL_AUDIT.json').stat().st_mtime_ns < (RUN/'execution/ALL_CHOICES_FREEZE.json').stat().st_mtime_ns
assert {(v['object_id'],v['recipe']) for v in report['results']}==set(selected)
assert report['actual_attempts']==len(selected)<=12
if report['results']:
    assert (RUN/'execution/ALL_CHOICES_FREEZE.json').stat().st_mtime_ns<min(Path(v['trial']['execution_dir']).stat().st_mtime_ns for v in report['results'])
hashes={}
for doc in (protocol,planning,read(OUT/'STUDY_PROTOCOL.json')):
    for key in ('source_sha256','input_sha256','frozen_files'):
        for p,h in doc.get(key,{}).items():
            if p not in hashes:hashes[p]=sha(p)
            assert hashes[p]==h,p
rows=[]
for item in report['results']:
    oid=item['object_id'];trial=item['trial'];planned=selected[(oid,item['recipe'])]
    assert item['actual_grasps']==1 and trial['trial_id']==planned['trial_id']
    binding=read(trial['binding_json'])
    digest=hashlib.sha256(json.dumps({k:v for k,v in binding.items() if k!='trial_binding_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert trial['trial_binding_sha256']==planned['trial_binding_sha256']==binding['trial_binding_sha256']==digest
    d=Path(trial['execution_dir']);pd=Path(trial['source_plan_dir'])
    e=read(d/'evaluation.json');plan=read(pd/'plan.json');t=load(d/'trace.npz');cmd=load(pd/'commands.npz')
    assert sha(d/'trace.npz')==e['trace_sha256']
    assert sha(pd/'commands.npz')==e['commands_sha256']==trial['commands_sha256']==plan['commands_sha256']
    assert sha(pd/'plan.json')==e['plan_sha256']
    pool=load(Path(plan['arguments']['candidates'])/'candidates.npz')
    pool_names=list(pool['independent_motor_names']);control_names=list(cmd['joint_names'])
    close(plan['palm'],pool['proposal_palm'][0])
    # Reconstruct every independent closure motor from frozen source endpoints.
    motors=plan['hand_motor_names'];columns=[control_names.index(name) for name in motors]
    pcols=[pool_names.index(name) for name in motors]
    opening=pool['proposal_q'][0,pcols];closing=pool['proposal_q_close'][0,pcols]
    close(plan['open_motors'],opening);close(plan['closed_motors'],closing)
    hand=np.repeat(opening[None],len(cmd['phase']),axis=0)
    ix=np.flatnonzero(cmd['phase']=='close')
    hand[ix]=opening+(np.arange(1,len(ix)+1)/len(ix))[:,None]*(closing-opening)
    hand[np.isin(cmd['phase'],['hold','lift'])]=closing
    close(hand,cmd['joint_target'][:,columns])
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
    for phase,count in [('settle',50),('close',250),('hold',50)]:
        assert int(np.sum(cmd['phase']==phase))==count
    for phase,minimum in [('home_to_pregrasp',750),('approach',500),('lift',500)]:
        assert int(np.sum(cmd['phase']==phase))>=minimum
        assert int(np.sum(cmd['phase']==phase))==plan['segment_timing'][phase]['steps']
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
    rows.append(dict(object_id=oid,recipe=item['recipe'],trial_id=trial['trial_id'],task_success=success,label=trial.get('label'),label_eligible=trial['label_eligible'],steps=n,expected_steps=expected,complete_command_targets_all_equal=True,actual_joint_count=len(names),complete_arm_velocity_and_zero_hand_velocity_targets_equal=True,original_phase_and_palm_arrays_equal=True,all_trace_finite=True,native20um=True,gates=gates,source_mesh_vertices=len(verts),final_reference_rise_m=float(rise[-1]) if rise is not None else None,final_source_bottom_clearance_m=float(bottom[-1]),qualifying_steps=int(qualified.sum()),longest_pickup_s=longest*.004,palm_tracking_maximum_m=posmax,base_drift_from_first_recorded_state_m=basepos,failure_reason=e['failure_reason'],source_evaluation_sha256=sha(d/'evaluation.json'),source_trace_sha256=sha(d/'trace.npz'),source_commands_sha256=sha(pd/'commands.npz')))
assert sum(x['steps'] for x in rows)==report['physical_steps']
metrics={}
for recipe in recipes:
    cases=[v for v in rows if v['recipe']==recipe]
    expected=dict(objects=6,actual_attempts=len(cases),successes=sum(v['task_success'] for v in cases),
        actual_failures=sum(not v['task_success'] for v in cases),
        no_plan=sum(v['recipe']==recipe and v['selected'] is None for v in choices['conditions']))
    assert expected==report['metrics'][recipe],(recipe,expected,report['metrics'][recipe])
    assert expected['successes']+expected['actual_failures']+expected['no_plan']==6
    metrics[recipe]=expected
receipt=dict(schema='boya_augmented_arm6_independent_physics_audit_v1',status='complete',
    protocol_sha256=sha(RUN/'protocol.json'),execution_results_sha256=sha(RUN/'execution/RESULTS.json'),
    physical_steps=report['physical_steps'],actual_attempts=report['actual_attempts'],metrics=metrics,rows=rows,
    unique_source_input_files_verified=len(hashes),source_sha256={str(Path(__file__).resolve()):sha(__file__)},
    new_physics_steps=0,new_training_updates=0,new_network_forwards=0,
    limits=['Coarse convex physics, source visual vertices only for measurement',
            'Native nonhand contact and base-reference gates use recorded maxima; full contact points not stored',
            'Actual command/mimic/velocity/palm echo independently checked; not an independent IK solver'])
(OUT/'INDEPENDENT_PHYSICS_AUDIT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status='complete',physical_steps=report['physical_steps'],metrics=metrics)))
