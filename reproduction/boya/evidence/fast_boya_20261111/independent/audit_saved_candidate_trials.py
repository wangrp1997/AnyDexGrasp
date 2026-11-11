"""Independent saved-data audit. No simulator, helper execution or network."""
from pathlib import Path
import hashlib
import json
import datetime
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
OUT = ROOT/'outputs/boya_fast_audit_20261111_v1'
RUN = ROOT/'outputs/boya_candidate_diagnostic_20261111_v2'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=2e-12): assert np.allclose(a,b,atol=tol,rtol=0), (a,b)
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}

p=read(RUN/'protocol.json')
r=read(RUN/'RESULTS.json')
prepared=read(RUN/'PREPARED.json')
baseline=read(OUT/'IMMUTABLE_BASELINE.json')
assert r['status']=='complete' and r['errors']==[]
assert r['protocol_sha256']==prepared['protocol_sha256']==sha(RUN/'protocol.json')
verified={}
for group in (p['source_sha256'],p['input_sha256'],prepared['jobs_sha256'],baseline['input_sha256']):
    for path,digest in group.items():
        assert sha(path)==digest,path
        verified[path]=digest
assert p['new_training_updates']==p['new_CGR_forwards']==0
assert not p['models_changed'] and not p['training_writeback']
assert not p['retry'] and p['physics_attempt_budget']==9
expected={'egad_A3':[36,80,95],'egad_A6':[22,26,33],'egad_E0':[1,2,3]}
assert len(r['results'])==len(p['cases'])==9
assert len({x['object_id'] for x in r['results']})==9
assert {k:sorted(x['candidate_rank'] for x in r['results'] if x['source_object_id']==k) for k in expected}==expected
rows=[]
bound_files={str(RUN/'protocol.json'):sha(RUN/'protocol.json'),str(RUN/'RESULTS.json'):sha(RUN/'RESULTS.json')}
for case in r['results']:
    job=read(RUN/'jobs'/(case['object_id']+'.json'))
    ep=Path(case['evaluation']); e=read(ep); d=ep.parent
    assert sha(ep)==case['evaluation_sha256']
    for file,key in [('trace.npz','trace_sha256'),('commands.npz','commands_sha256')]:
        assert sha(d/file)==e[key]
    t=load(d/'trace.npz'); cmd=load(d/'commands.npz'); pool=load(Path(job['candidate_dir'])/'candidates.npz')
    native=read(d/'native_audit.json')
    assert native['within_20um'] and native['physics_steps']==0
    assert e['complete'] and not e['failure_reason']
    assert e['total_physics_steps']==case['physical_steps']==len(t['time_s'])==2100
    assert e['state_writes_during_execution']==e['root_pose_writes_during_execution']==e['object_state_writes_during_execution']==0
    assert e['new_training_updates']==e['new_CGR_forwards']==0 and not e['training_writeback']
    assert e['hand_candidate_sha256']==sha(Path(job['candidate_dir'])/'candidates.npz')
    assert e['protocol_ref']['sha256']==sha(RUN/'protocol.json')
    assert all(np.isfinite(v).all() for v in t.values() if v.dtype.kind in 'fiu')
    for key in ('joint_target','velocity_target','desired_palm','phase','time_s','joint_names'):
        assert np.array_equal(t[key],cmd[key]),key
    assert np.array_equal(t['time_s'],np.arange(1,2101)*.004)
    phases=np.repeat(['prehold','approach','close','hold','lift'],[800,500,250,50,500])
    assert np.array_equal(t['phase'],phases)
    names=list(t['joint_names']); xml=ET.parse(d/'boya_translation_gantry.urdf').getroot()
    targets={}
    fraction=np.arange(1,251,dtype=float)/250
    for i,name in enumerate(pool['independent_motor_names']):
        opened,closed=pool['proposal_q'][0,i],pool['proposal_q_close'][0,i]
        val=np.full(2100,opened,dtype=float)
        val[1300:1550]=(1-fraction)*opened+fraction*closed
        val[1550:]=closed
        targets[name]=val
    for joint in xml.findall('joint'):
        name=joint.get('name'); m=joint.find('mimic')
        if m is not None:
            targets[name]=targets[m.get('joint')]*float(m.get('multiplier','1'))+float(m.get('offset','0'))
    hand_names=[n for n in names if n not in ('diagnostic_slide_x','diagnostic_slide_y','diagnostic_slide_z')]
    for name in hand_names:
        if name not in targets: targets[name]=np.zeros(2100)
        assert np.array_equal(t['joint_target'][:,names.index(name)],targets[name].astype(np.float32)),name
    palm=pool['proposal_palm'][0]; direction=pool['proposal_approach_direction'][0]
    pre=palm[:3,3]-.1*direction
    desired=np.repeat(palm[None],2100,axis=0)
    desired[:800,:3,3]=pre
    q=np.arange(1,501,dtype=float)/500; u=q**3*(10-15*q+6*q*q)
    desired[800:1300,:3,3]=pre+.1*u[:,None]*direction
    desired[1600:,:3,3]+=u[:,None]*np.array([0,0,.1])
    close(t['desired_palm'],desired)
    axis=[names.index('diagnostic_slide_'+a) for a in 'xyz']
    close(t['joint_target'][:,axis],((desired[:,:3,3]-pre)@palm[:3,:3]).astype(np.float32),2e-8)
    velocity=np.zeros_like(t['joint_target'])
    velocity[1:,axis]=np.diff(t['joint_target'][:,axis],axis=0)/.004
    assert np.array_equal(velocity,t['velocity_target'])
    limmax=mimicmax=0.
    for joint in xml.findall('joint'):
        name=joint.get('name')
        if name not in names: continue
        vals=t['joint_pos'][:,names.index(name)]; lim=joint.find('limit'); m=joint.find('mimic')
        if lim is not None:
            limmax=max(limmax,float(np.maximum(np.maximum(float(lim.get('lower'))-vals,vals-float(lim.get('upper'))),0).max()))
        if m is not None:
            mimicmax=max(mimicmax,float(np.abs(vals-t['joint_pos'][:,names.index(m.get('joint'))]*float(m.get('multiplier','1'))-float(m.get('offset','0'))).max()))
    posmax=float(np.linalg.norm(t['palm_pose'][:,:3,3]-desired[:,:3,3],axis=1).max())
    angmax=float(Rotation.from_matrix(t['palm_pose'][:,:3,:3]@desired[:,:3,:3].transpose(0,2,1)).magnitude().max())
    penmax=max(float(t['object_penetration_m'].max()),float(t['robot_penetration_m'].max()))
    close(limmax,e['maxima']['joint_limit_violation_rad'],2e-7)
    close(mimicmax,e['maxima']['mimic_error_rad'],2e-7)
    close(posmax,e['maxima']['palm_position_error_m'])
    close(angmax,e['maxima']['palm_orientation_error_rad'],1e-7)
    close(penmax,e['maxima']['penetration_m'])
    assert limmax<=.002 and mimicmax<=.002 and penmax<=.001 and posmax<=.03 and angmax<=np.pi/12
    assert all(e['gates'].values()) and len(e['gates'])==7
    obj=job['object_row']; assert sha(obj['mesh_path'])==obj['mesh_sha256']
    verts=np.array([[float(x) for x in line.split()[1:4]] for line in Path(obj['mesh_path']).read_text().splitlines() if line.startswith('v ')])*obj['scale']
    poses=t['object_pose_xyzw'].astype(float); rz=Rotation.from_quat(poses[:,3:]).as_matrix()[:,2,:]
    bottom=np.array([np.min(verts@z)+pose[2] for z,pose in zip(rz,poses)])
    rise=poses[:,2]-poses[1599,2]
    bilateral=(t['finger_loads_n'][:,0]>.001)&(t['finger_loads_n'][:,1:].sum(1)>.001)
    qualified=(phases=='lift')&(bottom>=.005)&(rise>=.005)&bilateral
    current=longest=0
    for q in qualified:
        current=current+1 if q else 0
        longest=max(current,longest)
    success=bool(qualified[-1] and rise[-1]>=.05 and all(e['gates'].values()))
    pickup=e['tabletop_pickup']
    assert success==e['task_success']==case['task_success']
    assert int(qualified.sum())==pickup['qualifying_steps']
    close(longest*.004,pickup['longest_observed_pickup_run_s'])
    close(rise[-1],pickup['final_reference_rise_m'])
    close(bottom[-1],pickup['final_bottom_clearance_m'])
    assert bool(qualified[-1])==pickup['final_state_still_lifted_and_bearing']
    for file in ('trace.npz','commands.npz','native_audit.json','evaluation.json','boya_translation_gantry.urdf'):
        bound_files[str(d/file)]=sha(d/file)
    rows.append(dict(object_id=case['object_id'],steps=2100,success=success,all_hand_command_count=len(hand_names),all_hand_commands_equal=True,desired_palm_equal=True,finite_trace=True,native20um=True,all_seven_gates=True,source_mesh_vertex_count=len(verts),final_reference_rise_m=float(rise[-1]),final_source_bottom_clearance_m=float(bottom[-1]),qualifying_steps=int(qualified.sum()),longest_pickup_seconds=longest*.004,max_palm_position_error_m=posmax))
assert sum(x['success'] for x in rows)==r['success']==1
assert sum(x['steps'] for x in rows)==r['physical_steps']==18900
receipt=dict(schema='anydex_boya_independent_candidate_physical_audit_v1',complete=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(RUN/'protocol.json'),results_sha256=sha(RUN/'RESULTS.json'),verified_unique_source_input_job_files=len(verified),immutable_baseline_26_unchanged=True,source_input_job_sha256=verified,artifact_sha256=bound_files,results=rows,success=1,physical_steps=18900,new_physics_steps=0,new_training_steps=0,scope='Independent saved-data reconstruction; existing actual nine trials only, not benchmark top1',limitations=['Native/base/nonhand-contact gates bound to frozen reports; raw all-step contact lists not saved','Coarse collision proxies and controlled wrist remain, not full-arm acceptance','A6 viable later pose proves a candidate exists, not general ranking improvement'])
target=OUT/'CANDIDATE_PHYSICAL_AUDIT.json'
target.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(complete=True,success=1,physical_steps=18900,verified_unique_files=len(verified),output=str(target))))
