"""Independent saved-trace audit of proxy and published-position timing controls."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');B=ROOT/'reproduction/boya';sys.path.insert(0,str(B))
from run_egad_floating_three import sha,read,save
OLD=ROOT/'outputs/egad_floating_three_20261108_v1'
PROXY=ROOT/'outputs/boya_contact_proxy_alignment_20261108_v2'
TIMING=ROOT/'outputs/boya_position_timing_alignment_20261108_v1'

def source_rise(trace,mesh):
 pose=trace['object_pose_xyzw'];r=Rotation.from_quat(pose[:,3:]).as_matrix()
 z=np.asarray(mesh.vertices)@r[:,2,:].T+pose[:,2]
 bottom=z.min(0)
 hold=np.flatnonzero(trace['phase']=='hold')[-1];lift=np.flatnonzero(trace['phase']=='lift')
 return dict(final_bottom_m=float(bottom[-1]),lift_max_bottom_m=float(bottom[lift].max()),
  final_reference_rise_m=float(pose[-1,2]-pose[hold,2]),bottom_above50mm_lift_steps=int((bottom[lift]>.05).sum()),
  qualifying_lift_steps=int(((bottom[lift]>=.005)&((pose[lift,2]-pose[hold,2])>=.005)&(trace['finger_loads_n'][lift,0]>.001)&(trace['finger_loads_n'][lift,1:].sum(1)>.001)).sum()))

def load(path):
 with np.load(path,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}

def main(out):
 result=dict(complete=False,proxy=[],timing=[],input_hash_count=0,physics_steps=0,new_training_updates=0,original_external_result_unchanged='1/12')
 for r in (PROXY,TIMING):
  p=read(r/'protocol.json');d=read(r/'RESULTS.json');assert d['status']=='complete'and not d['errors']
  assert d['protocol_sha256']==sha(r/'protocol.json')
  for sec in ('source_sha256','input_sha256'):
   for path,h in p[sec].items():assert sha(path)==h,path
   result['input_hash_count']+=len(p[sec])
  for row in d['results']:
   oid=row['object_id'];ev=read(row['evaluation']);execution=Path(row['evaluation']).parent
   assert sha(row['evaluation'])==row['evaluation_sha256'] and sha(execution/'trace.npz')==ev['trace_sha256']
   assert sha(execution/'commands.npz')==ev['commands_sha256']
   assert ev['total_physics_steps']==2100 and all(ev['gates'].values())
   assert ev['state_writes_during_execution']==ev['object_state_writes_during_execution']==ev['root_pose_writes_during_execution']==0
   job=read(r/'jobs'/(oid+'.json'));tr=load(execution/'trace.npz');cmd=load(execution/'commands.npz')
   assert len(tr['time_s'])==2100 and np.isfinite(tr['joint_pos']).all() and np.isfinite(tr['object_pose_xyzw']).all()
   np.testing.assert_array_equal(tr['joint_target'],cmd['joint_target']);np.testing.assert_array_equal(tr['desired_palm'],cmd['desired_palm'])
   summary=dict(object_id=oid,task_success=ev['task_success'],all7_gates=True,
    palm_tracking_max_error_m=ev['maxima']['palm_position_error_m'],physical_steps=2100,
    independent_source_vertices=source_rise(tr,trimesh.load_mesh(job['object_row']['mesh_path'],process=False)))
   assert abs(summary['independent_source_vertices']['final_reference_rise_m']-ev['tabletop_pickup']['final_reference_rise_m'])<1e-8
   assert summary['independent_source_vertices']['qualifying_lift_steps']==ev['tabletop_pickup']['qualifying_steps']
   if r==PROXY:
    if oid.endswith('_fine'):
     sourceid=job['source_object_id'];base=(PROXY/'execution/egad_A0_coarse')if sourceid=='egad_A0'else(OLD/'execution'/sourceid)
     coarse=load(base/'commands.npz');np.testing.assert_array_equal(cmd['joint_target'],coarse['joint_target']);np.testing.assert_array_equal(cmd['desired_palm'],coarse['desired_palm'])
     summary['all2100_command_targets_bitwise_identical_to_coarse']=True
    result['proxy'].append(summary)
   else:
    base=(PROXY/'execution/egad_A0_coarse')if oid=='egad_A0'else(OLD/'execution'/oid)
    coarse=load(base/'commands.npz');close=cmd['phase']=='close'
    np.testing.assert_array_equal(cmd['joint_target'][~close],coarse['joint_target'][~close]);np.testing.assert_array_equal(cmd['desired_palm'],coarse['desired_palm']);np.testing.assert_array_equal(cmd['velocity_target'],coarse['velocity_target'])
    summary['only_close_phase_targets_changed']=True
    summary['changed_rows']=int(np.any(cmd['joint_target']!=coarse['joint_target'],axis=1).sum())
    assert summary['changed_rows']==240
    result['timing'].append(summary)
   result['physics_steps']+=2100
 # Explicitly retain the no-step failed launch, never count it as 5 grasp failures.
 failed=read(PROXY.with_name('boya_contact_proxy_alignment_20261108_v1')/'RESULTS.json')
 assert failed['actual_grasps']==failed['physical_steps']==0
 result['retained_zero_step_launch_error_count']=len(failed['errors'])
 result['complete']=True;result['source_sha256']={str(Path(__file__).resolve()):sha(__file__)}
 save(out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
