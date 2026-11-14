"""Read-only XML/NumPy FK check; no original FK/IK/planning-model import."""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
OUT=ROOT/'outputs/boya_author_fixed_goal_probe_20261114_v1'
protocol=json.loads((OUT/'PROTOCOL.json').read_text())
result=json.loads((OUT/'RESULT.json').read_text())
plan=json.loads(Path(protocol['source_plan']['path']).read_text())
robot=ET.parse(plan['arguments']['robot_urdf']).getroot()
children={j.find('child').get('link') for j in robot.findall('joint')}
roots={n.get('name') for n in robot.findall('link')}-children
assert len(roots)==1

def origin(node):
    value=np.eye(4)
    if node is None:return value
    value[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
    r,p,y=np.fromstring(node.get('rpy','0 0 0'),sep=' ')
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    value[:3,:3]=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])@np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return value

frames={next(iter(roots)):np.eye(4)}
pending=list(robot.findall('joint'))
while pending:
    left=[]
    for joint in pending:
        parent,child=joint.find('parent').get('link'),joint.find('child').get('link')
        if parent not in frames:left.append(joint);continue
        motion=np.eye(4)
        if joint.get('type')!='fixed':
            axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ');axis=axis/np.linalg.norm(axis)
            x,y,z=axis;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            angle=result['solved_raw28'][joint.get('name')]
            motion[:3,:3]=np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)
        frames[child]=frames[parent]@origin(joint.find('origin'))@motion
    assert len(left)<len(pending)
    pending=left
palm=frames['palm']
error=float(abs(palm-np.asarray(result['actual_world_T_palm'])).max())
assert error<2e-14
target=np.asarray(protocol['target_world_T_palm'])
position=float(np.linalg.norm(palm[:3,3]-target[:3,3]))
assert abs(position-result['residual']['position_error_m'])<2e-14
srdf=ET.parse(Path(plan['arguments']['reference_root'])/'ur_by_moveit_config/config/ur_with_dexhand.srdf').getroot()
home=srdf.find("group_state[@name='home'][@group='ur_arm']")
assert {j.get('name'):float(j.get('value')) for j in home.findall('joint')}==protocol['reference_HOME_arm']
with np.load(Path(protocol['source_plan']['path']).parent/'commands.npz',allow_pickle=False) as z:
    names=list(map(str,z['joint_names']))
    boundary=z['joint_target'][protocol['closed_boundary_command_index']]
    assert z['phase'][protocol['closed_boundary_command_index']]=='transport_bin'
    assert z['phase'][protocol['closed_boundary_command_index']+1]=='release'
    assert all(result['solved_raw28'][n]==boundary[names.index(n)] for n in protocol['fixed_non_arm_names'])
assert result['solver_observation']['calls']==1 and result['original_ik_audit']['nfev']==13
assert result['status']=='null_geometry' and not (OUT/'FIXED_GOAL_ARM_ENDPOINT.json').exists()
assert hashlib.sha256(Path(protocol['fixed_release']['path']).read_bytes()).hexdigest()=='43223161bf53e4fa40335960e1b51185320646c8ee38f03793f2bea4fed7522c'
value=dict(schema='anydex_boya_fixed_goal_probe_independent_numeric_audit_v1',complete=True,
    method='Original URDF XML with independent NumPy RzRyRx/Rodrigues transforms; no original FK or IK call',
    world_T_palm_max_matrix_error=error,independent_position_error_m=position,
    original_SRDF_HOME_exact=True,all18_closed_non_arm_values_exact=True,
    fixed_release_sha256_exact=True,one_original_ik_call_with_13_nfev=True,
    null_geometry_has_no_clear_goal_artifact=True,IK_calls=0,Kit_launches=0,physics_steps=0,labels=0)
path=Path(__file__).with_suffix('.json')
with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(value))
