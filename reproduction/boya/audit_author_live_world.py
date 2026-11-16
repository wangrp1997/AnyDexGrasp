"""Read-only full original numerical audits with measured live entry binding.

Only the cold-zero-velocity entry and capture actor namespace assumptions
change. Original all-contact, force, mesh, command and pickup checks remain.
"""
import argparse
import ast
import copy
from pathlib import Path
import time
import numpy as np
import author_collection_batch as B
import audit_author_resident_collection as C
import audit_author_collection_runtime as A
from trial_record import json_sha


def capture_auditor():
    tree=ast.parse(Path(C.__file__).read_text());fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='capture_audit'))
    changed=dict(roots=0,targets=0,pairs=0)
    class Adapt(ast.NodeTransformer):
        def visit_Assign(self,n):
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='roots':
                changed['roots']+=1;n.value=ast.parse("observed['persistent_world']['active_paths']",mode='eval').body
            return self.generic_visit(n)
        def visit_Compare(self,n):
            text=ast.unparse(n)
            if text=="set(sensor['target_collision_paths']) == set(roots) | floor":
                changed['targets']+=1;return ast.parse("set(sensor['target_collision_paths']) == set(roots) | floor | set(observed['persistent_world']['robot_contact_paths'])",mode='eval').body
            return self.generic_visit(n)
        def visit_BoolOp(self,n):
            if ast.unparse(n)=='body in roots and other in set(roots) | floor':
                changed['pairs']+=1;return ast.parse("body in roots and (other in set(roots) | floor or (other in observed['persistent_world']['robot_contact_paths'] and force <= .001))",mode='eval').body
            return self.generic_visit(n)
    fn=Adapt().visit(fn)
    if changed!={'roots':1,'targets':1,'pairs':1}:raise ValueError('Original capture audit assumptions changed: '+str(changed))
    scope=dict(vars(C));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),__file__+':capture','exec'),scope)
    return scope['capture_audit']


def runtime_auditor():
    tree=ast.parse(Path(A.__file__).read_text());fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='audit_runtime'))
    count=0
    class Adapt(ast.NodeTransformer):
        def visit_Compare(self,n):
            nonlocal count
            if ast.unparse(n)=='np.max(np.abs(velocity)) <= 2e-05':
                count+=1;return ast.parse("np.array_equal(velocity, _array(next(o for o in scene['objects'] if o['object_id'] == key)['observed_twist'], (6,)))",mode='eval').body
            return self.generic_visit(n)
    fn=Adapt().visit(fn)
    if count!=1:raise ValueError('Original cold-zero-velocity audit changed')
    scope=dict(vars(A));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),__file__+':runtime','exec'),scope)
    return scope['audit_runtime']


def capture(path,protocol):
    result=capture_auditor()(path);p=B.read(protocol);observed=B.read(Path(path)/'observed_scene.json')
    metadata=observed['persistent_world'];template=B.read(B.read(p['inputs']['geometry_template']['path'])['scene']['path'])
    ids=[o['object_id']for o in observed['objects']]
    expected=['/World/Object'if key==template['target_object_id']else'/World/AuthorScene/'+key for key in ids]
    if metadata['active_paths']!=expected or metadata['world_builds']!=1:raise ValueError('Capture source-to-bank identity mismatch')
    if not metadata['robot_contact_paths']or not all(x.startswith('/World/Robot/')for x in metadata['robot_contact_paths']):raise ValueError('Original robot contact coverage missing')
    live=B.read(Path(path)/'LIVE_STATE.json')
    with np.load(Path(path)/'initialization_trace.npz',allow_pickle=False)as z:
        for i,key in enumerate(ids):
            np.testing.assert_array_equal(live['objects'][key]['pose_xyzw'],z['pose_xyzw'][-1,i]);np.testing.assert_array_equal(live['objects'][key]['velocity'],z['velocity'][-1,i])
    result.update(live_state=B.binding(Path(path)/'LIVE_STATE.json'),persistent_world=metadata,entry_semantics='Measured velocities preserved through observation; zeroing forbidden')
    return result


def runtime(prepared,split):
    began=time.monotonic();tree=ast.parse(Path(A.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='audit_trial')
    scope=dict(vars(A),audit_runtime=runtime_auditor());exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),__file__+':audit_trial','exec'),scope)
    result=scope['audit_trial'](prepared,split_manifest=split)
    if result['status']!='pass':return result
    prepared=Path(prepared);ready=B.read(prepared/'READY.json');plan=B.read(ready['plan']['path']);binding=plan['persistent_world']
    B.verify({'observation':binding['observation'],'source_plan':binding['source_plan']})
    live=B.read(binding['observation']['path']);setup=B.read(prepared/'execution/MULTI_OBJECT_SETUP.json')
    proof=setup['world_continuity']
    if proof['state_sha256']!=json_sha(live)or proof['sim_step']!=live['sim_step']or proof['actor_pose_or_velocity_writes']!=0 or proof['world_builds']!=1:
        raise ValueError('Same-world no-reset observation proof failed')
    entry=B.read(prepared/'execution/initial_state_audit.json')
    if entry['mode']!='bound_measured_live_state_without_reset':raise ValueError('Native executor used cold entry')
    np.testing.assert_array_equal(entry['actual_initial_speeds']['object'],live['object_velocity'])
    for key,state in setup['entry'].items():
        np.testing.assert_array_equal(state['pose_xyzw'],live['objects'][key]['pose_xyzw']);np.testing.assert_array_equal(state['velocity'],live['objects'][key]['velocity'])
    old=B.read(binding['source_plan']['path'])
    if B.sha(Path(binding['source_plan']['path']).parent/'commands.npz')!=old['commands_sha256']:raise ValueError('Original source commands changed')
    with np.load(Path(binding['source_plan']['path']).parent/'commands.npz',allow_pickle=False)as source,np.load(ready['commands']['path'],allow_pickle=False)as actual:
        for key in source.files:
            if key in ('initial_position','collection_execution_spec_sha256'):continue
            if source[key].tobytes()!=actual[key].tobytes()or source[key].dtype!=actual[key].dtype:raise ValueError('Original native command changed: '+key)
        names=list(map(str,actual['joint_names']))
        for j,name in enumerate(names):
            if name in ('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'):
                if source['initial_position'][j]!=actual['initial_position'][j]:raise ValueError('Prepared arm initial state changed')
    result['audit_inputs_sha256'].update({str(Path(p).resolve()):B.sha(p)for p in (binding['observation']['path'],binding['source_plan']['path'],prepared/'execution/initial_state_audit.json',__file__)})
    result.update(live_entry=proof,original_commands_and_physics_gates_preserved=True,live_initial_hand_only=True,wall_s=time.monotonic()-began)
    return result


if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('stage',choices=['capture','runtime'])
    q.add_argument('--capture-dir',type=Path);q.add_argument('--protocol',type=Path);q.add_argument('--prepared-dir',type=Path)
    q.add_argument('--split-manifest',type=Path);q.add_argument('--output',type=Path,required=True);a=q.parse_args()
    import traceback
    try:result=capture(a.capture_dir,a.protocol)if a.stage=='capture'else runtime(a.prepared_dir,a.split_manifest)
    except BaseException:result=dict(status='failed',error=traceback.format_exc())
    result['live_auditor']=B.binding(__file__);B.save(a.output,result,exclusive=True);print(result['status'],flush=True)
    if result['status']!='pass':raise SystemExit(1)
