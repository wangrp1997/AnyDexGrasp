"""Static MuJoCo command/FK adapter inspired by DexGraspBench's MjHO.

No integration or label generation. Compile the original combined URDF and
check saved AnyDex commands before considering a new physical backend.
Reference: https://github.com/JYChen18/DexGraspBench/blob/main/src/util/hand_util.py
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np
import author_collection_batch as B
import author_transport as T

NATIVE_ROBOT=T.CANON/'canondex/robots/ur16e_boya.py'


def native_arm_parameters(names,mode='isaaclab',*,effort_limits):
    """Read the existing config function without importing Kit/IsaacLab."""
    if mode!='isaaclab':raise ValueError('This preflight only supports the saved native isaaclab arm mode')
    tree=ast.parse(NATIVE_ROBOT.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_arm_actuators')
    scope=dict(ImplicitActuatorCfg=lambda **kw:kw,ARM_ACTUATION_MODES=('isaaclab','source_tabletop'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(NATIVE_ROBOT),'exec'),scope)
    groups=scope['_arm_actuators'](mode);result={}
    for name in names:
        matches=[g for g in groups.values()if any(re.fullmatch(pattern,name)for pattern in g['joint_names_expr'])]
        if len(matches)!=1:raise ValueError('One original arm drive group required: '+name)
        result[name]=dict(stiffness=float(matches[0]['stiffness']),damping=float(matches[0]['damping']),
            viscous_friction=float(matches[0].get('viscous_friction',0.)),
            joint_effort_limit=float(matches[0].get('joint_effort_limit',effort_limits[name])),armature=float(matches[0].get('armature',0.)))
    return result


def joint_addresses(model,names):
    import mujoco
    if len(names)!=len(set(names)):raise ValueError('Duplicate command joint')
    result=[]
    for name in names:
        index=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if index<0 or model.jnt_type[index]!=mujoco.mjtJoint.mjJNT_HINGE:raise ValueError('Unknown/non-hinge joint: '+name)
        result.append(int(model.jnt_qposadr[index]))
    return np.asarray(result,dtype=np.int64)


def put_state(model,data,names,targets,mimics):
    values=np.asarray(targets,dtype=np.float64)
    if values.shape!=(len(names),)or not np.isfinite(values).all():raise ValueError('Finite named target vector required')
    data.qpos[joint_addresses(model,names)]=values
    state=dict(zip(names,map(float,values)))
    for child,(driver,multiplier,offset)in mimics.items():
        if child in state:raise ValueError('Passive mimic must not be a motor command')
        if driver not in state:raise ValueError('Missing mimic driver')
        data.qpos[joint_addresses(model,[child])]=state[driver]*multiplier+offset


def set_viscous_damping(joint,value):
    shape=np.asarray(joint.damping).shape
    if shape==():joint.damping=float(value)
    elif shape==(3,):joint.damping=[float(value),0.,0.]
    else:raise ValueError('Unsupported installed MuJoCo damping representation')


def build_robot(plan,names):
    import mujoco
    T._backend_path()
    from canondex.planning.model import PlanningModel
    from canondex.robots.hand_actuation import hand_drive_parameters
    from canondex.robots.reference import load_reference
    args=plan['arguments'];urdf=Path(args['robot_urdf']);planning=PlanningModel(urdf)
    if len(names)!=24 or set(names)!=set(planning.active_names):raise ValueError('All original24 motors required')
    physics=B.read(args['physics_config']);spec=mujoco.MjSpec.from_file(str(urdf))
    spec.compiler.fusestatic=False;spec.option.timestep=.004
    spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags|=mujoco.mjtDisableBit.mjDSBL_GRAVITY
    reference=load_reference(Path(args['reference_root']))
    base=next(b for b in spec.worldbody.bodies if b.name==planning.fk.root)
    base.pos=reference.base_transform[:3,3]
    quat=np.zeros(4);mujoco.mju_mat2Quat(quat,reference.base_transform[:3,:3].ravel());base.quat=quat
    joints={j.name:j for j in spec.joints}
    arm_names=[n for n in names if n.endswith('_joint')]
    parameters=native_arm_parameters(arm_names,args.get('arm_actuation_mode','isaaclab'),
        effort_limits={n:float(planning.joints[n].find('limit').get('effort'))for n in arm_names})
    hand=hand_drive_parameters(physics['hand_damping_mode'],Path(args['hand_scene']))
    native_tree=ast.parse(NATIVE_ROBOT.read_text())
    armature=next(ast.literal_eval(n.value)for n in native_tree.body if isinstance(n,ast.Assign)
                  and any(isinstance(x,ast.Name)and x.id=='FINGER_ARMATURE'for x in n.targets))
    parameters.update({n:dict(v,armature=armature)for n,v in hand.items()})
    for name in names:
        p=parameters[name];set_viscous_damping(joints[name],p['viscous_friction']);joints[name].armature=p['armature']
        joints[name].actfrcrange=[-p['joint_effort_limit'],p['joint_effort_limit']];joints[name].actfrclimited=1
        spec.add_actuator(name=name,target=name,trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,gainprm=[p['stiffness']]+[0.]*9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,biasprm=[0.,-p['stiffness'],-p['damping']]+[0.]*7,
            ctrllimited=0,forcelimited=1,forcerange=[-p['joint_effort_limit'],p['joint_effort_limit']])
    for child,(driver,multiplier,offset)in planning.mimics.items():
        set_viscous_damping(joints[child],parameters[child]['viscous_friction']);joints[child].armature=armature
        spec.add_equality(name='mimic_'+child,type=mujoco.mjtEq.mjEQ_JOINT,name1=child,name2=driver,
                          data=[offset,multiplier]+[0.]*9)
    begin=time.monotonic();model=spec.compile();elapsed=time.monotonic()-begin
    if(model.nq,model.nv,model.nu,model.neq)!=(28,28,24,4):raise ValueError('Original joint/actuator/mimic dimensions changed')
    return spec,model,planning,reference,parameters,elapsed


def inspect_prepared(prepared,output):
    import mujoco
    prepared=Path(prepared).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ready=B.read(prepared/'READY.json');B.verify({k:ready[k]for k in ('plan','commands','scene','command_binding')})
    plan=B.read(ready['plan']['path']);T.verify_pins()
    with np.load(ready['commands']['path'],allow_pickle=False)as z:
        names=z['joint_names'].tolist();targets=z['joint_target'].copy();phases=z['phase'].copy()
    spec,model,planning,reference,parameters,compile_s=build_robot(plan,names)
    data=mujoco.MjData(model);indices={0,len(targets)-1}
    for phase in dict.fromkeys(phases.tolist()):
        found=np.flatnonzero(phases==phase);indices.update([int(found[0]),int(found[len(found)//2]),int(found[-1])])
    errors=[];bodies=0
    for i in sorted(indices):
        put_state(model,data,names,targets[i],planning.mimics);mujoco.mj_kinematics(model,data)
        expected=planning.fk.transforms(planning.command_state(dict(zip(names,map(float,targets[i])))),reference.base_transform)
        for name,pose in expected.items():
            body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
            if body<0:raise ValueError('Original FK link missing: '+name)
            actual=np.eye(4);actual[:3,:3]=data.xmat[body].reshape(3,3);actual[:3,3]=data.xpos[body]
            errors.append(float(abs(actual-pose).max()));bodies+=1
    if max(errors)>1e-10 or data.time!=0.:raise ValueError('Static FK changed or physics integrated')
    (output/'robot.xml').write_text(spec.to_xml())
    result=dict(status='static_contract_pass',schema='anydex_boya_mujoco_preflight_v1',mujoco=mujoco.__version__,
        inputs=dict(ready=B.binding(prepared/'READY.json'),plan=ready['plan'],commands=ready['commands']),
        source=dict(adapter=B.binding(__file__),native_robot=B.binding(NATIVE_ROBOT)),
        command_rows=len(targets),active_motors=24,actual_joints=28,mimic_equalities=4,
        collision_geoms=model.ngeom,meshes=model.nmesh,compile_wall_s=compile_s,
        fk_command_states=len(indices),fk_link_checks=bodies,max_fk_matrix_error=max(errors),
        drive_parameters=parameters,physics_steps=0,Kit_starts=0,new_labels=0,training_updates=0,
        production_ready=False,dynamic_equivalence_validated=False,native_geometry_tolerance_validated=False,
        scope='Static model/command interface only; no object scene, integration, grasp labels or speedup certification')
    B.save(output/'RESULTS.json',result,exclusive=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared-dirs',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if not 1<=len(a.prepared_dirs)<=2:p.error('Only one/two fixed static preflights')
    for i,directory in enumerate(a.prepared_dirs):
        v=inspect_prepared(directory,a.output_dir/f'{i:02d}')
        print({k:v[k]for k in ('status','actual_joints','collision_geoms','compile_wall_s','max_fk_matrix_error','physics_steps')},flush=True)
