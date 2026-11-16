"""Static three-object scene adapter using DexGraspBench's source-mesh pattern.

Reuse the actual AnyDex measured scene, original CoACD parts and24 commands.
No stepping API or label output is provided. This is not a production backend.
"""
import argparse
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import author_mujoco_preflight as ROBOT

B,T=ROBOT.B,ROBOT.T
_OWNER=None


def write_controls(model,data,names,targets):
    import mujoco
    values=np.asarray(targets,dtype=np.float64)
    if len(names)!=len(set(names))or values.shape!=(len(names),)or not np.isfinite(values).all():
        raise ValueError('Unique finite named actuator targets required')
    indices=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n)for n in names]
    if any(i<0 for i in indices):raise ValueError('Motor missing in compiled model')
    if model.nu!=len(names):raise ValueError('Do not omit independent motors')
    data.ctrl[indices]=values
    return np.asarray(indices,dtype=np.int64)


def add_scene(spec,scene):
    import mujoco
    from run_template_arm_positive import object_asset
    if scene['bin']is not None or len(scene['objects'])!=3:raise ValueError('Original three-body no-bin scene required')
    spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[2.,2.,.1],
                           friction=[.6,.02,.0001],condim=4)
    records=[]
    for row in scene['objects']:
        asset=object_asset(SimpleNamespace(object_root=Path(row['object_root'])),np.asarray(row['world_T_object']),row['scale'])
        parts=asset.collision_parts
        volume=sum(float(p.volume)for p in parts)
        if not parts or not np.isfinite(volume)or volume<=0:raise ValueError('Positive source collision volume required')
        pose=np.asarray(row['world_T_object']);quat=np.zeros(4);mujoco.mju_mat2Quat(quat,pose[:3,:3].ravel())
        body=spec.worldbody.add_body(name=row['object_id'],pos=pose[:3,3],quat=quat)
        body.add_freejoint(name=row['object_id']+'_free')
        for i,part in enumerate(parts):
            name=row['object_id']+'_part_'+str(i)
            spec.add_mesh(name=name,uservert=np.asarray(part.vertices).ravel(),userface=np.asarray(part.faces).ravel())
            if not np.isfinite(part.volume)or part.volume<=0:raise ValueError('Positive volume per source convex piece required')
            body.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,mass=.1*float(part.volume)/volume,
                          friction=[.6,.02,.0001],condim=4)
        records.append(dict(object_id=row['object_id'],source_parts=len(parts),expected_mass_kg=.1,
                            initial_world_pose=pose.tolist(),source_mesh=B.binding(row['source_mesh'])))
    return records


def inspect_prepared(prepared,output):
    import mujoco
    global _OWNER
    if _OWNER is not None:raise RuntimeError('Only one owning preflight thread')
    _OWNER=threading.get_ident();prepared=Path(prepared);ready=B.read(prepared/'READY.json')
    scene=T.validate_scene(ready['scene']['path'],B.read(prepared/'LIFT_PIPELINE_PROTOCOL.json').get('scene_archive_sources'))
    original=ROBOT.build_robot;details={}
    def build(plan,names):
        spec,_,planning,reference,parameters,robot_s=original(plan,names)
        objects=add_scene(spec,scene)
        for geom in spec.geoms:geom.friction=[.6,.02,.0001];geom.condim=4
        start=time.monotonic();model=spec.compile();elapsed=time.monotonic()-start
        if(model.nq,model.nv,model.nu,model.neq)!=(49,46,24,4):raise ValueError('Robot and three free-object dimensions differ')
        data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
        for obj in objects:
            index=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj['object_id'])
            pose=np.eye(4);pose[:3,:3]=data.xmat[index].reshape(3,3);pose[:3,3]=data.xpos[index]
            obj.update(actual_mass_kg=float(model.body_mass[index]),pose_error=float(abs(pose-obj['initial_world_pose']).max()))
            if abs(obj['actual_mass_kg']-.1)>1e-12 or obj['pose_error']>1e-10:raise ValueError('Source mass/initial pose changed')
            data.xfrc_applied[index,2]=-.98
        with np.load(ready['commands']['path'],allow_pickle=False)as z:
            for row in z['joint_target']:
                indices=write_controls(model,data,names,row)
                if not np.array_equal(data.ctrl[indices],row):raise ValueError('Original motor target was clipped/reordered')
            command_rows=len(z['joint_target'])
        if data.time!=0.:raise ValueError('Static adapter unexpectedly integrated')
        details.update(objects=objects,total_qpos=model.nq,total_velocity=model.nv,total_collision_geoms=model.ngeom,
            original_motor_rows_checked=command_rows,all_motor_targets_exact=True,robot_compile_s=robot_s,
            three_object_compile_s=elapsed,declared_per_object_force_n=[0.,0.,-.98],friction=dict(tangent=.6,torsional=.02,rolling=.0001,condim=4),
            friction_source='DexGraspBench-style explicit setting; equivalence to PhysX friction not established')
        return spec,model,planning,reference,parameters,robot_s+elapsed
    try:
        ROBOT.build_robot=build;robot=ROBOT.inspect_prepared(prepared,output)
    finally:ROBOT.build_robot=original;_OWNER=None
    result=dict(status='static_scene_contract_pass',schema='anydex_boya_mujoco_scene_preflight_v1',
        source=B.binding(__file__),robot_preflight=B.binding(Path(output)/'RESULTS.json'),measured_scene=ready['scene'],
        **details,physics_steps=0,Kit_starts=0,new_labels=0,training_updates=0,production_ready=False,
        dynamic_and_contact_equivalence_validated=False,source_geometry_native_tolerance_validated=False,
        speedup_measured=False)
    B.save(Path(output)/'SCENE_PREFLIGHT.json',result,exclusive=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared-dirs',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if not 1<=len(a.prepared_dirs)<=2:p.error('One/two static scene checks only')
    for i,directory in enumerate(a.prepared_dirs):
        v=inspect_prepared(directory,a.output_dir/f'{i:02d}')
        print({k:v[k]for k in ('status','total_qpos','total_collision_geoms','original_motor_rows_checked','physics_steps')},flush=True)
