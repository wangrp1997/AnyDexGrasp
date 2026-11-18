"""Finite saved-plan test of one world, object exchange and physical return.

No RGBD inference, candidate selection, collection labels or training. The
MuJoCo backend/contact settings differ explicitly from the old PhysX control.
"""
import argparse
import os
from pathlib import Path
import time
import json
import shutil


def observe(world, phase, index, clock_start, legal):
    import mujoco
    import numpy as np
    import author_mujoco_control as C
    import author_fast_geometry_checks as F
    mujoco.mj_copyData(world.observed,world.model,world.data)
    mujoco.mj_forward(world.model,world.observed)
    contacts=C.contact_rows(world.model,world.observed,world.mapping)
    views=C.object_views(contacts,world.active); states={}
    robot_paths=sorted({p for p in world.mapping.values() if p.startswith('/World/Robot/')})
    for key in world.active:
        body=world.bodyids[key];quat=np.zeros(4)
        mujoco.mju_mat2Quat(quat,world.observed.xmat[body])
        pose=np.r_[world.observed.xpos[body],quat[1:],quat[0]]
        vel=np.zeros(6);mujoco.mj_objectVelocity(world.model,world.observed,mujoco.mjtObj.mjOBJ_BODY,body,vel,0)
        states[key]=dict(pose_xyzw=pose.tolist(),velocity=np.r_[vel[3:],vel[:3]].tolist(),
            **C.T.object_robot_loads('/World/'+key,views,robot_paths,legal),
            **F.source_metrics(world.assets[key].mesh.vertices,pose))
    return dict(step=index,time_s=index*.004,native_time_s=float(world.data.time),world_step=world.steps,
        phase=phase,objects=states,contacts=contacts,
        joint_position=world.data.qpos[world.qindices].tolist(),joint_velocity=world.data.qvel[world.vindices].tolist(),
        base_transform=C.pose_matrix(world.model,world.observed,mujoco.mj_name2id(world.model,mujoco.mjtObj.mjOBJ_BODY,world.planning.fk.root)).tolist(),
        palm_transform=C.pose_matrix(world.model,world.observed,mujoco.mj_name2id(world.model,mujoco.mjtObj.mjOBJ_BODY,'palm')).tolist())


def main():
    os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared-dirs',type=Path,nargs='+',required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--max-steps',type=int,default=17500);a=p.parse_args()
    if not 1<=len(a.prepared_dirs)<=2 or not 1<=a.max_steps<=17500:p.error('At most two explicit saved plans and 17500 ticks')
    import numpy as np
    import author_mujoco_control as C
    C.T._backend_path()
    from author_mujoco_persistent_world import World
    from author_compiled_kinematics import CompiledKinematics
    from canondex.grasp.template_gripper_adapter import TemplateFamily
    import mujoco
    B=C.B;root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();ready=[];plans=[];scenes=[];arrays=[];inputs={}
    for i,directory in enumerate(a.prepared_dirs):
        r=B.read(directory/'READY.json');B.verify({k:r[k] for k in ('plan','commands','scene','command_binding')})
        plan=B.read(r['plan']['path'])
        for field in ('source_sha256','extra_input_sha256'):
            B.verify({str(k):dict(path=k,sha256=v) for k,v in plan.get(field,{}).items()})
        with np.load(r['commands']['path'],allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
        ready.append(r);plans.append(plan);scenes.append(B.read(r['scene']['path']));arrays.append(arr)
        inputs[str(i)]=B.binding(directory/'READY.json')
    sources={name:B.binding(B.HERE/name) for name in (Path(__file__).name,'author_mujoco_persistent_world.py',
        'author_mujoco_control.py','author_mujoco_preflight.py','author_mujoco_scene_preflight.py',
        'author_mujoco_source_filters.py','author_fast_geometry_checks.py','author_compiled_kinematics.py',
        'author_static_kinematics.py','author_planning_reuse.py','author_mujoco_return.py')}
    archive=root/'source_archive';archive.mkdir()
    for name,row in sources.items():shutil.copyfile(row['path'],archive/name)
    B.save(root/'PROTOCOL.json',dict(inputs=inputs,sources=sources,max_steps=a.max_steps,max_grasps=len(arrays),
        max_world_builds=1,max_explicit_scene_resets=1,retries=0,Kit_starts=0,exclude_training=True,
        new_training_labels=0,new_CGR_forwards=0,RGBD_frames=0,
        contact_source='Source XML mimic parameters plus explicit DexGraspBench hand solref/solimp and friction solver; source 0.5ms XML timestep is not claimed reproduced',
        scope='Saved-plan backend/lifecycle diagnostic; no complete acquisition throughput certification'),exclusive=True)
    names=arrays[0]['joint_names'].tolist();world=None;results=[];error=None;return_steps=0;returns=[]
    try:
        with CompiledKinematics().installed():
            world=World(plans[0],names,arrays[0]['initial_position'],scenes,max_steps=a.max_steps)
            (root/'model.xml').write_text(world.spec.to_xml())
            for case,(r,plan,scene,arr) in enumerate(zip(ready,plans,scenes,arrays)):
                if arr['joint_names'].tolist()!=names:raise ValueError('Saved motor orders differ')
                if case:
                    # Remove exchanged objects, then physically retrace the prior
                    # path and align to the next start. Never write robot state.
                    from author_mujoco_return import return_to_start
                    before_return=world.steps
                    try:
                        returned=return_to_start(world,arrays[case-1]['joint_target'],arr['initial_position'],root/f'return_{case:02d}.jsonl')
                    finally:return_steps+=world.steps-before_return
                    returns.append(returned);B.save(root/f'RETURN_{case:02d}.json',returned,exclusive=True)
                    if not returned['pass_']:raise RuntimeError('Physical return failed original numeric gates')
                exchange_begin=time.monotonic();world.exchange(scene);exchange_s=time.monotonic()-exchange_begin
                world.begin_round();clock=float(world.data.time)
                family=TemplateFamily.load(Path(plan['arguments']['package']),Path(plan['arguments']['hand_urdf']),Path(plan['arguments']['hand_scene']),plan['gesture_id'])
                legal={world.planning.fk.fixed_attachment(n,np.eye(4))[0] for n in family.hand.links}
                velocity=C.source_velocity_targets(arr['joint_target'],names,world.arm_names)
                rows=[];settled={};stop=None;step_s=observer_s=write_s=0.;mark_case=time.monotonic()
                with (root/f'case_{case:02d}.jsonl').open('x') as log:
                    for k,(target,speed,phase) in enumerate(zip(arr['joint_target'],velocity,arr['phase']),1):
                        t=time.perf_counter();world.step(target,speed);step_s+=time.perf_counter()-t
                        t=time.perf_counter();row=observe(world,str(phase),k,clock,legal);observer_s+=time.perf_counter()-t
                        row.update(joint_target=target.tolist(),velocity_target=speed.tolist());rows.append(row)
                        if phase=='settle':settled={key:np.asarray(state['pose_xyzw'][:2]) for key,state in row['objects'].items()}
                        depth=max([0.]+[-c['separation_m'] for c in row['contacts']])
                        if depth>.005:stop='original_5mm_deep_collision_stop'
                        if C.T.preclosure_pushes(row['objects'],settled,str(phase)):stop='original_preclosure_push_stop'
                        t=time.perf_counter();log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');write_s+=time.perf_counter()-t
                        if stop:break
                        if k%250==0:
                            B.save(root/'COST.json',dict(steps=world.steps,return_steps=return_steps,grasps_started=case+1,labels=0))
                loop_wall_s=time.monotonic()-mark_case
                world.end_round()
                maxima,first=C.numeric_maxima(rows,world.planning,list(world.planning.joints),arr['desired_palm'],world.reference,legal,world.active)
                gates=C.source_gates(maxima,len(rows),len(arr['phase']))
                evaluation=C.LIFT.evaluate_lift(rows,world.active,.004)
                result=dict(case=case,steps=len(rows),expected_steps=len(arr['phase']),stop=stop,gates=gates,maxima=maxima,first_failure=first,
                    diagnostic_task_success=stop is None and all(gates.values()) and evaluation['task_success'],evaluation=evaluation,
                    exchange_wall_s=exchange_s,step_s=step_s,observer_s=observer_s,json_write_s=write_s,loop_wall_s=loop_wall_s,wall_s=time.monotonic()-mark_case)
                results.append(result);B.save(root/f'CASE_{case:02d}.json',result,exclusive=True)
                B.save(root/'COST.json',dict(steps=world.steps,return_steps=return_steps,grasps_started=case+1,labels=0))
                print({k:result[k] for k in ('case','steps','gates','diagnostic_task_success','step_s','observer_s','wall_s')},flush=True)
                if sum(int(w.number) for w in world.data.warning):raise RuntimeError('Native MuJoCo warnings')
    except BaseException:
        import traceback
        error=traceback.format_exc()
    finally:
        B.verify(sources);B.verify(inputs)
        B.save(root/'RESULTS.json',dict(status='complete' if error is None else 'failed',error=error,cases=results,
            world=world.receipt() if world else None,return_steps=return_steps,returns=returns,parent_wall_s=time.monotonic()-began,
            Kit_starts=0,new_training_labels=0,training_updates=0,production_ready=False,
            full_collection_validated=False),exclusive=True)
    if error:raise RuntimeError(error)


if __name__=='__main__':main()
