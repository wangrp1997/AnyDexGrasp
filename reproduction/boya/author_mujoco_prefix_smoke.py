"""One fixed20-step CPU smoke test of the new MuJoCo scene adapter.

Different explicit physical backend. No grasp label, collection batch, renderer,
Kit, retry or source execution marker is created. Full prefix contacts retained.
"""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np
import author_mujoco_scene_preflight as S

B,ROBOT=S.B,S.ROBOT
STEPS=20


def run(prepared,output):
    import mujoco
    prepared=Path(prepared).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    ready=B.read(prepared/'READY.json');plan=B.read(ready['plan']['path'])
    protocol=B.read(prepared/'LIFT_PIPELINE_PROTOCOL.json')
    sources={str(Path(p).resolve()):B.binding(p)for p in (__file__,S.__file__,ROBOT.__file__)}
    inputs={k:ready[k]for k in ('plan','commands','scene','command_binding')}
    inputs['preparation']=B.binding(prepared/'LIFT_PIPELINE_RESULTS.json')
    B.verify(inputs)
    for field in ('source_sha256','extra_input_sha256'):B.verify(B.normalized_bindings(plan.get(field,{})))
    B.save(output/'PROTOCOL.json',dict(schema='anydex_boya_mujoco_prefix_smoke_v1',max_steps=STEPS,
        phases=['settle'],physics_backend='MuJoCo CPU explicit change',sources=sources,inputs=inputs,
        retry_budget=0,Kit_starts=0,exclude_training=True,new_labels=0),exclusive=True)
    B.save(output/'STARTED.json',dict(started_unix_s=time.time()),exclusive=True)
    began=time.monotonic();steps=0;frames=[];error=None;step_s=0.;read_s=0.;start_clock=None
    try:
        scene=S.T.validate_scene(ready['scene']['path'],protocol.get('scene_archive_sources'))
        with np.load(ready['commands']['path'],allow_pickle=False)as z:
            names=z['joint_names'].tolist();initial=z['initial_position'].copy()
            commands=z['joint_target'][:STEPS].copy();phases=z['phase'][:STEPS].tolist()
        if len(commands)!=STEPS or set(phases)!={'settle'}:raise ValueError('Only first20 original settle commands allowed')
        spec,_,planning,_,_,_=ROBOT.build_robot(plan,names);S.add_scene(spec,scene)
        for geom in spec.geoms:geom.friction=[.6,.02,.0001];geom.condim=4
        model=spec.compile();data=mujoco.MjData(model)
        ROBOT.put_state(model,data,names,initial,planning.mimics)
        for row in scene['objects']:
            index=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,row['object_id'])
            data.xfrc_applied[index,2]=-.98
        S.write_controls(model,data,names,commands[0]);mujoco.mj_forward(model,data);start_clock=float(data.time)
        for command in commands:
            indices=S.write_controls(model,data,names,command)
            if not np.array_equal(data.ctrl[indices],command):raise ValueError('Frozen command changed')
            t=time.perf_counter();mujoco.mj_step(model,data);step_s+=time.perf_counter()-t;steps+=1
            if abs(data.time-start_clock-steps*.004)>1e-12 or not np.isfinite(data.qpos).all()or not np.isfinite(data.qvel).all():
                raise RuntimeError('Nonfinite state or internal reset/time discontinuity')
            t=time.perf_counter();contacts=[]
            for i in range(data.ncon):
                c=data.contact[i];force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
                contacts.append(dict(geom1=int(c.geom1),geom2=int(c.geom2),distance_m=float(c.dist),
                    position_world_m=c.pos.tolist(),contact_frame_world=c.frame.tolist(),force_contact_frame=force.tolist()))
            frames.append(dict(step=steps,time_s=float(data.time),phase='settle',qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),contacts=contacts))
            read_s+=time.perf_counter()-t
            B.save(output/'COST.json',dict(physics_steps_returned=steps,expected_steps=STEPS,labels=None))
        warnings=int(sum(int(w.number)for w in data.warning))
        if warnings:raise RuntimeError('MuJoCo warnings during tiny prefix: '+str(warnings))
        B.verify(inputs);B.verify(sources)
    except BaseException:error=traceback.format_exc()
    finally:
        B.save(output/'PREFIX_TRACE_AND_CONTACTS.json',dict(frames=frames,fields_dropped=False))
        result=dict(status='prefix_smoke_complete'if error is None and steps==STEPS else'stopped',error=error,
            physics_steps_returned=steps,trace_frames=len(frames),dynamic_object_steps=3*steps,
            simulation_time_s=steps*.004,mj_step_wall_s=step_s,full_prefix_readback_wall_s=read_s,
            total_wall_s=time.monotonic()-began,contact_points=sum(len(x['contacts'])for x in frames),
            Kit_starts=0,CGR_forwards=0,new_labels=0,training_updates=0,retries=0,
            exclude_training=True,physical_backend_changed=True,full_grasp_gates_validated=False,
            throughput_speedup_certified=False,scope='One20-step settle-prefix interface smoke only; not an original physical execution certificate')
        B.save(output/'RESULTS.json',result,exclusive=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();v=run(a.prepared_dir,a.output_dir)
    print(v,flush=True)
    if v['status']!='prefix_smoke_complete':raise SystemExit(1)
