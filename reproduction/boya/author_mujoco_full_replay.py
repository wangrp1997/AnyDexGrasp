"""One fixed original g1 plan on MuJoCo CPU. Diagnostic only, zero retries.

No CGR, batch collection, training, renderer, Kit or eligible label output.
Observations use copyData+forward to align contacts/kinematics with end-state
qpos without perturbing the live integration data or applying runtime resets.
"""
import argparse
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import author_mujoco_control as C

B, T, L = C.B, C.T, C.LIFT


def run(prepared, output):
    import mujoco
    began = time.monotonic()
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    ready = B.read(prepared/'READY.json')
    plan = B.read(ready['plan']['path'])
    inputs = {k: ready[k] for k in ('plan', 'commands', 'scene', 'command_binding', 'candidate')}
    for field in ('source_sha256', 'extra_input_sha256'):
        inputs.update(B.normalized_bindings(plan.get(field, {})))
    sources = {str(Path(p).resolve()): B.binding(p) for p in
        (__file__, C.__file__, C.R.__file__, C.S.__file__, C.FILTERS.__file__, L.__file__, T.__file__, C.EXECUTOR)}
    B.verify(inputs); B.verify(sources)
    with np.load(ready['commands']['path'], allow_pickle=False) as z:
        names = z['joint_names'].tolist(); initial = z['initial_position'].copy()
        targets = z['joint_target'].copy(); phases = z['phase'].tolist(); desired = z['desired_palm'].copy()
    if len(targets) != 3389 or plan['gesture_id'] != 1:
        raise ValueError('Only the original fixed g1 3389-step diagnostic is authorized here')
    protocol = dict(schema='anydex_boya_mujoco_full_replay_v1', inputs=inputs, sources=sources,
        max_steps=len(targets), max_attempts=1, retries=0, Kit_starts=0, CGR_forwards=0,
        physics_backend='MuJoCo CPU 3.11 explicit change from native PhysX', exclude_training=True,
        contact_settings='Unchanged from source-filtered prototype: implicitfast/default solver, condim4, friction .6/.02/.0001',
        control_corrections=['original six-arm float32 velocity targets', 'original THJ4/THJ3 0.6Nm override'],
        timestep_s=.004, original_seven_numeric_gates=True, lift_policy=L.POLICY,
        early_stop='Original >5mm penetration or >3cm preclosure XY displacement of any original body',
        contact_schema='Every end-state-copy mjContact with native geom/body ids, frame, wrench6, distance and solver parameters; not PhysX filtered contract',
        observation='mj_copyData plus mj_forward on separate end-state data; live simulation data not modified',
        source_geometry_native_20um_validated=False, production_ready=False)
    B.save(output/'PROTOCOL.json', protocol, exclusive=True)
    rows=[]; steps=0; stop=None; failure=None; step_s=0.; observation_s=0.; check_s=0.; write_s=0.
    B.save(output/'STARTED.json', dict(started_unix_s=time.time()), exclusive=True)
    try:
        scene = L.validate_scene(ready['scene']['path'], B.read(prepared/'LIFT_PIPELINE_PROTOCOL.json').get('scene_archive_sources'))
        spec, model, planning, reference, parameters, arm_names, excludes, objects = C.build(plan, names, scene)
        velocity = C.source_velocity_targets(targets, names, arm_names)
        from canondex.grasp.template_gripper_adapter import TemplateFamily
        from run_template_arm_positive import object_asset
        args = plan['arguments']
        family = TemplateFamily.load(Path(args['package']), Path(args['hand_urdf']), Path(args['hand_scene']), plan['gesture_id'])
        legal = {planning.fk.fixed_attachment(n, np.eye(4))[0] for n in family.hand.links}
        ids = [r['object_id'] for r in scene['objects']]
        assets = {r['object_id']: object_asset(SimpleNamespace(object_root=Path(r['object_root'])),
                  np.asarray(r['world_T_object']), r['scale']) for r in scene['objects']}
        mapping = C.body_mapping(model, planning, ids)
        robot_paths = sorted({p for p in mapping.values() if p.startswith('/World/Robot/')})
        joint_names = list(planning.joints)
        qindices = C.R.joint_addresses(model, joint_names)
        jindices = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        vindices = model.jnt_dofadr[jindices]
        bodyids = {key: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, key) for key in ids}
        data, observed = mujoco.MjData(model), mujoco.MjData(model)
        C.R.put_state(model, data, names, initial, planning.mimics)
        for body in bodyids.values():
            if abs(float(model.body_mass[body])-.1) > 1e-12:
                raise ValueError('Original object mass changed')
            data.xfrc_applied[body, 2] = -.98
        C.controls(model, data, names, targets[0], velocity[0], arm_names)
        mujoco.mj_forward(model, data)
        entry_errors = []
        for obj in objects:
            actual = C.pose_matrix(model, data, bodyids[obj['object_id']])
            entry_errors.append(float(abs(actual-np.asarray(obj['initial_world_pose'])).max()))
        if max(entry_errors) > 1e-10 or np.max(abs(data.qvel)) != 0. or data.time != 0.:
            raise ValueError('Measured scene cold entry changed')
        setup = dict(mujoco=mujoco.__version__, command_joint_names=names, measured_joint_names=joint_names,
            arm_names=arm_names, legal_hand_owners=sorted(legal), objects=objects, mapping=mapping,
            drive_parameters=parameters, source_excludes=excludes, model_dimensions=dict(nq=model.nq,nv=model.nv,nu=model.nu,ngeom=model.ngeom),
            source_initial_pose_max_error=max(entry_errors), solver_iterations=model.opt.iterations,
            explicit_new_backend=True, no_runtime_state_writes=True)
        (output/'model.xml').write_text(spec.to_xml())
        B.save(output/'SETUP.json', setup, exclusive=True)
        settled={}
        with (output/'TRACE.jsonl').open('x') as trace:
            loop_start=time.monotonic()
            for i, (phase, target) in enumerate(zip(phases, targets)):
                t=time.perf_counter()
                C.controls(model, data, names, target, velocity[i], arm_names)
                control_s=time.perf_counter()-t
                t=time.perf_counter(); mujoco.mj_step(model, data); step_s+=time.perf_counter()-t; steps+=1
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time-steps*.004)>1e-10:
                    raise RuntimeError('Nonfinite state or simulation time discontinuity')
                t=time.perf_counter()
                mujoco.mj_copyData(observed, model, data); mujoco.mj_forward(model, observed)
                contacts=C.contact_rows(model, observed, mapping); views=C.object_views(contacts, ids)
                states={}
                for key, body in bodyids.items():
                    quat=np.zeros(4); mujoco.mju_mat2Quat(quat, observed.xmat[body])
                    pose=np.r_[observed.xpos[body], quat[1:], quat[0]]
                    vel=np.zeros(6); mujoco.mj_objectVelocity(model, observed, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
                    states[key]=dict(pose_xyzw=pose.tolist(),velocity=np.r_[vel[3:],vel[:3]].tolist(),
                        **T.object_robot_loads('/World/'+key, views, robot_paths, legal),
                        **L.source_metrics(assets[key].mesh.vertices, pose))
                base=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, planning.fk.root)
                row=dict(step=steps,time_s=float(data.time),phase=phase,objects=states,contacts=contacts,
                    joint_position=data.qpos[qindices].tolist(),joint_velocity=data.qvel[vindices].tolist(),
                    joint_target=target.tolist(),velocity_target=velocity[i].tolist(),qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),
                    base_transform=C.pose_matrix(model,observed,base).tolist(),
                    palm_transform=C.pose_matrix(model,observed,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'palm')).tolist())
                observation_s+=time.perf_counter()-t
                t=time.perf_counter(); rows.append(row)
                if phase=='settle': settled={key:np.asarray(state['pose_xyzw'][:2]) for key,state in states.items()}
                depth=max([0.]+[-c['separation_m'] for c in contacts])
                if depth>.005: stop='multi_object_deep_collision_stop'
                pushed=T.preclosure_pushes(states,settled,phase)
                if pushed: stop='source_object_pushed_before_closure:'+','.join(pushed)
                check_s+=time.perf_counter()-t
                t=time.perf_counter(); trace.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n'); write_s+=time.perf_counter()-t
                if steps%250==0 or stop or steps==len(targets):
                    trace.flush()
                    B.save(output/'COST.json',dict(steps_returned=steps,three_body_steps=3*steps,contacts=sum(len(r['contacts']) for r in rows),labels=None))
                    print(json.dumps(dict(step=steps,phase=phase,depth_m=depth,stop=stop)),flush=True)
                if stop: break
            loop_s=time.monotonic()-loop_start
        warnings=sum(int(w.number) for w in data.warning)
        if warnings: raise RuntimeError('MuJoCo warning count '+str(warnings))
        t=time.perf_counter()
        maxima,first=C.numeric_maxima(rows,planning,joint_names,desired,reference,legal,ids)
        check_s+=time.perf_counter()-t
        gates=C.source_gates(maxima,steps,len(targets))
        lift=L.evaluate_lift(rows,ids,.004)
        B.verify(inputs); B.verify(sources)
        result=dict(status='complete' if steps==len(targets) and stop is None else 'physical_stop',
            gates=gates,maxima=maxima,first_numeric_gate_failure=first,observed_lift=lift,
            diagnostic_physical_gates_pass=all(gates.values()) and stop is None,
            diagnostic_task_success=all(gates.values()) and stop is None and lift['task_success'],
            warnings=warnings,replay_loop_wall_s=loop_s)
    except BaseException:
        failure=traceback.format_exc();result=dict(status='runtime_error',error=failure)
    result.update(total_wall_s=time.monotonic()-began,physics_steps_returned=steps,recorded_frames=len(rows),
        dynamic_object_steps=steps*3,physical_stop=stop,mj_step_wall_s=step_s,observation_wall_s=observation_s,
        checks_wall_s=check_s,journal_write_wall_s=write_s,contact_points=sum(len(r['contacts']) for r in rows),
        Kit_starts=0,CGR_forwards=0,new_labels=0,training_updates=0,retries=0,exclude_training=True,
        production_ready=False,physical_backend_changed=True,source_geometry_native_20um_validated=False,
        complete_PhysX_contact_contract_compatible=False,throughput_speedup_certified=False)
    B.save(output/'RESULTS.json',result,exclusive=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ('observed_lift',)},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();r=run(a.prepared_dir,a.output_dir)
    if r['status']=='runtime_error':raise SystemExit(1)
