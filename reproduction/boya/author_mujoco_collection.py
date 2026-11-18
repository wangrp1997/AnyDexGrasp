"""Finite fresh-RGBD/CGR/planning/physical grasp cycles in one MuJoCo world."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import author_collection_batch as B

REFERENCE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/fast_geometry_world_control_20261117_v1/PROTOCOL.json')
SOURCES=('author_mujoco_collection.py','author_mujoco_camera.py','author_mujoco_persistent_world.py',
    'author_mujoco_representation.py','author_mujoco_return.py','replay_author_persistent_world.py',
    'author_mujoco_control.py','author_mujoco_preflight.py','author_mujoco_scene_preflight.py','author_mujoco_source_filters.py',
    'author_planning_reuse.py','author_static_kinematics.py','author_compiled_kinematics.py','author_resident_prepare.py',
    'author_live_world_contract.py','author_fast_geometry_checks.py','run_pilot_representation.py',
    'author_contact_pipeline.py','author_lift_pipeline.py','settle_s40_scenes.py','render_s40_observations.py',
    'author_mujoco_source_options.py','author_mujoco_owner.py','author_collection_wave_resources.py',
    'author_mujoco_lift_policy.py','author_mujoco_clock.py')


def freeze(root,prepared,*,source_options=False):
    from author_mujoco_lift_policy import POLICY
    if len(prepared)!=2:raise ValueError('Exactly two explicitly frozen source layouts required')
    old=B.read(REFERENCE);B.verify(old['inputs']);B.verify(old['sources'])
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    inputs={k:old['inputs'][k] for k in ('python','checkpoint','backend','calibration','cloud_manifest','nvidia_smi')}
    rows=[]
    for i,path in enumerate(prepared):
        ready=B.read(Path(path)/'READY.json');B.verify({k:ready[k] for k in ('scene','plan','commands','command_binding')})
        inputs['template:'+str(i)]=B.binding(Path(path)/'READY.json')
        scene=B.read(ready['scene']['path']);destination=root/'scenes'/f'{i:02d}'/'source_scene.json'
        B.save(destination,scene,exclusive=True);inputs['scene:'+str(i)]=B.binding(destination)
        rows.append(dict(index=i,scene_id=scene['scene_id'],scene=inputs['scene:'+str(i)]))
    sources={name:B.binding(B.HERE/name) for name in SOURCES};archive=root/'source_archive';archive.mkdir()
    for name,b in sources.items():shutil.copyfile(b['path'],archive/name)
    domain=copy.deepcopy(old['domain'])
    domain['sensor_adapter']['description']='Calibrated MuJoCo optical camera; inherited unchanged numeric width/workspace/approach predicates, new renderer/physics domain'
    p=dict(schema='anydex_boya_mujoco_fresh_collection_v1',rows=rows,inputs=inputs,sources=sources,
        env=dict(old['env'],MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
        inherited_env=old['inherited_env'],native_deps=old['native_deps'],gpu_uuid=old['gpu_uuid'],
        seed=old['queue']['seed'],domain=domain,max_world_builds=1,max_explicit_resets=1,max_camera_builds=1,
        max_environment_steps=17500,max_grasp_attempts=2,max_CGR_forwards=2,max_planning_jobs=2,
        source_hand_solver_options=bool(source_options),max_native_integration_steps=17500*(8 if source_options else 1),
        max_execution_steps_per_job=8000,settling_steps_per_round=750,retry_budget=0,
        exclude_training=True,formal_training_eligible=False,training_updates=0,
        physics_domain='Independent MuJoCo candidate, original source drives and commands, changed contact/geometry/solver',
        source_physics_config_semantics='Planning/source drive provenance, not a claim of active PhysX simulation',
        native_20um_and_PhysX_filtered_contract_validated=False,full_collection_qualification_complete=False)
    p['acceptance_policy']=copy.deepcopy(POLICY)
    B.save(root/'PROTOCOL.json',p,exclusive=True);return p


def execute(world,prepared,output):
    import mujoco
    import author_mujoco_control as C
    from replay_author_persistent_world import observe
    from canondex.grasp.template_gripper_adapter import TemplateFamily
    import json
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ready=B.read(Path(prepared)/'READY.json');B.verify({k:ready[k] for k in ('scene','plan','commands','command_binding')})
    plan=B.read(ready['plan']['path']);scene=B.read(ready['scene']['path'])
    with np.load(ready['commands']['path'],allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
    if arr['joint_names'].tolist()!=world.names or len(arr['phase'])>8000:raise ValueError('Finite original command order/budget changed')
    family=TemplateFamily.load(Path(plan['arguments']['package']),Path(plan['arguments']['hand_urdf']),Path(plan['arguments']['hand_scene']),plan['gesture_id'])
    legal={world.planning.fk.fixed_attachment(n,np.eye(4))[0] for n in family.hand.links}
    velocity=C.source_velocity_targets(arr['joint_target'],world.names,world.arm_names)
    rows=[];settled={};stop=None;step_s=observe_s=write_s=0.;began=time.monotonic();world.begin_round()
    start=world.steps
    with (output/'TRACE.jsonl').open('x') as log:
        for i,(target,speed,phase) in enumerate(zip(arr['joint_target'],velocity,arr['phase']),1):
            t=time.perf_counter();world.step(target,speed);step_s+=time.perf_counter()-t
            t=time.perf_counter();row=observe(world,str(phase),i,start*.004,legal);observe_s+=time.perf_counter()-t
            row.update(joint_target=target.tolist(),velocity_target=speed.tolist());rows.append(row)
            if phase=='settle':settled={k:np.asarray(s['pose_xyzw'][:2]) for k,s in row['objects'].items()}
            if max([0.]+[-c['separation_m'] for c in row['contacts']])>.005:stop='original_5mm_deep_collision_stop'
            if C.T.preclosure_pushes(row['objects'],settled,str(phase)):stop='original_preclosure_push_stop'
            t=time.perf_counter();log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');write_s+=time.perf_counter()-t
            if i%250==0:B.save(output/'COST.json',dict(steps=i,world_steps=world.steps,labels=0))
            if stop:break
    loop_s=time.monotonic()-began;world.end_round()
    maxima,first=C.numeric_maxima(rows,world.planning,list(world.planning.joints),arr['desired_palm'],world.reference,legal,world.active)
    gates=C.source_gates(maxima,len(rows),len(arr['phase']));lift=C.LIFT.evaluate_lift(rows,world.active,.004)
    from author_mujoco_lift_policy import evaluate
    acceptance=evaluate(rows,world.active,maxima,len(arr['phase']),stop=stop)
    warnings=sum(int(w.number) for w in world.data.warning)
    if warnings:raise RuntimeError('Native MuJoCo warnings during execution: '+str(warnings))
    result=dict(status='complete' if len(rows)==len(arr['phase']) else 'physical_stop',steps=len(rows),expected_steps=len(arr['phase']),
        start_world_step=start,end_world_step=world.steps,stop=stop,gates=gates,maxima=maxima,first_failure=first,
        diagnostic_physical_gates_pass=stop is None and all(gates.values()),
        diagnostic_task_success=acceptance['diagnostic_task_success'],evaluation=lift,
        original_diagnostic_task_success=stop is None and all(gates.values()) and lift['task_success'],
        acceptance=acceptance,policy_admission_pass=acceptance['policy_admission_pass'],
        step_s=step_s,observer_s=observe_s,json_write_s=write_s,loop_wall_s=loop_s,wall_s=time.monotonic()-began,
        native_warning_count=warnings,trace=B.binding(output/'TRACE.jsonl'),new_training_labels=0)
    B.save(output/'RESULTS.json',result,exclusive=True);return result,arr


def run(root):
    if (Path(root)/'STATIC_FIXTURE.json').exists():raise ValueError('Static fixture cannot execute a physical collection')
    import author_mujoco_control as C
    import author_collection_wave_resources as R
    import author_contact_pilot as OLD
    import author_resident_prepare as P
    root=Path(root).resolve();p=B.read(root/'PROTOCOL.json');B.verify(p['inputs']);B.verify(p['sources'])
    from author_mujoco_lift_policy import POLICY
    if p.get('acceptance_policy')!=POLICY:raise ValueError('An explicitly frozen current lift acceptance policy is required')
    if p['retry_budget']!=0 or len(p['rows'])!=2 or p['max_environment_steps']!=17500:raise ValueError('Finite development protocol changed')
    P.bind_backend(dict(backend=p['inputs']['backend']['path']))
    from author_compiled_kinematics import PlanningSession
    from author_mujoco_persistent_world import World
    from author_mujoco_camera import capture
    from author_mujoco_representation import Stream
    from author_mujoco_return import return_to_start
    from author_mujoco_owner import scope
    from author_live_world_contract import bind_live_plan
    import author_contact_pipeline as M
    began=time.monotonic();world=engine=session=None;rounds=[];error=None;cost=dict(grasp_attempts=0,CGR_forwards=0,planning_jobs=0)
    templates=[B.read(p['inputs']['template:'+str(i)]['path']) for i in range(2)]
    plan=B.read(templates[0]['plan']['path']);scenes=[B.read(r['scene']['path']) for r in p['rows']]
    with np.load(templates[0]['commands']['path']) as z:names=z['joint_names'].tolist();initial=z['initial_position'].copy()
    with scope(root,lock=OLD.LEGACY_LOCK,registry=OLD.REGISTRY):
        try:
            session=PlanningSession();cfg=B.read(p['inputs']['backend']['path'])
            session.bind(Path(cfg[n]) for n in ('robot_urdf','hand_urdf','hand_scene','package'))
            with session.installed():
                world=World(plan,names,initial,scenes,max_steps=p['max_environment_steps'],camera=True,source_options=p['source_hand_solver_options'])
                (root/'model.xml').write_text(world.spec.to_xml())
                args=SimpleNamespace(checkpoint=Path(p['inputs']['checkpoint']['path']),native_deps=Path(p['native_deps']),
                    device='cuda:0',seed=p['seed'],num_seed=2048,voxel_size=.005,depth_min=.2,depth_max=.65)
                engine=Stream(args)
                for i,scene in enumerate(scenes):
                    directory=root/'scenes'/f'{i:02d}';round_begin=time.monotonic()
                    cap=capture(world,scene,directory/'capture',initial)
                    snapshot=(world.data.qpos.copy(),world.data.qvel.copy(),world.steps,float(world.data.time))
                    representation=engine.forward(directory/'capture/observation.json',directory/'representation');cost['CGR_forwards']=engine.engine.calls
                    request=dict(scene=str(directory/'capture/observed_scene.json'),observation=str(directory/'capture/observation.json'),
                        representation_dir=str(directory/'representation'),seed=p['seed'],output_dir=str(directory/'contact'),
                        calibration=p['inputs']['calibration']['path'],cloud_manifest=p['inputs']['cloud_manifest']['path'],backend=p['inputs']['backend']['path'])
                    domain=directory/'DOMAIN.json';B.save(domain,p['domain'],exclusive=True);request['domain']=str(domain)
                    cost['planning_jobs']+=1;prepared=P.prepare(request,session)
                    np.testing.assert_array_equal(world.data.qpos,snapshot[0]);np.testing.assert_array_equal(world.data.qvel,snapshot[1])
                    if (world.steps,float(world.data.time))!=snapshot[2:]:raise ValueError('CGR or planning altered the photographed world')
                    row=dict(index=i,capture=cap,representation_wall_s=representation['timing_seconds']['total_frame_wall'],
                        planning=prepared,execution=None,physical_return=None,world_state_exact_since_photo=True)
                    if prepared['status']=='ready':
                        bind=bind_live_plan(directory/'contact/pipeline',directory/'capture/LIVE_STATE.json',directory/'live');M.audit_mapping(directory/'live')
                        cost['grasp_attempts']+=1;execution,arr=execute(world,directory/'live',directory/'execution');row['execution']=execution
                        returned=return_to_start(world,arr['joint_target'],initial,directory/'return.jsonl');row['physical_return']=returned
                        B.save(directory/'RETURN.json',returned,exclusive=True)
                        if not returned['pass_']:raise RuntimeError('Physical return failed original gates')
                    elif prepared['status']=='infrastructure_failure':raise RuntimeError('Original planning infrastructure failed')
                    row['round_wall_s']=time.monotonic()-round_begin;rounds.append(row)
                    B.save(directory/'ROUND.json',row,exclusive=True)
                    B.save(root/'PROGRESS.json',dict(rounds_finished=len(rounds),world_steps=world.steps,**cost))
                    print(dict(round=i,status=prepared['status'],world_steps=world.steps,round_wall_s=row['round_wall_s'],
                        gates=None if row['execution'] is None else row['execution']['gates']),flush=True)
        except BaseException:error=traceback.format_exc()
        finally:
            if world is not None and world.camera is not None:world.camera.close()
            B.verify(p['sources']);B.verify(p['inputs'])
            result=dict(status='complete' if error is None and len(rounds)==2 else 'failed',error=error,rounds=rounds,cost=cost,
                world=None if world is None else world.receipt(),representation=None if engine is None else engine.receipt(),
                planning=None if session is None else session.receipt(),parent_wall_s=time.monotonic()-began,
                Kit_starts=0,new_training_labels=0,training_updates=0,retries=0,exclude_training=True,
                full_collection_qualified=error is None and len(rounds)==2 and all(r['execution'] and r['execution']['policy_admission_pass'] for r in rounds),
                production_ready=False)
            B.save(root/'RESULTS.json',result,exclusive=True)
    print(dict(status=result['status'],wall_s=result['parent_wall_s'],error=error),flush=True)
    if error:raise RuntimeError(error)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','run'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared-dirs',type=Path,nargs=2)
    p.add_argument('--source-options',action='store_true');a=p.parse_args()
    if a.action=='freeze':freeze(a.root,a.prepared_dirs,source_options=a.source_options)
    else:
        protocol=B.read(a.root/'PROTOCOL.json');marker=B.sha(a.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_MUJOCO_COLLECTION_ENV')!=marker:
            env=B._environment(protocol);env['ANYDEX_MUJOCO_COLLECTION_ENV']=marker
            os.execve(protocol['inputs']['python']['path'],[protocol['inputs']['python']['path'],'-B',str(Path(__file__).resolve()),'run','--root',str(a.root.resolve())],env)
        run(a.root)
