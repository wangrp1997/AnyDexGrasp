"""Fixed functional collection acceptance slots; no retries or training."""
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
import author_mujoco_collection as BASE
import author_mujoco_sample as SAMPLE

QUEUE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/manual_start_20261116_v2/batch/BATCH_PROTOCOL.json')
TEMPLATE=B.ROOT/'outputs/boya_planning_reuse_20261117_v2/original/00/pipeline/READY.json'
INDICES=(3,4,10,12)
FULL_INDICES=(3,4,6,7,10,12)


def freeze(root, *, rigid_joint_limits=False, all_ready_scenes=False, object_dynamics=False):
    import author_collection_scenes as SCENES
    old=B.read(BASE.REFERENCE);B.verify(old['inputs']);B.verify(old['sources'])
    queue=B.read(QUEUE);ready=B.read(TEMPLATE)
    selected=FULL_INDICES if all_ready_scenes else INDICES
    if all_ready_scenes and (not rigid_joint_limits or tuple(i for i,r in enumerate(queue['queue']['scenes']) if r['status']=='ready')!=FULL_INDICES):
        raise ValueError('The six original ready scenes and explicit stiff-limit fix are required')
    if object_dynamics and not all_ready_scenes:raise ValueError('Source inertia/table rolling requires the declared six-scene acceptance')
    B.verify({k:ready[k] for k in ('scene','plan','commands','command_binding')})
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    inputs={k:old['inputs'][k] for k in ('python','checkpoint','backend','calibration','cloud_manifest','nvidia_smi')}
    inputs.update({'template:0':B.binding(TEMPLATE),'source_queue':B.binding(QUEUE),'split_manifest':queue['queue']['split_manifest']})
    rows=[]
    for index,source_index in enumerate(selected):
        row=queue['queue']['scenes'][source_index]
        B.verify({'spec':row['scene_spec']})
        if row['status']!='ready':raise ValueError('Frozen layout was not ready')
        directory=root/'scenes'/f'{index:02d}'
        scene=SCENES.build_scene(row['scene_spec']['path'],directory/'build')
        B.save(directory/'source_scene.json',scene,exclusive=True)
        inputs['scene:'+str(index)]=B.binding(directory/'source_scene.json')
        inputs['spec:'+str(index)]=row['scene_spec']
        rows.append(dict(index=index,source_queue_index=source_index,scene_id=row['scene_id'],split=row['split'],
            scene=inputs['scene:'+str(index)],source_object_ids=row['source_object_ids'],source_object_digests=row['source_object_digests']))
    groups={s:{h for r in rows if r['split']==s for h in r['source_object_digests'].values()} for s in ('fit','internal')}
    if groups['fit']&groups['internal']:raise ValueError('Whole-scene split leakage')
    names=(*BASE.SOURCES,'author_mujoco_acceptance.py','author_mujoco_sample.py','author_collection_scenes.py',
        'audit_author_mujoco_collection.py','author_training.py','author_collect.py','author_lift.py','author_protocol.py',
        'author_mujoco_joint_limits.py','author_mujoco_object_dynamics.py')
    sources={n:B.binding(B.HERE/n) for n in names};archive=root/'source_archive';archive.mkdir()
    for n,b in sources.items():shutil.copyfile(b['path'],archive/n)
    budget=80000 if all_ready_scenes else 48000
    schema='anydex_boya_mujoco_collection_acceptance_v3' if object_dynamics else ('anydex_boya_mujoco_collection_acceptance_v2' if all_ready_scenes else 'anydex_boya_mujoco_collection_acceptance_v1')
    p=dict(schema=schema,rows=rows,inputs=inputs,sources=sources,
        env=dict(old['env'],MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
        inherited_env=old['inherited_env'],native_deps=old['native_deps'],gpu_uuid=old['gpu_uuid'],
        seed=2026111601,domain=copy.deepcopy(old['domain']),max_world_builds=1,max_explicit_resets=1,
        max_camera_builds=1,max_environment_steps=budget,max_native_integration_steps=budget*8,
        source_hand_solver_options=True,rigid_joint_limits=bool(rigid_joint_limits),object_dynamics=bool(object_dynamics),max_grasp_attempts=len(selected),max_CGR_forwards=len(selected),max_planning_jobs=len(selected),
        max_execution_steps_per_job=8000,min_settling_steps=750,max_settling_steps=2500,
        retry_budget=0,training_updates=0,sample_policy=copy.deepcopy(SAMPLE.POLICY),
        selection='All six ready slots from the existing frozen queue; no outcome replacement' if all_ready_scenes else 'First two ready fit and first two ready internal slots from the existing fixed queue; no outcome replacement',
        acceptance_only=True,formal_production_quota_frozen=False,old_datasets_merged=False)
    B.save(root/'PROTOCOL.json',p,exclusive=True);return p


def run(root):
    if (Path(root)/'STATIC_FIXTURE.json').exists():raise ValueError('Static fixture cannot run physical acceptance')
    import author_contact_pilot as OLD
    import author_resident_prepare as P
    from author_mujoco_owner import scope
    root=Path(root).resolve();p=B.read(root/'PROTOCOL.json');B.verify(p['inputs']);B.verify(p['sources'])
    allowed={'anydex_boya_mujoco_collection_acceptance_v1':(48000,INDICES),'anydex_boya_mujoco_collection_acceptance_v2':(80000,FULL_INDICES),'anydex_boya_mujoco_collection_acceptance_v3':(80000,FULL_INDICES)}
    if p['schema'] not in allowed or p['sample_policy']!=SAMPLE.POLICY:
        raise ValueError('Explicit functional acceptance protocol required')
    budget,indices=allowed[p['schema']]
    if p['retry_budget']!=0 or p['max_environment_steps']!=budget or tuple(r['source_queue_index'] for r in p['rows'])!=indices:
        raise ValueError('Finite acceptance budget or scene selection changed')
    if p['schema'].endswith(('_v2','_v3')) and p.get('rigid_joint_limits') is not True:raise ValueError('Six-scene acceptance requires the explicit stiff-limit fix')
    if p['schema'].endswith('_v3') and p.get('object_dynamics') is not True:raise ValueError('v3 requires source inertia and explicit table rolling')
    P.bind_backend(dict(backend=p['inputs']['backend']['path']))
    from author_compiled_kinematics import PlanningSession
    from author_mujoco_persistent_world import World
    from author_mujoco_camera import capture, InitializationUnstable
    from author_mujoco_representation import Stream
    from author_mujoco_return import return_to_start
    from author_live_world_contract import bind_live_plan
    import author_contact_pipeline as M
    began=time.monotonic();world=engine=session=None;rounds=[];error=None
    cost=dict(grasp_attempts=0,CGR_forwards=0,planning_jobs=0)
    template=B.read(p['inputs']['template:0']['path']);plan=B.read(template['plan']['path'])
    scenes=[B.read(r['scene']['path']) for r in p['rows']]
    with np.load(template['commands']['path'],allow_pickle=False) as z:names=z['joint_names'].tolist();initial=z['initial_position'].copy()
    with scope(root,lock=OLD.LEGACY_LOCK,registry=OLD.REGISTRY):
        try:
            session=PlanningSession();cfg=B.read(p['inputs']['backend']['path'])
            session.bind(Path(cfg[n]) for n in ('robot_urdf','hand_urdf','hand_scene','package'))
            with session.installed():
                world=World(plan,names,initial,scenes,max_steps=p['max_environment_steps'],camera=True,source_options=True,
                    rigid_joint_limits=p.get('rigid_joint_limits',False),object_dynamics=p.get('object_dynamics',False))
                (root/'model.xml').write_text(world.spec.to_xml())
                args=SimpleNamespace(checkpoint=Path(p['inputs']['checkpoint']['path']),native_deps=Path(p['native_deps']),
                    device='cuda:0',seed=p['seed'],num_seed=2048,voxel_size=.005,depth_min=.2,depth_max=.65)
                engine=Stream(args)
                for spec,scene in zip(p['rows'],scenes):
                    i=spec['index'];directory=root/'scenes'/f'{i:02d}';round_begin=time.monotonic()
                    row=dict(index=i,split=spec['split'],capture=None,execution=None,physical_return=None)
                    if world.steps+p['max_settling_steps']>world.max_steps:
                        row['record']=SAMPLE.write_record(directory,spec,terminal='budget_not_executed')
                    else:
                        try:
                            row['capture']=capture(world,scene,directory/'capture',initial,
                                min_steps=p['min_settling_steps'],max_steps=p['max_settling_steps'])
                        except InitializationUnstable as exc:
                            row['record']=SAMPLE.write_record(directory,spec,terminal='initialization_unstable',error=str(exc))
                        else:
                            snapshot=(world.data.qpos.copy(),world.data.qvel.copy(),world.steps,float(world.data.time))
                            representation=engine.forward(directory/'capture/observation.json',directory/'representation')
                            cost['CGR_forwards']=engine.engine.calls
                            row['representation_wall_s']=representation['timing_seconds']['total_frame_wall']
                            request=dict(scene=str(directory/'capture/observed_scene.json'),observation=str(directory/'capture/observation.json'),
                                representation_dir=str(directory/'representation'),seed=p['seed'],output_dir=str(directory/'contact'),
                                calibration=p['inputs']['calibration']['path'],cloud_manifest=p['inputs']['cloud_manifest']['path'],backend=p['inputs']['backend']['path'])
                            domain=directory/'DOMAIN.json';B.save(domain,p['domain'],exclusive=True);request['domain']=str(domain)
                            cost['planning_jobs']+=1;prepared=P.prepare(request,session);row['planning']=prepared
                            np.testing.assert_array_equal(world.data.qpos,snapshot[0]);np.testing.assert_array_equal(world.data.qvel,snapshot[1])
                            if (world.steps,float(world.data.time))!=snapshot[2:]:raise ValueError('Planning/CGR altered photographed world')
                            row['world_state_exact_since_photo']=True
                            if prepared['status']=='ready':
                                bind_live_plan(directory/'contact/pipeline',directory/'capture/LIVE_STATE.json',directory/'live');M.audit_mapping(directory/'live')
                                ready=B.read(directory/'live/READY.json')
                                with np.load(ready['commands']['path']) as z:steps=len(z['phase'])
                                # Reserve the entire grasp AND its physical return before dispatch.
                                if steps>8000 or world.steps+2*steps+100>world.max_steps:
                                    row['record']=SAMPLE.write_record(directory,spec,terminal='budget_not_executed')
                                else:
                                    cost['grasp_attempts']+=1
                                    execution,arr=BASE.execute(world,directory/'live',directory/'execution');row['execution']=execution
                                    # Durable outcome precedes robot return; a later return error cannot erase it.
                                    row['record']=SAMPLE.write_record(directory,spec,execution=execution)
                                    if execution['status']!='complete':raise RuntimeError('Execution stopped before complete lift')
                                    returned=return_to_start(world,arr['joint_target'],initial,directory/'return.jsonl')
                                    row['physical_return']=returned;B.save(directory/'RETURN.json',returned,exclusive=True)
                                    if not returned['pass_']:raise RuntimeError('Physical return failed original gates')
                            elif prepared['status']=='infrastructure_failure':raise RuntimeError('Original preparation infrastructure failed')
                            else:row['record']=SAMPLE.write_record(directory,spec,terminal='planning_'+prepared['status'])
                    row['round_wall_s']=time.monotonic()-round_begin;rounds.append(row)
                    B.save(directory/'ROUND.json',row,exclusive=True)
                    B.save(root/'PROGRESS.json',dict(rounds_finished=len(rounds),world_steps=world.steps,**cost))
                    print(dict(round=i,status=row['record']['status'],label=row['record']['label'],world_steps=world.steps,wall_s=row['round_wall_s']),flush=True)
        except BaseException:error=traceback.format_exc()
        finally:
            if world is not None and world.camera is not None:world.camera.close()
            # Every fixed slot gets a terminal record, including slots never dispatched.
            for spec in p['rows']:
                directory=root/'scenes'/f"{spec['index']:02d}"
                if not (directory/'RECORD.json').exists():
                    SAMPLE.write_record(directory,spec,terminal='controller_error' if spec['index']==len(rounds) else 'not_executed_after_controller_error',error=error)
            B.verify(p['sources']);B.verify(p['inputs'])
            records=[B.read(root/'scenes'/f"{r['index']:02d}"/'RECORD.json') for r in p['rows']]
            result=dict(status='complete' if error is None and len(rounds)==len(p['rows']) else 'failed',error=error,rounds=rounds,cost=cost,
                records=records,world=None if world is None else world.receipt(),representation=None if engine is None else engine.receipt(),
                planning=None if session is None else session.receipt(),parent_wall_s=time.monotonic()-began,
                Kit_starts=0,eligible_acceptance_labels=sum(r['eligible'] for r in records),training_updates=0,retries=0,
                production_ready=False,old_datasets_merged=False)
            B.save(root/'RESULTS.json',result,exclusive=True)
    print(dict(status=result['status'],wall_s=result['parent_wall_s'],error=error),flush=True)
    return result


def dry_consumer(dataset, output):
    import torch
    from author_training import validate_arrays,CanonicalDataset,components
    with np.load(dataset,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    fit,check=validate_arrays(data);torch.manual_seed(20261117)
    cls,_,loss_cls,_=components();cfg=dict(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5)
    model=cls(**cfg).eval();criterion=loss_cls(**cfg,train_type=1).eval()
    before={k:v.clone() for k,v in model.state_dict().items()};results=[]
    for indices in (fit,check):
        loader=torch.utils.data.DataLoader(CanonicalDataset(data,indices,1),batch_size=128,shuffle=False)
        with torch.no_grad():
            for batch in loader:
                prediction,end_points=model(batch);loss,_,_=criterion(end_points)
                if prediction.shape!=(len(batch['result']),1,20) or not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                    raise ValueError('Published consumer returned malformed/nonfinite predictions or loss')
                results.append(dict(rows=len(batch['result']),loss=float(loss),output_shape=list(prediction.shape)))
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Dry consumer changed weights')
    result=dict(status='published_head_consumer_pass',fit=len(fit),internal=len(check),
        dataset=B.binding(dataset),weights_unchanged=True,optimizer_updates=0,backward_calls=0,training_updates=0,
        forward_checks=results,initial_head='Fresh random weights, not a trained/scored collection policy')
    B.save(output,result,exclusive=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','run','export','consumer'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stiff-joint-limits',action='store_true')
    parser.add_argument('--all-ready-scenes',action='store_true');parser.add_argument('--source-object-dynamics',action='store_true');a=parser.parse_args()
    if (a.stiff_joint_limits or a.all_ready_scenes or a.source_object_dynamics) and a.action!='freeze':parser.error('Options must be frozen before running')
    if a.action=='freeze':print(freeze(a.root,rigid_joint_limits=a.stiff_joint_limits,all_ready_scenes=a.all_ready_scenes,object_dynamics=a.source_object_dynamics)['schema'])
    elif a.action=='export':print(SAMPLE.export(a.root,a.root/'export'))
    elif a.action=='consumer':print(dry_consumer(a.root/'export/gesture_1.npz',a.root/'CONSUMER_CHECK.json'))
    else:
        protocol=B.read(a.root/'PROTOCOL.json');marker=B.sha(a.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_ACCEPTANCE_ENV')!=marker:
            env=B._environment(protocol);env['ANYDEX_ACCEPTANCE_ENV']=marker
            os.execve(protocol['inputs']['python']['path'],[protocol['inputs']['python']['path'],'-B',str(Path(__file__).resolve()),'run','--root',str(a.root.resolve())],env)
        result=run(a.root)
        if result['status']!='complete':sys.exit(1)
