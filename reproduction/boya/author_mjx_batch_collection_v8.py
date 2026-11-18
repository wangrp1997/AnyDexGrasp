"""Eight fixed diagnostic slots through fresh RGBD/CGR/plans/GPU grasps/return.

Synchronous GPU batches keep observation binding without stepping a photographed
world during planning. All idle work is counted. No scene/candidate retry or
training. Comparison modes use the same eight slots; unused lanes remain idle.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import author_collection_batch as B

REFERENCE=B.ROOT/'outputs/boya_mujoco_acceptance_20261117_v3'
DEPS=B.ROOT/'outputs/boya_mjx_dependencies_20261118_v2'
WARP=B.ROOT/'outputs/boya_mjx_warp_compat_20261118_v1'
GPU='GPU-4d3badb8-5fb0-b880-f08e-6407f0093e86'
SEED=2026111801


def freeze(root,*,environments=4,source_protocol=None,remaining_only=False,prior_native_steps=0,additional_inputs=None):
    import author_collection_scenes as S
    import author_mujoco_collection as BASE
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    old=B.read(REFERENCE/'PROTOCOL.json');B.verify(old['inputs']);B.verify(old['sources'])
    rows=[];inputs=copy.deepcopy(old['inputs']);inputs['reference_protocol']=B.binding(REFERENCE/'PROTOCOL.json')
    if source_protocol:
        prior=B.read(source_protocol)
        if len(prior['rows'])!=8:raise ValueError('Same eight fixed slots required')
        inputs['comparison_source_protocol']=B.binding(source_protocol)
        selected=prior['rows'][4:] if remaining_only else prior['rows']
        if remaining_only:
            for spec in selected:
                if (Path(source_protocol).parent/'scenes'/f"{spec['index']:02d}"/'execution/TRACE.jsonl').exists():raise ValueError('Previously executed slots cannot be retried as remaining work')
        for spec in selected:
            row=copy.deepcopy(spec);row['source_slot_index']=spec['index'];row['index']=len(rows)
            if row.get('scene'):
                B.verify({'scene':row['scene']});path=root/'scenes'/f"{row['index']:02d}"/'source_scene.json'
                B.save(path,B.read(row['scene']['path']),exclusive=True);row['scene']=B.binding(path)
            rows.append(row)
    else:
        # Six existing ready layouts, all included independent of old outcomes.
        for src in (0,1,4,5,2,3):
            spec=copy.deepcopy(old['rows'][src]);index=len(rows);scene=B.read(spec['scene']['path'])
            path=root/'scenes'/f'{index:02d}'/'source_scene.json';B.save(path,scene,exclusive=True)
            spec.update(index=index,scene=B.binding(path),origin='all_six_existing_ready_layouts',source_acceptance_slot=src,status='ready')
            rows.append(spec)
        queue=B.read('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/manual_start_20261116_v2/batch/BATCH_PROTOCOL.json')['queue']
        catalog=B.read(queue['catalog']['path']);inputs['scene_catalog']=queue['catalog']
        for split in ('fit','internal'):
            spec=S.sample_scene(catalog,split,0,SEED);spec['catalog']=queue['catalog']
            index=len(rows);path=root/'specs'/(spec['scene_id']+'.json');B.save(path,spec,exclusive=True)
            row=dict(index=index,scene_id=spec['scene_id'],split=split,status=spec['status'],scene=None,
                scene_spec=B.binding(path),source_object_ids=spec['source_object_ids'],source_object_digests=spec['source_object_digests'],origin='one_fixed_fresh_draw_no_replacement')
            if spec['status']=='ready':
                scene=S.build_scene(path,root/'scenes'/f'{index:02d}'/'build');dest=root/'scenes'/f'{index:02d}'/'source_scene.json'
                B.save(dest,scene,exclusive=True);row['scene']=B.binding(dest)
            rows.append(row)
    for row in rows:
        if row.get('scene'):inputs['scene:'+str(row['index'])]=row['scene']
    names=set(BASE.SOURCES)|{Path(__file__).name,'author_mjx_live_batch_v4.py','author_mjx_compatibility.py','author_mjx_fast_camera.py','author_mjx_graph_staging.py',
        'author_mujoco_joint_limits.py','author_mujoco_object_dynamics.py','author_mujoco_sample.py',
        'author_collection_scenes.py','author_live_world_contract.py','author_scaling_control.py','author_mjx_measured_planning.py'}
    sources={n:B.binding(B.HERE/n)for n in sorted(names)}
    archive=root/'source_archive';archive.mkdir()
    for n,b in sources.items():shutil.copyfile(b['path'],archive/n)
    waves=(len(rows)+environments-1)//environments;max_ticks=waves*(2500+8000+8100)
    prior_spent=0
    if source_protocol and (Path(source_protocol).parent/'RESULTS.json').exists():
        previous=B.read(Path(source_protocol).parent/'RESULTS.json')
        if previous['status']=='failed' and (remaining_only or previous['cost']['grasp_attempts']==0):
            prior_spent=previous['world']['native_environment_integrations']+prior.get('prior_native_environment_steps',0)
            inputs['infrastructure_prior_result']=B.binding(Path(source_protocol).parent/'RESULTS.json')
    stage_cap=1190400
    if remaining_only:max_ticks=min(max_ticks,(stage_cap-prior_spent)//(environments*8))
    else:prior_spent=0
    if prior_native_steps:
        if remaining_only or environments!=1 or prior_native_steps<0:raise ValueError('Prior comparison budget is only for a fresh single mode')
        prior_spent=int(prior_native_steps);max_ticks=min(max_ticks,(stage_cap-prior_spent)//8)
        if max_ticks<=0:raise ValueError('Original single-mode budget exhausted')
    if additional_inputs:inputs.update(additional_inputs)
    p=dict(GPU_graph_mode='WARP_STAGED_EX',prior_native_steps_is_upper_bound=bool(prior_native_steps),prior_native_environment_steps=prior_spent,schema='anydex_boya_mjx_full_collection_pilot_v1',rows=rows,inputs=inputs,sources=sources,
        env=dict(old['env'],ANYDEX_CONTACT_SPOOL_ROOT=str(root/'contact_spool')),inherited_env=old['inherited_env'],
        native_deps=old['native_deps'],gpu_uuid=GPU,seed=SEED,domain=copy.deepcopy(old['domain']),
        environments=environments,max_global_control_ticks=max_ticks,max_native_environment_steps=max_ticks*8*environments,
        max_CGR_forwards=len(rows),max_planning_jobs=len(rows),max_grasp_attempts=len(rows),remaining_only=remaining_only,stage_native_budget=(stage_cap if (remaining_only or prior_native_steps) else max_ticks*8*environments),pilot_native_budget_reference=stage_cap,min_settle=750,max_settle=2500,
        max_execution_controls=8000,retries=0,source_dt_s=.0005,control_dt_s=.004,
        GPU_noslip_iterations=0,GPU_ccd_tolerance_m=1e-6,robot_initializations_per_environment=1,
        planning_processes=3,stage_timeout_s=1800,exclude_training=True,formal_production_quota_frozen=False,
        sample_policy='Original functional 0/1/null classification; independent MJX cohort',
        unused_environment_work_counted=True,comparison_underfilled=environments>8,training_updates=0)
    B.save(root/'PROTOCOL.json',p,exclusive=True);return p


def write_record(root,spec,*,execution=None,terminal=None,error=None):
    import author_mujoco_sample as S
    directory=root/'scenes'/f"{spec['index']:02d}"
    decision=S.classify(execution) if execution is not None else dict(status=terminal,label=None,eligible=False)
    bindings={}
    if execution is not None:
        ready=B.read(directory/'live/READY.json');bindings={k:ready[k]for k in ('scene','candidate','commands','command_binding','plan')}
        bindings.update(execution=B.binding(directory/'execution/RESULTS.json'),trace=execution['trace'],
            observation=B.binding(directory/'capture/observation.json'),representation=B.binding(directory/'representation/RESULTS.json'))
    record=dict(schema='anydex_boya_mjx_functional_record_v1',scene_id=spec['scene_id'],split=spec['split'],gesture_id=1,
        **decision,error=error,bindings=bindings,backend='MJX-Warp',domain=B.binding(root/'PROTOCOL.json'),
        exclude_training=True,formal_training_eligible=False,training_updates=0)
    B.save(directory/'RECORD.json',record,exclusive=True);return record


def planning_batch(root,requests,p,pool):
    from author_collection_wave_resources import process_identity
    groups=[requests[i::3]for i in range(3)];groups=[g for g in groups if g]
    def job(pair):
        number,items=pair;directory=root/f'worker_{number}';directory.mkdir(parents=True,exist_ok=False)
        request=directory/'REQUESTS.json';B.save(request,[r for _,r in items],exclusive=True)
        command=[sys.executable,'-B',str(B.HERE/'author_mjx_measured_planning.py'),'--requests',str(request),'--output',str(directory/'RESULTS.json')]
        B.save(directory/'DISPATCH.json',dict(command=command,requests=B.binding(request)),exclusive=True)
        with pool.stage(('prepare',),directory):
            with (directory/'process.log').open('x')as log:
                process=pool.popen(command,cwd=B.ROOT,env=B._environment(p),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity=process_identity(process.pid);B.save(directory/'PROCESS.json',dict(identity=identity),exclusive=True)
                try:code=process.wait(timeout=p['stage_timeout_s'])
                except BaseException:
                    if process_identity(process.pid)==identity:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                    raise
                if code:raise RuntimeError('CPU planning process failed: '+str(directory))
        result=B.read(directory/'RESULTS.json')
        return [(index,row)for (index,_),row in zip(items,result['jobs'])]
    result={}
    with ThreadPoolExecutor(max_workers=3)as executor:
        for values in executor.map(job,enumerate(groups)):result.update(values)
    return result


def settle(batch,assigned,root,p,initial):
    from settle_s40_scenes import window_metrics
    windows={lane:dict(poses=deque(maxlen=250),velocity=deque(maxlen=250),support=deque(maxlen=250),checks=[])for lane in assigned}
    logs={};last={};start=batch.ticks;began=time.monotonic()
    try:
        for lane,(spec,scene)in assigned.items():
            batch.exchange(lane,scene);directory=root/'scenes'/f"{spec['index']:02d}"/'capture';directory.mkdir(parents=True,exist_ok=False)
            logs[lane]=(directory/'initialization.jsonl').open('x')
        targets=np.broadcast_to(initial,batch.targets.shape).copy();speed=np.zeros_like(targets)
        for index in range(1,p['max_settle']+1):
            batch.step(targets,speed)
            for lane in assigned:
                row=batch.observe(lane,'capture_settle',index);last[lane]=row;window=windows[lane];view=batch.views[lane]
                window['poses'].append([row['objects'][k]['pose_xyzw']for k in view.active]);window['velocity'].append([row['objects'][k]['velocity']for k in view.active])
                window['support'].append({k:any('/World/'+k in c['bodies']and '/World/ground'in c['bodies']and c['normal_force_n']>.001 for c in row['contacts'])for k in view.active})
                logs[lane].write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            if index>=p['min_settle']and index%250==0:
                for lane in assigned:
                    w=windows[lane];view=batch.views[lane]
                    stability=[window_metrics(np.asarray(w['poses'])[:,j],np.asarray(w['velocity'])[:,j])for j in range(len(view.active))]
                    support={k:sum(s[k]for s in w['support'])for k in view.active}
                    w['checks'].append(dict(steps=index,stability=stability,supported_ticks=support,pass_=all(s['stable']for s in stability)and all(support.values())))
                print({'phase':'settle','global_tick':batch.ticks,'settle_ticks':index,'stable':[windows[i]['checks'][-1]['pass_']for i in assigned]},flush=True)
                if all(w['checks'][-1]['pass_']for w in windows.values()):break
    finally:
        for stream in logs.values():stream.close()
    result={}
    for lane,(spec,scene)in assigned.items():
        value=windows[lane]['checks'][-1];directory=root/'scenes'/f"{spec['index']:02d}"/'capture'
        result[lane]=value
        B.save(directory/'SETTLING.json',dict(steps=index,min_steps=p['min_settle'],max_steps=p['max_settle'],
            checks=windows[lane]['checks'],start_world_step=start,end_world_step=batch.ticks,
            stability=value['stability'],supported_ticks=value['supported_ticks'],wall_s=time.monotonic()-began,
            batch_barrier=True),exclusive=True)
    return result,last,index


def evaluate_execution(view,rows,arr,stop,output,elapsed):
    import author_mujoco_control as C
    maxima,first=C.numeric_maxima(rows,view.planning,list(view.planning.joints),arr['desired_palm'],view.reference,view.legal_hand,view.active)
    gates=C.source_gates(maxima,len(rows),len(arr['phase']));complete=len(rows)==len(arr['phase'])and stop is None
    # Partial execution is an invalid/null record, never an evaluated negative.
    lift=C.LIFT.evaluate_lift(rows,view.active,.004)if complete else dict(task_success=False,incomplete=True)
    result=dict(status='complete'if complete else'physical_stop',steps=len(rows),expected_steps=len(arr['phase']),stop=stop,
        gates=gates,maxima=maxima,first_failure=first,evaluation=lift,native_warning_count=0,
        native_warning_contract='All-step finite state, overflow/capacity and exact float32 tick checks; CPU warning array not used',
        trace=B.binding(output/'TRACE.jsonl'),wall_s=elapsed,backend='MJX-Warp',all_GPU_contacts_retained=True,
        CPU_contact_solver_used=False,new_training_labels=0)
    B.save(output/'RESULTS.json',result,exclusive=True);return result


def motion(batch,prepared,root,p,initial):
    import author_mujoco_control as C
    import mujoco
    states={};began=time.monotonic();result={}
    for lane,(spec,arr)in prepared.items():
        directory=root/'scenes'/f"{spec['index']:02d}";out=directory/'execution';out.mkdir()
        batch.views[lane].phase='active'
        states[lane]=dict(spec=spec,arr=arr,velocity=C.source_velocity_targets(arr['joint_target'],batch.base.names,batch.base.arm_names),
            phase='execute',index=0,rows=[],stream=(out/'TRACE.jsonl').open('x'),out=out,settled={},stop=None)
    iteration=0
    try:
        while any(s['phase']!='done'for s in states.values()):
            targets=np.broadcast_to(initial,batch.targets.shape).copy();speeds=np.zeros_like(targets)
            for lane,s in states.items():
                if s['phase']=='done':continue
                a=s['arr']['joint_target']if s['phase']=='execute'else s['return_targets']
                v=s['velocity']if s['phase']=='execute'else s['return_velocity']
                targets[lane]=a[s['index']];speeds[lane]=v[s['index']]
            batch.step(targets,speeds);iteration+=1
            for lane,s in states.items():
                if s['phase']=='done':continue
                s['index']+=1;phase=str(s['arr']['phase'][s['index']-1])if s['phase']=='execute'else'physical_return'
                row=batch.observe(lane,phase,s['index']);row.update(joint_target=targets[lane].tolist(),velocity_target=speeds[lane].tolist(),
                    applied_position_target_float32=targets[lane].astype(np.float32).tolist(),applied_velocity_target_float32=speeds[lane].astype(np.float32).tolist())
                s['rows'].append(row);s['stream'].write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                if phase=='settle':s['settled']={k:np.asarray(v['pose_xyzw'][:2])for k,v in row['objects'].items()}
                if max([0.]+[-c['separation_m']for c in row['contacts']])>.005:s['stop']='original_5mm_deep_collision_stop'
                if s['phase']=='execute'and C.T.preclosure_pushes(row['objects'],s['settled'],phase):s['stop']='original_preclosure_push_stop'
                length=len(s['arr']['phase'])if s['phase']=='execute'else len(s['return_targets'])
                if s['index']==length or s['stop']:
                    s['stream'].close();spec=s['spec'];directory=root/'scenes'/f"{spec['index']:02d}";view=batch.views[lane]
                    if s['phase']=='execute':
                        execution=evaluate_execution(view,s['rows'],s['arr'],s['stop'],s['out'],time.monotonic()-began)
                        record=write_record(root,spec,execution=execution);result[lane]=dict(execution=execution,record=record)
                        print({'phase':'grasp_finished','scene':spec['index'],'status':record['status'],'label':record['label'],'tick':batch.ticks},flush=True)
                        view.phase='boundary';batch.exchange(lane,dict(scene_id='external_clear_between_rounds',objects=[]))
                        previous=s['arr']['joint_target'][:s['index']]
                        s['return_targets']=np.r_[previous[::-1],np.linspace(previous[0],initial,101)[1:]]
                        s['return_velocity']=C.source_velocity_targets(s['return_targets'],view.names,view.arm_names)
                        s['return_desired']=[view.planning.fk.transforms(view.planning.command_state(dict(zip(view.names,t))),view.reference.base_transform)['palm']for t in s['return_targets']]
                        s.update(phase='return',index=0,rows=[],stop=None,stream=(directory/'return.jsonl').open('x'))
                    else:
                        maxima,first=C.numeric_maxima(s['rows'],view.planning,list(view.planning.joints),s['return_desired'],view.reference,set(),list(view.bodyids))
                        gates=C.source_gates(maxima,len(s['rows']),len(s['return_targets']))
                        returned=dict(steps=len(s['rows']),expected_steps=len(s['return_targets']),stop=s['stop'],gates=gates,maxima=maxima,
                            first_failure=first,pass_=s['stop']is None and all(gates.values()),trace=B.binding(directory/'return.jsonl'),
                            robot_state_writes=0,scene_reset_calls=0,clock_reset=False,
                            final_motor_error_rad=float(np.max(np.abs(view.data.qpos[C.R.joint_addresses(view.model,view.names)]-initial))))
                        B.save(directory/'RETURN.json',returned,exclusive=True);result[lane]['physical_return']=returned;s['phase']='done'
                        print({'phase':'return_finished','scene':spec['index'],'pass':returned['pass_'],'tick':batch.ticks},flush=True)
            if iteration%250==0:B.save(root/'PROGRESS.json',dict(phase='motion',global_tick=batch.ticks,states={str(k):dict(phase=s['phase'],index=s['index'])for k,s in states.items()}))
    finally:
        for s in states.values():
            if not s['stream'].closed:s['stream'].close()
    return result


def run(root):
    import author_collection_wave_resources as R
    import author_resident_prepare as P
    root=Path(root).resolve();p=B.read(root/'PROTOCOL.json');B.verify(p['inputs']);B.verify(p['sources'])
    if (root/'RUN_STARTED.json').exists():raise ValueError('No relaunch/retry of a dispatched collection root')
    P.bind_backend(dict(backend=p['inputs']['backend']['path']))
    from author_compiled_kinematics import CompiledKinematics
    from author_mjx_live_batch_v4 import Batch,photograph
    from author_mujoco_representation import Stream
    from author_live_world_contract import bind_live_plan
    import author_contact_pipeline as M
    template=B.read(p['inputs']['template:0']['path']);plan=B.read(template['plan']['path'])
    scenes=[B.read(r['scene']['path'])for r in p['rows']if r.get('scene')]
    with np.load(template['commands']['path'])as z:names=z['joint_names'].tolist();initial=z['initial_position'].copy()
    began=time.monotonic();batch=engine=None;waves=[];error=None;cost=dict(CGR_forwards=0,planning_jobs=0,grasp_attempts=0)
    lock=B.ROOT/'outputs/author_collection_gpu0.lock';registry=lock.with_suffix('.wave_active.json')
    with R.lease(root/'DRIVER.lock'),R.lease(lock)as lease,R.active_run(root,registry,B.binding(root/'PROTOCOL.json')):
        B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid()),started_unix_s=time.time()),exclusive=True)
        pool=R.ResourcePool(root/'resource_slots',lease,B.binding(root/'PROTOCOL.json'))
        try:
            with CompiledKinematics().installed():
                batch=Batch(plan,names,initial,scenes,count=p['environments'],max_ticks=p['max_global_control_ticks'])
                batch.journal=(root/'GPU_JOURNAL.jsonl').open('x')
                import mujoco
                (root/'MODEL.xml').write_text(batch.base.spec.to_xml())
                B.save(root/'MODEL_RECEIPT.json',dict(model=B.binding(root/'MODEL.xml'),model_format='MJCF source spec; GPU numerical changes explicit below',GPU_CCD=batch.compatibility,GPU_graph_mode='WARP_STAGED_EX',
                    source_template=p['inputs']['template:0'],camera_count=1),exclusive=True)
                args=SimpleNamespace(checkpoint=Path(p['inputs']['checkpoint']['path']),native_deps=Path(p['native_deps']),device='cuda:0',
                    seed=p['seed'],num_seed=2048,voxel_size=.005,depth_min=.2,depth_max=.65)
                engine=Stream(args)
                for wave_start in range(0,len(p['rows']),p['environments']):
                    mark=time.monotonic();assigned={}
                    for lane,spec in enumerate(p['rows'][wave_start:wave_start+p['environments']]):
                        if not spec.get('scene'):write_record(root,spec,terminal='placement_null');continue
                        assigned[lane]=(spec,B.read(spec['scene']['path']))
                    if not assigned:continue
                    B.save(root/'PROGRESS.json',dict(phase='settle',wave_start=wave_start,global_tick=batch.ticks))
                    stable,last,ticks=settle(batch,assigned,root,p,initial);snapshot=batch.snapshot();requests=[];ready={};rounds={}
                    for lane,(spec,scene)in assigned.items():
                        directory=root/'scenes'/f"{spec['index']:02d}";row=dict(index=spec['index'],lane=lane,capture=None,execution=None)
                        rounds[lane]=row
                        if not stable[lane]['pass_']:row['record']=write_record(root,spec,terminal='initialization_unstable');continue
                        row['capture']=photograph(batch,lane,scene,directory/'capture',last[lane],stable[lane]['stability'],ticks)
                        rep=engine.forward(directory/'capture/observation.json',directory/'representation');cost['CGR_forwards']+=1
                        row['CGR_wall_s']=rep['timing_seconds']['total_frame_wall']
                        domain=directory/'DOMAIN.json';B.save(domain,p['domain'],exclusive=True)
                        request=dict(scene=str(directory/'capture/observed_scene.json'),observation=str(directory/'capture/observation.json'),
                            representation_dir=str(directory/'representation'),seed=p['seed'],output_dir=str(directory/'contact'),domain=str(domain),
                            calibration=p['inputs']['calibration']['path'],cloud_manifest=p['inputs']['cloud_manifest']['path'],backend=p['inputs']['backend']['path'])
                        requests.append((spec['index'],request));cost['planning_jobs']+=1
                    batch.assert_snapshot(snapshot)
                    B.save(root/'PROGRESS.json',dict(phase='planning',wave_start=wave_start,jobs=len(requests),global_tick=batch.ticks))
                    plans=planning_batch(root/'planning'/f'{wave_start:02d}',requests,p,pool)if requests else{}
                    batch.assert_snapshot(snapshot)
                    for lane,(spec,scene)in assigned.items():
                        index=spec['index'];directory=root/'scenes'/f'{index:02d}'
                        if index not in plans:continue
                        prepared=plans[index];rounds[lane]['planning']=prepared;rounds[lane]['state_exact_since_photograph']=True
                        if prepared['status']=='ready':
                            bind_live_plan(directory/'contact/pipeline',directory/'capture/LIVE_STATE.json',directory/'live');M.audit_mapping(directory/'live')
                            r=B.read(directory/'live/READY.json')
                            with np.load(r['commands']['path'],allow_pickle=False)as z:arr={k:z[k].copy()for k in z.files}
                            if len(arr['phase'])>8000:raise ValueError('Original complete command budget exceeded')
                            ready[lane]=(spec,arr);cost['grasp_attempts']+=1
                        elif prepared['status']=='infrastructure_failure':raise RuntimeError('Original planning infrastructure failed')
                        else:rounds[lane]['record']=write_record(root,spec,terminal='planning_'+prepared['status'])
                    outcome=motion(batch,ready,root,p,initial)if ready else{}
                    for lane,(spec,_)in assigned.items():
                        if lane in outcome:rounds[lane].update(outcome[lane])
                        B.save(root/'scenes'/f"{spec['index']:02d}"/'ROUND.json',rounds[lane],exclusive=True)
                        batch.views[lane].phase='boundary'
                        if batch.views[lane].active:batch.exchange(lane,dict(scene_id='external_clear_between_rounds',objects=[]))
                    waves.append(dict(first_slot=wave_start,rounds=list(rounds.values()),wall_s=time.monotonic()-mark,world_tick=batch.ticks))
                    B.save(root/'WAVES.json',waves)
                    if any(not x['physical_return']['pass_']for x in outcome.values()):raise RuntimeError('Physical return failed; future wave remains null')
        except BaseException:error=traceback.format_exc()
        finally:
            if batch is not None:batch.close()
            for spec in p['rows']:
                if not (root/'scenes'/f"{spec['index']:02d}"/'RECORD.json').exists():write_record(root,spec,terminal='controller_error_or_not_executed',error=error)
            B.verify(p['inputs']);B.verify(p['sources'])
            records=[B.read(root/'scenes'/f"{r['index']:02d}"/'RECORD.json')for r in p['rows']]
            result=dict(status='complete'if error is None else'failed',error=error,waves=waves,records=records,cost=cost,
                world=None if batch is None else batch.receipt(),representation=None if engine is None else engine.receipt(),
                parent_wall_s=time.monotonic()-began,labels=sum(r['eligible']for r in records),training_updates=0,
                production_ready=False,formal_quota_frozen=False,full_collection_speedup_proven=False)
            B.save(root/'RESULTS.json',result,exclusive=True)
            B.save(root/'OWNER_SCOPE_FINISHED.json',dict(process=R.process_identity(os.getpid()),finished_unix_s=time.time(),controller_scope_exited=True),exclusive=True)
    print({'status':result['status'],'error':error,'wall_s':result['parent_wall_s']},flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--environments',type=int,choices=(1,2,4,16,64),default=4)
    parser.add_argument('--prior-native-environment-steps',type=int,default=0);parser.add_argument('--source-protocol',type=Path);parser.add_argument('--remaining-only',action='store_true');args=parser.parse_args()
    if args.action=='freeze':print(freeze(args.root,environments=args.environments,source_protocol=args.source_protocol,remaining_only=args.remaining_only,prior_native_steps=args.prior_native_environment_steps)['max_native_environment_steps'])
    else:
        p=B.read(args.root/'PROTOCOL.json');marker=B.sha(args.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_MJX_COLLECTION_ENV')!=marker:
            env=B._environment(p);env.update(PYTHONPATH=os.pathsep.join((str(WARP),str(DEPS),env['PYTHONPATH'])),
                CUDA_VISIBLE_DEVICES=GPU,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                WARP_CACHE_PATH=str(args.root.resolve()/'warp_cache'),ANYDEX_MJX_COLLECTION_ENV=marker)
            os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'run','--root',str(args.root.resolve())],env)
        if run(args.root)['status']!='complete':raise SystemExit(1)
