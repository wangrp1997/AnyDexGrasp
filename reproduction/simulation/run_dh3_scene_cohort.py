"""Stream original-DH3 batch top1 results to at most three physics clients.

No inference/control source is modified. One shared-model inference subprocess
runs every frozen RGB-D frame in the declared order. Each finalized original
selection is dispatched at most once, with immutable input hashes. Worker
stepSimulation wrappers only count original calls in a shared 16-byte counter;
no candidate search, reset/regrasp, or video encoding is added.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import traceback

SCRIPT_START = time.monotonic()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):return None


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def counted_steps(path):
    try:
        raw=Path(path).read_bytes()
        attempted,completed=struct.unpack('<QQ',raw)
        return dict(attempted_python_step_calls=attempted,completed_python_step_calls=completed,
                    possible_in_flight_step_calls=max(0,attempted-completed))
    except (OSError,struct.error):
        return dict(attempted_python_step_calls=0,completed_python_step_calls=0,
                    possible_in_flight_step_calls=0)


def native_environment(repo, python, gpu):
    """Source the existing local loader; no dependencies/environments installed."""
    loader=repo/'outputs/native_deps/env.sh'
    env=os.environ.copy()
    if not loader.is_file():raise FileNotFoundError('Existing native loader missing: '+str(loader))
    keys=['CUDA_HOME','PATH','PYTHONPATH','LD_LIBRARY_PATH','OMP_NUM_THREADS',
          'OPENBLAS_NUM_THREADS','MAX_JOBS','TORCH_CUDA_ARCH_LIST']
    snippet='import json,os; print(json.dumps({k:os.environ[k] for k in '+repr(keys)+' if k in os.environ}))'
    command=['bash','--noprofile','--norc','-c','source "$1"; "$2" -B -c "$3"',
             'anydex-native-loader',str(loader),str(python),snippet]
    loaded=subprocess.run(command,capture_output=True,text=True,check=True,env=env)
    env.update(json.loads(loaded.stdout))
    native=repo/'outputs/native_deps'
    env['PYTHONPATH']=str(native/'python_overlay')+os.pathsep+env.get('PYTHONPATH','')
    env['LD_LIBRARY_PATH']=str(native/'os_libs')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    env['CUDA_VISIBLE_DEVICES']=str(gpu)
    return env


def resolve_scenes(args):
    declared=[]
    if args.scene_list:
        source=args.scene_list.resolve();document=json.loads(source.read_text())
        values=document if isinstance(document,list) else document['scenes']
        for value in values:
            if isinstance(value,str):path=value;name=None
            else:
                path=value.get('scene_manifest',value.get('manifest',value.get('path')))
                if path is None and value.get('directory'):path=str(Path(value['directory'])/'scene_manifest.json')
                name=value.get('scene_id',value.get('id'))
            if path is None:raise ValueError('Scene-list row has no manifest path')
            path=Path(path)
            if not path.is_absolute():path=source.parent/path
            declared.append((name,path.resolve()))
    declared.extend((None,path.resolve()) for path in (args.scene_manifest or []))
    if not declared:raise ValueError('Provide --scene-list or repeat --scene-manifest')
    seen=set();records=[]
    for name,path in declared:
        if not path.is_file():raise FileNotFoundError(str(path))
        key=path.parent.name
        if key in seen:raise ValueError('Original batch requires unique resolved scene directory names: '+key)
        seen.add(key)
        scene=json.loads(path.read_text())
        # This is provenance only, never network geometry/pose input.
        input_files={}
        for filename in ('color.png','depth.png','meta.mat'):
            file=path.parent/filename
            if not file.is_file():raise FileNotFoundError(str(file))
            input_files[filename]=dict(path=str(file),bytes=file.stat().st_size,sha256=sha(file))
        records.append(dict(scene_id=name or key,batch_frame_name=key,scene_manifest=str(path),
            scene_manifest_sha256=sha(path),scene_signature_sha256=scene['scene_signature_sha256'],
            scene_module=scene.get('scene_module','synthetic_scene'),input_files=input_files,
            status='awaiting_inference',terminal=False,physics_invocations=0,
            actual_physical_attempts=0,completed_physics_steps=0,possible_in_flight_steps=0,
            final_top1_selected=None,not_entered_physics_reason=None))
    return records


def physics_worker(job_path):
    """Original runner entry with exact normally-completed step accounting."""
    job=json.loads(Path(job_path).read_text());started=time.monotonic();start_unix=time.time()
    receipt=dict(status='starting',job_path=str(Path(job_path).resolve()),job_sha256=sha(job_path),
        scene_manifest_sha256=job['scene_manifest_sha256'],selection_sha256=job['selection_sha256'],
        initial_scene_frozen_per_client=True,dispatch_attempt_number=1,
        physics_invocations=1,control_source_modified=False,counting_wrapper_only=True,
        worker_start_unix=start_unix,completed_physics_steps=0,possible_in_flight_steps=0)
    original_step=None;physics=None;counter=None;stream=None
    code=1
    try:
        for path,expected in job['bound_files'].items():
            if sha(path)!=expected:raise RuntimeError('Frozen worker input changed: '+path)
        repo=Path(job['repo_root'])
        sys.path[:0]=[str(repo/'reproduction/simulation'),str(repo/'reproduction/runtime')]
        spec=importlib.util.spec_from_file_location('cohort_original_scene_runner',job['physics_script'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        physics=module.p
        stream=Path(job['step_counter']).open('r+b',buffering=0)
        counter=mmap.mmap(stream.fileno(),16,access=mmap.ACCESS_WRITE)
        original_step=physics.stepSimulation
        attempted=completed=0
        def count_step(*values,**kwargs):
            nonlocal attempted,completed
            attempted+=1;struct.pack_into('<QQ',counter,0,attempted,completed)
            result=original_step(*values,**kwargs)
            completed+=1;struct.pack_into('<QQ',counter,0,attempted,completed)
            return result
        physics.stepSimulation=count_step
        sys.argv=[job['physics_script'],'--adapter-manifest',job['adapter_manifest'],
                  '--scene-manifest',job['scene_manifest'],'--selection-results',job['selection_results'],
                  '--output-dir',job['physics_output_dir']]
        module.main()
        result=read_json(Path(job['physics_output_dir'])/'RESULTS.json')
        if result is None:raise RuntimeError('Original physics runner saved no final RESULTS.json')
        if result['physics_steps']!=completed:raise RuntimeError('Original trace step count differs from observed actual stepSimulation calls')
        if result.get('actual_time_step_s')!=1/1920:raise RuntimeError('Original accepted 1920 Hz setting changed')
        receipt.update(status='completed',result_status=result['status'],actual_step_count_exact=True,
            physics_RESULT_sha256=sha(Path(job['physics_output_dir'])/'RESULTS.json'))
        code=0
    except BaseException as exc:
        receipt.update(status='physics_execution_error',error_type=type(exc).__name__,error=str(exc),
                       traceback=traceback.format_exc())
    finally:
        if physics is not None and original_step is not None:physics.stepSimulation=original_step
        if counter is not None:counter.flush();counter.close()
        if stream is not None:stream.close()
        progress=counted_steps(job['step_counter'])
        receipt.update(**progress,completed_physics_steps=progress['completed_python_step_calls'],
            possible_in_flight_steps=progress['possible_in_flight_step_calls'],
            actual_physical_attempts=int(progress['completed_python_step_calls']>0),
            worker_wall_seconds=time.monotonic()-started,worker_end_unix=time.time(),
            video_encoding_executed=False,
            accounting_boundary='Completed calls are exact for normal completion. On abrupt process interruption the mmap counter is a completed-step lower bound plus at most one in-flight call; no unknown step is silently treated as zero.')
        atomic_json(job['worker_receipt'],receipt)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp'))
    parser.add_argument('--scene-list',type=Path)
    parser.add_argument('--scene-manifest',type=Path,action='append')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--python',type=Path,default=Path('/home/wangrenpeng/boya-inhand-wm/.venv/bin/python'))
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--max-physics-workers',type=int,choices=[1,2,3],default=3)
    parser.add_argument('--poll-seconds',type=float,default=.25)
    parser.add_argument('--adapter-manifest',type=Path)
    parser.add_argument('--asset-root',type=Path)
    args=parser.parse_args()
    if not .05<=args.poll_seconds<=5:parser.error('poll-seconds must be .05..5')
    repo=args.repo_root.resolve()
    python=args.python.absolute()  # Preserve virtualenv interpreter path.
    output=args.output_dir.resolve()
    inference_script=repo/'reproduction/runtime/run_official_dh3_batch.py'
    physics_script=repo/'reproduction/simulation/run_dh3_scene.py'
    adapter=(args.adapter_manifest or repo/'outputs/dh3_geometry_adapter_20261103_v2/ADAPTER.json').resolve()
    asset_root=(args.asset_root or repo/'outputs/official_dh3_collision_assets_20261103_v2/assets').resolve()
    if output.exists():parser.error('Use a fresh output directory; each original top1 is dispatched at most once')
    for path in [python,inference_script,physics_script,adapter]:
        if not path.is_file():parser.error('Missing existing prerequisite: '+str(path))
    scenes=resolve_scenes(args)
    output.mkdir(parents=True);(output/'logs').mkdir();(output/'jobs').mkdir();(output/'progress').mkdir()
    frozen_sources={str(path):sha(path) for path in [Path(__file__).resolve(),inference_script,physics_script,adapter]}
    frozen=dict(scene_order=[row['scene_id'] for row in scenes],scenes=scenes,
        declared_scene_list=({'path':str(args.scene_list.resolve()),'sha256':sha(args.scene_list)} if args.scene_list else None),
        sources=frozen_sources,seed=args.seed,max_physics_workers=args.max_physics_workers,
        same_original_top1_only=True,physical_attempt_limit_per_scene=1,actual_hz=1920,
        arm_execution=False,network_ground_truth_input=False,video_encoding_during_pipeline=False)
    atomic_json(output/'FROZEN_INPUTS.json',frozen)
    env=native_environment(repo,python,args.gpu)
    worker_env=env.copy();worker_env['CUDA_VISIBLE_DEVICES']=''
    logs={};active={};pending=[];events=[];interrupted=[None]
    pipeline=None;pipeline_start=None;pipeline_end=None;last_physics_end=None;peak=0
    report=dict(scope='Frozen original-DH3 RGB-D batch top1 to independent scene physics pipeline',
        status='starting',repo_root=str(repo),output_dir=str(output),requested_scenes=len(scenes),
        max_physics_workers=args.max_physics_workers,peak_concurrent_physics_workers=0,
        seed=args.seed,same_original_top1_only=True,per_scene_physics_invocation_limit=1,
        actual_hz=1920,arm_execution=False,training_steps=0,video_encoding_executed=False,
        inference_log=str(output/'logs/inference.log'),scenes=scenes,
        frozen_inputs_sha256=sha(output/'FROZEN_INPUTS.json'),frozen_sources=frozen_sources,
        timing_seconds={},physical_success_rate_claim_scope='Only this predeclared independent simulation cohort; never the original paper benchmark')
    def event(kind,**values):
        row=dict(event=kind,unix=time.time(),scheduler_elapsed_s=time.monotonic()-SCRIPT_START,**values)
        events.append(row)
        with (output/'EVENTS.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def snapshot():
        nonlocal peak
        for row in scenes:
            if row.get('step_counter'):
                steps=counted_steps(row['step_counter'])
                row['completed_physics_steps']=steps['completed_python_step_calls']
                row['possible_in_flight_steps']=steps['possible_in_flight_step_calls']
                row['actual_physical_attempts']=int(row['completed_physics_steps']>0)
        peak=max(peak,len(active));report['peak_concurrent_physics_workers']=peak
        report['aggregate']=dict(final_status_counts=dict(collections.Counter(row['status'] for row in scenes)),
            finalized_scenes=sum(row['terminal'] for row in scenes),original_top1_selected=sum(row['final_top1_selected'] is True for row in scenes),
            physics_invocations=sum(row['physics_invocations'] for row in scenes),
            actual_physical_attempts=sum(row['actual_physical_attempts'] for row in scenes),
            completed_physics_steps=sum(row['completed_physics_steps'] for row in scenes),
            possible_in_flight_steps=sum(row['possible_in_flight_steps'] for row in scenes),
            valid_object_lifted=sum(row['status']=='object_lifted' for row in scenes),
            observed_lifted=sum(row.get('lift_observed') is True for row in scenes),
            not_entered_physics=sum(row['terminal'] and row['physics_invocations']==0 for row in scenes),
            active_physics_workers=len(active),queued_physics_workers=len(pending))
        report['timing_seconds']['scheduler_wall_so_far']=time.monotonic()-SCRIPT_START
        if pipeline_start is not None:
            report['timing_seconds']['pipeline_and_physics_wall_so_far']=time.monotonic()-pipeline_start
        atomic_json(output/'RESULTS.json',report)
    def no_physics(row,status,reason):
        row.update(status=status,terminal=True,not_entered_physics_reason=reason)
        event('scene_final_without_physics',scene_id=row['scene_id'],status=status,reason=reason)
    def stop(signum,_frame):
        interrupted[0]=signum
    previous_signals={signum:signal.signal(signum,stop) for signum in [signal.SIGINT,signal.SIGTERM]}
    try:
        snapshot()
        command=[str(python),'-B',str(inference_script),'--output-dir',str(output/'inference'),
                 '--seed',str(args.seed),'--asset-root',str(asset_root)]
        for row in scenes:command+=['--scene-manifest',row['scene_manifest']]
        report['inference_command']=command
        logs['inference']=(output/'logs/inference.log').open('wb')
        pipeline_start=time.monotonic();report['pipeline_start_unix']=time.time()
        pipeline=subprocess.Popen(command,cwd=repo,env=env,stdout=logs['inference'],stderr=subprocess.STDOUT)
        report.update(status='running',inference_pid=pipeline.pid)
        event('inference_started',pid=pipeline.pid,frames=len(scenes))
        last_snapshot=0
        while True:
            changed=False
            if interrupted[0] is not None:
                if pipeline.poll() is None:pipeline.terminate()
                for state in active.values():
                    if state['process'].poll() is None:state['process'].terminate()
            for key,state in list(active.items()):
                process=state['process'];returncode=process.poll()
                if returncode is None:continue
                completed_at=time.monotonic();last_physics_end=completed_at;row=state['row']
                logs[key].close()
                worker=read_json(row['worker_receipt']);result_path=Path(row['physics_output_dir'])/'RESULTS.json'
                result=read_json(result_path);progress=counted_steps(row['step_counter'])
                row.update(physics_returncode=returncode,physics_process_wall_seconds=completed_at-state['started'],
                    physics_finished_unix=time.time(),completed_physics_steps=progress['completed_python_step_calls'],
                    possible_in_flight_steps=progress['possible_in_flight_step_calls'],
                    actual_physical_attempts=int(progress['completed_python_step_calls']>0),terminal=True)
                if worker is not None:row['worker_result']=worker
                if result is not None:
                    row.update(raw_physics_result_status=result['status'],physics_RESULTS_sha256=sha(result_path),
                        lift_observed=result.get('lift_observed'),port_stability_pass=result.get('port_stability_pass'),
                        object_final_lift_m=result.get('object_final_lift_m'),
                        object_final_lowest_world_z_m=result.get('object_final_lowest_world_z_m'))
                if returncode==0 and worker and worker.get('status')=='completed' and result:
                    if result['status']=='object_lifted' and result.get('lift_observed') and result.get('port_stability_pass'):
                        row['status']='object_lifted'
                    elif not result.get('port_stability_pass'):
                        row['status']='physics_port_invalid'
                    else:row['status']='object_not_lifted'
                else:
                    row['status']='physics_execution_error'
                    row['error_reason']=(worker or {}).get('error','Worker exited without a complete normal receipt; inspect retained log/counter')
                    if progress['possible_in_flight_step_calls']:
                        row['step_accounting_boundary']='Completed lower bound plus possible in-flight call retained; abrupt termination prevents an exact completed-step total'
                event('physics_finished',scene_id=row['scene_id'],status=row['status'],returncode=returncode,
                      completed_physics_steps=row['completed_physics_steps'],wall_seconds=row['physics_process_wall_seconds'])
                del active[key];changed=True
            batch=read_json(output/'inference/RESULTS.json') or {}
            frame_states={row['name']:row for row in batch.get('frames',[])}
            for row in scenes:
                if row['status']!='awaiting_inference':continue
                frame=frame_states.get(row['batch_frame_name'])
                if not frame or frame.get('status')=='starting':continue
                if frame.get('status')=='failed':
                    no_physics(row,'inference_error',frame.get('error','Original inference frame failed'))
                    row['inference_failure']=frame;changed=True;continue
                frame_dir=output/'inference'/row['batch_frame_name'];run_path=frame_dir/'RESULTS.json'
                run=read_json(run_path)
                if run is None or frame.get('RESULTS_sha256')!=sha(run_path):continue
                row.update(inference_output_dir=str(frame_dir),inference_RESULTS_sha256=sha(run_path),
                    inference_frame_status=frame['status'],inference_frame_wall_seconds=frame.get('wall_seconds'),
                    inference_stage_timing_seconds=run.get('timing_seconds'),
                    hand_stages=run.get('hand_stage',{}).get('stages'),
                    collision_stages=run.get('collision_stage',{}).get('result',{}).get('stages'))
                selected_path=frame_dir/'selection/RESULTS.json';selected=read_json(selected_path)
                if not frame.get('selected'):
                    row['final_top1_selected']=False
                    no_physics(row,'no_original_candidate',run.get('stop_reason',frame.get('stop_reason',frame['status'])))
                    changed=True;continue
                if selected is None:continue
                if run.get('selection_stage',{}).get('RESULTS_sha256')!=sha(selected_path):continue
                if run.get('scene_binding',{}).get('manifest_sha256')!=row['scene_manifest_sha256']:
                    no_physics(row,'scene_binding_error','Final inference scene binding differs from frozen cohort manifest')
                    changed=True;continue
                if selected.get('selected') is not True:continue
                candidate=Path(selected['output']['path'])
                if not candidate.is_file() or sha(candidate)!=selected['output']['sha256']:
                    no_physics(row,'selection_binding_error','Final original top1 NPZ is missing or changed')
                    changed=True;continue
                row.update(status='queued_for_physics',final_top1_selected=True,
                    selection_results=str(selected_path),selection_sha256=sha(selected_path),
                    selected_candidate_npz=str(candidate),selected_candidate_sha256=sha(candidate),
                    selected_hand=selected.get('selected_hand'),selection_ready_unix=time.time(),
                    selection_ready_after_pipeline_start_s=time.monotonic()-pipeline_start)
                pending.append(row);changed=True
                event('original_top1_ready',scene_id=row['scene_id'],selection_results=str(selected_path))
            while pending and len(active)<args.max_physics_workers and interrupted[0] is None:
                row=pending.pop(0);key=row['batch_frame_name']
                try:
                    for path,digest in frozen_sources.items():
                        if sha(path)!=digest:raise RuntimeError('Frozen source/adapter changed: '+path)
                    if sha(row['scene_manifest'])!=row['scene_manifest_sha256']:raise RuntimeError('Frozen scene manifest changed')
                    if sha(row['selection_results'])!=row['selection_sha256']:raise RuntimeError('Original top1 selection changed')
                    counter_path=output/'progress'/(key+'.steps.bin');counter_path.write_bytes(bytes(16))
                    worker_receipt=output/'progress'/(key+'.worker.json')
                    physics_dir=output/'physics'/key;job_path=output/'jobs'/(key+'.json')
                    bound={str(physics_script):frozen_sources[str(physics_script)],str(adapter):frozen_sources[str(adapter)],
                        str(Path(__file__).resolve()):frozen_sources[str(Path(__file__).resolve())],
                        row['scene_manifest']:row['scene_manifest_sha256'],
                        row['selection_results']:row['selection_sha256'],row['selected_candidate_npz']:row['selected_candidate_sha256']}
                    job=dict(repo_root=str(repo),physics_script=str(physics_script),adapter_manifest=str(adapter),
                        scene_manifest=row['scene_manifest'],scene_manifest_sha256=row['scene_manifest_sha256'],
                        selection_results=row['selection_results'],selection_sha256=row['selection_sha256'],
                        physics_output_dir=str(physics_dir),step_counter=str(counter_path),worker_receipt=str(worker_receipt),bound_files=bound)
                    atomic_json(job_path,job)
                    logs[key]=(output/'logs'/('physics_'+key+'.log')).open('wb')
                    command=[str(python),'-B',str(Path(__file__).resolve()),'--internal-physics-worker',str(job_path)]
                    started=time.monotonic()
                    process=subprocess.Popen(command,cwd=repo,env=worker_env,stdout=logs[key],stderr=subprocess.STDOUT)
                    active[key]=dict(process=process,row=row,started=started)
                    row.update(status='physics_running',physics_invocations=1,physics_pid=process.pid,
                        physics_output_dir=str(physics_dir),physics_log=str(output/'logs'/('physics_'+key+'.log')),
                        step_counter=str(counter_path),worker_receipt=str(worker_receipt),job_path=str(job_path),
                        physics_started_unix=time.time(),physics_dispatch_after_pipeline_start_s=started-pipeline_start,
                        physics_command=command)
                    event('physics_started',scene_id=row['scene_id'],pid=process.pid,
                          concurrency=len(active),selection_sha256=row['selection_sha256'])
                except Exception as exc:
                    no_physics(row,'physics_dispatch_error',str(exc));row['dispatch_traceback']=traceback.format_exc()
                changed=True
            returncode=pipeline.poll()
            exit_observed_now=False
            if returncode is not None:
                if pipeline_end is None:
                    exit_observed_now=True
                    pipeline_end=time.monotonic();report['inference_returncode']=returncode
                    report['inference_finished_unix']=time.time()
                    report['inference_batch_status']=batch.get('status')
                    report['inference_model_cache']=batch.get('cache')
                    report['timing_seconds']['inference_subprocess_wall']=pipeline_end-pipeline_start
                    event('inference_finished',returncode=returncode,status=batch.get('status'))
                # Final inference receipts are immutable now. Pending scenes
                # without one are explicit infrastructure failures, not omitted.
                for row in scenes:
                    if row['status']=='awaiting_inference' and not exit_observed_now:
                        no_physics(row,'inference_not_completed',
                            'Batch exited without a complete, matching final frame/selection receipt')
                        changed=True
            if interrupted[0] is not None:
                for row in pending:
                    no_physics(row,'interrupted_before_physics','Cohort interruption requested; no retry dispatched')
                pending.clear()
            if changed or time.monotonic()-last_snapshot>=1:
                snapshot();last_snapshot=time.monotonic()
            if returncode is not None and not active and not pending and all(row['terminal'] for row in scenes):break
            time.sleep(args.poll_seconds)
        if logs.get('inference'):logs['inference'].close()
        end=max(pipeline_end or pipeline_start,last_physics_end or pipeline_start)
        report['timing_seconds'].update(pipeline_and_last_physics_wall=end-pipeline_start,
            pipeline_start_to_last_physics_wall=(last_physics_end-pipeline_start if last_physics_end is not None else None),
            scheduler_total_wall=time.monotonic()-SCRIPT_START)
        report.update(status='interrupted' if interrupted[0] is not None else 'cohort_finished',
            cohort_finished_unix=time.time(),interruption_signal=interrupted[0],
            media_boundary='Actual concurrent simulation frames are retained by each physics runner; video encoding deliberately remains a separate later operation',
            physics_step_accounting='Original stepSimulation calls counted transparently; normal workers require exact equality with original saved trace/result. Abrupt exits retain a lower bound and possible in-flight call.',
            no_candidate_retry=True,no_scene_reset_regrasp=True,source_or_control_changes=False)
        snapshot()
        print(json.dumps(dict(status=report['status'],aggregate=report['aggregate'],timing_seconds=report['timing_seconds'])),flush=True)
        return 0
    except BaseException as exc:
        report.update(status='scheduler_failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        for state in active.values():
            if state['process'].poll() is None:state['process'].terminate()
        if pipeline is not None and pipeline.poll() is None:pipeline.terminate()
        snapshot()
        raise
    finally:
        for signum,handler in previous_signals.items():signal.signal(signum,handler)
        for stream in logs.values():
            if not stream.closed:stream.close()


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--internal-physics-worker':
        raise SystemExit(physics_worker(sys.argv[2]))
    raise SystemExit(main())
