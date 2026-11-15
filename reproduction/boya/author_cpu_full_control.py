"""One CPU serial/full parallel pair selected by the frozen prefix rule."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import author_cpu_prefix_control as P
import author_cpu_prefix_audit as PA

C,B,R,G=P.C,P.B,P.R,P.G
SCHEMA='anydex_boya_author_cpu_full_control_v1'
PROTOCOL='CPU_FULL_PROTOCOL.json'
RESULTS='CPU_FULL_RESULTS.json'


def freeze(root,prefix_audit):
    audit=B.read(prefix_audit)
    C.need(audit['schema']==PA.SCHEMA and audit['status']=='prefix_integrity_pass',
           'Passing independent prefix audit required')
    B.verify(B.normalized_bindings(audit['inputs_sha256']))
    source=P.verify(Path(audit['groups'][0]['workers'][0]['path']).parents[3])
    decision=PA.select(audit['groups'],source['full_followup_selection'])
    C.need(decision==audit['selection'] and decision['full_followup_allowed'],
           'Frozen throughput rule did not qualify a parallel followup')
    selected=decision['selected_parallel_count'];C.need(selected in (4,8),'Only reviewed parallel sizes')
    root=Path(root).resolve();sources=copy.deepcopy(source['sources'])
    for name in ('author_cpu_full_control.py','author_cpu_full_audit.py',
                 'author_scaling_full_execution.py','author_gpu_vector_audit.py',
                 'audit_author_collection_runtime.py'):
        sources[name]=B.binding(B.HERE/name)
    inputs=copy.deepcopy(source['inputs']);inputs['prefix_audit']=B.binding(prefix_audit)
    groups=[]
    for count in (1,selected):
        groups.append(dict(name=f'cpu_{count}',mode='cpu',count=count,workers=[
            dict(worker_id=f'worker_{i:02d}',prepared_dir=source['prepared_template']['path'],lanes=1,
                stage_dir=str(root/f'cpu_{count}'/f'worker_{i:02d}'),
                output_dir=str(root/f'cpu_{count}'/f'worker_{i:02d}'/'result')) for i in range(count)]))
    p=dict(schema=SCHEMA,task=B.TASK,groups=groups,selected_count=selected,
        selection=decision,selection_rule=source['full_followup_selection'],
        full_trial=True,command_steps=3024,prefix_ticks=0,retry_budget=0,stage_timeout_s=900.,
        total_worker_attempt_budget=1+selected,total_integrated_environment_steps=3024*(1+selected),
        total_dynamic_object_steps=3*3024*(1+selected),groups_are_sequential=True,
        diagnostic_only=True,label=None,raw_label=None,exclude_training=True,formal_training_eligible=False,
        automatic_training=False,resource_limits=dict(kit=selected),
        gpu_uuid=source['gpu_uuid'],gpu_identity=source['gpu_identity'],
        env=source['env'],inherited_env=source['inherited_env'],sources=sources,inputs=inputs,
        prepared_template=source['prepared_template'],worker_source=B.binding(B.HERE/'author_scaling_full_execution.py'),
        cancel_sigint_grace_s=30.,cancel_sigterm_grace_s=5.,cancel_sigkill_grace_s=5.,
        sampling_interval_s=2.,created_unix_s=time.time(),
        limitations=['Known template repetitions cannot become training labels or estimate random-scene yield',
            'One full serial/parallel comparison without independent replication',
            'No automatic production collection, training or external evaluation'])
    B.verify(inputs);B.verify(sources);root.mkdir(parents=True,exist_ok=False)
    p['source_archive']=C.archive_sources(root,sources);B.save(root/PROTOCOL,p,exclusive=True)
    return verify(root)


def verify(root):
    root=Path(root).resolve();p=B.read(root/PROTOCOL);n=p['selected_count']
    C.need(p['schema']==SCHEMA and n in (4,8) and p['full_trial'] is True
        and p['command_steps']==3024 and p['prefix_ticks']==0 and p['retry_budget']==0
        and p['total_worker_attempt_budget']==n+1 and p['total_integrated_environment_steps']==3024*(n+1)
        and p['exclude_training'] is True and p['stage_timeout_s']==900.
        and p['resource_limits']==dict(kit=n),'Frozen full-trial budget differs')
    C.need([(g['name'],g['mode'],g['count']) for g in p['groups']]==
        [(f'cpu_{k}','cpu',k) for k in (1,n)],'Frozen full serial then parallel order required')
    for g in p['groups']:
        C.need(len(g['workers'])==g['count'],'Frozen full worker count changed')
        for i,w in enumerate(g['workers']):
            d=root/g['name']/f'worker_{i:02d}'
            C.need(w==dict(worker_id=f'worker_{i:02d}',prepared_dir=p['prepared_template']['path'],
                lanes=1,stage_dir=str(d),output_dir=str(d/'result')),'Full worker escaped its source/output')
    B.verify(p['inputs']);B.verify(p['sources']);B.verify({'archive':p['source_archive']})
    for row in B.read(p['source_archive']['path'])['sources']:
        B.verify({'source':row['source'],'archive':row['archive']})
    C.need(p['env']['CUDA_VISIBLE_DEVICES']==p['gpu_uuid'] and p['env']['NVIDIA_VISIBLE_DEVICES']=='0',
           'Only physical GPU0 may serve the CPU workers')
    audit=B.read(p['inputs']['prefix_audit']['path'])
    C.need(PA.select(audit['groups'],p['selection_rule'])==p['selection']==audit['selection'],
           'Performance-based full selection changed')
    return p


def admit(r,worker,p):
    n=p['command_steps']
    C.need(r['status']=='diagnostic_full' and r['complete'] is True and r['full_trial'] is True
        and r['prefix_ticks']==0 and r['expected_command_steps']==n
        and r['mode']=='cpu' and r['device']=='cpu' and r['lanes']==1
        and r['label'] is None and type(r['raw_label']) is int and r['raw_label'] in (0,1)
        and r['exclude_training'] is True and r['formal_training_eligible'] is False,
        'Complete excluded CPU full diagnostic required')
    for key in ('physical_clock_steps','returned_physical_steps','actual_simulation_step_counter',
                'physical_steps_lower_bound','physical_steps_upper_bound',
                'actual_integrated_environment_steps','recorded_command_environment_steps',
                'integrated_environment_steps_lower_bound','integrated_environment_steps_upper_bound'):
        C.need(r[key]==n,'Full step accounting mismatch: '+key)
    C.need(r['observations_completed_steps_per_lane']==[n] and r['physical_steps_unknown'] is False
        and r['inflight_step_unconfirmed'] is False and all(r['original_gates'].values()),
        'Original full gates or exact completed costs failed')
    C.need(r['prepared_template']['ready']==p['prepared_template']['ready']
        and r['prepared_template']['commands']==p['prepared_template']['commands'], 'Original preparation changed')
    B.verify({'group':r['group_receipt'],'profile':r['profile']})
    B.verify(B.normalized_bindings(r['source_sha256']))
    C.need(Path(r['group_receipt']['path'])==Path(worker['output_dir'])/'GPU_VECTOR_EXECUTION.json',
           'Full receipt belongs to another output')
    return r


def stage(worker,group,p,pool,sampler,stop):
    directory=Path(worker['stage_dir']);directory.mkdir(parents=True,exist_ok=False)
    output=Path(worker['output_dir'])
    command=[sys.executable,'-B',p['worker_source']['path'],'--prepared-dir',worker['prepared_dir'],
        '--output-dir',str(output),'--protocol',p['_protocol_path']]
    value=dict(status='starting',complete=False,command=command,worker_id=worker['worker_id'],
        started_unix_s=time.time(),process=None,retry_count=0,signals=[],physical_steps_unknown=True)
    B.save(directory/'STARTED.json',value,exclusive=True);began=time.monotonic();process=None
    try:
        B.verify(p['inputs']);B.verify(p['sources'])
        with pool.stage(('kit',),directory):
            with (directory/'process.log').open('xb') as stream:
                process=pool.popen(command,cwd=B.ROOT,env=B._environment(p),stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True)
                value.update(status='running',process=R.process_identity(process.pid));sampler.add(value['process'])
                B.save(directory/'STAGE.json',value);deadline=time.monotonic()+p['stage_timeout_s']
                while process.poll() is None:
                    if stop.is_set() or time.monotonic()>=deadline:
                        value['cancellation_reason']='group_stop' if stop.is_set() else 'stage_timeout'
                        value['signals']=C.finish_owned(process,value['process'],p);break
                    try:process.wait(timeout=.2)
                    except subprocess.TimeoutExpired:continue
                C.need(process.returncode==0 and value.get('cancellation_reason') is None,'Full worker failed/canceled')
                r=admit(B.read(output/C.WORKER_RECEIPT),worker,p)
                C.need(r['worker_pid']==process.pid,'Full receipt PID changed')
                value.update(status='diagnostic_full',complete=True,receipt=B.binding(output/C.WORKER_RECEIPT),
                    physical_steps_known=r['physical_clock_steps'],physical_steps_unknown=False,
                    physical_cost={k:r[k] for k in C.COST_FIELDS})
            B.verify(p['inputs']);B.verify(p['sources'])
    except BaseException:
        stop.set();value.update(status='infrastructure_failure',complete=False,error=traceback.format_exc())
        if process is not None and process.poll() is None:
            try:value['signals']+=C.finish_owned(process,value['process'],p)
            except BaseException:value['cleanup_error']=traceback.format_exc()
        if (output/C.WORKER_RECEIPT).is_file():
            partial=B.read(output/C.WORKER_RECEIPT)
            value.update(partial_receipt=B.binding(output/C.WORKER_RECEIPT),
                physical_steps_known=partial.get('physical_clock_steps'),
                physical_steps_unknown=partial.get('physical_steps_unknown',True),
                physical_cost={k:partial.get(k) for k in C.COST_FIELDS})
    value.update(wall_s=time.monotonic()-began,finished_unix_s=time.time(),
        returncode=None if process is None else process.poll())
    B.save(directory/'STAGE.json',value)
    return dict(worker_id=worker['worker_id'],status=value['status'],stage=B.binding(directory/'STAGE.json'),
        receipt=value.get('receipt'),partial_receipt=value.get('partial_receipt'),process=value.get('process'))


def run_group(root,group,p,pool,sampler):
    began=time.monotonic();start=time.time();stop=threading.Event();workers=[]
    with ThreadPoolExecutor(max_workers=group['count']) as executor:
        pending=[executor.submit(stage,w,group,p,pool,sampler,stop) for w in group['workers']]
        for future in as_completed(pending):
            worker=future.result();workers.append(worker)
            if worker['status']!='diagnostic_full':stop.set()
    workers.sort(key=lambda w:w['worker_id'])
    value=dict(name=group['name'],count=group['count'],mode='cpu',workers=workers,
        status='full_complete' if all(w['status']=='diagnostic_full' for w in workers) else 'stopped_after_failure',
        started_unix_s=start,finished_unix_s=time.time(),wall_s=time.monotonic()-began,
        label=None,exclude_training=True)
    B.save(Path(root)/group['name']/'GROUP_RESULTS.json',value,exclusive=True)
    return value


def run(root):
    root=Path(root).resolve();p=verify(root);C.need(not (root/'RUN_STARTED.json').exists(),'No full retries/resume')
    C.need(C.gpu_identity(p['inputs']['nvidia_smi']['path'],p['gpu_uuid'])==p['gpu_identity'],
           'Frozen GPU0 identity changed')
    start=time.monotonic();groups=[];error=None
    with R.lease(root/'DRIVER.lock'),R.lease(G.LOCK) as fd:
        with R.active_run(root,G.REGISTRY,B.binding(root/PROTOCOL)):
            B.save(root/'RUN_STARTED.json',dict(protocol=B.binding(root/PROTOCOL),
                process=R.process_identity(os.getpid()),unix_s=time.time()),exclusive=True)
            pool=C.ScalingPool(root/'resource_slots',fd,B.binding(root/PROTOCOL))
            sampler=C.ResourceSampler(root,p);sampler.start()
            try:
                for group in p['groups']:
                    value=run_group(root,group,dict(p,_protocol_path=str(root/PROTOCOL)),pool,sampler)
                    groups.append(value)
                    if value['status']!='full_complete':break
            except BaseException:error=traceback.format_exc()
            finally:sampler.close()
            value=dict(schema=SCHEMA,status='complete' if len(groups)==2 and error is None
                and all(g['status']=='full_complete' for g in groups) else 'stopped_after_failure',
                protocol=B.binding(root/PROTOCOL),groups=groups,error=error,wall_s=time.monotonic()-start,
                resource_samples=B.binding(sampler.path),exclude_training=True,formal_training_eligible=False,
                label=None,raw_label=None,new_training_labels=0,training_updates=0,retries=0)
            B.verify(p['inputs']);B.verify(p['sources']);B.save(root/RESULTS,value,exclusive=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--prefix-audit',type=Path)
    args=parser.parse_args()
    value=freeze(args.output_dir,args.prefix_audit) if args.action=='freeze' else run(args.output_dir)
    print(json.dumps(dict(status=value.get('status','frozen'),workers=value.get('total_worker_attempt_budget'),
        groups=len(value['groups'])),indent=2))
    if args.action=='run' and value['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
