"""Freeze and run CPU 1/4/8-process prefixes through existing stage controls."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import author_scaling_control as C

B,R,G=C.B,C.R,C.G
SCHEMA='anydex_boya_author_cpu_prefix_control_v1'
PROTOCOL='CPU_PREFIX_PROTOCOL.json'
RESULTS='CPU_PREFIX_RESULTS.json'
COUNTS=(1,4,8)


def freeze(root):
    from author_gpu_vector_execution import load_jobs,source_bindings
    old=B.read(C.REFERENCE);prepared=Path(old['jobs'][0]['prepared_dirs'][0]).resolve()
    job=load_jobs([prepared])[0]
    C.need(job['command_steps']==3024,'Frozen g6 plan required')
    sources=copy.deepcopy(old['sources'])
    sources.update({p:dict(path=p,sha256=h) for p,h in source_bindings().items()})
    for name in ('author_cpu_prefix_control.py','author_cpu_prefix_audit.py',
                 'author_scaling_execution.py','author_scaling_control.py',
                 'author_scaling_audit.py','author_gpu_vector_profile.py'):
        sources[name]=B.binding(B.HERE/name)
    inputs=copy.deepcopy(old['inputs']);inputs['reference_protocol']=B.binding(C.REFERENCE)
    inputs.update({'g6_'+k:v for k,v in job['bindings'].items()})
    smi=shutil.which('nvidia-smi');C.need(smi is not None,'Declared GPU0 sampler required')
    inputs['nvidia_smi']=B.binding(smi)
    root=Path(root).resolve()
    groups=[]
    for count in COUNTS:
        groups.append(dict(name=f'cpu_{count}',mode='cpu',count=count,workers=[
            dict(worker_id=f'worker_{i:02d}',prepared_dir=str(prepared),lanes=1,
                stage_dir=str(root/f'cpu_{count}'/f'worker_{i:02d}'),
                output_dir=str(root/f'cpu_{count}'/f'worker_{i:02d}'/'result')) for i in range(count)]))
    p=dict(schema=SCHEMA,task=B.TASK,diagnostic_only=True,label=None,raw_label=None,
        exclude_training=True,formal_training_eligible=False,automatic_training=False,
        groups=groups,prefix_ticks=100,retry_budget=0,stage_timeout_s=900.,
        total_worker_attempt_budget=13,total_worker_physical_clock_steps=1300,
        total_integrated_environment_steps=1300,total_dynamic_object_steps=3900,
        resource_limits=dict(kit=8),groups_are_sequential=True,
        cancel_sigint_grace_s=30.,cancel_sigterm_grace_s=5.,cancel_sigkill_grace_s=5.,
        sampling_interval_s=2.,gpu_uuid=old['gpu_uuid'],
        gpu_identity=C.gpu_identity(smi,old['gpu_uuid']),
        env=dict(old['env']),inherited_env=old['inherited_env'],sources=sources,inputs=inputs,
        worker_source=B.binding(C.WORKER),
        prepared_template=dict(path=str(prepared),ready=B.binding(prepared/'READY.json'),
            commands=job['bindings']['commands']),created_unix_s=time.time(),
        full_followup_selection=dict(minimum_both_rate_gain=1.10,tie_relative_band=.05,
            score='min(prefix throughput / cpu1, startup-inclusive throughput / cpu1)',
            eligibility='All prefix audits pass; measured prefix peak concurrency >= 2',
            tie_break='Lowest process count within 5 percent of best score',
            full_attempt_budget_max=9,full_command_steps_per_worker=3024,
            full_requires_separate_frozen_protocol=True),
        limitations=['Known repeated-template prefixes do not add labels or estimate collection yield',
            'Independent CPU clocks; aggregate steps are not a shared clock',
            'Execution intervals may be staggered; actual spans and occupancy are measured',
            'No nominal worker-count speedup or full collection ETA'])
    B.verify(inputs);B.verify(sources)
    root.mkdir(parents=True,exist_ok=False)
    p['source_archive']=C.archive_sources(root,sources)
    B.save(root/PROTOCOL,p,exclusive=True)
    return verify(root)


def verify(root):
    root=Path(root).resolve();p=B.read(root/PROTOCOL)
    C.need(p['schema']==SCHEMA and p['prefix_ticks']==100 and p['retry_budget']==0
        and p['total_worker_attempt_budget']==13 and p['total_integrated_environment_steps']==1300
        and p['stage_timeout_s']==900. and p['resource_limits']=={'kit':8}
        and p['exclude_training'] is True,'Frozen CPU prefix budget required')
    C.need([(g['name'],g['mode'],g['count']) for g in p['groups']]==
        [(f'cpu_{n}','cpu',n) for n in COUNTS],'Frozen CPU 1/4/8 order required')
    for g in p['groups']:
        C.need(len(g['workers'])==g['count'],'Frozen worker count required')
        for i,w in enumerate(g['workers']):
            directory=root/g['name']/f'worker_{i:02d}'
            C.need(w==dict(worker_id=f'worker_{i:02d}',prepared_dir=p['prepared_template']['path'],
                lanes=1,stage_dir=str(directory),output_dir=str(directory/'result')),
                'A worker escaped the frozen source/output contract')
    B.verify(p['inputs']);B.verify(p['sources']);B.verify({'archive':p['source_archive']})
    archive=B.read(p['source_archive']['path'])
    C.need({(r['source']['path'],r['source']['sha256']) for r in archive['sources']}==
        {(r['path'],r['sha256']) for r in p['sources'].values()},'Source archive incomplete')
    for row in archive['sources']:B.verify({'source':row['source'],'archive':row['archive']})
    C.need(p['env']['CUDA_VISIBLE_DEVICES']==p['gpu_uuid'] and p['env']['NVIDIA_VISIBLE_DEVICES']=='0',
        'Only GPU0 may serve the CPU Kit render/asset services')
    return p


def run(root):
    root=Path(root).resolve();p=verify(root)
    C.need(not (root/'RUN_STARTED.json').exists(),'Started diagnostic is never retried')
    C.need(C.gpu_identity(p['inputs']['nvidia_smi']['path'],p['gpu_uuid'])==p['gpu_identity'],'GPU0 changed')
    start=time.monotonic();groups=[];error=None
    with R.lease(root/'DRIVER.lock'),R.lease(G.LOCK) as fd:
        with R.active_run(root,G.REGISTRY,B.binding(root/PROTOCOL)):
            B.save(root/'RUN_STARTED.json',dict(protocol=B.binding(root/PROTOCOL),
                process=R.process_identity(os.getpid()),unix_s=time.time()),exclusive=True)
            pool=C.ScalingPool(root/'resource_slots',fd,B.binding(root/PROTOCOL))
            sampler=C.ResourceSampler(root,p);sampler.start()
            try:
                for group in p['groups']:
                    B.verify(p['inputs']);B.verify(p['sources'])
                    value=C.run_group(root,group,dict(p,_protocol_path=str(root/PROTOCOL)),pool,sampler)
                    groups.append(value)
                    if value['status']!='partial_scaling_complete':break
            except BaseException:error=traceback.format_exc()
            finally:sampler.close()
            result=dict(schema=SCHEMA,status='complete' if len(groups)==3 and error is None
                and all(g['status']=='partial_scaling_complete' for g in groups) else 'stopped_after_failure',
                protocol=B.binding(root/PROTOCOL),groups=groups,error=error,wall_s=time.monotonic()-start,
                resource_samples=B.binding(sampler.path),exclude_training=True,formal_training_eligible=False,
                label=None,raw_label=None,new_training_labels=0,training_updates=0,retries=0,
                full_lift_certified=False,full_collection_ETA=None)
            B.verify(p['inputs']);B.verify(p['sources']);B.save(root/RESULTS,result,exclusive=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run']);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();result=freeze(args.output_dir) if args.action=='freeze' else run(args.output_dir)
    print(json.dumps(dict(status=result.get('status','frozen'),groups=len(result['groups']),
        protocol=str(args.output_dir/PROTOCOL)),indent=2))
    if args.action=='run' and result['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
