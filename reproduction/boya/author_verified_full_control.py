"""Four once-only complete CPU diagnostic trials after storage-only readiness."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import subprocess
import time
import traceback

import author_cpu_full_control as F
import author_contact_storage_control as STORAGE

B,R,C=F.B,F.R,F.C
SCHEMA='anydex_boya_verified_full_control_v1'
PROTOCOL='VERIFIED_FULL_PROTOCOL.json'
BASE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')
REFERENCE=BASE/'cpu_full_control_20261115_v1/CPU_FULL_PROTOCOL.json'


def freeze(root,readiness):
    ready=B.read(readiness)
    STORAGE.verify(Path(readiness).resolve().parent)
    if ready['schema']!=STORAGE.SCHEMA or ready['status']!='verified_storage_control'or ready['physics_steps']!=0 or ready['actual_Kit_starts']!=1:
        raise ValueError('Verified empty-Kit storage-only readiness required')
    for stage in ready['stages']:
        B.verify({'result':stage['result']});value=B.read(stage['result']['path'])
        if not value['groups'][-1]['all_verified']:raise ValueError('Every new storage writer must pass')
        if value['mode']=='kit'and(value['native_before']!=value['native_after']or value['native_before']['simulation_context_exists']
            or value['native_before']['timeline_playing']or value['native_before']['physics_scenes']):
            raise ValueError('Storage readiness must not integrate or create a simulation context')
    old=B.read(REFERENCE);B.verify(old['inputs']);B.verify(old['sources'])
    root=Path(root).resolve();sources=copy.deepcopy(old['sources'])
    for name in('author_verified_full_control.py','author_verified_full_execution.py','author_verified_full_audit.py',
                'author_verified_execution.py','author_contact_compressor.py','author_verified_contact_journal.py'):
        sources[name]=B.binding(B.HERE/name)
    inputs=copy.deepcopy(old['inputs']);inputs['storage_readiness']=B.binding(readiness)
    workers=[dict(worker_id=f'worker_{i:02d}',prepared_dir=old['prepared_template']['path'],lanes=1,
        stage_dir=str(root/'cpu_4'/f'worker_{i:02d}'),output_dir=str(root/'cpu_4'/f'worker_{i:02d}'/'result'))for i in range(4)]
    p=dict(schema=SCHEMA,groups=[dict(name='cpu_4',count=4,mode='cpu',workers=workers)],selected_count=4,
        command_steps=3024,prefix_ticks=0,full_trial=True,retry_budget=0,stage_timeout_s=900.,
        total_worker_attempt_budget=4,total_integrated_environment_steps=12096,total_dynamic_object_steps=36288,
        sources=sources,inputs=inputs,prepared_template=old['prepared_template'],
        worker_source=sources['author_verified_full_execution.py'],gpu_uuid=old['gpu_uuid'],gpu_identity=old['gpu_identity'],
        env=dict(old['env'],ANYDEX_CONTACT_SPOOL_ROOT='/tmp/anydex_verified_full_control_20261115_v1'),
        inherited_env=old['inherited_env'],resource_limits=dict(kit=4),sampling_interval_s=2.,
        cancel_sigint_grace_s=30.,cancel_sigterm_grace_s=5.,cancel_sigkill_grace_s=5.,
        diagnostic_only=True,label=None,raw_label=None,exclude_training=True,formal_training_eligible=False,
        automatic_training=False,audit_workers=2,
        scope='Known untouched3024-command template, each four processes once; storage-only overlay; not failed production-record retries or training labels')
    root.mkdir(parents=True,exist_ok=False);p['source_archive']=C.archive_sources(root,sources)
    B.save(root/PROTOCOL,p,exclusive=True);return verify(root)


def verify(root):
    root=Path(root).resolve();p=B.read(root/PROTOCOL)
    if(p['schema']!=SCHEMA or p['command_steps']!=3024 or p['selected_count']!=4
        or p['retry_budget']!=0 or p['total_worker_attempt_budget']!=4 or p['total_integrated_environment_steps']!=12096
        or p['stage_timeout_s']!=900. or p['exclude_training']is not True or p['audit_workers']!=2
        or len(p['groups'])!=1 or p['resource_limits']!=dict(kit=4)
        or len(p['groups'][0]['workers'])!=4 or p['groups'][0]['name']!='cpu_4'
        or p['groups'][0]['count']!=4 or p['groups'][0]['mode']!='cpu'
        or p['env']['CUDA_VISIBLE_DEVICES']!=p['gpu_uuid']or p['env']['NVIDIA_VISIBLE_DEVICES']!='0'):
        raise ValueError('Frozen four-trial storage-validation budget changed')
    B.verify(p['inputs']);B.verify(p['sources']);B.verify({'archive':p['source_archive']})
    for row in B.read(p['source_archive']['path'])['sources']:B.verify({'source':row['source'],'copy':row['archive']})
    for i,w in enumerate(p['groups'][0]['workers']):
        d=root/'cpu_4'/f'worker_{i:02d}'
        if w!=dict(worker_id=f'worker_{i:02d}',prepared_dir=p['prepared_template']['path'],lanes=1,stage_dir=str(d),output_dir=str(d/'result')):
            raise ValueError('Worker escaped frozen template/output')
    return p


def audit_one(root,worker,p,pool):
    i=worker['worker_id'];directory=root/'audits'/i;directory.mkdir(parents=True,exist_ok=False)
    output=directory/'AUDIT.json';command=[p['inputs']['python']['path'],'-B',p['sources']['author_verified_full_audit.py']['path'],
        '--worker',worker['receipt']['path'],'--output',str(output)]
    result=dict(worker_id=i,status='failed',started_unix_s=time.time())
    with pool.stage(('audit',),directory):
        with(directory/'process.log').open('xb')as log:
            process=pool.popen(command,cwd=B.ROOT,env=B._environment(p),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            result['process']=R.process_identity(process.pid);code=process.wait(timeout=900.)
        if output.exists():result['audit']=B.binding(output)
        result.update(returncode=code,status='pass'if code==0 and B.read(output)['status']=='full_integrity_pass'else'failed',finished_unix_s=time.time())
    B.save(directory/'STAGE.json',result,exclusive=True);return result


def run(root):
    root=Path(root).resolve();p=verify(root);group=None;audits=[];error=None;began=time.monotonic()
    if C.gpu_identity(p['inputs']['nvidia_smi']['path'],p['gpu_uuid'])!=p['gpu_identity']:
        raise ValueError('Frozen physical GPU0 identity changed')
    with R.lease(root/'DRIVER.lock'),R.lease(F.G.LOCK)as fd:
        if(root/'RUN_STARTED.json').exists():raise FileExistsError('No physical validation retry')
        with R.active_run(root,F.G.REGISTRY,B.binding(root/PROTOCOL)):
            B.save(root/'RUN_STARTED.json',dict(protocol=B.binding(root/PROTOCOL),process=R.process_identity(__import__('os').getpid()),unix_s=time.time()),exclusive=True)
            pool=C.ScalingPool(root/'resource_slots',fd,B.binding(root/PROTOCOL));sampler=C.ResourceSampler(root,p);sampler.start()
            try:
                group=F.run_group(root,p['groups'][0],dict(p,_protocol_path=str(root/PROTOCOL)),pool,sampler)
                if group['status']=='full_complete':
                    with ThreadPoolExecutor(max_workers=2)as executor:
                        audits=list(executor.map(lambda w:audit_one(root,w,p,pool),group['workers']))
            except BaseException:error=traceback.format_exc()
            finally:sampler.close()
            costs=[]
            for path in root.glob('cpu_4/*/result/env_0/execution/CONTACT_STORAGE_COST.json'):
                value=B.read(path);costs.append(dict(binding=B.binding(path),runtime_steps=value['original_runtime_steps'],trace_rows=value['observed_trace_rows'],contact_records=value['contact_records']))
            result=dict(schema=SCHEMA,status='full_integrity_pass'if group and group['status']=='full_complete'and len(audits)==4
                and all(a['status']=='pass'for a in audits)and not error else'stopped',group=group,audits=audits,error=error,
                cost_snapshots=costs,wall_s=time.monotonic()-began,label=None,exclude_training=True,new_training_labels=0,
                training_updates=0,retries=0,resource_samples=B.binding(sampler.path))
            verify(root);B.save(root/'VERIFIED_FULL_RESULTS.json',result,exclusive=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--readiness',type=Path);a=p.parse_args();result=freeze(a.root,a.readiness)if a.action=='freeze'else run(a.root)
    print({k:result.get(k)for k in('status','wall_s','new_training_labels')})
    if a.action=='run'and result['status']!='full_integrity_pass':raise SystemExit(1)
