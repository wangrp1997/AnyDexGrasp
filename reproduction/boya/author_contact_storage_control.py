"""Frozen concurrent storage replay, before/inside one empty Kit; zero physics."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import author_pipeline_collection as P
import author_contact_compressor as W

B,R,C=P.B,P.R,P.C
SCHEMA='anydex_boya_contact_storage_control_v1'
PROTOCOL='CONTACT_STORAGE_PROTOCOL.json'
BASE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')
REFERENCE=BASE/'production_pilot_20261115_continuation_v1/run/PRODUCTION_PROTOCOL.json'
GOOD_TRIAL=BASE/'production_pilot_20261115_continuation_v1/run/windows/01/scenes/0000011_s40_lift_fit_00006_s2026111501/g1'


def freeze(root):
    old=B.read(REFERENCE);B.verify(old['inputs']);B.verify(old['sources'])
    window=B.read(B.read(old['window_index']['path'])['windows'][0]['protocol']['path'])
    B.verify({'runtime_gpu_sampler':window['inputs']['nvidia_smi']})
    certificate=B.read(GOOD_TRIAL/'RUNTIME_AUDIT.json')
    if certificate['status']!='pass':raise ValueError('Previously certified contact source required')
    pointer=B.read(GOOD_TRIAL/'execution/MULTI_OBJECT_CONTACTS.json');B.verify({'data':pointer['data']})
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    B.save(root/'REQUEST.json',dict(writers=4,records=1024,algorithms=['original_gzip','verified_isolated'],
        modes=['standalone','kit'],max_Kit_attempts=1,max_physics_steps=0,retry_budget=0,
        labels=None,exclude_training=True,policy='Replay existing complete data bytes; no simulation, candidates, training or label admission'),exclusive=True)
    raw=root/'REFERENCE.jsonl';logical=hashlib.sha256();size=0;records=0
    with gzip.open(pointer['data']['path'],'rb')as source,raw.open('xb')as output:
        for i,line in enumerate(source,1):
            row=json.loads(line)
            canonical=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
            if canonical!=line or row['step']!=i:raise ValueError('Certified source is not exact canonical complete JSONL')
            output.write(line);logical.update(line);size+=len(line);records=i
            if i==1024:break
        output.flush();os.fsync(output.fileno())
    if records!=1024:raise ValueError('Fixed source prefix unavailable')
    fixture=dict(raw=B.binding(raw),logical=dict(sha256=logical.hexdigest(),bytes=size,records=records),
        original_pointer=B.binding(GOOD_TRIAL/'execution/MULTI_OBJECT_CONTACTS.json'),
        original_certificate=B.binding(GOOD_TRIAL/'RUNTIME_AUDIT.json'),new_physics_steps=0,new_labels=0)
    sources=copy.deepcopy(old['sources'])
    for name in('author_contact_compressor.py','author_verified_contact_journal.py',
                'author_contact_storage_probe.py','author_contact_storage_control.py'):
        sources[name]=B.binding(B.HERE/name)
    inputs=copy.deepcopy(old['inputs']);inputs.update(reference_protocol=B.binding(REFERENCE),
        fixture=B.binding(raw),source_pointer=fixture['original_pointer'],source_certificate=fixture['original_certificate'],
        nvidia_smi=window['inputs']['nvidia_smi'])
    p=dict(schema=SCHEMA,fixture=fixture,inputs=inputs,sources=sources,
        source_sha256={v['path']:v['sha256']for v in sources.values()},writers=4,
        spool_root='/tmp/anydex_contact_storage_control_20261115_v1',modes=['standalone','kit'],
        max_Kit_attempts=1,max_physics_steps=0,retry_budget=0,stage_timeout_s=900.,
        cancel_sigint_grace_s=30.,cancel_sigterm_grace_s=5.,cancel_sigkill_grace_s=5.,sampling_interval_s=2.,
        gpu_uuid=P.GPU_UUID,gpu_identity=C.gpu_identity(inputs['nvidia_smi']['path'],P.GPU_UUID),
        env=window['env'],inherited_env=window['inherited_env'],resource_limits=dict(kit=1),
        label=None,exclude_training=True,automatic_training=False,synthetic_contact_replays_not_physics=True)
    p['source_archive']=C.archive_sources(root,sources);B.save(root/PROTOCOL,p,exclusive=True)
    return verify(root)


def verify(root):
    root=Path(root).resolve();p=B.read(root/PROTOCOL)
    if(p['schema']!=SCHEMA or p['writers']!=4 or p['fixture']['logical']['records']!=1024
        or p['modes']!=['standalone','kit']or p['max_Kit_attempts']!=1 or p['max_physics_steps']!=0
        or p['retry_budget']!=0 or p['stage_timeout_s']!=900. or p['resource_limits']!=dict(kit=1)
        or p['exclude_training']is not True or p['gpu_uuid']!=P.GPU_UUID
        or p['env']['CUDA_VISIBLE_DEVICES']!=P.GPU_UUID or p['env']['NVIDIA_VISIBLE_DEVICES']!='0'):
        raise ValueError('Frozen storage-only budget/GPU0 identity changed')
    B.verify(p['inputs']);B.verify(p['sources']);B.verify({'archive':p['source_archive']})
    archive=B.read(p['source_archive']['path'])
    for row in archive['sources']:B.verify({'original':row['source'],'copy':row['archive']})
    if {(r['source']['path'],r['source']['sha256'])for r in archive['sources']}!={(r['path'],r['sha256'])for r in p['sources'].values()}:
        raise ValueError('Runtime archive incomplete')
    if B.sha(p['fixture']['raw']['path'])!=p['fixture']['logical']['sha256']:raise ValueError('Exact replay fixture changed')
    return p


def run(root):
    root=Path(root).resolve();p=verify(root);stages=[];error=None;began=time.monotonic()
    with R.lease(root/'DRIVER.lock'),R.lease(P.OLD.LEGACY_LOCK)as fd:
        if(root/'RUN_STARTED.json').exists():raise FileExistsError('No storage-probe restart')
        with R.active_run(root,P.OLD.REGISTRY,B.binding(root/PROTOCOL)):
            B.save(root/'RUN_STARTED.json',dict(protocol=B.binding(root/PROTOCOL),process=R.process_identity(os.getpid()),unix_s=time.time()),exclusive=True)
            pool=P.ExpandedPool(root/'resource_slots',fd,B.binding(root/PROTOCOL));sampler=C.ResourceSampler(root,p);sampler.start()
            try:
                for mode in p['modes']:
                    verify(root);directory=root/'stages'/mode;directory.mkdir(parents=True,exist_ok=False)
                    output=root/mode
                    command=[p['inputs']['python']['path'],'-B',p['sources']['author_contact_storage_probe.py']['path'],
                        '--root',str(output),'--protocol',str(root/PROTOCOL),'--mode',mode]
                    stage=dict(mode=mode,status='starting',command=command,physics_steps=0,retry_count=0,started_unix_s=time.time())
                    B.save(directory/'STARTED.json',stage,exclusive=True)
                    with pool.stage(('kit',)if mode=='kit'else(),directory):
                        with(directory/'process.log').open('xb')as log:
                            process=pool.popen(command,env=B._environment(p),cwd=B.ROOT,stdout=log,
                                stderr=__import__('subprocess').STDOUT,start_new_session=True)
                            stage.update(status='running',process=R.process_identity(process.pid));sampler.add(stage['process'])
                            B.save(directory/'STAGE.json',stage)
                            code=process.wait(timeout=p['stage_timeout_s'])
                        stage.update(returncode=code,status='complete'if code==0 else'failed',finished_unix_s=time.time())
                    if(output/'RESULTS.json').exists():stage['result']=B.binding(output/'RESULTS.json')
                    B.save(directory/'STAGE.json',stage);stages.append(stage)
                    if code!=0 or B.read(output/'RESULTS.json')['status']!='verified_storage':break
            except BaseException:error=traceback.format_exc()
            finally:sampler.close()
            result=dict(schema=SCHEMA,status='verified_storage_control'if len(stages)==2 and not error and all(s['status']=='complete'for s in stages)else'stopped',
                stages=stages,error=error,wall_s=time.monotonic()-began,physics_steps=0,max_Kit_attempts=1,
                actual_Kit_starts=sum(B.read(s['result']['path'])['Kit_starts']for s in stages if s.get('result')),
                resource_samples=B.binding(sampler.path),label=None,exclude_training=True,new_training_labels=0)
            B.save(root/'CONTACT_STORAGE_RESULTS.json',result,exclusive=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();result=freeze(a.root)if a.action=='freeze'else run(a.root)
    print({k:result.get(k)for k in('status','physics_steps','actual_Kit_starts')})
    if a.action=='run'and result['status']!='verified_storage_control':raise SystemExit(1)
