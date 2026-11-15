"""First numerical audits of four saved executions; never launches physics.

The original controller selected a Kit-only resource pool for audits. Its error
and unspawned quarantine incidents remain unchanged. This separate protocol
uses the original two-audit ResourcePool and the unchanged full numerical CLI.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import time
import traceback

import author_verified_full_control_v2 as F

OLD=F.OLD
B,R,C=OLD.B,OLD.R,OLD.C
SCHEMA='anydex_boya_verified_saved_full_audits_v1'


def freeze(root, execution_root):
    execution_root=Path(execution_root).resolve();p=F.verify(execution_root)
    result=B.read(execution_root/'VERIFIED_FULL_RESULTS.json')
    if result['group']['status']!='full_complete'or result['audits']!=[]:
        raise ValueError('Four complete saved executions with no numeric audit required')
    if any((execution_root/'audits').glob('*/AUDIT.json'))or any((execution_root/'audits').glob('*/STAGE.json')):
        raise ValueError('An original audit may already have started')
    incidents=list((execution_root/'resource_slots/quarantine').glob('*.json'))
    if not incidents:raise ValueError('Expected original unspawned audit-pool incidents')
    for file in incidents:
        value=B.read(file)
        if value.get('spawn_attempted')is not False or value.get('pgid')is not None:
            raise ValueError('Cannot classify original audit launch as unspawned')
    sources=dict(p['sources']);sources[Path(__file__).name]=B.binding(__file__)
    inputs=dict(p['inputs'],execution_protocol=B.binding(execution_root/OLD.PROTOCOL),
                stopped_execution_result=B.binding(execution_root/'VERIFIED_FULL_RESULTS.json'))
    for i,file in enumerate(incidents):inputs[f'unspawned_incident_{i}']=B.binding(file)
    value=dict(schema=SCHEMA,execution_root=str(execution_root),workers=result['group']['workers'],
               sources=sources,inputs=inputs,env=p['env'],inherited_env=p['inherited_env'],
               audit_workers=2,audit_attempts=4,Kit_starts=0,physics_steps=0,retries=0,
               new_training_labels=0,exclude_training=True)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    value['source_archive']=C.archive_sources(root,sources)
    B.save(root/'PROTOCOL.json',value,exclusive=True)
    return verify(root)


def verify(root):
    root=Path(root).resolve();p=B.read(root/'PROTOCOL.json')
    if(p['schema']!=SCHEMA or p['audit_workers']!=2 or p['audit_attempts']!=4 or p['Kit_starts']!=0
       or p['physics_steps']!=0 or p['retries']!=0 or p['exclude_training']is not True or len(p['workers'])!=4):
        raise ValueError('Saved-result audit budget changed')
    B.verify(p['inputs']);B.verify(p['sources']);B.verify({'archive':p['source_archive']})
    for row in B.read(p['source_archive']['path'])['sources']:
        B.verify({'source':row['source'],'archive':row['archive']})
    return p


def run(root):
    root=Path(root).resolve();p=verify(root);audits=[];error=None;began=time.monotonic()
    with R.lease(root/'DRIVER.lock')as fd:
        if(root/'RUN_STARTED.json').exists():raise FileExistsError('No audit retry')
        B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid()),protocol=B.binding(root/'PROTOCOL.json')),exclusive=True)
        # The original ResourcePool supports audit=2; ScalingPool supports Kit only.
        pool=R.ResourcePool(root/'resource_slots',fd,B.binding(root/'PROTOCOL.json'))
        try:
            with ThreadPoolExecutor(max_workers=2)as executor:
                audits=list(executor.map(lambda w:OLD.audit_one(root,w,p,pool),p['workers']))
        except BaseException:error=traceback.format_exc()
        result=dict(schema=SCHEMA,status='saved_full_integrity_pass'if len(audits)==4 and all(a['status']=='pass'for a in audits)and not error else'stopped',
                    audits=audits,error=error,wall_s=time.monotonic()-began,Kit_starts=0,physics_steps=0,
                    new_training_labels=0,training_updates=0,retries=0,production_resumed=False,
                    original_execution_result_preserved=True)
        verify(root);B.save(root/'RESULTS.json',result,exclusive=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--execution-root',type=Path)
    a=parser.parse_args();value=freeze(a.root,a.execution_root)if a.action=='freeze'else run(a.root)
    print({k:value.get(k)for k in ('status','wall_s','physics_steps')},flush=True)
    if a.action=='run'and value['status']!='saved_full_integrity_pass':raise SystemExit(1)
