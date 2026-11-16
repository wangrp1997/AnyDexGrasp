"""Review an unspawned CGR interruption, then finish only the remaining scene.

Original source/protocol/stages remain immutable. Reuses the already complete
first label and second capture. No capture, trial or started stage is retried.
"""
import argparse
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import time
import traceback

import author_contact_pilot as P

B,R=P.B,P.R


def review(root):
    p=P.verify(root);started=B.read(root/'RUN_STARTED.json');old=root/'resource_slots/observation/0.json'
    state=B.read(old);stage=root/'scenes/01/stages/representation'
    if (state['state']!='reserved'or state.get('process')is not None
        or state['stage_dir']!=str(stage)or (stage/'STAGE.json').exists()or(stage/'process.log').exists()):
        raise ValueError('Only the source-proven pre-Popen interruption is resumable')
    identities=[started['process']];slots=[]
    for file in (root/'resource_slots').glob('*/*.json'):
        value=B.read(file);slots.append(dict(path=str(file),value=value))
        if value.get('process'):identities.append(value['process'])
        if file!=old and value['state']!='released':raise ValueError('Another unresolved slot')
    for identity in identities:
        if R.process_identity(identity['pid'])==identity or R.group_alive(identity['pgid']):
            raise ValueError('An original process group is still alive')
    first=B.read(root/'scenes/00/OUTCOME.json')
    if first['status']!='labelled':raise ValueError('Completed audited first label required')
    for name in ('build','capture','capture_audit'):
        receipt=B.read(root/'scenes/01/stages'/name/'STAGE.json')
        if receipt['status']!='complete':raise ValueError('Second capture prefix must be complete')
        B.verify(receipt['outputs'])
    audit=root/'scenes/01/capture/INDEPENDENT_CAPTURE_AUDIT.json'
    if B.read(audit)['status']!='pass':raise ValueError('Saved second capture must be certified')
    evidence=dict(status='reviewed_unspawned_CGR_dispatch',driver=started,resource_slots_before=slots,
        no_original_process_groups_alive=True,CGR_STAGE_and_process_log_absent=True,
        proof='Pinned B._stage persists STAGE and opens process.log before the only Popen; neither exists',
        original_worker_adapters=B.binding(R.__file__),original_stage_source=B.binding(B.__file__),
        second_capture_audit=B.binding(audit),first_outcome=B.binding(root/'scenes/00/OUTCOME.json'),
        consumed_Kit_starts=3,consumed_environment_steps=5317,remaining_Kit_starts=1,
        remaining_executions=1,max_remaining_execution_steps=8000,review_unix_s=time.time())
    path=root/'INTERRUPTION_REVIEW.json';B.save(path,evidence,exclusive=True)
    state.update(state='released',finished_unix_s=time.time(),resolution=B.binding(path),
        resolution_reason='Reviewed pre-Popen dispatch, no stage or process log; every known original group exited')
    B.save(old,state);R.assert_quiescent(root)
    return p


@contextmanager
def preserve_completed_prefix():
    original_save,original_stage=B.save,P.stage
    def save(path,value,*,exclusive=False):
        if exclusive and Path(path).name=='REPRESENTATION_MANIFEST.json'and Path(path).exists():
            if B.read(path)!=value:raise ValueError('Saved representation manifest changed')
            return
        return original_save(path,value,exclusive=exclusive)
    def stage(directory,name,command,expected,p):
        if name=='representation':name='representation_first_dispatch_after_interruption'
        return original_stage(directory,name,command,expected,p)
    B.save,P.stage=save,stage
    try:yield
    finally:B.save,P.stage=original_save,original_stage


def run(root):
    root=Path(root).resolve();new=root/'continuation';new.mkdir(exist_ok=False)
    with R.lease(new/'DRIVER.lock'),R.lease(P.LEGACY_LOCK)as fd:
        p=review(root);p=copy.deepcopy(p);p['sources'][Path(__file__).name]=B.binding(__file__)
        protocol=dict(schema='anydex_boya_preclosure_remaining_scene_continuation_v1',source_protocol=B.binding(root/'PROTOCOL.json'),
            reviewed_interruption=B.binding(root/'INTERRUPTION_REVIEW.json'),implementation=B.binding(__file__),
            remaining_scene_index=1,new_capture_starts=0,max_new_Kit_starts=1,max_new_environment_steps=8000,
            original_max_Kit_starts=4,original_max_environment_steps=17500,started_stage_retries=0,
            source_queue_unchanged=True,source_bindings=p['sources'])
        B.save(new/'PROTOCOL.json',protocol,exclusive=True)
        with R.active_run(new,P.REGISTRY,B.binding(new/'PROTOCOL.json')):
            B.save(new/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid()),started_unix_s=time.time()),exclusive=True)
            pool=R.ResourcePool(new/'resource_slots',fd,B.binding(new/'PROTOCOL.json'))
            with preserve_completed_prefix(),P.classify_new_stages(),R.install_worker_adapters(pool,P.LEGACY_LOCK):
                second=P.one(root,1,p)
            rows=[B.read(root/'scenes/00/OUTCOME.json'),second];P.verify(root);B.verify(p['sources'])
            result=dict(schema=P.SCHEMA,status='complete'if second['status']!='infrastructure_failure'else'stopped',rows=rows,
                wall_s=time.time()-B.read(root/'RUN_STARTED.json')['started_unix_s'],new_labels=sum(r['status']=='labelled'for r in rows),
                controller_was_interrupted=True,continuation_protocol=B.binding(new/'PROTOCOL.json'),
                interruption_review=B.binding(root/'INTERRUPTION_REVIEW.json'),training_updates=0,bulk_collection_started=False,retries=0)
            B.save(new/'RESULTS.json',result,exclusive=True);B.save(root/'RESULTS.json',result,exclusive=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();r=run(args.root);print({k:r[k]for k in ('status','new_labels','wall_s')},flush=True)
