"""Finite full-pipeline 1/16/64 comparison on the same eight fixed slots.

Every mode includes fresh camera/CGR/planning, physical return, complete native
journal, independent audit and diagnostic export. Unused lanes count as work.
"""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import author_collection_batch as B
import author_collection_wave_resources as R

ENTRY=B.HERE/'author_mjx_batch_collection_v8.py'
AUDIT=B.HERE/'audit_author_mjx_collection.py'
FINISH=B.HERE/'finish_author_mjx_collection.py'
MODES=(1,16,64)


def freeze(root, source, interruption_review, staging_result):
    import author_mjx_batch_collection_v8 as C
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    review=B.read(interruption_review);check=B.read(staging_result)
    if not review['owner_and_group_exited']or not review['global_GPU_lock_available']:raise ValueError('Interrupted owner not released')
    if check['status']!='read_only_staging_contract_verified'or not check['complete_native_field_copy_restored']:raise ValueError('Complete native copy contract required')
    prior=review['native_cost_interval'][1]+check['native_environment_steps_total_including_previous_checks']
    additional=dict(interrupted_single_mode=B.binding(interruption_review),graph_staging_check=B.binding(staging_result))
    modes=[]
    for n in MODES:
        destination=root/f'env_{n:03d}'
        p=C.freeze(destination,environments=n,source_protocol=source,prior_native_steps=(prior if n==1 else 0),additional_inputs=additional)
        cache=Path(source).parent/'warp_cache'
        shutil.copytree(cache,destination/'warp_cache')
        modes.append(dict(environments=n,root=str(destination),protocol=B.binding(destination/'PROTOCOL.json'),
                          max_native_environment_steps=p['max_native_environment_steps']))
    p=dict(schema='anydex_mjx_full_pipeline_comparison_v1',modes=modes,
           implementation={p.name:B.binding(p)for p in (Path(__file__),ENTRY,AUDIT,FINISH)},
           same_eight_slots=True,randomized_order=False,replicates=1,GPU_graph_mode="WARP_STAGED_EX",prior_failed_single_native_upper_bound=prior,interruption_review=B.binding(interruption_review),staging_result=B.binding(staging_result),
           high_modes_underfilled=True,all_idle_work_counted=True,retries=0,
           formal_collection=False,training_updates=0)
    B.save(root/'COMPARISON_PROTOCOL.json',p,exclusive=True)
    return p


def dispatch(command, directory):
    directory.mkdir(parents=True,exist_ok=False)
    B.save(directory/'DISPATCH.json',dict(command=command),exclusive=True)
    start=time.monotonic();began=time.time()
    with (directory/'process.log').open('x')as log:
        process=subprocess.Popen(command,cwd=B.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=R.process_identity(process.pid)
        B.save(directory/'PROCESS.json',dict(process=identity,started_unix_s=began),exclusive=True)
        try:code=process.wait(timeout=5400)
        except BaseException:
            if R.process_identity(process.pid)==identity:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)
            raise
    alive=R.group_alive(identity['pgid'])
    result=dict(returncode=code,wall_s=time.monotonic()-start,process=identity,
                process_group_exited=not alive,finished_unix_s=time.time())
    B.save(directory/'FINISHED.json',result,exclusive=True)
    if code or alive:raise RuntimeError('Stage did not complete cleanly: '+str(directory))
    return result


def run(root, number):
    root=Path(root).resolve();p=B.read(root/'COMPARISON_PROTOCOL.json')
    B.verify(p['implementation']);mode=next(x for x in p['modes']if x['environments']==number)
    B.verify({'protocol':mode['protocol']});directory=Path(mode['root'])
    if (directory/'PIPELINE_STARTED.json').exists():raise ValueError('No mode retry')
    B.save(directory/'PIPELINE_STARTED.json',dict(process=R.process_identity(os.getpid()),unix_s=time.time()),exclusive=True)
    start=time.monotonic();stages={}
    try:
        for key,command in (
            ('collect',[sys.executable,'-B',str(ENTRY),'run','--root',str(directory)]),
            ('audit',[sys.executable,'-B',str(AUDIT),'--root',str(directory)]),
            ('export',[sys.executable,'-B',str(FINISH),'--root',str(directory)])):
            stages[key]=dispatch(command,directory/'timing'/key)
        result=B.read(directory/'RESULTS.json');audit=B.read(directory/'READ_ONLY_AUDIT.json')
        export=B.read(directory/'export/MANIFEST.json');elapsed=time.monotonic()-start
        eligible=export['records'];records=result['records']
        summary=dict(status='complete',environments=number,slots=len(records),labels=eligible,
                     positive=export['positive'],negative=export['negative'],null=sum(r['label']is None for r in records),
                     full_pipeline_wall_s=elapsed,slots_per_hour=3600*len(records)/elapsed,
                     qualified_labels_per_hour=3600*eligible/elapsed,stages=stages,
                     grasp_attempts=result['cost']['grasp_attempts'],cost=result['cost'],
                     native_environment_integrations=result['world']['native_environment_integrations'],
                     all_complete_grasps_pass=audit['all_complete_grasps_pass'],
                     high_mode_underfilled=number>8,source_result=B.binding(directory/'RESULTS.json'),
                     source_audit=B.binding(directory/'READ_ONLY_AUDIT.json'),source_export=B.binding(directory/'export/MANIFEST.json'),
                     formal_training_eligible=False,training_updates=0)
    except BaseException as error:
        B.save(directory/'PIPELINE_FAILURE.json',dict(error=repr(error),stages=stages,wall_s=time.monotonic()-start),exclusive=True)
        raise
    B.save(directory/'FULL_PIPELINE_RESULTS.json',summary,exclusive=True)
    print(summary,flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','run'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-protocol',type=Path)
    p.add_argument('--environments',type=int,choices=MODES);p.add_argument('--interruption-review',type=Path);p.add_argument('--staging-result',type=Path);a=p.parse_args()
    if a.action=='freeze':freeze(a.root,a.source_protocol,a.interruption_review,a.staging_result)
    else:run(a.root,a.environments)
