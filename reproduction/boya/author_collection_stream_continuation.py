"""One explicitly authorized extra Kit; reuse the saved complete cold control."""
import argparse
import copy
import os
from pathlib import Path
import time
import traceback

import author_collection_stream_control as D
import author_collection_service_bound as SERVICE

B,R=D.B,D.R
SOURCE=D.BASE/'stream_collection_control_20261116_v1'
SCHEMA='anydex_boya_prebound_service_continuation_v1'


class SavedRoots:
    """Read the original cold artifacts without moving or rewriting them."""
    def __init__(self,root):self.root=Path(root)
    def __truediv__(self,key):
        parts=Path(key).parts
        if not parts or parts[0]not in ('cold','resident')or '..'in parts:raise ValueError('Comparison path outside fixed roots')
        base=SOURCE if parts[0]=='cold'else self.root
        return base.joinpath(*parts)


def freeze(root):
    D.verify(SOURCE);cold=B.read(SOURCE/'cold/RESULTS.json')
    failed=B.read(SOURCE/'resident/runtime/WORKER_RECEIPT.json')
    if cold['status']!='complete'or len(cold['audits'])!=2 or failed['status']!='closed_after_failure'or failed['Kit_starts']!=1:
        raise ValueError('Require the original complete cold and retained failed resident controls')
    if B.read(SOURCE/'resident/scenes/00/g1/LIFT_PIPELINE_EXECUTION.json')['physics_steps']!=0:
        raise ValueError('Previous failed execution cost differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);mode=root/'resident';mode.mkdir()
    p=copy.deepcopy(B.read(SOURCE/'resident/PROTOCOL.json'))
    sources=dict(p['sources']);sources[Path(__file__).name]=B.binding(__file__)
    sources[Path(SERVICE.__file__).name]=B.binding(SERVICE.__file__)
    sources['service_entry']=B.binding(SERVICE.__file__)
    jobs=[dict(role='capture',scene=str(mode/'scenes'/f'{i:02d}'/'build/scene.json'),
               output_dir=str(mode/'scenes'/f'{i:02d}'/'capture'))for i in range(2)]
    jobs += [dict(role='execute',prepared_dir=str(mode/'scenes'/f'{i:02d}'/'g1'))for i in range(2)]
    manifest=mode/'REPRESENTATION_MANIFEST.json'
    B.save(manifest,dict(objects=[dict(object_id=row['scene_id'],observation_json=str(mode/'scenes'/f'{i:02d}'/'capture/observation.json'),
        output_dir=str(mode/'representation/representations'/row['scene_id']))for i,row in enumerate(p['rows'])]),exclusive=True)
    approval=dict(source='User reply to fixed-budget request',answer='追加1次Kit完成比较',additional_Kit_starts=1,
                  max_new_environment_steps=17500,exclude_training=True,cold_baseline_reused=True)
    B.save(root/'APPROVAL.json',approval,exclusive=True)
    p.update(sources=sources,worker_jobs=jobs,max_Kit_starts=1,continuation_schema=SCHEMA)
    p['inputs']=dict(p['inputs'],stream_manifest=B.binding(manifest),additional_budget_approval=B.binding(root/'APPROVAL.json'),
        saved_cold_result=B.binding(SOURCE/'cold/RESULTS.json'),previous_failed_control=B.binding(SOURCE/'RESULTS.json'))
    p['env']=dict(p['env'],ANYDEX_CONTACT_SPOOL_ROOT='/dev/shm/anydex_stream_bound_control_20261116_v1/resident')
    p['source_archive']=D.REFERENCE.C.archive_sources(mode,sources)
    B.save(mode/'PROTOCOL.json',p,exclusive=True)
    control=dict(schema=SCHEMA,protocol=B.binding(mode/'PROTOCOL.json'),approval=B.binding(root/'APPROVAL.json'),
        previous_control=B.binding(SOURCE/'CONTROL.json'),previous_result=B.binding(SOURCE/'RESULTS.json'),
        saved_cold_result=B.binding(SOURCE/'cold/RESULTS.json'),additional_Kit_starts=1,
        known_previous_environment_steps=10138,max_new_environment_steps=17500,max_combined_environment_steps=27638,
        original_environment_step_budget=35000,exclude_training=True,retry_budget=0,new_training_labels=0,training_updates=0)
    B.save(root/'CONTROL.json',control,exclusive=True);return verify(root)


def verify(root):
    root=Path(root).resolve();control=B.read(root/'CONTROL.json')
    if(control['schema']!=SCHEMA or control['additional_Kit_starts']!=1 or control['max_new_environment_steps']!=17500
       or control['max_combined_environment_steps']!=27638 or control['original_environment_step_budget']!=35000
       or control['exclude_training']is not True or control['retry_budget']!=0):raise ValueError('Additional finite budget changed')
    B.verify({k:control[k]for k in ('protocol','approval','previous_control','previous_result','saved_cold_result')})
    approval=B.read(control['approval']['path'])
    if approval['additional_Kit_starts']!=1 or approval['answer']!='追加1次Kit完成比较':raise ValueError('Explicit one-Kit approval required')
    p=B.read(control['protocol']['path']);B.verify(p['inputs']);B.verify(p['sources']);B.verify({'archive':p['source_archive']})
    for row in B.read(p['source_archive']['path'])['sources']:B.verify({'source':row['source'],'archive':row['archive']})
    old=B.read(SOURCE/'resident/PROTOCOL.json')
    if(p['rows']!=old['rows']or p['mode']!='resident'or p['max_Kit_starts']!=1 or p['max_environment_steps']!=17500
       or p['max_execution_steps_per_job']!=8000 or p['exclude_training']is not True or p['diagnostic_only']is not True
       or p['retry_budget']!=0 or p['sources']['service_entry']!=B.binding(SERVICE.__file__)
       or [j['role']for j in p['worker_jobs']]!=['capture','capture','execute','execute']):raise ValueError('Original two-scene/g1 scope changed')
    for i,j in enumerate(p['worker_jobs']):
        expected=root/'resident/scenes'/f'{i%2:02d}'
        if j['role']=='capture':
            if j['scene']!=str(expected/'build/scene.json')or j['output_dir']!=str(expected/'capture'):raise ValueError('Capture output escaped new root')
        elif j['prepared_dir']!=str(expected/'g1'):raise ValueError('Do not resume original failed executions')
    return control


def run(root):
    root=Path(root).resolve();control=verify(root);p=B.read(control['protocol']['path']);began=time.monotonic()
    if D.REFERENCE.C.gpu_identity(p['inputs']['nvidia_smi']['path'],p['gpu_uuid'])!=p['gpu_identity']:raise ValueError('Physical GPU0 changed')
    with R.lease(root/'DRIVER.lock'),R.lease(D.REFERENCE.P.OLD.LEGACY_LOCK)as fd:
        if(root/'RUN_STARTED.json').exists():raise FileExistsError('One extra Kit only; never restart a started control')
        with R.active_run(root,D.REFERENCE.P.OLD.REGISTRY,B.binding(root/'CONTROL.json')):
            B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid()),started_unix_s=time.time()),exclusive=True)
            pool=R.ResourcePool(root/'resource_slots',fd,B.binding(root/'CONTROL.json'))
            previous={k:os.environ.get(k)for k in p['env']};os.environ.update(p['env'])
            try:
                with R.install_worker_adapters(pool,D.REFERENCE.P.OLD.LEGACY_LOCK):result=D.run_mode(root/'resident',pool,fd)
            finally:
                for k,v in previous.items():
                    if v is None:os.environ.pop(k,None)
                    else:os.environ[k]=v
            comparison=None
            if result['status']=='complete':
                try:
                    comparison=D.compare(SavedRoots(root),[B.read(control['saved_cold_result']['path']),result])
                    comparison['scope']='Saved completed cold baseline versus new prebound resident group; fixed two scenes/g1, order/cache effects remain, not production yield/ETA'
                except BaseException:comparison=dict(comparison_qualified=False,error=traceback.format_exc())
            value=dict(schema=SCHEMA,status='complete'if comparison and comparison['comparison_qualified']else'stopped',
                resident=result,saved_cold_result=control['saved_cold_result'],comparison=comparison,wall_s=time.monotonic()-began,
                exclude_training=True,new_training_labels=0,training_updates=0,retries=0)
            verify(root);B.save(root/'RESULTS.json',value,exclusive=True)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();v=freeze(a.root)if a.action=='freeze'else run(a.root)
    print({k:v.get(k)for k in ('status','wall_s','comparison')},flush=True)
    if a.action=='run'and v['status']!='complete':raise SystemExit(1)
