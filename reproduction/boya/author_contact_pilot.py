"""Two fixed fresh g1 scenes with preclosure mapping, no retries or training."""
import argparse
import copy
from contextlib import contextmanager
import os
from pathlib import Path
import time
import traceback

import author_collection_batch as B
import author_collection_wave_resources as R
import author_scaling_control as C
import author_contact_pipeline as M
import author_verified_contact_journal as J

SCHEMA='anydex_boya_preclosure_two_scene_pilot_v1'
LEGACY_LOCK=B.ROOT/'outputs/author_collection_gpu0.lock'
REGISTRY=LEGACY_LOCK.with_suffix('.wave_active.json')
GPU_UUID='GPU-4d3badb8-5fb0-b880-f08e-6407f0093e86'


def freeze(source_batch,root):
    source=Path(source_batch)/'BATCH_PROTOCOL.json';p=copy.deepcopy(B.read(source))
    if list(Path(source_batch).rglob('STAGE.json')):raise ValueError('Source manual queue must be unstarted')
    root=Path(root).resolve();rows=[copy.deepcopy(p['queue']['scenes'][i])for i in (3,4)]
    if not all(r['status']=='ready'and r['split']=='fit'for r in rows):raise ValueError('Original fixed fresh layouts changed')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('author_contact_pipeline.py','author_fast_contact_alignment.py','author_contact_pilot.py',
        'author_verified_execution.py','author_verified_contact_journal.py','author_contact_compressor.py',
        'audit_author_resident_collection.py','author_collection_wave_resources.py','author_scaling_control.py'):
        p['sources'][name]=B.binding(B.HERE/name)
    p['sources']['execution_entry']=B.binding(M.__file__)
    p['inputs']['source_manual_batch']=B.binding(source)
    p['inputs']['nvidia_smi']=B.binding('/usr/bin/nvidia-smi')
    p.update(schema=SCHEMA,rows=rows,source_queue_indices=[3,4],gestures=[1],max_Kit_starts=4,
        max_capture_attempts=2,max_execution_attempts=2,max_execution_steps_per_job=8000,
        max_environment_steps=17500,retry_budget=0,automatic_training=False,old_labels_merged=False,
        label_scope='Separate new preclosure g1 pilot; never automatically merged into old mapping data',
        opponent_order=list(M.OPPONENTS),source_candidate_budget=1,planning_fallback=False,
        stage_timeout_s=1800.,gpu_uuid=GPU_UUID,gpu_identity=C.gpu_identity('/usr/bin/nvidia-smi',GPU_UUID),
        closure='Original open-to-wrap 250 steps, then 50 hold; virtual preclosure is not execution stop')
    p['env']=dict(p['env'],CUDA_VISIBLE_DEVICES=GPU_UUID,NVIDIA_VISIBLE_DEVICES=GPU_UUID,
        ANYDEX_CONTACT_SPOOL_ROOT=str(root/'contact_spool'))
    B.verify(p['inputs']);B.verify(p['sources']);p['source_archive']=C.archive_sources(root,p['sources'])
    B.save(root/'PROTOCOL.json',p,exclusive=True)
    return p


def verify(root):
    p=B.read(Path(root)/'PROTOCOL.json')
    if (p['schema']!=SCHEMA or p['source_queue_indices']!=[3,4]or p['gestures']!=[1]
        or p['max_Kit_starts']!=4 or p['max_environment_steps']!=17500 or p['retry_budget']!=0
        or p['max_execution_steps_per_job']!=8000 or p['automatic_training']is not False):
        raise ValueError('Fixed pilot budget changed')
    B.verify(p['inputs']);B.verify(p['sources']);B.verify({'archive':p['source_archive']})
    for item in B.read(p['source_archive']['path'])['sources']:B.verify({'source':item['source'],'copy':item['archive']})
    return p


@contextmanager
def classify_new_stages():
    original=R.classify
    def classify(command,protocol):
        if str(B.HERE/'author_contact_pipeline.py')in command:return ('prepare',)if 'prepare'in command else ('kit',)
        if str(B.HERE/'audit_author_resident_collection.py')in command:return ('audit',)
        return original(command,protocol)
    R.classify=classify
    try:yield
    finally:R.classify=original


def stage(root,name,command,expected,p):
    result=B._stage(root/'stages'/name,command,expected,p)
    if result['status']!='complete':raise RuntimeError(name+': '+str(result.get('error')))
    return result


def one(root,index,p):
    row=p['rows'][index];directory=root/'scenes'/f'{index:02d}';capture=directory/'capture';py=p['inputs']['python']['path']
    result=dict(index=index,source_queue_index=p['source_queue_indices'][index],scene_id=row['scene_id'],status='started',label=None)
    try:
        build=directory/'build'
        stage(directory,'build',[py,'-B',p['sources']['scene_entry']['path'],'build','--scene-spec',row['scene_spec']['path'],
            '--output-dir',str(build)],[build/'scene.json'],p)
        B._scene_objects(build/'scene.json',row)
        stage(directory,'capture',[py,'-B',p['sources']['capture_entry']['path'],'--scene',str(build/'scene.json'),
            '--output-dir',str(capture)],[capture/n for n in ('RESULTS.json','observation.json','observed_scene.json','CONTACT_JOURNAL.json')],p)
        result['initialization_ticks']=750;B._scene_objects(capture/'observed_scene.json',row)
        output=capture/'INDEPENDENT_CAPTURE_AUDIT.json'
        stage(directory,'capture_audit',[py,'-B',str(B.HERE/'audit_author_resident_collection.py'),'capture',
            '--capture-dir',str(capture),'--output',str(output)],[output],p)
        if B.read(output)['status']!='pass':raise ValueError('Complete initialization audit failed')
        manifest=directory/'REPRESENTATION_MANIFEST.json';rep=directory/'representation'
        B.save(manifest,dict(objects=[dict(object_id=row['scene_id'],observation_json=str(capture/'observation.json'),
            observation_sha256=B.sha(capture/'observation.json'))]),exclusive=True)
        target=rep/'representations'/row['scene_id']
        stage(directory,'representation',[py,'-B',str(B.HERE/'run_collection_representation.py'),'--manifest',str(manifest),
            '--run-dir',str(rep),'--device','cuda:0','--seed',str(p['queue']['seed']),'--checkpoint',p['inputs']['checkpoint']['path'],
            '--native-deps',p['native_deps'],'--depth-min','.2','--depth-max','.65'],[rep/'BATCH_REPRESENTATION.json',target/'RESULTS.json',target/'raw_camera_candidates.npz'],p)
        receipt=B.read(rep/'BATCH_REPRESENTATION.json')
        if receipt['network_forward_count']!=1 or receipt['hand_head_load_count']!=0:raise ValueError('Original single CGR required')
        domain=directory/'DOMAIN.json';B.save(domain,p['domain'],exclusive=True);outer=directory/'g1_contact';prepared=outer/'pipeline'
        stage(directory,'prepare',[py,'-B',str(B.HERE/'author_contact_pipeline.py'),'prepare','--scene',str(capture/'observed_scene.json'),
            '--observation',str(capture/'observation.json'),'--representation-dir',str(target),'--seed',str(p['queue']['seed']),
            '--calibration',p['inputs']['calibration']['path'],'--cloud-manifest',p['inputs']['cloud_manifest']['path'],
            '--backend',p['inputs']['backend']['path'],'--domain',str(domain),'--output-dir',str(outer)],[prepared/'LIFT_PIPELINE_RESULTS.json'],p)
        prep=B.read(prepared/'LIFT_PIPELINE_RESULTS.json');result['preparation']=B.binding(prepared/'LIFT_PIPELINE_RESULTS.json')
        if prep['status']!='ready':result['status']=prep['status'];return result
        mapping=M.audit_mapping(prepared);B.save(outer/'INDEPENDENT_MAPPING_AUDIT.json',mapping,exclusive=True)
        ready=B.read(prepared/'READY.json');plan=B.read(ready['plan']['path'])
        if plan['total_command_steps']>p['max_execution_steps_per_job']:raise ValueError('Prepared execution exceeds fixed 8000-step budget')
        result['planned_execution_steps']=plan['total_command_steps']
        stage(directory,'execute',[py,'-B',str(B.HERE/'author_contact_pipeline.py'),'execute','--prepared-dir',str(prepared)],
            [prepared/'COLLECTION_RECORD.json',prepared/'LIFT_PIPELINE_EXECUTION.json'],p)
        output=prepared/'RUNTIME_AUDIT.json'
        stage(directory,'runtime_audit',[py,'-B',p['sources']['audit_entry']['path'],'--prepared-dir',str(prepared),
            '--split-manifest',p['queue']['split_manifest']['path'],'--output',str(output)],[output],p)
        with M.installed():trial=B.validate_trial(prepared,row)
        B.certify_audit(B.read(output),trial,row)
        J.verify_pointer(B.read(prepared/'execution/MULTI_OBJECT_CONTACTS.json'))
        result.update(status='labelled',label=trial['label'],trial=trial,runtime_audit=B.binding(output),mapping_audit=B.binding(outer/'INDEPENDENT_MAPPING_AUDIT.json'))
    except BaseException:
        result.update(status='infrastructure_failure',error=traceback.format_exc())
        if isinstance(__import__('sys').exc_info()[1],(KeyboardInterrupt,SystemExit)):raise
    finally:B.save(directory/'OUTCOME.json',result,exclusive=True)
    return result


def run(root):
    root=Path(root).resolve();p=verify(root);began=time.monotonic();rows=[]
    if C.gpu_identity(p['inputs']['nvidia_smi']['path'],p['gpu_uuid'])!=p['gpu_identity']:raise ValueError('Physical GPU0 changed')
    with R.lease(root/'DRIVER.lock'),R.lease(LEGACY_LOCK)as fd,R.active_run(root,REGISTRY,B.binding(root/'PROTOCOL.json')):
        B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid()),started_unix_s=time.time()),exclusive=True)
        pool=R.ResourcePool(root/'resource_slots',fd,B.binding(root/'PROTOCOL.json'))
        with classify_new_stages(),R.install_worker_adapters(pool,LEGACY_LOCK):
            for index in range(2):
                rows.append(one(root,index,p));B.save(root/'PROGRESS.json',dict(rows=rows,wall_s=time.monotonic()-began))
                if rows[-1]['status']=='infrastructure_failure':break
        verify(root)
        result=dict(schema=SCHEMA,status='complete'if len(rows)==2 and all(r['status']!='infrastructure_failure'for r in rows)else'stopped',
            rows=rows,wall_s=time.monotonic()-began,new_labels=sum(r['status']=='labelled'for r in rows),
            training_updates=0,bulk_collection_started=False,retries=0)
        B.save(root/'RESULTS.json',result,exclusive=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source-batch',type=Path)
    a=parser.parse_args();result=freeze(a.source_batch,a.root)if a.action=='freeze'else run(a.root)
    print({k:result.get(k)for k in ('schema','status','wall_s','new_labels')},flush=True)
