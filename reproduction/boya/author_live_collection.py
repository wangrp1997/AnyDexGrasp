"""Finite same-world collection control, persistent CGR, pipelined audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import author_contact_pilot as OLD
import author_collection_service_client as CLIENT
import author_stream_representation as CGR
import author_live_world_contract as K
import author_contact_pipeline as M

B,R,C=OLD.B,OLD.R,OLD.C


def freeze(source_batch,geometry_template,root,indices=(3,4)):
    source=Path(source_batch)/'BATCH_PROTOCOL.json';p=copy.deepcopy(B.read(source));root=Path(root).resolve()
    if not 1<=len(indices)<=8 or len(set(indices))!=len(indices):raise ValueError('One to eight fixed distinct scene indices required')
    rows=[copy.deepcopy(p['queue']['scenes'][i])for i in indices]
    if not all(r['status']=='ready'for r in rows):raise ValueError('This finite world control requires already fixed ready layouts')
    root.mkdir(parents=True,exist_ok=False);bank={};camera=None
    import author_collection_scenes as SCENES
    static_started=time.monotonic()
    for i,row in enumerate(rows):
        scene=SCENES.build_scene(row['scene_spec']['path'],root/'scenes'/f'{i:02d}'/'build')
        scene['state_boundary']='One persistent robot/table/camera world; actor placement only between rounds; measured state preserved from photograph to execution'
        scene['persistent_world_schema']=K.SCHEMA
        for name in ('author_live_world.py','author_live_world_contract.py','author_live_collection_service.py'):
            scene['source_sha256'][str((B.HERE/name).resolve())]=B.sha(B.HERE/name)
        B.save(root/'scenes'/f'{i:02d}'/'build/scene.json',scene)
        row['built_scene']=B.binding(root/'scenes'/f'{i:02d}'/'build/scene.json')
        # Frozen specs carry complete source rows; no geometry selection by outcome.
        if camera is None:camera=scene['camera']
        if scene['camera']!=camera:raise ValueError('A persistent world has one fixed calibrated camera')
        for obj in scene['objects']:
            key=obj['object_id']
            if key in bank:
                a,b=copy.deepcopy(bank[key]),copy.deepcopy(obj);a.pop('world_T_object');b.pop('world_T_object')
                if a!=b:raise ValueError('One source identity has different bank geometry')
            bank[key]=obj
    template=B.read(B.read(geometry_template)['scene']['path'])
    primary=next(o for o in template['objects']if o['object_id']==template['target_object_id'])
    bank.setdefault(primary['object_id'],primary)
    for name in ('author_live_collection.py','author_live_collection_service.py','author_live_world.py','author_live_world_contract.py',
        'audit_author_live_world.py','author_contact_pipeline.py','author_fast_contact_alignment.py',
        'author_collection_service.py','author_collection_service_bound.py','author_resident_execution.py',
        'author_collection_service_client.py','author_stream_representation.py','author_collection_wave_resources.py',
        'author_scaling_control.py','author_verified_execution.py','author_verified_contact_journal.py','author_contact_compressor.py',
        'audit_author_resident_collection.py'):
        p['sources'][name]=B.binding(B.HERE/name)
    p['sources'].update(service_entry=B.binding(B.HERE/'author_live_collection_service.py'),audit_entry=B.binding(B.HERE/'audit_author_live_world.py'))
    p['inputs'].update(geometry_template=B.binding(geometry_template),source_manual_batch=B.binding(source),nvidia_smi=B.binding('/usr/bin/nvidia-smi'))
    for i,row in enumerate(rows):p['inputs']['built_scene:'+str(i)]=row['built_scene']
    for key in ('scene','plan','commands','candidate','protocol'):
        p['inputs']['geometry_template:'+key]=B.read(geometry_template)[key]
    p.update(schema=K.SCHEMA,rows=rows,source_queue_indices=list(indices),object_bank=list(bank.values()),camera=camera,
        gestures=[1],total_scene_budget=len(rows),total_gesture_attempt_budget=len(rows),max_Kit_starts=1,
        max_execution_steps_per_job=8000,max_environment_steps=len(rows)*8750,retry_budget=0,
        diagnostic_only=True,exclude_training=True,formal_training_eligible=False,automatic_training=False,
        stage_timeout_s=1800.,service_idle_timeout_s=600.,service_wall_timeout_s=3600.,
        cancel_sigint_grace_s=30.,cancel_sigterm_grace_s=10.,cancel_sigkill_grace_s=5.,
        gpu_uuid=OLD.GPU_UUID,gpu_identity=C.gpu_identity('/usr/bin/nvidia-smi',OLD.GPU_UUID),
        physics_change='Same CPU PhysX settings, continuous solver/world state and measured nonzero entry velocities replace cold recreation; trajectories not claimed equal',
        old_data_merged=False,static_build_wall_s=time.monotonic()-static_started)
    p['env']=dict(p['env'],CUDA_VISIBLE_DEVICES=OLD.GPU_UUID,NVIDIA_VISIBLE_DEVICES=OLD.GPU_UUID,ANYDEX_CONTACT_SPOOL_ROOT=str(root/'contact_spool'))
    p['worker_jobs']=[];items=[]
    for i,row in enumerate(rows):
        scene=root/'scenes'/f'{i:02d}'
        p['worker_jobs'].extend([dict(role='capture',round_index=i,scene=str(scene/'build/scene.json'),output_dir=str(scene/'capture')),
            dict(role='execute',round_index=i,prepared_dir=str(scene/'live'))])
        items.append(dict(object_id=row['scene_id'],observation_json=str(scene/'capture/observation.json'),output_dir=str(root/'representation/representations'/row['scene_id'])))
    manifest=root/'REPRESENTATION_MANIFEST.json';B.save(manifest,dict(objects=items),exclusive=True);p['inputs']['stream_manifest']=B.binding(manifest)
    p['source_archive']=C.archive_sources(root,p['sources']);B.verify(p['inputs']);B.verify(p['sources'])
    B.save(root/'PROTOCOL.json',p,exclusive=True);return p


def verify(root):
    p=B.read(Path(root)/'PROTOCOL.json')
    if p['schema']!=K.SCHEMA or p['max_Kit_starts']!=1 or p['max_environment_steps']!=len(p['rows'])*8750 or p['exclude_training']is not True or p['retry_budget']!=0:raise ValueError('Finite same-world control changed')
    if [j['role']for j in p['worker_jobs']]!=['capture','execute']*len(p['rows']):raise ValueError('Continuous round order changed')
    B.verify(p['inputs']);B.verify(p['sources']);B.verify({'archive':p['source_archive']});return p


@contextmanager
def classifiers():
    old=R.classify
    def classify(command,p):
        if str(B.HERE/'author_contact_pipeline.py')in command:return ('prepare',)
        return old(command,p)
    R.classify=classify
    try:yield
    finally:R.classify=old


def stage(directory,name,command,expected,p):
    result=B._stage(directory/'stages'/name,command,expected,p)
    if result['status']!='complete':raise RuntimeError(name+': '+str(result.get('error')))
    return result


def audit_execution(root,index,p):
    scene=root/'scenes'/f'{index:02d}';prepared=scene/'live';output=prepared/'RUNTIME_AUDIT.json'
    if not(prepared/'COLLECTION_RECORD.json').exists():return dict(index=index,status='not_executed',label=None)
    stage(scene,'runtime_audit',[p['inputs']['python']['path'],'-B',p['sources']['audit_entry']['path'],'runtime',
        '--prepared-dir',str(prepared),'--split-manifest',p['queue']['split_manifest']['path'],'--output',str(output)],[output],p)
    audit=B.read(output)
    trial=B.validate_trial(prepared,p['rows'][index]);B.certify_audit(audit,trial,p['rows'][index])
    return dict(index=index,status='pass',label=trial['label'],trial=trial,audit=B.binding(output),exclude_training=True)


def run(root):
    root=Path(root).resolve();p=verify(root);client=None;began=time.monotonic();error=None;audits=[];jobs=[];futures=[]
    if C.gpu_identity('/usr/bin/nvidia-smi',p['gpu_uuid'])!=p['gpu_identity']:raise ValueError('Physical GPU0 changed')
    with R.lease(root/'DRIVER.lock'),R.lease(OLD.LEGACY_LOCK)as fd,R.active_run(root,OLD.REGISTRY,B.binding(root/'PROTOCOL.json')):
        B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid()),started_unix_s=time.time()),exclusive=True)
        pool=R.ResourcePool(root/'resource_slots',fd,B.binding(root/'PROTOCOL.json'));previous={k:os.environ.get(k)for k in p['env']};os.environ.update(p['env'])
        try:
            with classifiers(),R.install_worker_adapters(pool,OLD.LEGACY_LOCK),ThreadPoolExecutor(max_workers=2)as executor:
                py=p['inputs']['python']['path']
                for i,row in enumerate(p['rows']):
                    scene=root/'scenes'/f'{i:02d}';build=scene/'build'
                    B._scene_objects(build/'scene.json',row)
                args=SimpleNamespace(manifest=root/'REPRESENTATION_MANIFEST.json',run_dir=root/'representation',checkpoint=Path(p['inputs']['checkpoint']['path']),
                    native_deps=Path(p['native_deps']),device='cuda:0',seed=p['queue']['seed'],num_seed=2048,voxel_size=.005,depth_min=.2,depth_max=.65,prepare_only=False)
                args.run_dir.mkdir();engine=CGR.RepresentationStream(args,B.read(args.manifest)['objects'])
                client=CLIENT.ResidentClient(root,root/'PROTOCOL.json',pool,fd)
                for i,row in enumerate(p['rows']):
                    scene=root/'scenes'/f'{i:02d}';capture=scene/'capture'
                    client.submit(2*i);job=client.wait(2*i);jobs.append(job)
                    output=capture/'INDEPENDENT_CAPTURE_AUDIT.json'
                    stage(scene,'capture_audit',[py,'-B',p['sources']['audit_entry']['path'],'capture','--capture-dir',str(capture),'--protocol',str(root/'PROTOCOL.json'),'--output',str(output)],[output],p)
                    if B.read(output)['status']!='pass':raise ValueError('Same-world capture audit failed')
                    engine.forward(i);domain=scene/'DOMAIN.json';B.save(domain,p['domain'],exclusive=True);contact=scene/'contact';base=contact/'pipeline'
                    stage(scene,'prepare',[py,'-B',str(B.HERE/'author_contact_pipeline.py'),'prepare','--scene',str(capture/'observed_scene.json'),'--observation',str(capture/'observation.json'),
                        '--representation-dir',str(root/'representation/representations'/row['scene_id']),'--seed',str(p['queue']['seed']),
                        '--calibration',p['inputs']['calibration']['path'],'--cloud-manifest',p['inputs']['cloud_manifest']['path'],'--backend',p['inputs']['backend']['path'],
                        '--domain',str(domain),'--output-dir',str(contact)],[base/'LIFT_PIPELINE_RESULTS.json'],p)
                    if B.read(base/'LIFT_PIPELINE_RESULTS.json')['status']=='ready':
                        K.bind_live_plan(base,capture/'LIVE_STATE.json',scene/'live');M.audit_mapping(scene/'live')
                    else:
                        (scene/'live').mkdir();B.save(scene/'live/NOT_EXECUTED.json',dict(source_preparation=B.binding(base/'LIFT_PIPELINE_RESULTS.json'),physics_steps=0),exclusive=True)
                    client.submit(2*i+1);job=client.wait(2*i+1);jobs.append(job)
                    futures.append(executor.submit(audit_execution,root,i,p))
                    B.save(root/'PROGRESS.json',dict(rounds_finished=i+1,total_rounds=len(p['rows']),jobs=jobs,wall_s=time.monotonic()-began))
                client.close();client=None;audits=[f.result()for f in futures]
        except BaseException:error=traceback.format_exc()
        finally:
            if client is not None:client.close(abort=True)
            for f in futures:
                if f.done():
                    try:
                        value=f.result()
                        if value not in audits:audits.append(value)
                    except BaseException:pass
            for k,v in previous.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
            verify(root)
            receipt=B.read(root/'runtime/WORLD_RECEIPT.json')if(root/'runtime/WORLD_RECEIPT.json').exists()else None
            result=dict(schema=K.SCHEMA,status='complete'if error is None and len(audits)==len(p['rows'])else'stopped',error=error,jobs=jobs,audits=audits,world=receipt,
                wall_s=time.monotonic()-began,new_training_labels=0,diagnostic_positive=sum(a.get('label')==1 for a in audits),diagnostic_negative=sum(a.get('label')==0 for a in audits),exclude_training=True,
                training_updates=0,retries=0,bulk_started=False)
            B.save(root/'RESULTS.json',result,exclusive=True)
    return result


if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('stage',choices=['freeze','run']);q.add_argument('--root',type=Path,required=True)
    q.add_argument('--source-batch',type=Path);q.add_argument('--geometry-template',type=Path);q.add_argument('--queue-indices',nargs='+',type=int,default=[3,4]);a=q.parse_args()
    if a.stage=='run':
        # Native extension lookup uses the process startup environment. Set it
        # before any model/CUDA initialization and before taking run leases.
        marker=B.sha(a.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_LIVE_PROTOCOL_ENV')!=marker:
            import sys
            env=B._environment(verify(a.root));env['ANYDEX_LIVE_PROTOCOL_ENV']=marker
            os.execve(env.get('ANYDEX_LIVE_PYTHON',sys.executable),[sys.executable,'-B',str(Path(__file__).resolve()),'run','--root',str(a.root.resolve())],env)
    result=freeze(a.source_batch,a.geometry_template,a.root,a.queue_indices)if a.stage=='freeze'else run(a.root)
    print({k:result.get(k)for k in ('schema','status','wall_s','diagnostic_positive','diagnostic_negative','error')},flush=True)
