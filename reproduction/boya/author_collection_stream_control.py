"""Paired complete cold/streamed controls, two fixed scenes and g1, no training."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import author_production_tail as REFERENCE
import author_collection_service_client as CLIENT
import author_stream_representation as CGR
import author_contact_compressor as W

B,R=REFERENCE.B,REFERENCE.R
SCHEMA='anydex_boya_collection_stream_control_v1'
BASE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')
SOURCE=BASE/'production_tail_20261115_v1/run'


def freeze(root):
    reference=REFERENCE.verify(SOURCE)
    template=REFERENCE.verify_window(Path(B.read(reference['window_index']['path'])['windows'][0]['protocol']['path']).parent)
    root=Path(root).resolve();rows=[copy.deepcopy(reference['queue']['scenes'][i])for i in (1,9)]
    if not all(x['split']=='fit'and x['status']=='ready'for x in rows):raise ValueError('Fixed two known fit controls changed')
    root.mkdir(parents=True,exist_ok=False);sources=dict(reference['sources'])
    for name in ('author_collection_service.py','author_collection_service_client.py','author_stream_representation.py',
                 'author_collection_stream_control.py','run_collection_representation.py'):
        sources[name]=B.binding(B.HERE/name)
    sources.update(service_entry=sources['author_collection_service.py'],execution_entry=B.binding(B.HERE/'author_verified_execution.py'))
    groups=[]
    for mode in ('cold','resident'):
        directory=root/mode;directory.mkdir();jobs=[]
        for i,row in enumerate(rows):
            scene=directory/'scenes'/f'{i:02d}'
            jobs.append(dict(role='capture',scene=str(scene/'build/scene.json'),output_dir=str(scene/'capture')))
        for i,row in enumerate(rows):jobs.append(dict(role='execute',prepared_dir=str(directory/'scenes'/f'{i:02d}'/'g1')))
        manifest=directory/'REPRESENTATION_MANIFEST.json'
        items=[dict(object_id=row['scene_id'],observation_json=str(directory/'scenes'/f'{i:02d}'/'capture/observation.json'),
                    output_dir=str(directory/'representation/representations'/row['scene_id']))for i,row in enumerate(rows)]
        B.save(manifest,dict(objects=items),exclusive=True)
        p=copy.deepcopy(template);p.update(schema=SCHEMA,mode=mode,rows=rows,sources=sources,queue=reference['queue'],
            worker_jobs=jobs,diagnostic_only=True,exclude_training=True,formal_training_eligible=False,
            automatic_training=False,max_Kit_starts=4 if mode=='cold'else 1,
            max_capture_attempts=2,max_execution_attempts=2,max_execution_steps_per_job=8000,
            max_environment_steps=17500,stage_timeout_s=1200.,service_idle_timeout_s=600.,service_wall_timeout_s=2400.,
            retry_budget=0,service_source_reference=B.binding(SOURCE/REFERENCE.PROTOCOL))
        p['env']=dict(p['env'],ANYDEX_CONTACT_SPOOL_ROOT=f'/dev/shm/anydex_stream_control_20261116_v1/{mode}')
        p['inputs']=dict(p['inputs'],stream_manifest=B.binding(manifest),source_reference=B.binding(SOURCE/REFERENCE.PROTOCOL))
        p['source_archive']=REFERENCE.C.archive_sources(directory,sources)
        B.save(directory/'PROTOCOL.json',p,exclusive=True);groups.append(dict(mode=mode,protocol=B.binding(directory/'PROTOCOL.json')))
    p=dict(schema=SCHEMA,groups=groups,scenes=2,gesture_ids=[1],max_Kit_starts=5,max_environment_steps=35000,
           retry_budget=0,exclude_training=True,new_training_labels=0,training_updates=0,sources=sources)
    B.save(root/'CONTROL.json',p,exclusive=True);return verify(root)


def verify(root):
    root=Path(root).resolve();p=B.read(root/'CONTROL.json')
    if(p['schema']!=SCHEMA or p['scenes']!=2 or p['gesture_ids']!=[1]or p['max_Kit_starts']!=5
       or p['max_environment_steps']!=35000 or p['retry_budget']!=0 or p['exclude_training']is not True):
        raise ValueError('Finite paired control changed')
    B.verify(p['sources'])
    for group in p['groups']:
        B.verify({'protocol':group['protocol']});cp=B.read(group['protocol']['path']);B.verify(cp['inputs']);B.verify(cp['sources'])
        B.verify({'archive':cp['source_archive']})
        for source in B.read(cp['source_archive']['path'])['sources']:B.verify({'source':source['source'],'copy':source['archive']})
        if cp['exclude_training']is not True or [j['role']for j in cp['worker_jobs']]!=['capture','capture','execute','execute']:
            raise ValueError('Diagnostic four-job order changed')
    return p


def stage(directory,command,expected,p):
    result=B._stage(directory,command,expected,p)
    if result['status']!='complete':raise RuntimeError('Original external stage failed: '+str(result.get('error')))
    return result


def prepare(root,index,p,engine):
    row=p['rows'][index];scene=root/'scenes'/f'{index:02d}';capture=scene/'capture'
    cp=copy.deepcopy(p);cp['sources']['audit_entry']=p['sources']['audit_author_resident_collection.py']
    output=capture/'INDEPENDENT_CAPTURE_AUDIT.json'
    stage(scene/'stages/capture_audit',[p['inputs']['python']['path'],'-B',cp['sources']['audit_entry']['path'],
         'capture','--capture-dir',str(capture),'--output',str(output)],[output],cp)
    if B.read(output)['status']!='pass':raise ValueError('Original initialization audit failed')
    engine.forward(index)
    domain=scene/'DOMAIN.json';B.save(domain,p['domain'],exclusive=True);target=scene/'g1'
    command=[p['inputs']['python']['path'],'-B',str(B.HERE/'author_lift_pipeline.py'),'prepare',
             '--scene',str(capture/'observed_scene.json'),'--observation',str(capture/'observation.json'),
             '--representation-dir',str(root/'representation/representations'/row['scene_id']),
             '--gesture-id','1','--source-type',str(B.SOURCE_TYPES[1]),'--seed',str(p['queue']['seed']),
             '--calibration',p['inputs']['calibration']['path'],'--cloud-manifest',p['inputs']['cloud_manifest']['path'],
             '--backend',p['inputs']['backend']['path'],'--domain',str(domain),'--output-dir',str(target)]
    stage(scene/'stages/prepare_g1',command,[target/'LIFT_PIPELINE_RESULTS.json'],p)
    if B.read(target/'LIFT_PIPELINE_RESULTS.json')['status']!='ready':raise ValueError('Fixed control has no original plan; no fallback')
    ready=B.read(target/'READY.json');plan=B.read(ready['plan']['path'])
    if plan['total_command_steps']>p['max_execution_steps_per_job']:raise ValueError('Frozen execution-step budget exceeded')


def audit_execution(root,index,p):
    scene=root/'scenes'/f'{index:02d}';target=scene/'g1';output=target/'RUNTIME_AUDIT.json'
    stage(scene/'stages/audit_g1',[p['inputs']['python']['path'],'-B',p['sources']['audit_entry']['path'],
        '--prepared-dir',str(target),'--split-manifest',p['queue']['split_manifest']['path'],'--output',str(output)],[output],p)
    result=B.read(output)
    if result['status']!='pass':raise ValueError('Original numeric execution certificate failed')
    trial=B.validate_trial(target,p['rows'][index]);B.certify_audit(result,trial,p['rows'][index])
    REFERENCE.verified_contact_artifacts(trial)
    return dict(certificate=B.binding(output),trial=trial)


def run_mode(root,pool,lease_fd):
    root=Path(root);p=B.read(root/'PROTOCOL.json');began=time.monotonic();error=None;audits=[];client=None
    B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid()),started_unix_s=time.time()),exclusive=True)
    try:
        for i,row in enumerate(p['rows']):
            scene=root/'scenes'/f'{i:02d}'
            stage(scene/'stages/build',[p['inputs']['python']['path'],'-B',p['sources']['scene_entry']['path'],
                'build','--scene-spec',row['scene_spec']['path'],'--output-dir',str(scene/'build')],[scene/'build/scene.json'],p)
        args=SimpleNamespace(manifest=root/'REPRESENTATION_MANIFEST.json',run_dir=root/'representation',
            checkpoint=Path(p['inputs']['checkpoint']['path']),native_deps=Path(p['native_deps']),device='cuda:0',
            seed=p['queue']['seed'],num_seed=2048,voxel_size=.005,depth_min=.2,depth_max=.65,prepare_only=False)
        args.run_dir.mkdir();items=B.read(args.manifest)['objects'];engine=CGR.RepresentationStream(args,items)
        klass=CLIENT.ColdClient if p['mode']=='cold'else CLIENT.ResidentClient
        client=klass(root,root/'PROTOCOL.json',pool,lease_fd)
        client.submit(0);client.wait(0)
        client.submit(1);prepare(root,0,p,engine)
        client.submit(2);client.wait(1);prepare(root,1,p,engine)
        client.submit(3)
        with ThreadPoolExecutor(max_workers=2)as executor:
            client.wait(2);first=executor.submit(audit_execution,root,0,p)
            client.wait(3);second=executor.submit(audit_execution,root,1,p)
            client.close();client=None
            audits=[first.result(),second.result()]
    except BaseException:error=traceback.format_exc()
    finally:
        if client is not None:
            try:client.close(abort=True)
            except BaseException:error=(error or'')+'\n'+traceback.format_exc()
    result=dict(status='complete'if error is None and len(audits)==2 else'stopped',error=error,audits=audits,
        wall_s=time.monotonic()-began,exclude_training=True,new_training_labels=0,training_updates=0,retries=0)
    B.save(root/'RESULTS.json',result,exclusive=True);return result


def compare(root,results):
    arrays=[];labels=[]
    for mode in ('cold','resident'):
        directory=root/mode;mode_arrays=[]
        for i in range(2):
            row=B.read(directory/'PROTOCOL.json')['rows'][i]
            files=[directory/'scenes'/f'{i:02d}'/'capture/native_rgbd.npz',
                   directory/'representation/representations'/row['scene_id']/'representation_outputs.npz',
                   directory/'representation/representations'/row['scene_id']/'raw_camera_candidates.npz',
                   Path(B.read(directory/'scenes'/f'{i:02d}'/'g1/READY.json')['commands']['path'])]
            values=[]
            for file in files:
                with np.load(file,allow_pickle=False)as z:
                    values.append({k:z[k].copy()for k in z.files if k not in
                                   ('collection_candidate_sha256','collection_execution_spec_sha256')})
            mode_arrays.append(values)
        arrays.append(mode_arrays);labels.append([a['trial']['label']for a in results[len(labels)]['audits']])
    matches=[]
    for i in range(2):
        for j in range(4):
            a,b=arrays[0][i][j],arrays[1][i][j]
            matches.append(dict(scene=i,component=j,exact=a.keys()==b.keys()and all(np.array_equal(v,b[k])for k,v in a.items())))
    worker=B.read(root/'resident/runtime/WORKER_RECEIPT.json')
    qualified=all(x['exact']for x in matches)and labels[0]==labels[1]and worker['Kit_starts']==1 and worker['status']=='closed'
    return dict(same_scientific_arrays=matches,same_labels=labels[0]==labels[1],diagnostic_labels=labels,
                resident_Kit_starts=worker['Kit_starts'],cold_Kit_starts=4,
                speedup=results[0]['wall_s']/results[1]['wall_s']if qualified else None,
                comparison_qualified=qualified,scope='One paired two-scene/g1 complete diagnostic; not population yield or a production ETA')


def run(root):
    root=Path(root).resolve();p=verify(root);results=[];comparison=None;began=time.monotonic()
    for group in p['groups']:
        cp=B.read(group['protocol']['path'])
        if REFERENCE.C.gpu_identity(cp['inputs']['nvidia_smi']['path'],cp['gpu_uuid'])!=cp['gpu_identity']:
            raise ValueError('Pinned physical GPU0 identity changed')
    with R.lease(root/'DRIVER.lock'),R.lease(REFERENCE.P.OLD.LEGACY_LOCK)as fd:
        if(root/'RUN_STARTED.json').exists():raise FileExistsError('No control retry')
        with R.active_run(root,REFERENCE.P.OLD.REGISTRY,B.binding(root/'CONTROL.json')):
            B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid())),exclusive=True)
            pool=R.ResourcePool(root/'resource_slots',fd,B.binding(root/'CONTROL.json'))
            for group in p['groups']:
                mode_root=Path(group['protocol']['path']).parent;cp=B.read(group['protocol']['path'])
                previous={k:os.environ.get(k)for k in cp['env']};os.environ.update(cp['env'])
                try:
                    with R.install_worker_adapters(pool,REFERENCE.P.OLD.LEGACY_LOCK):results.append(run_mode(mode_root,pool,fd))
                finally:
                    for k,v in previous.items():
                        if v is None:os.environ.pop(k,None)
                        else:os.environ[k]=v
                if results[-1]['status']!='complete':break
            if len(results)==2 and all(r['status']=='complete'for r in results):
                try:comparison=compare(root,results)
                except BaseException:comparison=dict(comparison_qualified=False,error=traceback.format_exc())
            result=dict(schema=SCHEMA,status='complete'if comparison and comparison['comparison_qualified']else'stopped',
                groups=results,comparison=comparison,wall_s=time.monotonic()-began,exclude_training=True,
                new_training_labels=0,training_updates=0,retries=0)
            verify(root);B.save(root/'RESULTS.json',result,exclusive=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();r=freeze(a.root)if a.action=='freeze'else run(a.root)
    print({k:r.get(k)for k in ('status','wall_s','comparison')},flush=True)
    if a.action=='run'and r['status']!='complete':raise SystemExit(1)
