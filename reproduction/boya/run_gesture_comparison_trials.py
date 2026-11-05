#!/usr/bin/env python3
"""Bound six additional gestures on the eight frozen pilot objects.

Preparation uses the existing read-only native arm admission, at most two rounds
per cell, at most four candidate preparations per round and eight per cell. Round
two starts with the still-unconsumed frozen pool indices. Execution is separately
frozen and explicitly launched; at most two object queues run in CPU resident
serial mode. A completed negative label never stops collection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'reproduction/boya/prepare_collection_queue.py').is_file()),Path('/home/wangrenpeng/AnyDexGrasp'))
sys.path.insert(0,str(ROOT/'reproduction/boya'))
from prepare_collection_queue import prepare_one

GESTURES=[2,3,5,7,9,10]
SCHEMA='anydex_boya_gesture_comparison_v1'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def now():return datetime.now(timezone.utc).isoformat()
def freeze(files,path,expected=None):
    path=Path(path).resolve();digest=sha(path)
    if expected is not None and digest!=expected:raise ValueError('Frozen SHA mismatch: '+str(path))
    if str(path) in files and files[str(path)]!=digest:raise ValueError('Frozen input changed: '+str(path))
    files[str(path)]=digest
    return digest
def verify(files):
    for p,h in files.items():
        if not Path(p).is_file() or sha(p)!=h:raise ValueError('Frozen input/source changed: '+p)

def remaining_indices(selected,consumed):
    if len(consumed)!=len(set(consumed)) or len(consumed)>8:raise ValueError('Consumed candidate budget/identity violated')
    return [int(i) for i in selected if int(i) not in set(consumed)][:min(4,8-len(consumed))]

def consumed_from(directory,result):
    used={int(r['index']) for r in result.get('plans',[])}
    for p in Path(directory).glob('prepare_*.log'):
        used.add(int(p.stem.rsplit('_',1)[1]))
    return sorted(used)

def run_cell(cell,plan_root,backend,stop,prepare_fn=prepare_one):
    began=time.monotonic();rows=[];consumed=[]
    result=dict(cell_id=cell['cell_id'],object_id=cell['object_id'],gesture_id=cell['gesture_id'],
        split=cell['split'],status='preparing',rounds=rows,consumed_indices=consumed,
        actual_physics_steps=0,max_actual_grasps=2,max_arm_prepares=8)
    for number in (1,2):
        if stop.is_set():
            result['status']='paused_before_next_round';break
        offered=remaining_indices(cell['selected_indices'],consumed)
        trial_id=f"{cell['cell_id']}__r{number:02d}"
        directory=plan_root/trial_id
        request=dict(trial_id=trial_id,object_id=cell['object_id'],gesture_id=cell['gesture_id'],split=cell['split'],extra_depth_index=0,round_number=number,offered_indices=offered,
            pool_status=cell['pool_status'],status='preparing',label=None,actual_physics_steps=0)
        rows.append(request)
        if cell['pool_status']=='no_candidate':
            request.update(status='no_hand_geometry_in_declared_budget',consumed_indices=[],
                no_candidate_json=cell['no_candidate_json'],no_candidate_sha256=cell['no_candidate_sha256'])
            continue
        if not offered:
            request.update(status='frozen_pool_exhausted',consumed_indices=[]);continue
        job=dict(trial_id=trial_id,object_id=cell['object_id'],gesture_id=cell['gesture_id'],
            split=cell['split'],observation_json=cell['observation_json'],source_npz=cell['source_npz'],
            calibration=cell['calibration'],candidate_dir=cell['pool'],candidate_indices=offered,
            extra_depth_index=0,max_arm_plans=min(4,8-len(consumed)))
        if cell.get('collision_refinement'):
            job.update(collision_refinement=cell['collision_refinement'],
                collision_refinement_sha256=cell['collision_refinement_sha256'])
        try:
            prepared=prepare_fn(job,plan_root,backend)
            used=consumed_from(directory,prepared)
            if not set(used)<=set(offered) or set(used)&set(consumed) or len(used)>4:
                raise ValueError('Preparation attempted repeated/out-of-request candidate')
            consumed.extend(used)
            request.update(prepared,round_number=number,offered_indices=offered,consumed_indices=used,
                label=None,actual_physics_steps=0,cell_id=cell['cell_id'],gesture_id=cell['gesture_id'],
                extra_depth_index=0,split=cell['split'])
            if prepared['status'] not in ('ready','no_full_arm_plan_in_declared_budget'):
                raise ValueError('Unexpected prepared round status: '+prepared['status'])
        except BaseException:
            partial=read(directory/'preparation.json') if (directory/'preparation.json').is_file() else {}
            used=consumed_from(directory,partial)
            request.update(status='infrastructure_error',error=traceback.format_exc(),
                consumed_indices=used,partial_preparation=partial)
            for index in used:
                if index not in consumed:consumed.append(index)
            result['status']='infrastructure_error';stop.set();break
    else:result['status']='prepared'
    result.update(arm_prepare_calls=len(consumed),admitted_plans=sum(r['status']=='ready' for r in rows),
        elapsed_wall_s=time.monotonic()-began)
    if len(consumed)>8 or result['admitted_plans']>2:raise ValueError('Cell preparation budget exceeded')
    return result


def describe_cells(args,protocol,only=None):
    objects=protocol['objects']
    if len(objects)!=8 or len({r['object_id'] for r in objects})!=8:
        raise ValueError('Exactly the eight unique frozen pilot objects required')
    files={};freeze(files,args.protocol)
    backend=Path(protocol['backend_config']).resolve();cal=Path(protocol['calibration']).resolve()
    freeze(files,backend);freeze(files,cal)
    cells=[]
    for obj in objects:
        oid=obj['object_id']
        if only is not None and not any(f'{oid}__g{gid:02d}' in only for gid in GESTURES):continue
        if re.fullmatch(r'[A-Za-z0-9_.-]+',oid) is None:raise ValueError('Unsafe object ID')
        observation=Path(obj.get('observation_json',obj.get('observation'))).resolve()
        source=Path(obj['representation_dir']).resolve()/'raw_camera_candidates.npz'
        freeze(files,observation);freeze(files,source)
        obs=read(observation)
        if obs['object_id']!=oid:raise ValueError('Observed object identity mismatch')
        repair=obj.get('collision_refinement')
        repair_path=None;repair_sha=None
        if repair:
            repair_path=str(Path(repair['path']).resolve());repair_sha=freeze(files,repair_path,repair['sha256'])
        for gid in GESTURES:
            if only is not None and f'{oid}__g{gid:02d}' not in only:continue
            pool=args.pool_root.resolve()/oid/f'g{gid}'
            generation=pool/'generation.json';null=pool/'no_candidate.json'
            if not generation.is_file() and not null.is_file():
                raise FileNotFoundError('Pool not complete yet: '+str(pool))
            cell=dict(cell_id=f'{oid}__g{gid:02d}',object_id=oid,gesture_id=gid,
                split='development',inherited_source_split=obj.get('source_split'),
                observation_json=str(observation),source_npz=str(source),calibration=str(cal),pool=str(pool),
                collision_refinement=repair_path,collision_refinement_sha256=repair_sha)
            if generation.is_file():
                gen=read(generation);freeze(files,generation)
                freeze(files,pool/'candidates.npz',gen['candidates_sha256'])
                if (gen.get('complete') is not True or gen['gesture_id']!=gid
                        or Path(gen['source_observation']).resolve()!=observation
                        or Path(gen['source_frame']).resolve()/ 'raw_camera_candidates.npz' !=source
                        or {r['additional_depth_index'] for r in gen['records']}!={0}):
                    raise ValueError('Geometry pool identity/depth/source mismatch')
                selected=list(map(int,gen['selected_indices']))
                if len(selected)>8 or len(selected)!=len(set(selected)):raise ValueError('Frozen pool selection must be unique, <=8')
                for p,h in gen['input_sha256'].items():freeze(files,p,h)
                cell.update(pool_status='ready',selected_indices=selected)
            else:
                gen=read(null);freeze(files,null)
                for key,p in [('source_candidates_sha256',source),('source_observation_sha256',observation),('calibration_sha256',cal)]:
                    freeze(files,p,gen[key])
                if gen['gesture_id']!=gid:raise ValueError('Null pool gesture mismatch')
                cell.update(pool_status='no_candidate',selected_indices=[],no_candidate_json=str(null),no_candidate_sha256=sha(null))
            cells.append(cell)
    return cells,files,backend


def bounded_schedule(items,workers,run_one,stop,on_result):
    """Do not pre-submit the whole cohort after an infrastructure failure."""
    iterator=iter(items);active={};done=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        def fill():
            while not stop.is_set() and len(active)<workers:
                try:item=next(iterator)
                except StopIteration:return
                active[executor.submit(run_one,item)]=item
        fill()
        while active:
            finished,_=wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                item=active.pop(future)
                try:value=future.result()
                except BaseException:
                    stop.set();value=dict(status='infrastructure_error',error=traceback.format_exc(),item=item,object_id=item.get('object_id'),cell_id=item.get('cell_id'),result={})
                done.append(value);on_result(value)
            fill()
    return done


def prepare(args):
    run=args.run_dir.resolve()
    if (run/'PREPARED.json').exists() or (run/'preparation_progress.json').exists():
        raise ValueError('Never replay an existing preparation segment')
    protocol=read(args.protocol)
    if len(protocol['objects'])!=8 or len({r['object_id'] for r in protocol['objects']})!=8:
        raise ValueError('Exactly eight frozen pilot objects required')
    cells=[dict(cell_id=f"{obj['object_id']}__g{gid:02d}",object_id=obj['object_id'],gesture_id=gid,
        pool=str(args.pool_root.resolve()/obj['object_id']/f'g{gid}'))
        for obj in protocol['objects'] for gid in GESTURES]
    inputs={};freeze(inputs,args.protocol)
    backend=Path(protocol['backend_config']).resolve();freeze(inputs,backend);freeze(inputs,protocol['calibration'])
    run.mkdir(parents=True,exist_ok=True);plans=run/'plans';plans.mkdir();(run/'cell_inputs').mkdir()
    sources={str(Path(__file__).resolve()):sha(__file__)}
    for name in ['prepare_collection_queue.py','execute_collection_queue.py','trial_record.py','native_arm.py','asset_adapter.py']:
        p=ROOT/'reproduction/boya'/name;sources[str(p)]=sha(p)
    spec=dict(schema=SCHEMA,created_at=now(),parent_protocol=str(args.protocol.resolve()),
        parent_protocol_sha256=sha(args.protocol),gesture_ids=GESTURES,cells=cells,
        backend_config=str(backend),limits=dict(rounds_per_cell=2,arm_per_round=4,arm_per_cell=8,
            actual_per_cell=2,prepare_workers=args.prepare_workers,execution_workers=args.execution_workers),
        input_sha256=inputs,source_sha256=sources,scope='Development comparison on the same eight pilot objects; six additional gestures, no new RGBD/CGR or learned head; all outcomes retained')
    spec_path=run/'PREPARATION_PROTOCOL.json';save(spec_path,spec)
    progress=dict(status='preparing',cells={},prepared_cells=0,started_at=now(),actual_physics_steps=0)
    save(run/'preparation_progress.json',progress);stop=threading.Event();start=time.monotonic()
    def completed(value):
        cid=value.get('cell_id',(value.get('item') or {}).get('cell_id','unknown'))
        progress['cells'][cid]=value;progress['prepared_cells']=len(progress['cells'])
        progress['elapsed_wall_s']=time.monotonic()-start
        save(run/'preparation_progress.json',progress)
        print('GESTURE_CELL_PREPARED '+json.dumps(dict(cell_id=cid,status=value['status'],
            admitted=value.get('admitted_plans',0),arm_calls=value.get('arm_prepare_calls',0))),flush=True)
    # Only ready pools occupy preparation workers. Geometry can finish other
    # cells while up to four independent native admissions are already running.
    pending=list(cells);active={};deadline=time.monotonic()+args.wait_pools_s
    with ThreadPoolExecutor(max_workers=args.prepare_workers) as executor:
        while pending or active:
            if not stop.is_set():
                for item in list(pending):
                    if len(active)>=args.prepare_workers:break
                    pool=Path(item['pool'])
                    if not (pool/'generation.json').is_file() and not (pool/'no_candidate.json').is_file():continue
                    pending.remove(item)
                    try:
                        resolved,files,_=describe_cells(args,protocol,only={item['cell_id']})
                        cell=resolved[0]
                        for path,digest in files.items():freeze(inputs,path,digest)
                        save(run/'cell_inputs'/(cell['cell_id']+'.json'),dict(cell=cell,input_sha256=files,source_sha256=sources))
                        active[executor.submit(run_cell,cell,plans,backend,stop)]=cell
                    except BaseException:
                        stop.set();completed(dict(item,status='infrastructure_error',rounds=[],error=traceback.format_exc()))
                        break
            if active:
                finished,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
                for future in finished:
                    item=active.pop(future)
                    try:value=future.result()
                    except BaseException:
                        stop.set();value=dict(item,status='infrastructure_error',rounds=[],error=traceback.format_exc())
                    completed(value)
            elif stop.is_set():break
            elif pending:
                if time.monotonic()>=deadline:
                    stop.set();progress['pool_wait_error']='Frozen pool wait deadline exceeded';break
                print('WAITING_POOLS '+str(len(pending)),flush=True);time.sleep(2)
    for cell in cells:
        cid=cell['cell_id']
        if cid not in progress['cells']:
            progress['cells'][cid]=dict(cell_id=cid,object_id=cell['object_id'],gesture_id=cell['gesture_id'],
                status='not_dispatched_after_infrastructure_error',rounds=[],actual_physics_steps=0)
    ordered=[progress['cells'][cell['cell_id']] for cell in cells]
    verify(inputs);verify(sources)
    trials=[];nulls=[];errors=[]
    for cell in ordered:
        if cell['status']!='prepared':errors.append(dict(cell_id=cell['cell_id'],status=cell['status'],error=cell.get('error')))
        for row in cell['rounds']:
            (trials if row['status']=='ready' else nulls).append(row)
    manifest=dict(schema='anydex_boya_collection_queue_v1',complete=not errors,trials=trials,nulls=nulls,
        errors=errors,jobs_json=str(spec_path),jobs_sha256=sha(spec_path),source_sha256=sources,
        inputs_unchanged=True,physical_steps=0,learned_Boya_head_loaded=False)
    manifest_path=run/'queue_manifest.json';save(manifest_path,manifest)
    object_queues=[];queues=run/'object_queues';queues.mkdir()
    for obj in read(args.protocol)['objects']:
        oid=obj['object_id'];sub=dict(manifest,trials=[r for r in trials if r['object_id']==oid],
            nulls=[r for r in nulls if r['trial_id'].startswith(oid+'__')])
        p=queues/(oid+'.json');save(p,sub)
        object_queues.append(dict(object_id=oid,manifest=str(p),manifest_sha256=sha(p),ready=len(sub['trials']),nulls=len(sub['nulls'])))
    prepared=dict(spec,status='prepared' if not errors else 'paused_infrastructure',cells=ordered,
        queue_manifest=str(manifest_path),queue_manifest_sha256=sha(manifest_path),object_queues=object_queues,
        preparation_wall_s=time.monotonic()-start,arm_prepare_calls=sum(c.get('arm_prepare_calls',0) for c in ordered),
        ready_plans=len(trials),null_rounds=len(nulls),errors=errors,actual_physics_steps=0,
        source_inputs_unchanged=True)
    save(run/'PREPARED.json',prepared)
    progress.update(status=prepared['status'],finished_at=now(),prepared_sha256=sha(run/'PREPARED.json'))
    save(run/'preparation_progress.json',progress)
    print(json.dumps({k:prepared[k] for k in ['status','ready_plans','null_rounds','arm_prepare_calls','preparation_wall_s','actual_physics_steps']},indent=2),flush=True)
    if errors:raise RuntimeError('Preparation infrastructure error; no physics dispatched')


def execution_infrastructure(result,returncode):
    return bool(returncode or result.get('error') or result.get('infrastructure_error')
        or result.get('status') not in ('complete','complete_with_null')
        or any(r.get('label') is None for r in result.get('trials',[])))


def execute(args):
    run=args.run_dir.resolve();path=run/'PREPARED.json'
    if not args.prepared_sha256 or sha(path)!=args.prepared_sha256:
        raise ValueError('Explicit matching --prepared-sha256 required before actual execution')
    prepared=read(path)
    if prepared['status']!='prepared':raise ValueError('Only a complete prepared cohort may execute')
    verify(prepared['input_sha256']);verify(prepared['source_sha256'])
    for queue in prepared['object_queues']:
        if sha(queue['manifest'])!=queue['manifest_sha256']:raise ValueError('Object execution queue changed')
    if (run/'EXECUTION_PROGRESS.json').exists():raise ValueError('Execution segment already exists; no replay')
    actual=run/'execution';actual.mkdir();stop=threading.Event();start=time.monotonic()
    progress=dict(schema=SCHEMA,status='running',started_at=now(),prepared_sha256=args.prepared_sha256,
        object_results={},max_cpu_object_workers=args.execution_workers,actual_physics_steps=0)
    save(run/'EXECUTION_PROGRESS.json',progress)
    def run_object(queue):
        oid=queue['object_id']
        if not queue['ready']:
            return dict(object_id=oid,status='no_ready_plans',returncode=0,result=dict(trials=[],preparation_nulls=read(queue['manifest'])['nulls']))
        output=actual/oid;command=[sys.executable,'-B',str(ROOT/'reproduction/boya/execute_collection_queue.py'),
            '--manifest',queue['manifest'],'--output-dir',str(output),'--device','cpu','--mode','serial','--batch-size','16']
        began=time.monotonic()
        with (actual/(oid+'.log')).open('xb') as log:
            proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1'))
        result=read(output/'RESULTS.json') if (output/'RESULTS.json').exists() else {}
        infra=execution_infrastructure(result,proc.returncode)
        if infra:stop.set()
        return dict(object_id=oid,status='infrastructure_error' if infra else 'complete',returncode=proc.returncode,
            wall_s=time.monotonic()-began,command=command,result=result,output_dir=str(output))
    def completed(row):
        progress['object_results'][row['object_id']]=row
        progress['elapsed_wall_s']=time.monotonic()-start
        progress['actual_physics_steps']=sum(r.get('result',{}).get('physical_steps',0) for r in progress['object_results'].values())
        save(run/'EXECUTION_PROGRESS.json',progress)
        print('GESTURE_OBJECT_EXECUTED '+json.dumps(dict(object_id=row['object_id'],status=row['status'])),flush=True)
    bounded_schedule(prepared['object_queues'],args.execution_workers,run_object,stop,completed)
    for queue in prepared['object_queues']:
        if queue['object_id'] not in progress['object_results']:
            progress['object_results'][queue['object_id']]=dict(object_id=queue['object_id'],status='not_dispatched_after_infrastructure_error',result={})
    results=[progress['object_results'][q['object_id']] for q in prepared['object_queues']]
    records=[];executed={}
    for item in results:
        for row in item['result'].get('trials',[]):executed[row['trial_id']]=row
        if item.get('output_dir') and (Path(item['output_dir'])/'PREPARED.json').is_file():
            for trial in read(Path(item['output_dir'])/'PREPARED.json')['trials']:
                record_path=Path(item['output_dir'])/'trials'/trial['output_identity']/'training_record.json'
                if record_path.is_file():
                    record=read(record_path)
                    if record['label_eligible']:
                        record['collection_context']=dict(trial_id=trial['trial_id'],object_id=trial['object_id'],
                            split=trial.get('split','development'),gesture_id=trial['gesture_id'],extra_depth_index=0,
                            benchmark_only=False,comparison_protocol_sha256=args.prepared_sha256)
                        records.append(record)
    cells=[]
    for cell in prepared['cells']:
        item=dict(cell);item['rounds']=[]
        for row in cell['rounds']:
            actual_row=executed.get(row['trial_id'])
            item['rounds'].append(dict(row,execution=actual_row,label=actual_row['label'] if actual_row else None,
                execution_status='not_dispatched' if row['status']=='ready' and actual_row is None else row['status']))
        cells.append(item)
    result=dict(progress,status='paused_infrastructure' if stop.is_set() else 'complete',finished_at=now(),
        cells=cells,object_results=results,wall_s=time.monotonic()-start,positive_labels=sum(r['result']==1 for r in records),
        negative_labels=sum(r['result']==0 for r in records),label_count=len(records),actual_physics_steps=sum(r.get('result',{}).get('physical_steps',0) for r in results),
        preparation_null_rounds=prepared['null_rounds'],source_inputs_unchanged=True,
        scope='New six-gesture development comparison; existing original pilot baseline remains separate; no Boya learned head')
    verify(prepared['input_sha256']);verify(prepared['source_sha256'])
    with (run/'training_records.jsonl').open('x') as f:
        for record in records:f.write(json.dumps(record,allow_nan=False)+'\n')
    save(run/'RESULTS.json',result);progress.update(status=result['status'],finished_at=result['finished_at']);save(run/'EXECUTION_PROGRESS.json',progress)
    print(json.dumps({k:result[k] for k in ['status','positive_labels','negative_labels','label_count','actual_physics_steps','wall_s']},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','execute'),required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'outputs/boya_pilot8_20261105_v2/protocol_v3.json')
    p.add_argument('--pool-root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--prepare-workers',type=int,default=4);p.add_argument('--execution-workers',type=int,default=2)
    p.add_argument('--wait-pools-s',type=float,default=1800)
    p.add_argument('--prepared-sha256')
    args=p.parse_args()
    if not 1<=args.prepare_workers<=4 or not 1<=args.execution_workers<=2:p.error('Preparation<=4, actual CPU workers<=2')
    if args.wait_pools_s<0:p.error('Nonnegative pool wait required')
    args.run_dir.resolve().mkdir(parents=True,exist_ok=True)
    with (args.run_dir.resolve()/'.comparison.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Comparison segment already running')
        (prepare if args.stage=='prepare' else execute)(args)


if __name__=='__main__':main()
