"""Continue proven unexecuted slots while preserving certified native task nulls.

Only the outcome adapter changes. Frozen commands, controllers, physical gates,
all prior attempts and all input/source files remain read-only. Known native
stops are consumed nulls, never labels or automatically replayed candidates.
"""
from __future__ import annotations
import argparse
import collections
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from run_collection_pipeline_v2 import (
    BOYA, ROOT, bounded_jobs, env, hashes, immutable, read, save, sha, verify,
    label_counts, split_records, execution_costs,
)
from resume_collection_execution_v3 import check_original as check_base_original
from terminal_collection_outcome import certify_known_task_stop

EXECUTOR = BOYA/'execute_collection_queue_v2.py'


def inspect_rows(rows, expected_ids=None):
    ids = [row['trial_id'] for row in rows]
    if len(ids) != len(set(ids)) or (expected_ids is not None and set(ids) != set(expected_ids)):
        raise ValueError('Missing, foreign or repeated actual trials')
    certificates = {}
    for row in rows:
        if row.get('label') in (0,1) and row.get('label_eligible') is True:
            if not row.get('guards') or not all(row['guards'].values()):
                raise ValueError('Eligible row has failed original execution guards')
            plan = read(Path(row['source_plan_dir'])/'plan.json')
            evaluation_path = Path(row['execution_dir'])/'evaluation.json'
            evaluation = read(evaluation_path)
            if (row.get('physics_steps') != plan['total_command_steps']
                    or evaluation.get('complete') is not True
                    or evaluation.get('gates',{}).get('complete_execution') is not True
                    or evaluation.get('total_physics_steps') != row['physics_steps']
                    or evaluation.get('commands_sha256') != row['commands_sha256']):
                raise ValueError('Inherited valid label lacks original full execution')
            verify({str(evaluation_path.parent/'trace.npz'):evaluation['trace_sha256']})
        elif row.get('label') is None:
            certificates[row['trial_id']] = certify_known_task_stop(row)
        else:
            raise ValueError('Malformed or ineligible non-null label')
    return certificates


def read_records(directory):
    values = []
    for split in ('train','validation'):
        with (directory/(split+'_records.jsonl')).open() as stream:
            values.extend(json.loads(line) for line in stream if line.strip())
    split_records(values)
    return values


def check_original(run, previous):
    protocol, queue, _, _ = check_base_original(run)
    previous_contract = read(previous/'PROTOCOL.json')
    if previous_contract.get('schema') != 'anydex_boya_collection_execution_resume_v3' or Path(previous_contract['original_run']).resolve() != run:
        raise ValueError('Expected stopped v3 continuation of this exact original run')
    verify(previous_contract['source_sha256']); verify(previous_contract['original_refs_sha256'])
    prior = read(previous/'RESULTS.json'); progress = read(previous/'PROGRESS.json')
    if prior.get('status') != 'infrastructure_stopped' or prior.get('complete') is not False or prior.get('unknown_execution_trial_ids'):
        raise ValueError('Previous run must be stopped with no unknown execution')
    rows = prior['trials']; ready = {row['trial_id']:row for row in queue['trials']}
    certificates = inspect_rows(rows)
    if any(row['trial_id'] not in ready for row in rows):
        raise ValueError('Foreign prior execution identity')
    consumed = {row['trial_id'] for row in rows}
    remaining = [row for row in queue['trials'] if row['trial_id'] not in consumed]
    if set(prior['unexecuted_ready_trial_ids']) != {row['trial_id'] for row in remaining}:
        raise ValueError('Prior consumed/unexecuted ready partition does not reconcile')
    if sum(row.get('physics_steps',0) for row in rows) != prior['physical_steps'] or len(rows) != prior['actual_grasps']:
        raise ValueError('Prior physical costs do not reconcile')
    records = read_records(previous)
    label_ids = {row['trial_id'] for row in rows if row.get('label_eligible') is True}
    if {row['collection_context']['trial_id'] for row in records} != label_ids or len(records) != len(label_ids):
        raise ValueError('Prior eligible records disagree with actual eligible attempts')
    objects = read(run/'objects.json')['objects']
    inherited = {}
    for obj in objects:
        oid = obj['object_id']
        observed = [row for row in rows if row['object_id'] == oid]
        old = progress['objects'].get(oid,{})
        if observed:
            expected = {row['trial_id'] for row in queue['trials'] if row['object_id'] == oid}
            if {row['trial_id'] for row in observed} != expected:
                raise ValueError('Partial prior object needs explicit per-object continuation contract')
            if old.get('result',{}).get('trials') != observed:
                raise ValueError('Prior per-object ledger differs from final original rows')
            if old.get('result',{}).get('source_sha256_unchanged') is not True or old.get('result',{}).get('input_sha256_unchanged') is not True:
                raise ValueError('Inherited object input/source status not confirmed')
            inherited[oid] = dict(object_id=oid,status='inherited_complete',output_dir=old.get('output_dir'),
                result=dict(status='complete_with_null' if any(row['label'] is None for row in observed) else 'complete',
                            trials=observed,input_sha256_unchanged=True,source_sha256_unchanged=True))
        elif any(row['object_id'] == oid for row in remaining):
            if old.get('status') != 'not_dispatched_after_infrastructure_error':
                raise ValueError('Remaining object not explicitly never-dispatched: '+oid)
            if (previous/'execution'/oid).exists():
                raise ValueError('Never-dispatched object has ambiguous execution outputs')
    return protocol,queue,inherited,remaining,objects,certificates,records


def freeze(run, previous, output):
    protocol,queue,inherited,remaining,objects,certificates,records = check_original(run,previous)
    refs = [run/p for p in ('protocol.json','objects.json','planning/queue_manifest.json','stages/planning.json')]
    refs += [previous/p for p in ('PROTOCOL.json','PROGRESS.json','RESULTS.json','train_records.jsonl','validation_records.jsonl')]
    for obj in inherited.values():
        for row in obj['result']['trials']:
            refs += [Path(row['execution_dir'])/'evaluation.json',Path(row['execution_dir'])/'trace.npz']
    rows = [row for obj in inherited.values() for row in obj['result']['trials']]
    contract = dict(schema='anydex_boya_collection_execution_resume_v4',original_run=str(run),previous_run=str(previous),
        original_protocol_sha256=sha(run/'protocol.json'),previous_protocol_sha256=sha(previous/'PROTOCOL.json'),
        original_refs_sha256=hashes(refs),
        source_sha256=hashes([Path(__file__).resolve(),BOYA/'terminal_collection_outcome.py',EXECUTOR,
            BOYA/'resume_collection_execution_v3.py',BOYA/'run_collection_pipeline_v2.py',BOYA/'trial_record.py']),
        inherited_objects=list(inherited),inherited_trial_count=len(rows),
        inherited_eligible_labels=len(records),inherited_terminal_null_count=len(certificates),
        inherited_physical_steps=sum(row['physics_steps'] for row in rows),
        remaining_trial_ids=[row['trial_id'] for row in remaining],remaining_trial_count=len(remaining),
        remaining_object_count=len({row['object_id'] for row in remaining}),
        slot_count=protocol['slot_count'],gesture_ids=protocol['gesture_ids'],split_slot_counts=protocol['split_slot_counts'],
        targets=protocol['valid_label_target_totals'],execution_workers=2,batch_size=16,device='cpu',mode='serial',
        amendment='Consume source-bound native ordinary early stops as certified terminal nulls; unknown infrastructure still stops. Never replay existing attempts, never turn nulls into training negatives. Original source commands, plans, controllers and all physical gates unchanged.',
        new_network_forwards=0,new_training_steps=0)
    path = output/'PROTOCOL.json'
    if path.exists():
        if read(path) != contract: raise ValueError('Frozen continuation differs; use a new version')
    else:
        if output.exists() and any(output.iterdir()): raise ValueError('New continuation output must be empty')
        output.mkdir(parents=True,exist_ok=True); save(path,contract)
        save(output/'INHERITED_TERMINAL_CERTIFICATES.json',certificates)
    return contract,protocol,queue,inherited,remaining,objects,certificates,records


def execute(run, previous, output, contract, protocol, queue, inherited, remaining, objects, certificates, records):
    if (output/'PROGRESS.json').exists(): raise ValueError('Continuation already dispatched; never replay outputs')
    for name in ('object_queues','execution','logs','terminal_certificates'): (output/name).mkdir()
    start = time.monotonic()
    progress = dict(schema='anydex_boya_collection_resume_progress_v4',status='running',original_run=str(run),previous_run=str(previous),
        protocol_sha256=sha(output/'PROTOCOL.json'),inherited_actual_grasps=contract['inherited_trial_count'],
        inherited_eligible_labels=contract['inherited_eligible_labels'],inherited_terminal_null_count=len(certificates),
        execution_workers=2,objects=dict(inherited),dispatch_events=[],peak_inflight_object_jobs=0)
    save(output/'PROGRESS.json',progress)
    jobs = []
    for obj in objects:
        oid = obj['object_id']; trials = [row for row in remaining if row['object_id'] == oid]
        if not trials:
            if oid not in inherited:
                progress['objects'][oid] = dict(object_id=oid,status='no_ready_plans',result=dict(trials=[]))
            continue
        sub = dict(queue,trials=trials,nulls=[row for row in queue['nulls'] if row['object_id']==oid],
                   parent_queue_manifest=str(run/'planning/queue_manifest.json'),parent_queue_manifest_sha256=sha(run/'planning/queue_manifest.json'))
        path = output/'object_queues'/(oid+'.json'); immutable(path,sub)
        jobs.append(dict(object_id=oid,manifest=str(path),manifest_sha256=sha(path),ready=len(trials)))
    save(output/'PROGRESS.json',progress)

    def one(job):
        oid = job['object_id']; verify(contract['source_sha256']); verify(protocol['source_sha256'])
        verify({job['manifest']:job['manifest_sha256']}); sub=read(job['manifest'])
        target = output/'execution'/oid
        argv = [sys.executable,'-B',str(EXECUTOR),'--manifest',job['manifest'],'--output-dir',str(target),
                '--device','cpu','--mode','serial','--batch-size','16']
        proc=None; error=None; began=time.monotonic(); cert={}
        try:
            with (output/'logs'/(oid+'.log')).open('xb') as log:
                proc=subprocess.run(argv,cwd=ROOT,env=env(),stdout=log,stderr=subprocess.STDOUT)
        except Exception: error=traceback.format_exc()
        doc=read(target/'RESULTS.json') if (target/'RESULTS.json').exists() else {}
        ok=(proc is not None and proc.returncode==0 and not doc.get('error') and not doc.get('infrastructure_error')
            and doc.get('status') in ('complete','complete_with_null')
            and doc.get('input_sha256_unchanged') is True and doc.get('source_sha256_unchanged') is True)
        try:
            if not ok: raise ValueError('Native worker/executor infrastructure not confirmed')
            cert=inspect_rows(doc.get('trials',[]),[row['trial_id'] for row in sub['trials']])
            verify(contract['source_sha256']); verify(protocol['source_sha256'])
        except Exception:
            error=(error or '')+traceback.format_exc(); ok=False
        doc={key:value for key,value in doc.items() if key in ('trials','status','error','infrastructure_error',
             'input_sha256_unchanged','source_sha256_unchanged','actual_grasps','physical_steps','positives','negatives','nulls')}
        return dict(object_id=oid,status='complete' if ok else 'infrastructure_error',infrastructure_error=not ok,
                    output_dir=str(target),result=doc,error=error,returncode=proc.returncode if proc else None,
                    terminal_certificates=cert,wall_s=time.monotonic()-began)

    def completed(job,value,bad):
        progress['objects'][job['object_id']]=value
        for tid,certificate in value.get('terminal_certificates',{}).items():
            certificates[tid]=certificate; save(output/'terminal_certificates'/(tid+'.json'),certificate)
        progress['terminal_null_count']=len(certificates);progress['wall_s']=time.monotonic()-start
        save(output/'PROGRESS.json',progress)
        print(json.dumps(dict(object_id=job['object_id'],status=value['status'],wall_s=progress['wall_s'])),flush=True)

    def dispatched(job,active):
        progress['peak_inflight_object_jobs']=max(progress['peak_inflight_object_jobs'],len(active))
        progress['dispatch_events'].append(dict(object_id=job['object_id'],ready=job['ready'],wall_s=time.monotonic()-start))
        save(output/'PROGRESS.json',progress)

    work=bounded_jobs(jobs,2,one,completed,dispatched)
    failed=any(row['failed'] for row in work)
    for job in jobs:
        progress['objects'].setdefault(job['object_id'],dict(object_id=job['object_id'],status='not_dispatched_after_infrastructure_error',result={}))
    allrows=[row for obj in progress['objects'].values() for row in obj.get('result',{}).get('trials',[])]
    byid={slot['trial_id']:slot for slot in protocol['slots']}
    records=list(records)
    for oid,obj in progress['objects'].items():
        if obj['status']=='inherited_complete' or not obj.get('output_dir'):continue
        for row in obj.get('result',{}).get('trials',[]):
            path=Path(obj['output_dir'])/'trials'/row.get('output_identity',row['trial_id'])/'training_record.json'
            if not path.exists():continue
            value=read(path)
            if value.get('label_eligible') is True and value.get('result') in (0,1):
                slot=byid[row['trial_id']]
                value['collection_context']=dict(trial_id=slot['trial_id'],object_id=oid,split=slot['split'],gesture_id=slot['gesture_id'],
                    extra_depth_index=slot['extra_depth_index'],benchmark_only=False,protocol_sha256=sha(run/'protocol.json'),
                    continuation_protocol_sha256=sha(output/'PROTOCOL.json'))
                records.append(value)
    order={slot['trial_id']:i for i,slot in enumerate(protocol['slots'])}
    allrows.sort(key=lambda row:order[row['trial_id']]); records.sort(key=lambda row:order[row['collection_context']['trial_id']])
    grouped=split_records(records); costs=execution_costs(allrows,[row['trial_id'] for row in queue['trials']])
    label_ids={row['trial_id'] for row in allrows if row.get('label_eligible') is True and row.get('label') in (0,1)}
    actual_ids={row['trial_id'] for row in allrows if row.get('physics_steps',0)>0}
    if len(records)!=len(label_ids) or {row['collection_context']['trial_id'] for row in records}!=label_ids:
        raise ValueError('Export records do not reconcile with eligible actual rows')
    unknown_task_ids=sorted(actual_ids-label_ids-set(certificates))
    if costs['not_dispatched_ready_trial_ids'] or costs['unknown_execution_trial_ids'] or unknown_task_ids:
        failed=True
    if not failed and (len(allrows)!=len(queue['trials']) or len(records)+len(certificates)!=len(queue['trials'])):
        raise ValueError('Final ready partition not eligible + certified terminal null')
    result=dict(schema='anydex_boya_collection_resume_results_v4',status='infrastructure_stopped' if failed else 'complete_with_null',
        complete=not failed,original_run=str(run),previous_run=str(previous),protocol_sha256=sha(output/'PROTOCOL.json'),
        inherited_actual_grasps=contract['inherited_trial_count'],new_actual_grasps=costs['actual_grasps']-contract['inherited_trial_count'],
        actual_grasps=costs['actual_grasps'],physical_steps=costs['physical_steps'],new_physical_steps=costs['physical_steps']-contract['inherited_physical_steps'],
        eligible_labels=len(records),positives=sum(row.get('label')==1 for row in allrows),negatives=sum(row.get('label')==0 for row in allrows),
        preparation_nulls=len(queue['nulls']),terminal_null_count=len(certificates),terminal_nulls=[dict(trial_id=tid,reason=value['reason'],
            physics_steps=value['physics_steps'],label=None,label_eligible=False) for tid,value in certificates.items()],
        terminal_certificates=certificates,unexecuted_ready_trial_ids=costs['not_dispatched_ready_trial_ids'],
        unknown_execution_trial_ids=costs['unknown_execution_trial_ids'],uncertified_task_trial_ids=unknown_task_ids,
        counts=label_counts(protocol,allrows),trials=allrows,wall_s=time.monotonic()-start,new_network_forwards=0,new_training_steps=0,
        source_sha256_unchanged=True,input_sha256_unchanged=True)
    verify(contract['source_sha256']);verify(contract['original_refs_sha256']);verify(protocol['source_sha256']);verify(protocol['input_sha256'])
    for filename,values in [('all_records.jsonl',records),('train_records.jsonl',grouped['train']),('validation_records.jsonl',grouped['validation'])]:
        with (output/filename).open('x') as stream:
            for value in values:stream.write(json.dumps(value,allow_nan=False)+'\n')
    save(output/'RESULTS.json',result);save(output/'collection_counts.json',{key:value for key,value in result.items() if key not in ('trials','terminal_certificates')})
    progress.update(status=result['status'],wall_s=result['wall_s'],terminal_null_count=len(certificates))
    save(output/'PROGRESS.json',progress)
    if failed:raise RuntimeError('Unknown infrastructure stop; preserve all evidence, no automatic replay')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-run',type=Path,required=True)
    parser.add_argument('--previous-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();run=args.original_run.resolve();previous=args.previous_run.resolve();output=args.output_dir.resolve()
    if output in (run,previous):raise ValueError('Continuation must use a separate new output directory')
    with (run.parent/('.'+run.name+'.pipeline.lock')).open('a') as base_lock, (previous.parent/('.'+previous.name+'.resume.lock')).open('a') as parent_lock:
        fcntl.flock(base_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (output.parent/('.'+output.name+'.resume.lock')).open('a') as own_lock:
            fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            frozen=freeze(run,previous,output)
            if args.dry_run:
                print(json.dumps({key:value for key,value in frozen[0].items() if key not in ('source_sha256','original_refs_sha256','remaining_trial_ids')},indent=2));return
            try:execute(run,previous,output,*frozen)
            except BaseException:
                path=output/'PROGRESS.json';progress=read(path) if path.exists() else {}
                progress.update(status='infrastructure_stopped',error=traceback.format_exc());save(path,progress);raise

if __name__=='__main__':main()
