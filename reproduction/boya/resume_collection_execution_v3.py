"""Continue only proven unexecuted slots from a stopped frozen collection.

Completed records and original plans remain read-only. A new executor accepts
native adaptive approach timing by binding each plan's full command arrays.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_collection_pipeline_v2 import (
    BOYA, ROOT, bounded_jobs, env, hashes, immutable, read, save, sha, verify,
    label_counts, split_records, execution_costs,
)

EXECUTOR = BOYA / 'execute_collection_queue_v2.py'


def check_original(run):
    protocol = read(run / 'protocol.json')
    if protocol['schema'] != 'anydex_boya_collection_pipeline_v2':
        raise ValueError('Expected frozen v2 collection')
    for section in ('source_sha256', 'input_sha256', 'manifest_sha256'):
        verify(protocol[section])
    stage = read(run / 'stages/planning.json')
    manifest = run / 'planning/queue_manifest.json'
    if not stage['complete'] or stage['protocol_sha256'] != sha(run / 'protocol.json'):
        raise ValueError('Planning stage is incomplete or foreign')
    verify({str(manifest): stage['artifacts_sha256'][str(manifest)]})
    queue = read(manifest)
    if not queue['complete'] or queue['errors'] or not queue['inputs_unchanged']:
        raise ValueError('Original queue not complete')
    result = read(run / 'execution/RESULTS.json')
    if result['status'] != 'infrastructure_stopped' or result.get('unknown_execution_trial_ids'):
        raise ValueError('Resume requires known stopped execution with no unknown trials')
    if not result['source_sha256_unchanged'] or not result['input_sha256_unchanged']:
        raise ValueError('Original execution source/input changed')
    completed = {}
    original_objects = read(run / 'execution/PROGRESS.json')['objects']
    for oid, obj in original_objects.items():
        if obj['status'] == 'complete':
            doc = obj['result']
            if not doc['input_sha256_unchanged'] or not doc['source_sha256_unchanged']:
                raise ValueError('Changed inherited execution: ' + oid)
            expected = {r['trial_id'] for r in queue['trials'] if r['object_id'] == oid}
            rows = doc['trials']
            if {r['trial_id'] for r in rows} != expected or len(rows) != len(expected):
                raise ValueError('Inherited object has missing/duplicate trials')
            for row in rows:
                if row.get('label') not in (0, 1) or row.get('label_eligible') is not True:
                    raise ValueError('Inherited label not eligible')
                plan = read(Path(row['source_plan_dir']) / 'plan.json')
                if row['physics_steps'] != plan['total_command_steps']:
                    raise ValueError('Inherited trace length differs from source plan')
                evpath = Path(row['execution_dir']) / 'evaluation.json'
                evaluation = read(evpath)
                verify({str(evpath.parent / 'trace.npz'): evaluation['trace_sha256']})
            completed[oid] = obj
        elif obj['status'] == 'infrastructure_error':
            # This version handles the observed pre-Kit command-length rejection.
            target = run / 'execution' / oid
            if not target.is_dir() or any(p.name != '.queue.lock' for p in target.iterdir()):
                raise ValueError('Cannot prove zero execution for failed object: ' + oid)
            log = (run / 'logs' / ('execution_' + oid + '.log')).read_text()
            if 'Exactly 2100 original approach/closure/hold/lift commands required' not in log:
                raise ValueError('Unknown infrastructure failure; independent diagnosis needed')
            if obj.get('result') or obj['process']['returncode'] == 0:
                raise ValueError('Ambiguous failed object state')
        elif obj['status'] == 'not_dispatched_after_infrastructure_error':
            if (run / 'execution' / oid).exists():
                raise ValueError('Undispatched object unexpectedly has outputs: ' + oid)
        else:
            raise ValueError('Unsupported original object state: ' + obj['status'])
    inherited_ids = {r['trial_id'] for obj in completed.values() for r in obj['result']['trials']}
    remaining = [r for r in queue['trials'] if r['trial_id'] not in inherited_ids]
    if {r['trial_id'] for r in remaining} != set(result['unexecuted_ready_trial_ids']):
        raise ValueError('Original unexecuted ledger does not reconcile')
    return protocol, queue, completed, remaining


def freeze(run, output):
    protocol, queue, completed, remaining = check_original(run)
    objects = read(run / 'objects.json')['objects']
    refs = [run / p for p in ('protocol.json', 'objects.json', 'planning/queue_manifest.json',
            'execution/PROGRESS.json', 'execution/RESULTS.json',
            'execution/train_records.jsonl', 'execution/validation_records.jsonl')]
    contract = dict(
        schema='anydex_boya_collection_execution_resume_v3', original_run=str(run),
        original_protocol_sha256=sha(run / 'protocol.json'), original_refs_sha256=hashes(refs),
        source_sha256=hashes([Path(__file__).resolve(), EXECUTOR]),
        inherited_objects=list(completed), inherited_trial_count=sum(len(x['result']['trials']) for x in completed.values()),
        remaining_trial_ids=[r['trial_id'] for r in remaining], remaining_trial_count=len(remaining),
        slot_count=protocol['slot_count'], gesture_ids=protocol['gesture_ids'],
        split_slot_counts=protocol['split_slot_counts'], targets=protocol['valid_label_target_totals'],
        execution_workers=2, batch_size=16, device='cpu', mode='serial',
        amendment='Only the adapter fixed-2100 assumption changes: execute and require all original per-plan commands and phases, including native adaptive approach timing. Original plan/command arrays, geometry, controller, physics gates, selection/splits/targets remain frozen. No replay of completed/unknown trials.',
        new_network_forwards=0, new_training_steps=0,
    )
    path = output / 'PROTOCOL.json'
    if path.exists():
        if read(path) != contract:
            raise ValueError('Frozen continuation differs; preserve it and use a new version')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('New continuation output must be empty')
        output.mkdir(parents=True, exist_ok=True)
        save(path, contract)
    return contract, protocol, queue, completed, remaining, objects


def execute(run, output, contract, protocol, queue, inherited, remaining, objects):
    if (output / 'PROGRESS.json').exists():
        raise ValueError('Continuation already dispatched; never replay existing outputs')
    (output / 'object_queues').mkdir()
    (output / 'execution').mkdir()
    (output / 'logs').mkdir()
    start = time.monotonic()
    progress = dict(schema='anydex_boya_collection_resume_progress_v3', status='running',
        original_run=str(run), protocol_sha256=sha(output / 'PROTOCOL.json'),
        inherited_actual_grasps=contract['inherited_trial_count'], execution_workers=2,
        objects={oid:dict(object_id=oid,status='inherited_complete',output_dir=obj.get('output_dir'),result={k:v for k,v in obj['result'].items() if k in ('trials','status','input_sha256_unchanged','source_sha256_unchanged')}) for oid,obj in inherited.items()},
        dispatch_events=[], peak_inflight_object_jobs=0)
    save(output / 'PROGRESS.json', progress)
    jobs = []
    for obj in objects:
        oid = obj['object_id']
        if oid in inherited:
            continue
        trials = [r for r in remaining if r['object_id'] == oid]
        nulls = [r for r in queue['nulls'] if r['object_id'] == oid]
        sub = dict(queue, trials=trials, nulls=nulls, parent_queue_manifest=str(run / 'planning/queue_manifest.json'),
                   parent_queue_manifest_sha256=sha(run / 'planning/queue_manifest.json'))
        path = output / 'object_queues' / (oid + '.json')
        immutable(path, sub)
        jobs.append(dict(object_id=oid, manifest=str(path), manifest_sha256=sha(path), ready=len(trials)))

    def one(job):
        oid = job['object_id']
        verify(contract['source_sha256']); verify(protocol['source_sha256'])
        verify({job['manifest']:job['manifest_sha256']})
        sub = read(job['manifest'])
        if not sub['trials']:
            return dict(object_id=oid,status='no_ready_plans', result=dict(trials=[]))
        target = output / 'execution' / oid
        argv = [sys.executable,'-B',str(EXECUTOR),'--manifest',job['manifest'],
                '--output-dir',str(target),'--device','cpu','--mode','serial','--batch-size','16']
        proc = None; error = None; began = time.monotonic()
        try:
            with (output / 'logs' / (oid + '.log')).open('xb') as log:
                proc = subprocess.run(argv,cwd=ROOT,env=env(),stdout=log,stderr=subprocess.STDOUT)
        except Exception:
            error = traceback.format_exc()
        doc = read(target / 'RESULTS.json') if (target / 'RESULTS.json').exists() else {}
        rows = doc.get('trials',[])
        expected = {r['trial_id']:read(Path(r['plan_dir'])/'plan.json')['total_command_steps'] for r in sub['trials']}
        ok = (proc is not None and proc.returncode == 0 and not doc.get('error') and not doc.get('infrastructure_error')
              and doc.get('status') in ('complete','complete_with_null')
              and doc.get('input_sha256_unchanged') is True and doc.get('source_sha256_unchanged') is True
              and len(rows) == len(expected) and {r['trial_id'] for r in rows} == set(expected)
              and all(r.get('label') in (0,1) and r.get('label_eligible') is True
                      and r.get('physics_steps') == expected[r['trial_id']] for r in rows))
        try:
            verify(contract['source_sha256']); verify(protocol['source_sha256'])
        except Exception:
            error = (error or '') + traceback.format_exc()
            ok = False
        doc = {k:v for k,v in doc.items() if k in ('trials','status','error','infrastructure_error','input_sha256_unchanged','source_sha256_unchanged','actual_grasps','physical_steps','positives','negatives','nulls')}
        return dict(object_id=oid,status='complete' if ok else 'infrastructure_error', infrastructure_error=not ok,
                    output_dir=str(target), result=doc,error=error,returncode=proc.returncode if proc else None,
                    wall_s=time.monotonic()-began)

    def completed(job,value,bad):
        progress['objects'][job['object_id']] = value
        progress['wall_s'] = time.monotonic()-start
        save(output / 'PROGRESS.json', progress)
        print(json.dumps(dict(object_id=job['object_id'],status=value['status'],wall_s=progress['wall_s'])),flush=True)

    def dispatched(job,active):
        progress['peak_inflight_object_jobs'] = max(progress['peak_inflight_object_jobs'],sum(bool(j['ready']) for j in active))
        progress['dispatch_events'].append(dict(object_id=job['object_id'],ready=job['ready'],wall_s=time.monotonic()-start))
        save(output / 'PROGRESS.json', progress)

    work = bounded_jobs(jobs,2,one,completed,dispatched)
    failed = any(r['failed'] for r in work)
    for job in jobs:
        progress['objects'].setdefault(job['object_id'],dict(object_id=job['object_id'],status='not_dispatched_after_infrastructure_error',result={}))
    allrows = [r for o in progress['objects'].values() for r in o.get('result',{}).get('trials',[])]
    byid = {s['trial_id']:s for s in protocol['slots']}
    records = []
    for split in ('train','validation'):
        with (run / 'execution' / (split+'_records.jsonl')).open() as f:
            records.extend(json.loads(line) for line in f if line.strip())
    for oid,obj in progress['objects'].items():
        if obj['status'] == 'inherited_complete' or not obj.get('output_dir'):
            continue
        for row in obj.get('result',{}).get('trials',[]):
            p = Path(obj['output_dir'])/'trials'/row.get('output_identity',row['trial_id'])/'training_record.json'
            if not p.exists():
                continue
            value = read(p)
            if value.get('label_eligible') is True and value.get('result') in (0,1):
                slot = byid[row['trial_id']]
                value['collection_context'] = dict(trial_id=slot['trial_id'],object_id=oid,split=slot['split'],
                    gesture_id=slot['gesture_id'],extra_depth_index=slot['extra_depth_index'],benchmark_only=False,
                    protocol_sha256=sha(run/'protocol.json'),continuation_protocol_sha256=sha(output/'PROTOCOL.json'))
                records.append(value)
    order = {slot['trial_id']:i for i,slot in enumerate(protocol['slots'])}
    allrows.sort(key=lambda row:order[row['trial_id']])
    records.sort(key=lambda row:order[row['collection_context']['trial_id']])
    grouped = split_records(records)
    costs = execution_costs(allrows,[r['trial_id'] for r in queue['trials']])
    if len(records) != sum(r.get('label_eligible') is True and r.get('label') in (0,1) for r in allrows):
        raise ValueError('Export records do not reconcile with execution receipts')
    if not failed and (costs['not_dispatched_ready_trial_ids'] or len(records) != len(queue['trials'])):
        failed = True
    counts = label_counts(protocol,allrows)
    result = dict(schema='anydex_boya_collection_resume_results_v3',status='infrastructure_stopped' if failed else 'complete_with_null',
        complete=not failed,original_run=str(run),protocol_sha256=sha(output/'PROTOCOL.json'),
        inherited_actual_grasps=contract['inherited_trial_count'],new_actual_grasps=costs['actual_grasps']-contract['inherited_trial_count'],
        actual_grasps=costs['actual_grasps'],physical_steps=costs['physical_steps'],
        positives=sum(r.get('label')==1 for r in allrows),negatives=sum(r.get('label')==0 for r in allrows),
        preparation_nulls=len(queue['nulls']),unexecuted_ready_trial_ids=costs['not_dispatched_ready_trial_ids'],
        unknown_execution_trial_ids=costs['unknown_execution_trial_ids'],counts=counts,trials=allrows,
        wall_s=time.monotonic()-start,new_network_forwards=0,new_training_steps=0)
    verify(contract['source_sha256']); verify(contract['original_refs_sha256'])
    for filename,values in [('all_records.jsonl',records),('train_records.jsonl',grouped['train']),('validation_records.jsonl',grouped['validation'])]:
        with (output/filename).open('x') as f:
            for value in values:f.write(json.dumps(value,allow_nan=False)+'\n')
    save(output/'RESULTS.json',result)
    save(output/'collection_counts.json',{k:v for k,v in result.items() if k!='trials'})
    progress.update(status=result['status'],wall_s=result['wall_s'])
    save(output/'PROGRESS.json',progress)
    if failed:
        raise RuntimeError('Continuation infrastructure stop; original and new evidence retained, no automatic replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    run = args.original_run.resolve(); output = args.output_dir.resolve()
    with (run.parent/('.'+run.name+'.pipeline.lock')).open('a') as original_lock:
        fcntl.flock(original_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (output.parent/('.'+output.name+'.resume.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            contract,protocol,queue,completed,remaining,objects = freeze(run,output)
            if args.dry_run:
                print(json.dumps({k:v for k,v in contract.items() if k not in ('source_sha256','original_refs_sha256','remaining_trial_ids')},indent=2));return
            try:
                execute(run,output,contract,protocol,queue,completed,remaining,objects)
            except BaseException:
                path = output/'PROGRESS.json'
                progress = read(path) if path.exists() else {}
                progress.update(status='infrastructure_stopped',error=traceback.format_exc())
                save(path,progress)
                raise


if __name__ == '__main__':
    main()
