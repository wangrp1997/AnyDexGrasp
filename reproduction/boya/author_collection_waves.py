"""Opt-in three-scene waves around the unchanged, source-pinned batch runner.

Freeze does no capture, planning or physics. Started workers are never restarted.
A wave is fully reconciled before labels are admitted in original scene/gesture
order. The parent schema deliberately cannot be run by author_collection_batch.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import author_collection_batch as B
import author_collection_wave_resources as R

SCHEMA = 'anydex_boya_author_collection_waves_v1'
BATCH_PIN = '153527cbbfa38a3533bf5759a06331ab1c1896da95f8a575d8b18f7e38240959'
WIDTH = 3
LEGACY_LOCK = B.ROOT/'outputs/author_collection_gpu0.lock'


def _batch_pin():
    if B.sha(B.__file__) != BATCH_PIN:
        raise ValueError('The parallel adapter requires the exact reviewed legacy batch source')


def _source_bindings():
    return {name: B.binding(B.HERE/name) for name in (
        'author_collection_waves.py', 'author_collection_wave_resources.py',
        'author_collection_wave_export.py', 'author_collection_batch.py')}


def freeze(queue_path, output, *, fit_per_gesture=1000, internal_per_gesture=200, **options):
    """Freeze PLAN -> child protocols -> child index -> final protocol, no hash cycle."""
    _batch_pin()
    queue_path = Path(queue_path).resolve(); output = Path(output).resolve()
    queue = B.validate_queue(queue_path)
    if any(type(n) is not int or n < 1 for n in (fit_per_gesture, internal_per_gesture)):
        raise ValueError('Positive fixed quotas required')
    if options.get('prepare_workers', 3) != 3:
        raise ValueError('This opt-in policy fixes three prepare workers and three scene workers')
    options['prepare_workers'] = 3
    options = {k: str(v) if isinstance(v, Path) else v for k, v in options.items()}
    output.mkdir(parents=True, exist_ok=False)
    sources = _source_bindings()
    plan = dict(schema=SCHEMA, task=B.TASK, source_queue=B.binding(queue_path), queue=queue,
        sources=sources, quotas=dict(fit=fit_per_gesture, internal=internal_per_gesture),
        batch_options=options, wave_width=WIDTH, resource_limits=R.LIMITS,
        scenes=[dict(queue_index=i, scene_id=row['scene_id'], split=row['split'],
            layout_seed=B.read(row['scene_spec']['path']).get('seed', queue['seed']),
            candidate_cgr_sampling_seed=queue['seed']) for i, row in enumerate(queue['scenes'])],
        seed_policy='Preserve original queue.seed unchanged for every child CGR/candidate call; no SeedSequence or layout mutation',
        admission_policy='Complete each wave; original scene index then gesture1,6,7,8; first eligible quota, independent of class',
        total_scene_budget=len(queue['scenes']), total_gesture_attempt_budget=4*len(queue['scenes']),
        old435_merged=False, automatic_training=False, gpu=0,
        physics_policy='At most3Kit; execute sim.step(render=False) may overlap capture; capture/CGR observation mutex1',
        retry_policy='No started worker or stage is relaunched; unresolved lifetimes quarantine all dispatch',
        freeze_cost_scope='Initial implementation reuses full B.freeze verification for every child; repeated large-file SHA cost is not optimized')
    B.save(output/'WAVE_PLAN.json', plan, exclusive=True)
    children = []
    for i, row in enumerate(queue['scenes']):
        directory = output/'children'/f'{i:07d}_{row["scene_id"]}'
        directory.mkdir(parents=True)
        child = copy.deepcopy(queue)
        child['scenes'] = [copy.deepcopy(row)]
        child['source_scene_queue'] = B.binding(queue_path)
        child['parent_queue_index'] = i
        child['layout_seed'] = plan['scenes'][i]['layout_seed']
        child['candidate_cgr_sampling_seed'] = queue['seed']
        child['inputs'] = B.normalized_bindings(child.get('inputs', {}))
        child['inputs'].update(wave_plan=B.binding(output/'WAVE_PLAN.json'), source_scene_queue=B.binding(queue_path))
        child['sources'] = B.normalized_bindings(child.get('sources', {}))
        child['sources'].update({'wave:'+k: v for k, v in sources.items()})
        B.save(directory/'QUEUE.json', child, exclusive=True)
        # Child quotas cannot stop any of its four gestures; admission belongs to parent.
        B.freeze(directory/'QUEUE.json', directory/'batch', fit_per_gesture=1,
            internal_per_gesture=1, **options)
        children.append(dict(queue_index=i, scene_id=row['scene_id'], split=row['split'],
            queue=B.binding(directory/'QUEUE.json'), protocol=B.binding(directory/'batch/BATCH_PROTOCOL.json')))
    B.save(output/'CHILD_FREEZE_INDEX.json', dict(schema=SCHEMA,
        plan=B.binding(output/'WAVE_PLAN.json'), children=children), exclusive=True)
    protocol = dict(schema=SCHEMA, task=B.TASK, plan=B.binding(output/'WAVE_PLAN.json'),
        child_index=B.binding(output/'CHILD_FREEZE_INDEX.json'), sources=sources,
        quotas=plan['quotas'], wave_width=WIDTH, resource_limits=R.LIMITS,
        total_scene_budget=len(children), total_gesture_attempt_budget=4*len(children),
        old435_merged=False, automatic_training=False)
    B.save(output/'WAVE_PROTOCOL.json', protocol, exclusive=True)
    B.save(output/'PROGRESS.json', progress(protocol, []))
    return protocol


def verify_protocol(root, *, child_index=None):
    root = Path(root).resolve(); protocol = B.read(root/'WAVE_PROTOCOL.json')
    if protocol.get('schema') != SCHEMA or protocol.get('task') != B.TASK:
        raise ValueError('Explicit wave protocol required; no legacy compatible parent view')
    _batch_pin(); B.verify(protocol['sources'])
    B.verify({k: protocol[k] for k in ('plan', 'child_index')})
    plan = B.read(protocol['plan']['path']); index = B.read(protocol['child_index']['path'])
    if index['plan'] != protocol['plan'] or plan['sources'] != protocol['sources']:
        raise ValueError('Wave freeze DAG differs')
    B.verify({'source_queue': plan['source_queue']})
    queue = B.read(plan['source_queue']['path'])
    if queue != plan['queue'] or protocol['quotas'] != plan['quotas']:
        raise ValueError('Parent queue or quotas changed')
    if (protocol['wave_width'] != WIDTH or protocol['resource_limits'] != R.LIMITS
            or len(index['children']) != len(queue['scenes'])):
        raise ValueError('Frozen three-scene budget changed')
    dependencies = {}
    selected = enumerate(index['children']) if child_index is None else [(child_index, index['children'][child_index])]
    for i, child in selected:
        B.verify({k: child[k] for k in ('queue', 'protocol')})
        cp = B.read(child['protocol']['path']); cq = B.read(child['queue']['path'])
        if (child['queue_index'] != i or cq['parent_queue_index'] != i
                or cq['seed'] != queue['seed'] or cq['scenes'] != [queue['scenes'][i]]
                or cq['inputs']['wave_plan'] != protocol['plan']
                or cp['inputs']['queue'] != child['queue'] or cp['queue'] != cq):
            raise ValueError('Single-scene child mapping or original sampling seed changed')
        for entry in list(cp['inputs'].values())+list(cp['sources'].values()):
            key = (str(Path(entry['path']).resolve()), entry['sha256'])
            dependencies[key] = entry
    # Shared checkpoint/asset bytes are checked once per verification call.
    B.verify({str(i): entry for i, entry in enumerate(dependencies.values())})
    return protocol, plan, index


def progress(protocol, rows):
    counts = {s: {str(g): 0 for g in B.GESTURES} for s in ('fit', 'internal')}
    positives = copy.deepcopy(counts)
    for row in rows:
        if row.get('label') is not None:
            counts[row['split']][str(row['gesture_id'])] += 1
            positives[row['split']][str(row['gesture_id'])] += row['label']
    captures = {}
    for row in rows:
        raw = row.get('raw_ledger')
        if raw:
            rr = B.read(raw['path'])
            capture = rr.get('artifact_bindings', {}).get('capture_stage')
            if capture: captures[row['queue_index']] = capture
    complete_capture = sum(B.read(c['path']).get('status') == 'complete' for c in captures.values())
    quota_met = all(counts[s][str(g)] >= protocol['quotas'][s] for s in counts for g in B.GESTURES)
    return dict(schema=SCHEMA, task=B.TASK, eligible_labels=counts, positive_labels=positives,
        ledger_rows=len(rows), quota_met=quota_met,
        status='quota_met' if quota_met else ('budget_exhausted_insufficient_labels' if len(rows) == protocol['total_gesture_attempt_budget'] else 'pending'),
        known_final_execution_physics_steps=sum(r.get('actual_physics_steps', 0) for r in rows),
        execution_starts_with_unknown_steps=sum(bool(r.get('execution_steps_unknown')) for r in rows),
        captured_initialization_ticks=complete_capture*750,
        captured_initialization_object_steps=complete_capture*2250,
        physics_cost_scope='All raw child outcomes including quota excess; known final execution receipts and complete initialization only; interrupted starts may have unknown cost',
        quota_excess_labels=sum(r['status'] == 'quota_excess_not_admitted' for r in rows),
        old435_merged=False, training_updates=0, old_head_forwards=0)


def ledger(root, protocol=None):
    root = Path(root); protocol = protocol or B.read(root/'WAVE_PROTOCOL.json')
    rows = []; verified_stages = set()
    for path in sorted((root/'ledger').glob('*.json')):
        row = B.read(path)
        if row['protocol_sha256'] != B.sha(root/'WAVE_PROTOCOL.json'):
            raise ValueError('Parent ledger protocol differs')
        B.verify(row.get('bindings', {}))
        if row.get('raw_ledger'):
            B.verify({'raw': row['raw_ledger']})
            raw = B.read(row['raw_ledger']['path']); B.verify(raw.get('artifact_bindings', {}))
            for artifact in raw.get('artifact_bindings', {}).values():
                if Path(artifact['path']).name == 'STAGE.json' and artifact['path'] not in verified_stages:
                    B.verify(B.read(artifact['path']).get('outputs', {}))
                    verified_stages.add(artifact['path'])
        rows.append(row)
    return rows


def _child_result(directory, child, status, error=None):
    """Bind complete or partial raw ledgers without inventing missing negative labels."""
    directory = Path(directory); batch = directory/'batch'
    cp = B.read(child['protocol']['path'])
    raw = B.ledger(batch, cp)
    bindings = {p.name: B.binding(p) for p in (batch/'PROGRESS.json', batch/'LEDGER.json') if p.exists()}
    bindings.update({f'ledger:g{r["gesture_id"]}': B.binding(batch/'ledger'/f'0000000_g{r["gesture_id"]}.json') for r in raw})
    result = dict(schema=SCHEMA, parent_queue_index=child['queue_index'], child_protocol=child['protocol'],
        status=status, error=error, raw_bindings=bindings, completed_gestures=[r['gesture_id'] for r in raw])
    return result


def worker(root, child_index, permit_path, lease_fd):
    root = Path(root).resolve(); protocol, plan, index = verify_protocol(root, child_index=child_index)
    child = index['children'][child_index]; permit = B.read(permit_path)
    if (permit['parent_protocol'] != B.binding(root/'WAVE_PROTOCOL.json')
            or permit['child'] != child or permit['child_index'] != child_index):
        raise ValueError('Worker requires an explicit final-parent dispatch permit')
    if R.process_identity(permit['parent']['pid']) != permit['parent']:
        raise ValueError('Dispatching parent identity is no longer alive')
    R.check_inherited_lease(lease_fd, LEGACY_LOCK, permit['legacy_inode'])
    directory = Path(child['protocol']['path']).parent.parent
    started = B.read(directory/'CHILD_STARTED.json')
    if started['permit'] != B.binding(permit_path):
        raise ValueError('Worker permit differs from the immutable start reservation')
    pool = R.ResourcePool(root/'resource_slots', lease_fd, permit['parent_protocol'])
    error = None; status = 'complete'
    try:
        with R.install_worker_adapters(pool, LEGACY_LOCK):
            B.run(directory/'batch')
    except BaseException:
        status = 'infrastructure_failure'; error = traceback.format_exc()
    result = _child_result(directory, child, status, error)
    B.save(directory/'WORKER_RESULT.json', result, exclusive=True)
    return result


def _reconcile(root, child):
    directory = Path(child['protocol']['path']).parent.parent
    result_path = directory/'WORKER_RESULT.json'
    state_path = directory/'CHILD_PROCESS.json'
    if state_path.exists():
        identity = B.read(state_path)['process']
        if R.process_identity(identity['pid']) == identity or R.group_alive(identity['pgid']):
            raise R.Quarantined('Worker or its process group remains alive: '+str(state_path))
    if result_path.exists():
        result = B.read(result_path)
        if result['child_protocol'] != child['protocol'] or result['parent_queue_index'] != child['queue_index']:
            raise ValueError('Worker result mapping changed')
        B.verify(result['raw_bindings'])
        return result
    marker = directory/'CHILD_STARTED.json'
    if not marker.exists(): return None
    if not state_path.exists():
        raise R.Quarantined('Started worker PID is unknown; it will not be relaunched: '+str(marker))
    state = B.read(state_path)
    current = R.process_identity(state['process']['pid'])
    if current == state['process']:
        raise R.Quarantined('Started worker is still alive; do not redispatch: '+str(state_path))
    # Process group descendants and stage reservations must also have disappeared.
    if R.group_alive(state['process']['pgid']):
        raise R.Quarantined('Worker group remains alive: '+str(state_path))
    for path in (root/'resource_slots').glob('*/*.json'):
        reservation = B.read(path)
        if reservation.get('state') != 'released':
            raise R.Quarantined('A stage reservation is unresolved: '+str(path))
    result = _child_result(directory, child, 'infrastructure_failure', 'Started worker died; no restart; missing outcomes remain null')
    B.save(result_path, result, exclusive=True)
    return result


def _admit(root, protocol, plan, child, result, rows):
    """Only parent-index order determines the first quota; raw costs are retained."""
    i = child['queue_index']; spec = plan['queue']['scenes'][i]
    directory = Path(child['protocol']['path']).parent.parent
    cp = B.read(child['protocol']['path']); raw_rows = B.ledger(directory/'batch', cp)
    raw = {r['gesture_id']: r for r in raw_rows}
    if len(raw) != len(raw_rows): raise ValueError('Duplicate child gesture ledger')
    counts = progress(protocol, rows)['eligible_labels']
    for g in B.GESTURES:
        path = root/'ledger'/f'{i:07d}_g{g}.json'
        if path.exists(): continue
        rr = raw.get(g)
        value = dict(schema=SCHEMA, task=B.TASK, protocol_sha256=B.sha(root/'WAVE_PROTOCOL.json'),
            queue_index=i, child_queue_index=0, scene_id=spec['scene_id'], split=spec['split'],
            gesture_id=g, trial_id=f'{spec["scene_id"]}:g{g}', label=None,
            status='infrastructure_failure', actual_physics_steps=0,
            execution_steps_unknown=rr is None,
            source_object_ids=spec['source_object_ids'], source_object_digests=spec['source_object_digests'],
            bindings=dict(child_protocol=child['protocol'], worker_result=B.binding(directory/'WORKER_RESULT.json')))
        if rr is not None:
            if (rr['queue_index'] != 0 or rr['scene_id'] != spec['scene_id'] or rr['split'] != spec['split']
                    or rr['source_object_digests'] != spec['source_object_digests']):
                raise ValueError('Child local-index0 differs from parent scene mapping')
            value.update(raw_ledger=B.binding(directory/'batch/ledger'/f'0000000_g{g}.json'),
                raw_label=rr.get('label'), status=rr['status'],
                actual_physics_steps=rr.get('actual_physics_steps', 0),
                execution_steps_unknown=rr.get('execution_steps_unknown', False))
            if rr.get('label') is not None:
                trial = B.validate_trial(Path(rr['trial']['record']['path']).parent, spec)
                if trial != rr['trial']: raise ValueError('Raw executed trial changed')
                B.certify_audit(B.read(rr['artifact_bindings']['runtime_audit']['path']), trial, spec)
                if counts[spec['split']][str(g)] < protocol['quotas'][spec['split']]:
                    value.update(label=rr['label'], trial=trial)
                    counts[spec['split']][str(g)] += 1
                else:
                    value['status'] = 'quota_excess_not_admitted'
        else:
            # A worker can die after physics but before its per-gesture ledger.
            # Preserve a final receipt's known cost while keeping its label null.
            scene_dir = directory/'batch/scenes'/f'0000000_{spec["scene_id"]}'
            stage = scene_dir/f'stages/execute_g{g}/STAGE.json'
            receipt = scene_dir/f'g{g}/LIFT_PIPELINE_EXECUTION.json'
            value['execution_steps_unknown'] = stage.exists()
            for name, artifact in (('interrupted_execution_stage', stage), ('interrupted_execution_receipt', receipt)):
                if artifact.exists(): value['bindings'][name] = B.binding(artifact)
            if receipt.exists():
                execution = B.read(receipt)
                steps = execution.get('physics_steps')
                if execution.get('status') not in (None, 'starting') and type(steps) is int and steps >= 0:
                    value.update(actual_physics_steps=steps, execution_steps_unknown=False)
        B.save(path, value, exclusive=True); rows.append(value)


def _quota_skip(root, protocol, child, rows):
    spec = B.read(child['protocol']['path'])['queue']['scenes'][0]
    for g in B.GESTURES:
        path = root/'ledger'/f'{child["queue_index"]:07d}_g{g}.json'
        if path.exists(): continue
        value = dict(schema=SCHEMA, task=B.TASK, protocol_sha256=B.sha(root/'WAVE_PROTOCOL.json'),
            queue_index=child['queue_index'], child_queue_index=0, scene_id=child['scene_id'],
            split=child['split'], gesture_id=g, trial_id=f'{child["scene_id"]}:g{g}',
            label=None, status='quota_reached', actual_physics_steps=0,
            execution_steps_unknown=False, source_object_ids=spec['source_object_ids'],
            source_object_digests=spec['source_object_digests'],
            bindings={'child_protocol':child['protocol']})
        B.save(path, value, exclusive=True); rows.append(value)


def run(root, *, max_waves=None):
    """Dispatch only unstarted children; no worker/scene/candidate retry."""
    root = Path(root).resolve(); protocol, plan, index = verify_protocol(root)
    if max_waves is not None and (type(max_waves) is not int or max_waves < 1):
        raise ValueError('Positive fixed wave prefix required')
    with R.lease(root/'DRIVER.lock'), R.lease(LEGACY_LOCK) as parent_fd, R.active_run(
            root, LEGACY_LOCK.with_suffix('.wave_active.json'), B.binding(root/'WAVE_PROTOCOL.json')):
        pool = R.ResourcePool(root/'resource_slots', parent_fd, B.binding(root/'WAVE_PROTOCOL.json'))
        pool.check_halt()
        rows = ledger(root, protocol)
        selected = index['children'][:max_waves*WIDTH] if max_waves else index['children']
        for offset in range(0, len(selected), WIDTH):
            wave = selected[offset:offset+WIDTH]
            processes = []
            counts = progress(protocol, rows)['eligible_labels']
            for child in wave:
                i = child['queue_index']; directory = Path(child['protocol']['path']).parent.parent
                if all((root/'ledger'/f'{i:07d}_g{g}.json').exists() for g in B.GESTURES): continue
                if (directory/'CHILD_STARTED.json').exists():
                    _reconcile(root, child); continue
                if all(counts[child['split']][str(g)] >= protocol['quotas'][child['split']] for g in B.GESTURES):
                    _quota_skip(root, protocol, child, rows); continue
                pool.check_halt()
                B.verify(protocol['sources'])
                B.verify({k:protocol[k] for k in ('plan','child_index')})
                B.verify({'selected_child':child['protocol'], 'selected_queue':child['queue']})
                permit = dict(schema=SCHEMA, parent_protocol=B.binding(root/'WAVE_PROTOCOL.json'),
                    child_index=i, child=child, nonce=uuid.uuid4().hex,
                    parent=R.process_identity(os.getpid()),
                    legacy_inode=[os.fstat(parent_fd).st_dev, os.fstat(parent_fd).st_ino])
                B.save(directory/'WORKER_PERMIT.json', permit, exclusive=True)
                B.save(directory/'CHILD_STARTED.json', dict(permit=B.binding(directory/'WORKER_PERMIT.json'),
                    state='started_no_retry', started_unix_s=time.time()), exclusive=True)
                py = B.read(child['protocol']['path'])['inputs']['python']['path']
                with (directory/'worker.log').open('xb') as log:
                    process = subprocess.Popen([py, '-B', str(Path(__file__).resolve()), 'worker', '--run-dir', str(root),
                        '--child-index', str(i), '--permit', str(directory/'WORKER_PERMIT.json'), '--lease-fd', str(parent_fd)],
                        cwd=B.ROOT, env=B._environment(B.read(child['protocol']['path'])),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(parent_fd,))
                    identity = R.process_identity(process.pid)
                    if identity is None:
                        identity = dict(pid=process.pid, pgid=process.pid, start_ticks=None)
                    B.save(directory/'CHILD_PROCESS.json', dict(process=identity), exclusive=True)
                processes.append((child, process))
            for child, process in processes:
                process.wait()
                directory = Path(child['protocol']['path']).parent.parent
                B.save(directory/'CHILD_RETURN.json', dict(returncode=process.returncode,
                    log=B.binding(directory/'worker.log')), exclusive=True)
            pool.check_halt()
            B.verify(protocol['sources'])
            B.verify({k:protocol[k] for k in ('plan','child_index')})
            for child in wave:
                i = child['queue_index']
                if all((root/'ledger'/f'{i:07d}_g{g}.json').exists() for g in B.GESTURES): continue
                result = _reconcile(root, child)
                if result is None: raise ValueError('Wave did not dispatch its frozen child')
                _admit(root, protocol, plan, child, result, rows)
            B.save(root/'LEDGER.json', dict(protocol=B.binding(root/'WAVE_PROTOCOL.json'), rows=rows))
            B.save(root/'PROGRESS.json', progress(protocol, rows))
        return progress(protocol, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    for key in ('queue', 'run-dir', 'calibration', 'cloud-manifest', 'backend'):
        f.add_argument('--'+key, type=Path, required=True)
    for key in ('scene-entry', 'capture-entry', 'execution-entry', 'audit-entry', 'checkpoint', 'native-deps', 'python'):
        f.add_argument('--'+key, type=Path)
    f.add_argument('--domain', type=Path)
    f.add_argument('--fit-per-gesture', type=int, default=1000)
    f.add_argument('--internal-per-gesture', type=int, default=200)
    f.add_argument('--stage-timeout-s', type=float, default=3600.)
    r = sub.add_parser('run'); r.add_argument('--run-dir', type=Path, required=True); r.add_argument('--max-waves', type=int)
    w = sub.add_parser('worker'); w.add_argument('--run-dir', type=Path, required=True)
    w.add_argument('--child-index', type=int, required=True); w.add_argument('--permit', type=Path, required=True); w.add_argument('--lease-fd', type=int, required=True)
    a = vars(p.parse_args()); command = a.pop('command')
    if command == 'freeze':
        queue = a.pop('queue'); output = a.pop('run_dir')
        if a['domain'] is not None: a['domain'] = B.read(a['domain'])
        result = freeze(queue, output, **{k: v for k, v in a.items() if v is not None})
    elif command == 'run': result = run(a['run_dir'], max_waves=a['max_waves'])
    else: result = worker(a['run_dir'], a['child_index'], a['permit'], a['lease_fd'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
