"""Diagnostic complete collection loop with the existing persistent CGR and Kit.

Freeze performs no inference, preparation or physics. Every selected scene is
freshly built/captured. The original CGR batch loads its model once; original
preparation and independent audits remain external stages. Ready executions use
one ResidentExecutionSession. This first adapter executes serially and retains
capture subprocesses. Scene groups may run in 1..3 separate resident processes;
the parent never starts Kit. Its outer ledger excludes every result from training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import traceback

import author_collection_batch as B
import author_collection_wave_resources as R

SCHEMA = 'anydex_boya_author_resident_collection_v1'
PROTOCOL = 'RESIDENT_COLLECTION_PROTOCOL.json'
LEGACY_LOCK = B.ROOT/'outputs/author_collection_gpu0.lock'
REGISTRY = LEGACY_LOCK.with_suffix('.wave_active.json')
BATCH_PIN = '153527cbbfa38a3533bf5759a06331ab1c1896da95f8a575d8b18f7e38240959'


def _base(protocol):
    if (protocol.get('schema') != B.SCHEMA or protocol.get('task') != B.TASK
            or protocol.get('gestures') != list(B.GESTURES)
            or protocol.get('max_candidates_per_gesture') != 1
            or protocol.get('max_executions_per_gesture') != 1
            or not 1 <= protocol.get('prepare_workers', 0) <= 3):
        raise ValueError('Original once-only four-gesture batch protocol required')
    if B.sha(B.__file__) != BATCH_PIN:
        raise ValueError('Reviewed original batch source required')
    B.verify(protocol['inputs']); B.verify(protocol['sources'])
    queue = B.validate_queue(protocol['inputs']['queue']['path'])
    if queue != protocol['queue']:
        raise ValueError('Source batch must bind the complete unchanged queue')
    return queue


def _freeze(base, source_protocol, indices, output_dir, resident_workers):
    queue = _base(base)
    if (not isinstance(indices, list) or not 1 <= len(indices) <= 512
            or any(type(i) is not int or not 0 <= i < len(queue['scenes']) for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Freeze 1..512 unique scene indices in original order')
    if type(resident_workers) is not int or not 1 <= resident_workers <= 3:
        raise ValueError('Explicit resident_workers1..3 required')
    output = Path(output_dir).resolve()
    protocol = copy.deepcopy(base)
    protocol.update(schema=SCHEMA, source_batch_protocol=source_protocol,
        source_queue=B.binding(base['inputs']['queue']['path']), source_scene_indices=indices,
        purpose='known_scene_complete_throughput_diagnostic', diagnostic_only=True,
        exclude_training=True, automatic_training=False, physics_concurrency=resident_workers,
        resident_workers=resident_workers,
        capture_resident=False, cgr_model_load_policy='one existing PersistentOfficialCGR per frozen manifest',
        execution_policy='one subprocess/session per scene; original-order ready plans within each scene; one attempt each',
        retry_policy='No run restart after RUN_STARTED; no stage/scene/candidate retry',
        freeze_created_unix_s=time.time(), total_scene_budget=len(indices),
        total_gesture_attempt_budget=4*len(indices), training_updates=0)
    protocol['queue']['scenes'] = [copy.deepcopy(queue['scenes'][i]) for i in indices]
    protocol['sources'].update({n: B.binding(B.HERE/n) for n in (
        'author_resident_collection.py', 'author_resident_execution.py',
        'author_collection_wave_resources.py')})
    # Existing resident/context helpers become runtime dependencies only in
    # this opt-in branch. Freeze their bytes before, rather than at, execution.
    import author_resident_execution as resident
    protocol['sources'].update(frozen_native_resident=B.binding(resident.NATIVE_RESIDENT),
        frozen_official_context=B.binding(resident.OFFICIAL_CONTEXT))
    if source_protocol:
        protocol['inputs']['source_batch_protocol'] = source_protocol
    B.verify(protocol['inputs']); B.verify(protocol['sources'])
    output.mkdir(parents=True, exist_ok=False)
    B.save(output/PROTOCOL, protocol, exclusive=True)
    B.save(output/'PROGRESS.json', progress(protocol, []))
    return protocol


def freeze_from_batch(source_batch, scene_indices, output_dir, *, resident_workers=1):
    """Reference old inputs only; never copy old observations/candidates/plans."""
    source = Path(source_batch).resolve()/'BATCH_PROTOCOL.json'
    return _freeze(B.read(source), B.binding(source), scene_indices, output_dir, resident_workers)


def freeze(queue_path, output_dir, *, max_scenes=None, resident_workers=1, **batch_options):
    """Freeze a new finite queue using the unchanged original metadata builder."""
    with tempfile.TemporaryDirectory(prefix='anydex_resident_freeze_') as temporary:
        base = B.freeze(queue_path, Path(temporary)/'batch', **batch_options)
        count = len(base['queue']['scenes']) if max_scenes is None else max_scenes
        if type(count) is not int or not 1 <= count <= len(base['queue']['scenes']):
            raise ValueError('Positive frozen queue prefix required')
        return _freeze(base, None, list(range(count)), output_dir, resident_workers)


def verify_protocol(root):
    root = Path(root).resolve(); protocol = B.read(root/PROTOCOL)
    if (protocol.get('schema') != SCHEMA or protocol.get('task') != B.TASK
            or protocol.get('exclude_training') is not True
            or protocol.get('diagnostic_only') is not True
            or protocol.get('automatic_training') is not False
            or type(protocol.get('resident_workers')) is not int
            or not 1 <= protocol['resident_workers'] <= 3
            or protocol.get('physics_concurrency') != protocol['resident_workers']):
        raise ValueError('Explicit diagnostic resident protocol required')
    if B.sha(B.__file__) != BATCH_PIN: raise ValueError('Original batch source changed')
    B.verify(protocol['inputs']); B.verify(protocol['sources'])
    B.verify({'source_queue': protocol['source_queue']})
    queue = B.read(protocol['source_queue']['path'])
    indices = protocol['source_scene_indices']
    if (indices != sorted(set(indices)) or not 1 <= len(indices) <= 512
            or protocol['queue']['scenes'] != [queue['scenes'][i] for i in indices]
            or protocol['queue']['seed'] != queue['seed']
            or protocol['gestures'] != list(B.GESTURES)
            or protocol['total_gesture_attempt_budget'] != 4*len(indices)
            or protocol['total_scene_budget'] != len(indices)
            or not 1 <= protocol['prepare_workers'] <= 3):
        raise ValueError('Original scene order/seed or finite budget changed')
    if protocol['source_batch_protocol']:
        source = B.read(protocol['source_batch_protocol']['path'])
        if _base(source) != queue: raise ValueError('Source batch queue differs')
        for key in ('domain', 'native_deps', 'env', 'inherited_env', 'stage_timeout_s'):
            if protocol[key] != source[key]: raise ValueError('Reference configuration changed: '+key)
    return protocol


def ledger(root):
    root = Path(root); rows = []
    for path in sorted((root/'ledger').glob('*.json')):
        row = B.read(path)
        if (row['protocol_sha256'] != B.sha(root/PROTOCOL) or row['label'] is not None
                or row['formal_training_eligible'] is not False or row['exclude_training'] is not True):
            raise ValueError('Diagnostic ledger binding/exclusion changed')
        B.verify(row['artifact_bindings'])
        for item in row['artifact_bindings'].values():
            if Path(item['path']).name == 'STAGE.json': B.verify(B.read(item['path']).get('outputs', {}))
        rows.append(row)
    return rows


def progress(protocol, rows, *, timings=None, started=None, finished=None):
    counts = {s: {str(g): 0 for g in B.GESTURES} for s in ('fit', 'internal')}
    positive = copy.deepcopy(counts)
    for row in rows:
        if row['raw_label'] is not None:
            counts[row['split']][str(row['gesture_id'])] += 1
            positive[row['split']][str(row['gesture_id'])] += row['raw_label']
    captures = {r['queue_index']: r['artifact_bindings']['capture_stage'] for r in rows
        if 'capture_stage' in r['artifact_bindings']}
    completed = sum(B.read(p['path']).get('status') == 'complete' for p in captures.values())
    return dict(schema=SCHEMA, task=B.TASK, status='complete' if len(rows) == protocol['total_gesture_attempt_budget'] else 'pending_or_incomplete',
        eligible_labels=counts, positive_labels=positive, counts_scope='audited diagnostic labels; all excluded from training',
        ledger_rows=len(rows), total_scene_budget=protocol['total_scene_budget'],
        total_gesture_attempt_budget=protocol['total_gesture_attempt_budget'],
        known_final_execution_physics_steps=sum(r['actual_physics_steps'] for r in rows),
        execution_starts_with_unknown_steps=sum(bool(r['execution_steps_unknown']) for r in rows),
        captured_initialization_ticks=completed*750, captured_initialization_object_steps=completed*2250,
        physics_cost_scope='Known final execution receipts and complete captures only; interrupted starts may have unknown cost',
        exclude_training=True, formal_training_eligible=False, old435_merged=False, training_updates=0,
        timing_seconds=timings or {}, started_unix_s=started, finished_unix_s=finished,
        timing_scope='Complete parent loop including capture/CGR, prepare, resident worker shutdown, audit and ledger; outer process startup/exit measured separately')


def _terminal(root, protocol, local_index, row, g, status, *, artifacts, trial=None, error=None, execution=None, attempted=False):
    # All post-execution SHA checks precede even an excluded diagnostic label.
    verify_protocol(root)
    B.verify(artifacts)
    execution = execution or {}
    value = dict(schema=SCHEMA, task=B.TASK, protocol_sha256=B.sha(root/PROTOCOL),
        queue_index=protocol['source_scene_indices'][local_index], local_queue_index=local_index,
        scene_id=row['scene_id'], split=row['split'], gesture_id=g, trial_id=f'{row["scene_id"]}:g{g}',
        source_object_ids=row['source_object_ids'], source_object_digests=row['source_object_digests'],
        scene_spec=row['scene_spec'], status=status, raw_label=None, diagnostic_label=None, label=None,
        exclude_training=True, formal_training_eligible=False, artifact_bindings=dict(artifacts), error=error,
        actual_physics_steps=execution.get('physics_steps', 0),
        execution_steps_unknown=bool(attempted and (execution.get('status') in (None, 'starting')
            or execution.get('_execution_steps_unknown', False))))
    if trial is not None:
        if trial['gesture_id'] != g: raise ValueError('Actual gesture differs')
        value.update(raw_label=trial['label'], diagnostic_label=trial['label'], trial=trial,
            actual_physics_steps=trial['actual_physics_steps'], execution_steps_unknown=False)
        value['artifact_bindings'].update({k: trial[k] for k in ('record', 'candidate', 'evaluation', 'commands', 'command_binding', 'scene')})
    B.save(root/'ledger'/f'{value["queue_index"]:07d}_g{g}.json', value, exclusive=True)
    return value


def _capture_outputs(capture, protocol):
    names = ['RESULTS.json', 'observation.json', 'observed_scene.json']
    entry = Path(protocol['sources']['capture_entry']['path']).name
    if entry in ('author_collection_capture.py', 'author_scene.py'):
        names += ['initialization_trace.npz', 'native_rgbd.npz', 'CONTACT_SENSOR_CONFIG.json',
            'CONTACT_SENSOR_CHECK.json', 'render_geometry_check.json', 'color.png', 'depth.png', 'meta.mat', 'PROGRESS.json']
        names += (['CONTACT_JOURNAL.json', 'initialization_contacts.jsonl.gz', 'CAPTURE_OVERLAY.json']
            if entry == 'author_collection_capture.py' else ['initialization_contacts.json'])
    return [capture/n for n in names]


def _run_locked(root, protocol, pool):
    started = time.time(); start = time.monotonic(); timings = {}
    B.save(root/'RUN_STARTED.json', dict(protocol=B.binding(root/PROTOCOL), started_unix_s=started,
        process=R.process_identity(os.getpid()), no_restart=True), exclusive=True)
    py = protocol['inputs']['python']['path']; rows = []; captured = []; prepared = []
    timing_lock = threading.Lock()
    fatal = None

    def stage(kind, directory, command, expected, stage_protocol=None):
        pool.check_halt()
        begin = time.monotonic()
        try: return B._stage(directory, command, expected, stage_protocol or protocol)
        finally:
            with timing_lock: timings[kind] = timings.get(kind, 0.)+time.monotonic()-begin

    def terminal(i, row, g, status, **kwargs):
        value = _terminal(root, protocol, i, row, g, status, **kwargs)
        rows.append(value)
        B.save(root/'PROGRESS.json', progress(protocol, rows, timings=timings, started=started))
        return value

    try:
        for i, row in enumerate(protocol['queue']['scenes']):
            directory = root/'scenes'/f'{protocol["source_scene_indices"][i]:07d}_{row["scene_id"]}'
            artifacts = {'scene_spec': row['scene_spec']}
            if B.read(row['scene_spec']['path']).get('status', row.get('status')) == 'placement_null':
                for g in B.GESTURES: terminal(i, row, g, 'placement_null', artifacts=artifacts)
                continue
            try:
                build = directory/'build'; capture = directory/'capture'
                result = stage('build', directory/'stages/build', [py, '-B', protocol['sources']['scene_entry']['path'],
                    'build', '--scene-spec', row['scene_spec']['path'], '--output-dir', str(build)], [build/'scene.json'])
                artifacts['build_stage'] = B.binding(directory/'stages/build/STAGE.json')
                if result['status'] != 'complete': raise ValueError('Build failed: '+str(result.get('error')))
                B._scene_objects(build/'scene.json', row)
                command = [py, '-B', protocol['sources']['capture_entry']['path']]
                if Path(command[-1]).name == 'author_scene.py': command.append('capture')
                result = stage('capture', directory/'stages/capture', command+['--scene', str(build/'scene.json'),
                    '--output-dir', str(capture)], _capture_outputs(capture, protocol))
                artifacts['capture_stage'] = B.binding(directory/'stages/capture/STAGE.json')
                if result['status'] != 'complete' or B.read(capture/'RESULTS.json').get('status') != 'complete':
                    raise ValueError('Capture failed: '+str(result.get('error')))
                B._scene_objects(capture/'observed_scene.json', row)
                observation = B.read(capture/'observation.json')
                if (observation['scene_sha256'] != B.sha(capture/'observed_scene.json')
                        or Path(observation['scene_json']).resolve() != (capture/'observed_scene.json').resolve()):
                    raise ValueError('Fresh observation binding differs')
                captured.append((i, row, directory, artifacts))
            except R.Quarantined: raise
            except Exception:
                for g in B.GESTURES: terminal(i, row, g, 'infrastructure_failure', artifacts=artifacts, error=traceback.format_exc())

        if captured:
            manifest = root/'REPRESENTATION_MANIFEST.json'; rep = root/'representation'
            B.save(manifest, dict(objects=[dict(object_id=row['scene_id'], observation_json=str(d/'capture/observation.json'),
                observation_sha256=B.sha(d/'capture/observation.json')) for _, row, d, _ in captured]), exclusive=True)
            expected = [rep/'BATCH_REPRESENTATION.json']
            for _, row, _, _ in captured:
                expected += [rep/'representations'/row['scene_id']/n for n in
                    ('RESULTS.json', 'raw_camera_candidates.npz', 'representation_outputs.npz')]
            result = stage('representation_batch', root/'stages/representation', [py, '-B', str(B.HERE/'run_collection_representation.py'),
                '--manifest', str(manifest), '--run-dir', str(rep), '--device', 'cuda:0', '--seed', str(protocol['queue']['seed']),
                '--checkpoint', protocol['inputs']['checkpoint']['path'], '--native-deps', protocol['native_deps'],
                '--depth-min', '.2', '--depth-max', '.65'], expected)
            representation_ok = result['status'] == 'complete'
            if representation_ok:
                report = B.read(rep/'BATCH_REPRESENTATION.json')
                representation_ok = (report.get('model_load_count') == 1 and report.get('network_forward_count') == len(captured)
                    and report.get('completed_forward_objects') == len(captured) and report.get('hand_head_load_count') == 0)
            for i, row, directory, artifacts in captured:
                artifacts['representation_stage'] = B.binding(root/'stages/representation/STAGE.json')
                if not representation_ok:
                    for g in B.GESTURES: terminal(i, row, g, 'infrastructure_failure', artifacts=artifacts, error='Persistent CGR batch incomplete')
                    continue
                domain = directory/'DOMAIN.json'; B.save(domain, protocol['domain'], exclusive=True)
                def prepare_one(g):
                    target = directory/f'g{g}'
                    return stage('prepare_sum', directory/f'stages/prepare_g{g}', [py, '-B', str(B.HERE/'author_lift_pipeline.py'), 'prepare',
                        '--scene', str(directory/'capture/observed_scene.json'), '--observation', str(directory/'capture/observation.json'),
                        '--representation-dir', str(rep/'representations'/row['scene_id']), '--gesture-id', str(g),
                        '--source-type', str(B.SOURCE_TYPES[g]), '--seed', str(protocol['queue']['seed']),
                        '--calibration', protocol['inputs']['calibration']['path'], '--cloud-manifest', protocol['inputs']['cloud_manifest']['path'],
                        '--backend', protocol['inputs']['backend']['path'], '--domain', str(domain), '--output-dir', str(target)],
                        [target/'LIFT_PIPELINE_RESULTS.json'])
                with ThreadPoolExecutor(max_workers=protocol['prepare_workers']) as executor:
                    results = list(executor.map(prepare_one, B.GESTURES))
                for g, result in zip(B.GESTURES, results):
                    bindings = dict(artifacts, prepare_stage=B.binding(directory/f'stages/prepare_g{g}/STAGE.json'))
                    target = directory/f'g{g}'
                    if result['status'] != 'complete':
                        terminal(i, row, g, 'infrastructure_failure', artifacts=bindings, error=result.get('error')); continue
                    preparation = B.read(target/'LIFT_PIPELINE_RESULTS.json')
                    bindings['preparation'] = B.binding(target/'LIFT_PIPELINE_RESULTS.json')
                    if preparation['status'] != 'ready':
                        status = preparation['status']
                        if status not in ('null_geometry', 'no_candidate', 'infrastructure_failure'): status = 'infrastructure_failure'
                        terminal(i, row, g, status, artifacts=bindings); continue
                    prepared.append((i, row, g, target, directory, bindings))

        if prepared:
            verify_protocol(root)
            groups = {}
            for item in prepared: groups.setdefault(item[0], []).append(item)
            # Only classification changes: all original execution source bindings
            # remain frozen, alongside the explicit new resident entry.
            execution_protocol = copy.deepcopy(protocol)
            execution_protocol['sources']['legacy_execution_entry'] = protocol['sources']['execution_entry']
            execution_protocol['sources']['execution_entry'] = protocol['sources']['author_resident_execution.py']
            def execute_group(items):
                i, row, _, _, directory, _ = items[0]
                output = directory/'resident_session'
                source_bindings = directory/'RESIDENT_SOURCE_BINDINGS.json'
                B.save(source_bindings, protocol['sources'], exclusive=True)
                command = [py, '-B', execution_protocol['sources']['execution_entry']['path'], 'execute-many',
                    '--prepared-dirs', *[str(item[3]) for item in items], '--output-dir', str(output),
                    '--source-bindings', str(source_bindings)]
                result = stage('execution_groups_sum', directory/'stages/resident_execute', command,
                    [output/'EXECUTE_MANY_RESULTS.json', output/'SESSION_RECEIPT.json', source_bindings], execution_protocol)
                return i, result
            begin = time.monotonic()
            with ThreadPoolExecutor(max_workers=protocol['resident_workers']) as executor:
                group_results = dict(executor.map(execute_group, groups.values()))
            timings['execution_parallel_wall'] = time.monotonic()-begin
            # Outcome speed/order never changes label admission order.
            for i, row, g, target, directory, artifacts in prepared:
                execution = {}; attempted = False
                try:
                    pool.check_halt(); verify_protocol(root)
                    group_stage = directory/'stages/resident_execute/STAGE.json'
                    artifacts['execute_stage'] = B.binding(group_stage)
                    group_result = directory/'resident_session/EXECUTE_MANY_RESULTS.json'
                    if group_result.exists(): artifacts['resident_group_results'] = B.binding(group_result)
                    session_receipt = directory/'resident_session/SESSION_RECEIPT.json'
                    if session_receipt.exists(): artifacts['resident_session_receipt'] = B.binding(session_receipt)
                    receipt = target/'LIFT_PIPELINE_EXECUTION.json'
                    attempted = receipt.exists() or (target/'RESIDENT_EXECUTION_ENTRY.json').exists()
                    if receipt.exists():
                        execution = B.read(receipt); artifacts['execution_receipt'] = B.binding(receipt)
                    for name in ('RESIDENT_EXECUTION_ENTRY.json', 'STREAMED_EXECUTION_ENTRY.json'):
                        if (target/name).exists(): artifacts[name] = B.binding(target/name)
                    # A later job or final close failure must not erase a prior
                    # complete, independently auditable job. Missing group/job
                    # completion evidence remains unknown, never a zero-step claim.
                    final_job = receipt.exists() and execution.get('status') not in (None, 'starting')
                    not_started = False
                    if group_result.exists():
                        jobs = B.read(group_result).get('jobs', [])
                        entries = [j for j in jobs if Path(j.get('prepared_dir', '')).resolve() == target.resolve()]
                        not_started = len(entries) == 1 and entries[0].get('status') == 'not_started'
                    if not_started:
                        if attempted or (target/'COLLECTION_RECORD.json').exists():
                            raise ValueError('Group not_started conflicts with an actual job entry')
                        terminal(i, row, g, 'execution_not_started', artifacts=artifacts, execution={}, attempted=False)
                        continue
                    if group_results[i]['status'] != 'complete' and not final_job:
                        attempted = True
                        raise ValueError('Resident subprocess failed; job lifetime/cost unresolved')
                    if execution.get('status') == 'infrastructure_failure':
                        raise ValueError('Original execution receipt reports infrastructure_failure')
                    verify_protocol(root); B.verify(artifacts)
                    record = target/'COLLECTION_RECORD.json'
                    if not record.exists() or B.read(record).get('label') is None:
                        terminal(i, row, g, 'executed_null' if attempted else 'execution_not_started',
                            artifacts=artifacts, execution=execution, attempted=attempted)
                        continue
                    audit = target/'RUNTIME_AUDIT.json'
                    result = stage('audit', directory/f'stages/audit_g{g}', [py, '-B', protocol['sources']['audit_entry']['path'],
                        '--prepared-dir', str(target), '--split-manifest', protocol['queue']['split_manifest']['path'], '--output', str(audit)], [audit])
                    artifacts['audit_stage'] = B.binding(directory/f'stages/audit_g{g}/STAGE.json')
                    if audit.exists(): artifacts['runtime_audit'] = B.binding(audit)
                    if result['status'] != 'complete': raise ValueError('Independent audit stage failed')
                    verify_protocol(root); B.verify(artifacts)
                    trial = B.validate_trial(target, row); B.certify_audit(B.read(audit), trial, row)
                    terminal(i, row, g, 'diagnostic_labelled', artifacts=artifacts, trial=trial, execution=execution, attempted=attempted)
                except R.Quarantined: raise
                except Exception:
                    receipt = target/'LIFT_PIPELINE_EXECUTION.json'
                    if receipt.exists(): execution = B.read(receipt)
                    execution['_execution_steps_unknown'] = (attempted and execution.get('complete') is not True
                        and not (target/'execution/LIFT_EVALUATION.json').exists())
                    terminal(i, row, g, 'infrastructure_failure', artifacts=artifacts, error=traceback.format_exc(), execution=execution, attempted=attempted)
    except BaseException:
        fatal = traceback.format_exc()
        raise
    finally:
        # Workers finalize all receipts before their unique real Kit close.
        timings['total_wall'] = time.monotonic()-start
        finished = time.time()
        result = progress(protocol, rows, timings=timings, started=started, finished=finished)
        result.update(error=fatal, parent_launches_kit=False)
        # Integrity failures may prohibit writing a ledger row. Preserve the
        # physical cost lower bound separately without certifying any label.
        recorded = {(r['local_queue_index'], r['gesture_id']) for r in rows}
        unadmitted = []
        for i, _, g, target, directory, _ in prepared:
            if (i, g) in recorded: continue
            receipt = target/'LIFT_PIPELINE_EXECUTION.json'
            if receipt.exists():
                value = B.read(receipt)
                unadmitted.append(dict(queue_index=protocol['source_scene_indices'][i], gesture_id=g,
                    receipt=B.binding(receipt), known_final_steps=value.get('physics_steps', 0),
                    execution_steps_unknown=value.get('status') in (None, 'starting') or (
                        value.get('complete') is not True and not (target/'execution/LIFT_EVALUATION.json').exists())))
            elif (directory/'stages/resident_execute/STAGE.json').exists():
                unadmitted.append(dict(queue_index=protocol['source_scene_indices'][i], gesture_id=g,
                    receipt=None, known_final_steps=0, execution_steps_unknown=True))
        result['unadmitted_execution_costs'] = unadmitted
        result['known_final_execution_physics_steps'] += sum(r['known_final_steps'] for r in unadmitted)
        result['execution_starts_with_unknown_steps'] += sum(r['execution_steps_unknown'] for r in unadmitted)
        B.save(root/'LEDGER.json', dict(protocol=B.binding(root/PROTOCOL), rows=sorted(rows, key=lambda r: (r['queue_index'], r['gesture_id']))))
        B.save(root/'PROGRESS.json', result)
        B.save(root/'RESIDENT_COLLECTION_RESULTS.json', result, exclusive=True)
    return result


def run(run_dir):
    """Run the frozen diagnostic once. Interrupted starts require review, no retry."""
    root = Path(run_dir).resolve(); protocol = verify_protocol(root)
    with R.lease(root/'DRIVER.lock'), R.lease(LEGACY_LOCK) as lease_fd:
        if (root/'RUN_STARTED.json').exists(): raise FileExistsError('Started diagnostic is never restarted')
        with R.active_run(root, REGISTRY, B.binding(root/PROTOCOL)):
            pool = R.ResourcePool(root/'resource_slots', lease_fd, B.binding(root/PROTOCOL))
            with R.install_worker_adapters(pool, LEGACY_LOCK):
                # Freeze the same environment for in-process execution as external stages.
                previous = {key: os.environ.get(key) for key in protocol['env']}
                os.environ.update(protocol['env'])
                try: return _run_locked(root, protocol, pool)
                finally:
                    for key, value in previous.items():
                        if value is None: os.environ.pop(key, None)
                        else: os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze-from-batch')
    f.add_argument('--source-batch', type=Path, required=True); f.add_argument('--scene-indices', type=int, nargs='+', required=True)
    f.add_argument('--output-dir', type=Path, required=True)
    f.add_argument('--resident-workers', type=int, default=1)
    r = sub.add_parser('run'); r.add_argument('--run-dir', type=Path, required=True)
    a = parser.parse_args()
    result = freeze_from_batch(a.source_batch, a.scene_indices, a.output_dir, resident_workers=a.resident_workers) if a.command == 'freeze-from-batch' else run(a.run_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
