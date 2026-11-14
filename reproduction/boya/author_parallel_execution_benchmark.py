"""Two once-only diagnostic executions using the existing wave resource pool.

Freeze is read-only apart from a fresh benchmark directory. Run reuses the
original execute/audit entries and slot/process-group machinery. These records
are explicitly excluded from collection, training and success-rate estimates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time
import traceback

import author_collection_batch as B
import author_collection_wave_resources as R
import audit_author_collection_runtime as A

SCHEMA = 'anydex_boya_parallel_execution_benchmark_v1'
COPIES = ('READY.json', 'LIFT_PIPELINE_PROTOCOL.json', 'LIFT_PIPELINE_RESULTS.json')
DEFAULT_BATCH = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development/batch_v1')
DEFAULT_SCENE = '0000001_s40_lift_fit_00001_s2026111401'
LOCK = B.ROOT/'outputs/author_collection_gpu0.lock'
REGISTRY = LOCK.with_suffix('.wave_active.json')
LIMITATION = ('Two known-success plans, one new execution each; historical serial '
    'stage walls are not a fresh randomized paired serial benchmark. No complete '
    'pipeline speedup, population success rate or new training labels is claimed.')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _original(prepared, split_manifest):
    """Check the audit's real pre-execution contract without launching anything."""
    prepared = Path(prepared).resolve()
    bound = A.Bindings()
    for name in COPIES:
        bound.add(prepared/name)
    ready = B.read(prepared/'READY.json')
    protocol = B.read(prepared/'LIFT_PIPELINE_PROTOCOL.json')
    prep = B.read(prepared/'LIFT_PIPELINE_RESULTS.json')
    _need(ready.get('task') == B.TASK and protocol.get('task') == B.TASK
          and protocol.get('bin', 'missing') is None and ready.get('status') == 'ready'
          and prep.get('status') == 'ready' and prep.get('complete') is True
          and prep['ready_sha256'] == B.sha(prepared/'READY.json'), 'Unchanged complete lift preparation required')
    _need(B.sha(prepared/'LIFT_PIPELINE_PROTOCOL.json') == ready['protocol']['sha256']
          and B.sha(ready['protocol']['path']) == ready['protocol']['sha256'],
          'Copied and absolute original protocols must be byte-identical')
    bound.tree(ready)
    for key in ('inputs', 'sources', 'capture_source_bindings'):
        bound.tree(protocol.get(key, {}))
    scene = B.read(ready['scene']['path'])
    ids, original, converted, split = A._identity(scene, split_manifest, bound)
    _need(protocol.get('original_object_ids') == ids and protocol.get('primary_is_candidate_target') is False
          and protocol.get('candidate_budget') == 1 and protocol.get('fallback_candidate') is False
          and ready['scene'] == protocol['inputs']['scene'], 'Fixed-first, all-body source contract differs')
    for path, digest in scene.get('source_sha256', {}).items():
        bound.add(protocol.get('scene_archive_sources', {}).get(str(Path(path).resolve()), path), digest)
    bound.mapping(scene.get('input_sha256', {}))
    plan = B.read(ready['plan']['path'])
    candidate = B.read(ready['candidate']['path'])
    command = B.read(ready['command_binding']['path'])
    for key in ('source_sha256', 'input_sha256', 'extra_input_sha256'):
        bound.mapping(plan.get(key, {}))
    _, _, _, counts = A._commands(prepared, ready, plan, candidate, command, bound)
    _need(command['execution_spec']['original_object_ids'] == ids, 'Command identities differ')
    # The prior audit is evidence of known success, not a request to re-audit
    # millions of historical contact points. Its original SHA bindings are read.
    old_audit = B.read(prepared/'RUNTIME_AUDIT.json')
    _need(old_audit.get('status') == 'pass' and old_audit.get('label') == 1
          and old_audit.get('formal_training_eligible') is True
          and old_audit['commands_sha256'] == ready['commands']['sha256'], 'Certified prior success required')
    bound.add(prepared/'RUNTIME_AUDIT.json')
    bound.mapping(old_audit['audit_inputs_sha256'])
    bound.verify()
    return dict(ready=ready, protocol=protocol, scene_id=scene['scene_id'], split=split,
        source_object_ids=ids, source_object_digests=original,
        converted_source_object_digests=converted, phase_counts=counts,
        candidate_sha256=candidate['candidate_sha256'], commands_sha256=ready['commands']['sha256'],
        dependencies=bound.files)


def freeze(output_dir, *, source_batch=DEFAULT_BATCH, stage_timeout_s=3600.):
    """Freeze precisely scene1 g1/g6; never import a simulator or run a model."""
    _need(math.isfinite(stage_timeout_s) and stage_timeout_s > 0, 'Positive finite timeout required')
    source_batch = Path(source_batch).resolve()
    batch_path = source_batch/'BATCH_PROTOCOL.json'
    batch = B.read(batch_path)
    _need(batch.get('task') == B.TASK and batch.get('gpu') == 0, 'Original GPU0 lift batch required')
    B.verify(batch['inputs']); B.verify(batch['sources'])
    split = batch['queue']['split_manifest']; B.verify({'split': split})
    dependencies = {str(batch_path): B.sha(batch_path)}
    rows = []
    for gesture in (1, 6):
        old = source_batch/'scenes'/DEFAULT_SCENE/f'g{gesture}'
        checked = _original(old, split['path'])
        stage_path = old.parent/'stages'/f'execute_g{gesture}'/'STAGE.json'
        stage = B.read(stage_path)
        _need(stage.get('status') == 'complete' and stage.get('returncode') == 0
              and stage.get('retry_count') == 0 and math.isfinite(stage['wall_s']) and stage['wall_s'] > 0,
              'Original successful single execution stage and measured wall required')
        _need(stage['command'] == [batch['inputs']['python']['path'], '-B', batch['sources']['execution_entry']['path'],
                                   '--prepared-dir', str(old)], 'Baseline used a different execution entry')
        B.verify(stage['outputs'])
        dependencies.update(checked.pop('dependencies'))
        dependencies[str(stage_path)] = B.sha(stage_path)
        rows.append(dict(job_id=f'g{gesture}', gesture_id=gesture, source_prepared_dir=str(old),
            copied_files={name:B.binding(old/name) for name in COPIES},
            baseline_stage=B.binding(stage_path), baseline_stage_wall_s=stage['wall_s'],
            **{k:v for k,v in checked.items() if k not in ('ready', 'protocol')}))
    _need(rows[0]['scene_id'] == rows[1]['scene_id'] and rows[0]['commands_sha256'] != rows[1]['commands_sha256'],
          'Two distinct g1/g6 commands in the fixed original scene required')
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    for row in rows:
        job = out/'jobs'/row['job_id']; job.mkdir(parents=True)
        for name, source in row['copied_files'].items():
            with (job/name).open('xb') as stream:
                stream.write(Path(source['path']).read_bytes())
            _need(B.sha(job/name) == source['sha256'], 'Byte-identical preparation copy required')
        row['prepared_dir'] = str(job)
    sources = dict(batch['sources'])
    sources.update(benchmark=B.binding(__file__), resources=B.binding(R.__file__),
                   batch_runner=B.binding(B.__file__), runtime_audit=B.binding(A.__file__))
    protocol = dict(schema=SCHEMA, task=B.TASK, purpose='diagnostic_only', exclude_training=True,
        formal_training_eligible=False, success_rate_evaluation=False,
        jobs=rows, job_count=2, executions_per_job=1, retry_budget=0,
        candidate_fallback=False, resource_limits=R.LIMITS, concurrent_jobs=2, gpu=0,
        physics_device='cpu', renderer_multi_gpu_enabled=False,
        inputs=batch['inputs'], sources=sources, inherited_env=batch['inherited_env'], env=batch['env'],
        stage_timeout_s=float(stage_timeout_s), split_manifest=split,
        original_batch=B.binding(batch_path), frozen_dependencies=dependencies,
        isolation='New directories and records only; all absolute plan/candidate/command bindings remain original read-only paths',
        audit_compatibility='Local protocol bytes equal READY absolute original protocol; new execute writes record/evaluation under new prepared_dir/execution',
        no_prepare=True, no_CGR=True, no_model_forward=True, no_training=True,
        limitation=LIMITATION, frozen_unix_s=time.time())
    B.save(out/'BENCHMARK_PROTOCOL.json', protocol, exclusive=True)
    B.save(out/'EXCLUDE_FROM_TRAINING.json', dict(schema=SCHEMA, exclude_training=True,
        reason='Performance diagnostic repeat of known-success commands; original scientific audits remain unmodified'), exclusive=True)
    verify(out)
    return protocol


def verify(directory):
    root = Path(directory).resolve(); protocol = B.read(root/'BENCHMARK_PROTOCOL.json')
    _need(protocol.get('schema') == SCHEMA and protocol.get('exclude_training') is True
          and protocol.get('purpose') == 'diagnostic_only' and protocol.get('job_count') == 2
          and protocol.get('executions_per_job') == 1 and protocol.get('retry_budget') == 0
          and protocol.get('resource_limits') == R.LIMITS and protocol.get('gpu') == 0,
          'Explicit finite GPU0 diagnostic protocol required')
    B.verify(protocol['inputs']); B.verify(protocol['sources'])
    for path, digest in protocol['frozen_dependencies'].items():
        _need(B.sha(path) == digest, 'Original frozen dependency changed: '+path)
    _need(len(protocol['jobs']) == 2 and [j['gesture_id'] for j in protocol['jobs']] == [1, 6], 'Fixed g1/g6 jobs required')
    for row in protocol['jobs']:
        prepared = Path(row['prepared_dir'])
        _need(row['job_id'] == f'g{row["gesture_id"]}'
              and prepared == root/'jobs'/row['job_id'], 'Job escaped fixed benchmark directory')
        _need(set(row['copied_files']) == set(COPIES), 'Exactly three copied preparation files required')
        for name, source in row['copied_files'].items():
            _need(name in COPIES and B.sha(prepared/name) == source['sha256'] == B.sha(source['path']),
                  'Original or copied preparation bytes changed')
    return protocol


def _job(root, protocol, row):
    """Original stages own once-only start guards, resource FDs and PGIDs."""
    job = Path(row['prepared_dir']); job_id = row['job_id']
    execution = B._stage(root/'stages'/f'execute_{job_id}',
        [protocol['inputs']['python']['path'], '-B', protocol['sources']['execution_entry']['path'],
         '--prepared-dir', str(job)], [job/'LIFT_PIPELINE_EXECUTION.json'], protocol)
    audit = None
    if execution.get('status') == 'complete' and (job/'COLLECTION_RECORD.json').exists():
        audit = B._stage(root/'stages'/f'audit_{job_id}',
            [protocol['inputs']['python']['path'], '-B', protocol['sources']['audit_entry']['path'],
             '--prepared-dir', str(job), '--split-manifest', protocol['split_manifest']['path'],
             '--output', str(job/'RUNTIME_AUDIT.json')], [job/'RUNTIME_AUDIT.json'], protocol)
    receipt = B.read(job/'LIFT_PIPELINE_EXECUTION.json') if (job/'LIFT_PIPELINE_EXECUTION.json').exists() else {}
    audit_result = B.read(job/'RUNTIME_AUDIT.json') if (job/'RUNTIME_AUDIT.json').exists() else {}
    result = dict(job_id=job_id, execution_stage=execution, audit_stage=audit,
        execution_receipt=B.binding(job/'LIFT_PIPELINE_EXECUTION.json') if receipt else None,
        audit_binding=B.binding(job/'RUNTIME_AUDIT.json') if audit_result else None,
        persisted_scientific_label=receipt.get('label'), independent_audit_status=audit_result.get('status'),
        physics_steps=receipt.get('physics_steps'),
        execution_steps_unknown=not receipt or receipt.get('status') == 'starting',
        exclude_training=True, formal_training_eligible=False)
    B.save(root/'results'/f'{job_id}.json', result, exclusive=True)
    return result


def _summary(protocol, rows, pair_wall_s):
    stages = [row['execution_stage'] for row in rows]
    complete = len(rows) == 2 and all(s.get('status') == 'complete' for s in stages)
    audited = complete and all(row.get('independent_audit_status') == 'pass' for row in rows)
    overlap = max(0., min(s['started_unix_s']+s['wall_s'] for s in stages)
                  - max(s['started_unix_s'] for s in stages)) if complete else None
    span = (max(s['started_unix_s']+s['wall_s'] for s in stages)
            - min(s['started_unix_s'] for s in stages)) if complete else None
    baseline = sum(j['baseline_stage_wall_s'] for j in protocol['jobs'])
    return dict(schema=SCHEMA, status='complete' if audited else 'infrastructure_or_audit_failure',
        purpose='diagnostic_only', exclude_training=True, formal_training_eligible=False,
        jobs=rows, known_physics_steps=sum(r['physics_steps'] or 0 for r in rows),
        execution_starts_with_unknown_steps=sum(r['execution_steps_unknown'] for r in rows),
        original_serial_execution_stage_wall_sum_s=baseline,
        new_execution_stage_span_s=span, execution_stage_overlap_s=overlap,
        actual_Kit_overlap_requires_independent_host_observer=True,
        observed_execution_stage_speedup=baseline/span if audited and span else None,
        benchmark_pair_wall_including_audits_s=pair_wall_s,
        speedup_scope='Execution stage span includes original per-stage validation/launch/cleanup; excludes new runtime audits, initial freeze and outer verification',
        limitation=LIMITATION, training_updates=0, new_training_labels=0,
        planning_calls=0, CGR_forwards=0, model_forwards=0, retry_count=0)


def run(directory):
    root = Path(directory).resolve(); protocol = verify(root)
    started_path = root/'BENCHMARK_STARTED.json'
    # A parent crash or uncertain child lifetime consumes this benchmark too.
    if started_path.exists():
        raise FileExistsError('A started benchmark is never restarted; review its existing evidence')
    binding = B.binding(root/'BENCHMARK_PROTOCOL.json')
    with R.lease(root/'DRIVER.lock'), R.lease(LOCK) as lease_fd:
        with R.active_run(root, REGISTRY, binding):
            B.save(started_path, dict(protocol=binding, process=R.process_identity(os.getpid()),
                started_unix_s=time.time(), policy='Exactly one attempt per frozen job, including infrastructure failures'), exclusive=True)
            pool = R.ResourcePool(root/'resource_slots', lease_fd, binding)
            begin = time.monotonic(); rows = []; failure = None
            try:
                with R.install_worker_adapters(pool, LOCK):
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        futures = [executor.submit(_job, root, protocol, row) for row in protocol['jobs']]
                        for future in futures:
                            try:
                                rows.append(future.result())
                            except BaseException:
                                failure = traceback.format_exc()
                result = _summary(protocol, rows, time.monotonic()-begin)
                verify(root)
            except BaseException:
                result = _summary(protocol, rows, time.monotonic()-begin)
                failure = traceback.format_exc()
            if failure:
                result.update(status='infrastructure_or_audit_failure', error=failure)
            result['protocol'] = binding
            B.save(root/'BENCHMARK_RESULTS.json', result, exclusive=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--output-dir', type=Path, required=True)
    f.add_argument('--source-batch', type=Path, default=DEFAULT_BATCH)
    f.add_argument('--stage-timeout-s', type=float, default=3600.)
    r = sub.add_parser('run'); r.add_argument('--benchmark-dir', type=Path, required=True)
    a = p.parse_args()
    result = freeze(a.output_dir, source_batch=a.source_batch, stage_timeout_s=a.stage_timeout_s) if a.command == 'freeze' else run(a.benchmark_dir)
    print(json.dumps(dict(schema=result['schema'], status=result.get('status', 'frozen'),
        exclude_training=True, job_count=2), indent=2))


if __name__ == '__main__':
    main()
