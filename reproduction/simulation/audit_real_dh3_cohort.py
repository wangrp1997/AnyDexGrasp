"""Read-only audit of one frozen original-DH3 real-mesh scene cohort.

No model, collision engine, renderer, physics client or hardware is imported.
All declared rows, including no-candidate/stage-error/zero-step outcomes, receive
an independent receipt. Saved outcomes are never replaced by a new attempt.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time
import traceback

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, label, condition, **detail):
        self.rows.append(dict(check=label, passed=bool(condition), **detail))
        return bool(condition)

    def call(self, label, function):
        try:
            value = function()
            self.check(label, True)
            return value
        except Exception as exc:
            self.check(label, False, error_type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc())
            return None

    @property
    def passed(self):
        return all(row['passed'] for row in self.rows)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_file(item):
    path = Path(item['path'])
    require(path.is_file(), 'Missing bound file: ' + str(path))
    if 'bytes' in item:
        require(path.stat().st_size == item['bytes'], 'Bound byte size changed: ' + str(path))
    require(sha(path) == item['sha256'], 'Bound SHA changed: ' + str(path))
    return record(path)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_attempt(function, *args):
    try:
        return dict(executed=True, passed=True, result=function(*args))
    except Exception as exc:
        return dict(executed=True, passed=False, error_type=type(exc).__name__,
                    error=str(exc), traceback=traceback.format_exc())


def counter(path):
    raw = Path(path).read_bytes()
    require(len(raw) == 16, 'Counter must contain exactly two uint64 values')
    attempted, completed = struct.unpack('<QQ', raw)
    require(completed <= attempted <= completed + 1, 'Invalid attempted/completed step counter')
    return dict(attempted_python_step_calls=attempted, completed_physics_steps=completed,
                possible_in_flight_steps=attempted - completed,
                actual_physical_attempts=int(completed > 0))


def scene_binding(row, frozen):
    for key in ('scene_id', 'batch_frame_name', 'scene_manifest', 'scene_manifest_sha256',
                'scene_signature_sha256', 'scene_module', 'input_files'):
        require(row.get(key) == frozen.get(key), 'Cohort frozen scene field changed: ' + key)
    path = Path(row['scene_manifest'])
    require(sha(path) == row['scene_manifest_sha256'], 'Scene manifest SHA changed')
    scene = read(path)
    signature = hashlib.sha256(json.dumps(dict(spec=scene['spec'], source_assets=scene['source_assets']),
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(signature == scene['scene_signature_sha256'] == row['scene_signature_sha256'],
            'Scene signature mismatch')
    for item in scene['files'] + scene['source_assets']:
        verify_file(item)
    for item in row['input_files'].values():
        verify_file(item)
    return dict(scene_manifest=record(path), scene_signature_sha256=signature,
                bound_scene_files=len(scene['files']), bound_source_assets=len(scene['source_assets']),
                observed_scene_module=scene.get('scene_module'))


def partial_pipeline_binding(run_dir, row, frame_row):
    """Check extant receipts even if an original gate stopped the pipeline."""
    path = run_dir / 'RESULTS.json'
    result = dict(frame_status=(frame_row or {}).get('status'), available_stage_receipts={})
    if not path.is_file():
        require(row['status'] in ('inference_error', 'inference_not_completed', 'interrupted_before_physics'),
                'Final frame receipt unexpectedly missing')
        result['final_frame_receipt_present'] = False
        return result
    run = read(path)
    result['final_frame_receipt_present'] = True
    result['final_frame_receipt'] = record(path)
    for expected in (row.get('inference_RESULTS_sha256'), (frame_row or {}).get('RESULTS_sha256')):
        if expected:
            require(sha(path) == expected, 'Final frame receipt SHA changed')
    for item_path, item in run.get('input_files', {}).items():
        verify_file(dict(path=item_path, **item))
    if 'scene_binding' in run:
        require(run['scene_binding']['manifest_sha256'] == row['scene_manifest_sha256'], 'Frame scene SHA')
        require(run['scene_binding']['scene_signature_sha256'] == row['scene_signature_sha256'], 'Frame signature')
        require(run['scene_binding']['manifest_or_object_pose_fed_to_network'] is False, 'Ground truth network input')
    for stage in ('frame', 'hand', 'collision', 'selection'):
        stage_path = run_dir / stage / 'RESULTS.json'
        stage_report = run.get(stage + '_stage', {})
        expected = stage_report.get('RESULTS_sha256')
        if expected:
            require(sha(stage_path) == expected, 'Stage receipt SHA: ' + stage)
        if stage_path.exists():
            result['available_stage_receipts'][stage] = record(stage_path)
    result['pipeline_status'] = run.get('status')
    result['original_stop_reason'] = run.get('stop_reason')
    result['timing_seconds'] = run.get('timing_seconds')
    return result


def worker_binding(row, run_dir, frozen_sources):
    invocation = row.get('physics_invocations', 0)
    require(invocation in (0, 1), 'More than one physics invocation')
    if invocation == 0:
        require(row.get('completed_physics_steps', 0) == 0, 'Uninvoked row contains physical steps')
        require(row.get('actual_physical_attempts', 0) == 0, 'Uninvoked row contains a physical attempt')
        require(row.get('possible_in_flight_steps', 0) == 0, 'Uninvoked row has an in-flight step')
        require(row.get('not_entered_physics_reason'), 'No explicit reason for not entering physics')
        # A pre-spawn dispatch error may have created a zero counter/job; retain it.
        existing_counter = run_dir / 'progress' / (row['batch_frame_name'] + '.steps.bin')
        extra = counter(existing_counter) if existing_counter.exists() else None
        require(extra is None or extra['attempted_python_step_calls'] == 0, 'Undeclared physical execution')
        return dict(physics_invocations=0, actual_physical_attempts=0, completed_physics_steps=0,
                    possible_in_flight_steps=0, pre_spawn_counter=extra,
                    not_entered_physics_reason=row['not_entered_physics_reason'])
    job_path = Path(row['job_path']); job = read(job_path)
    for path, expected in job['bound_files'].items():
        require(sha(path) == expected, 'Worker input SHA changed: ' + path)
    for path, expected in frozen_sources.items():
        if path in job['bound_files']:
            require(job['bound_files'][path] == expected, 'Worker source differs from frozen cohort')
    for key in ('scene_manifest', 'scene_manifest_sha256', 'selection_results', 'selection_sha256',
                'physics_output_dir', 'step_counter', 'worker_receipt'):
        require(job[key] == row[key], 'Worker job/row mismatch: ' + key)
    require(job['bound_files'][row['selected_candidate_npz']] == row['selected_candidate_sha256'], 'Candidate binding')
    require(sha(row['selected_candidate_npz']) == row['selected_candidate_sha256'], 'Candidate file SHA')
    selected = read(row['selection_results'])
    require(selected.get('selected') is True, 'Physics ran without an original selected top1')
    require(selected['output']['sha256'] == row['selected_candidate_sha256'], 'Top1 output binding')
    actual = counter(row['step_counter'])
    for key in ('completed_physics_steps', 'possible_in_flight_steps', 'actual_physical_attempts'):
        require(actual[key] == row[key], 'Counter/row mismatch: ' + key)
    receipt_path = Path(row['worker_receipt'])
    if not receipt_path.exists():
        require(row['status'] == 'physics_execution_error', 'Normally completed worker receipt missing')
        return dict(**actual, physics_invocations=1, job=record(job_path), worker_receipt_present=False,
                    accounting_boundary='Mmap completed-step lower bound; missing normal worker receipt retained')
    worker = read(receipt_path)
    require(worker == row.get('worker_result'), 'Worker receipt differs from scheduler copy')
    require(worker['job_sha256'] == sha(job_path), 'Worker job SHA')
    require(worker['dispatch_attempt_number'] == worker['physics_invocations'] == 1, 'Worker retry declared')
    require(worker['initial_scene_frozen_per_client'] is True, 'Initial scene not frozen')
    require(worker['control_source_modified'] is False and worker['counting_wrapper_only'] is True, 'Worker control boundary')
    for key in ('completed_physics_steps', 'possible_in_flight_steps', 'actual_physical_attempts', 'attempted_python_step_calls'):
        require(worker[key] == actual[key], 'Worker/counter mismatch: ' + key)
    require(worker['completed_python_step_calls'] == actual['completed_physics_steps'], 'Worker completed calls')
    out = dict(**actual, physics_invocations=1, job=record(job_path), worker_receipt=record(receipt_path),
               worker_status=worker['status'], worker_wall_seconds=worker['worker_wall_seconds'])
    physics_path = Path(row['physics_output_dir']) / 'RESULTS.json'
    if worker['status'] == 'completed':
        require(row['physics_returncode'] == 0, 'Normal worker nonzero exit')
        require(actual['possible_in_flight_steps'] == 0 and worker['actual_step_count_exact'] is True, 'Normal count not exact')
        physics = read(physics_path)
        require(sha(physics_path) == row['physics_RESULTS_sha256'] == worker['physics_RESULT_sha256'], 'Physics receipt SHA')
        with np.load(Path(row['physics_output_dir']) / 'trace.npz', allow_pickle=False) as arrays:
            require(len(arrays['step']) == physics['physics_steps'] == actual['completed_physics_steps'] == 19680,
                    'Normal actual calls/trace/result/full script step counts differ')
            require(np.array_equal(arrays['step'], np.arange(1, 19681)), 'Trace step sequence')
        out['physics_RESULTS'] = record(physics_path)
    else:
        require(row['status'] == 'physics_execution_error', 'Worker error concealed by final status')
        out['retained_worker_error'] = dict(error_type=worker.get('error_type'), error=worker.get('error'))
    return out


def event_audit(events, rows, result):
    active = set(); started = collections.Counter(); finished = collections.Counter()
    no_physics = collections.Counter(); ready = collections.Counter(); peak = 0
    by_id = {r['scene_id']: r for r in rows}; previous = -float('inf')
    for event in events:
        require(event['scheduler_elapsed_s'] >= previous, 'Event monotonic order changed')
        previous = event['scheduler_elapsed_s']; kind = event['event']; key = event.get('scene_id')
        if key is not None:
            require(key in by_id, 'Event refers to an undeclared scene')
        if kind == 'original_top1_ready':
            ready[key] += 1
        if kind == 'physics_started':
            require(key not in active and started[key] == 0 and ready[key] == 1, 'Repeated/unselected physics dispatch')
            started[key] += 1; active.add(key); peak = max(peak, len(active))
            require(event['concurrency'] == len(active) <= result['max_physics_workers'] <= 3, 'CPU physics concurrency exceeded')
            require(event['selection_sha256'] == by_id[key]['selection_sha256'], 'Event selection SHA')
        elif kind == 'physics_finished':
            require(key in active and finished[key] == 0, 'Unmatched/repeated physics completion')
            active.remove(key); finished[key] += 1
            require(event['completed_physics_steps'] == by_id[key]['completed_physics_steps'], 'Event step count')
            require(event['status'] == by_id[key]['status'], 'Event final outcome')
        elif kind == 'scene_final_without_physics':
            no_physics[key] += 1
            require(event['status'] == by_id[key]['status'], 'No-physics outcome changed')
    require(not active, 'Unfinished workers remain in completed event stream')
    require(peak == result['peak_concurrent_physics_workers'], 'Peak concurrency differs from event reconstruction')
    for row in rows:
        key = row['scene_id']; n = row.get('physics_invocations', 0)
        require(started[key] == finished[key] == n <= 1, 'Row/event invocation mismatch: ' + key)
        require(no_physics[key] == int(n == 0), 'Unaccounted no-physics terminal row: ' + key)
    return dict(event_count=len(events), reconstructed_peak_CPU_physics_workers=peak,
        physics_dispatches=dict(started), physics_finishes=dict(finished),
        explicit_unexecuted_rows=dict(no_physics), active_at_end=0,
        gpu_physics_batching=False, once_dispatched_per_scene=True)


def cache_audit(batch_dir):
    path = batch_dir / 'MODEL_CACHE.json'; cache = read(path); c = cache['counts']; events = cache['events']
    load_count = c['representation_checkpoint_file_loads'] + c['decision_checkpoint_file_loads']
    builds = c['representation_model_constructions'] + c['decision_model_constructions']
    requests = c['representation_constructor_requests'] + c['decision_constructor_requests']
    require(load_count == cache['model_load_count'] == builds <= 5, 'Actual model load/construction count')
    require(c['representation_model_constructions'] <= 1 and c['decision_model_constructions'] <= 4, 'Repeated model construction')
    require(c['strict_state_dict_assignments'] == builds, 'First strict assignments')
    require(c['identical_cached_strict_load_skips'] == requests - builds, 'Cached strict assignment skip count')
    require(c['constructor_CPU_RNG_replays'] + c['constructor_CPU_RNG_pre_mismatches'] == requests - builds, 'Constructor RNG accounting')
    require(c['constructor_CPU_RNG_pre_mismatches'] == 0, 'Constructor RNG state mismatch (not silently ignored)')
    rng_path = batch_dir / cache['constructor_CPU_rng_states']
    with np.load(rng_path, allow_pickle=False) as arrays:
        labels = list(arrays['model_labels'])
        recorded = [e for e in events if e['event'] == 'constructor_rng_recorded']
        require(len(labels) == len(recorded) == builds, 'Constructor RNG state inventory')
        for i, (label, event) in enumerate(zip(labels, recorded)):
            require(str(label) == event['model'], 'Constructor RNG model label')
            for suffix in ('before', 'after'):
                actual = hashlib.sha256(arrays['model_%d_%s' % (i, suffix)].tobytes()).hexdigest()
                require(actual == event['CPU_' + suffix + '_sha256'], 'Constructor RNG tensor SHA')
            require(event['CUDA_constructor_rng_unchanged'] is True, 'Constructor changed CUDA RNG')
    replay = [e for e in events if e['event'] == 'cached_constructor_rng_transition']
    require(len(replay) == c['constructor_CPU_RNG_replays'], 'RNG replay event count')
    require(all(e['incoming_CPU_state_equals_first_constructor_pre_state'] and e['recorded_CPU_post_state_replayed']
                for e in replay), 'RNG replay transition not recorded')
    for p, signature in cache['checkpoint_signatures'].items():
        verify_file(dict(path=p, **signature))
    return dict(model_cache=record(path), constructor_RNG_states=record(rng_path),
        actual_checkpoint_loads=load_count, actual_constructions=builds, counts=c,
        live_model_object_ids=cache['live_model_object_ids'],
        constructor_RNG_state_hashes_verified=True,
        forward_random_API_calls_observed={k: c[k] for k in ('torch_multinomial_calls', 'torch_randperm_calls')},
        boundary='Saved cache/RNG bookkeeping verified; model predictions are not recomputed by this audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path('/home/wangrenpeng/AnyDexGrasp'))
    parser.add_argument('--expected-scenes', type=int, default=12)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--allow-protocol-subset', action='store_true',
        help='Audit this frozen group as an explicit ordered subset of the full protocol; no merged-run claim')
    args = parser.parse_args(); start = time.monotonic()
    run_dir = args.run_dir.resolve(); out = args.output_dir.resolve(); repo = args.repo_root.resolve()
    if out.exists():
        parser.error('Retain previous receipts; use a new output directory')
    if out == run_dir or run_dir in out.parents:
        parser.error('Audit output must be outside the frozen run directory')
    out.mkdir(parents=True)
    checks = Checks(); sources = {}; result = None; rows = []; receipts = []
    summary = dict(scope=__doc__, audit_source=record(__file__), input_run=str(run_dir),
        expected_scenes=args.expected_scenes, additional_network_forwards=0,
        additional_collision_queries=0, additional_renders=0, additional_physical_steps=0,
        hardware_connections=0, all_declared_outcomes_retained=True)
    try:
        result = read(run_dir / 'RESULTS.json'); frozen = read(run_dir / 'FROZEN_INPUTS.json')
        rows = result['scenes']; originals = frozen['scenes']; ordered = [r['scene_id'] for r in rows]
        checks.check('all_expected_rows_retained_in_frozen_order', len(rows) == len(originals) == args.expected_scenes
                     and ordered == frozen['scene_order'] == [r['scene_id'] for r in originals]
                     and len(set(ordered)) == len(rows), observed_order=ordered)
        checks.check('cohort_accounting_terminal', result['status'] == 'cohort_finished'
                     and all(r.get('terminal') for r in rows), status=result['status'])
        checks.check('frozen_inputs_SHA', sha(run_dir / 'FROZEN_INPUTS.json') == result['frozen_inputs_sha256'])
        checks.check('frozen_sources_consistent', frozen['sources'] == result['frozen_sources'])
        for path, expected in frozen['sources'].items():
            checks.call('frozen_source_SHA:' + path, lambda p=path, h=expected: require(sha(p) == h, 'Frozen source SHA changed'))
        if frozen.get('declared_scene_list'):
            checks.call('declared_scene_list_SHA', lambda: verify_file(frozen['declared_scene_list']))
        protocol_path = args.protocol.resolve() if args.protocol else run_dir.parent / 'PROTOCOL.json'
        if protocol_path.exists():
            protocol = read(protocol_path); summary['protocol'] = record(protocol_path)
            if args.allow_protocol_subset:
                full_order = protocol['scene_names']
                checks.check('predeclared_protocol_ordered_subsequence',
                    all(scene_id in full_order for scene_id in ordered)
                    and ordered == [scene_id for scene_id in full_order if scene_id in set(ordered)])
                summary['protocol_subset_scope'] = dict(explicitly_enabled=True,
                    full_protocol_scene_count=len(full_order), audited_group_scene_count=len(ordered),
                    audited_group_scene_ids=ordered,
                    boundary='This receipt audits this group only. It is not the full protocol denominator or one combined scheduler/model process.')
            else:
                checks.check('predeclared_scene_order', ordered == protocol['scene_names'])
            checks.check('predeclared_CPU_worker_limit', result['max_physics_workers'] == protocol['physics']['max_simultaneous_CPU_workers'] <= 3)
        checks.check('one_top1_only_no_arm_no_training_no_pipeline_encoding',
            result['same_original_top1_only'] is True and frozen['same_original_top1_only'] is True
            and result['per_scene_physics_invocation_limit'] == frozen['physical_attempt_limit_per_scene'] == 1
            and result['actual_hz'] == frozen['actual_hz'] == 1920
            and result['arm_execution'] is frozen['arm_execution'] is False
            and frozen['network_ground_truth_input'] is False and result['training_steps'] == 0
            and result['video_encoding_executed'] is False)
        physics_source = repo / 'reproduction/simulation/audit_dh3_scene_results.py'
        frame_source = repo / 'reproduction/simulation/audit_synthetic_frame_binding.py'
        physics_auditor = module_at('cohort_saved_physics_auditor', physics_source)
        frame_auditor = module_at('cohort_saved_frame_auditor', frame_source)
        summary['reused_auditors'] = [record(physics_source), record(frame_source)]
        batch_path = run_dir / 'inference/RESULTS.json'
        batch = checks.call('batch_final_receipt_read', lambda: read(batch_path)) or {}
        frames = {f['name']: f for f in batch.get('frames', [])}
        if batch:
            checks.check('all_batch_rows_retained', [f['name'] for f in batch['frames']] == [r['batch_frame_name'] for r in rows])
            for p, h in batch.get('source_files', {}).items():
                checks.call('batch_source_SHA:' + p, lambda p=p, h=h: require(sha(p) == h, 'Batch source changed'))
            summary['batch_final_receipt'] = record(batch_path)
        frozen_by_id = {r['scene_id']: r for r in originals}
        for index, row in enumerate(rows):
            per = Checks(); scene_id = row['scene_id']; pipeline_dir = run_dir / 'inference' / row['batch_frame_name']
            row_receipt = dict(scene_id=scene_id, declared_index=index, final_status=row['status'],
                final_top1_selected=row.get('final_top1_selected'), physics_invocations=row.get('physics_invocations', 0),
                actual_physical_attempts=row.get('actual_physical_attempts', 0),
                completed_physics_steps=row.get('completed_physics_steps', 0),
                possible_in_flight_steps=row.get('possible_in_flight_steps', 0),
                not_entered_physics_reason=row.get('not_entered_physics_reason'),
                retained_error_reason=row.get('error_reason'), input_scheduler_row=row,
                additional_network_forwards=0, additional_collision_queries=0,
                additional_renders=0, additional_physical_steps=0)
            row_receipt['scene_binding'] = per.call('frozen_scene_RGBD_and_source_binding',
                lambda: scene_binding(row, frozen_by_id[scene_id]))
            row_receipt['partial_pipeline_binding'] = per.call('available_pipeline_stage_SHA_links',
                lambda: partial_pipeline_binding(pipeline_dir, row, frames.get(row['batch_frame_name'])))
            row_receipt['dispatch_and_steps'] = per.call('single_dispatch_job_counter_worker_trace_binding',
                lambda: worker_binding(row, run_dir, frozen['sources']))
            # Always invoke both original pure-file auditors. Missing outputs in an
            # explicitly unexecuted stage remain visible, rather than dropping rows.
            inf_audit = audit_attempt(frame_auditor.audit, Path(row['scene_manifest']).parent, pipeline_dir)
            if inf_audit['passed']:
                inf_audit['result']['timing_boundary'] = (
                    'This row uses its saved timing_seconds, not the old cube literals. '
                    'eleven_original_get_grasp_branches measures this frame\'s original eleven branches; '
                    'pipeline internal time includes this row\'s load/reuse, learned scoring, '
                    'asset preparation and collision as itemized. Shared-process/cohort '
                    'wall and overlapped CPU physics are reported separately at cohort level; '
                    'these costs must not be added as a serial total.')
                inf_audit['result']['timing_boundary_receipt_override'] = True
            inf_audit['coverage'] = ('complete_saved_original_pipeline' if inf_audit['passed'] else
                'incomplete_or_invalid_saved_pipeline')
            row_receipt['original_frame_auditor'] = inf_audit
            if row.get('final_top1_selected') is True:
                per.check('selected_row_full_saved_frame_audit', inf_audit['passed'])
            else:
                row_receipt['unexecuted_frame_audit_boundary'] = (
                    'Full original auditor was attempted; absent later artifacts are retained. '
                    'Available input/receipt hashes are checked separately. No selected top1 is invented.')
                per.check('unselected_row_not_physically_executed', row.get('physics_invocations', 0) == 0)
                late_selection = pipeline_dir / 'selection/RESULTS.json'
                expected_no_output = (inf_audit.get('error_type') == 'KeyError'
                    and inf_audit.get('error') == "'output'" and late_selection.exists()
                    and read(late_selection).get('selected') is False)
                if not inf_audit['passed'] and inf_audit.get('error_type') != 'FileNotFoundError' and not expected_no_output:
                    per.check('unselected_full_auditor_has_no_integrity_assertion_failure', False,
                              error_type=inf_audit.get('error_type'), error=inf_audit.get('error'))
            physics_dir = Path(row.get('physics_output_dir', run_dir / 'physics' / row['batch_frame_name']))
            phy_audit = audit_attempt(physics_auditor.audit, physics_dir)
            phy_audit['coverage'] = 'complete_saved_physics' if phy_audit['passed'] else 'unexecuted_or_incomplete_saved_physics'
            row_receipt['original_physics_auditor'] = phy_audit
            normal = row.get('worker_result', {}).get('status') == 'completed'
            if normal:
                per.check('normal_worker_full_independent_physics_audit', phy_audit['passed'])
                if phy_audit['passed']:
                    recomputed = phy_audit['result']
                    expected = ('object_lifted' if recomputed['validated_grasp'] else
                        ('physics_port_invalid' if not recomputed['port_stability_pass'] else 'object_not_lifted'))
                    per.check('independently_recomputed_final_outcome', expected == row['status'])
            else:
                row_receipt['unexecuted_or_error_physics_audit_boundary'] = (
                    'Full original auditor was attempted; a missing/partial trace is not claimed '
                    'as a completed grasp. Worker/counter evidence retains zero or partial steps.')
            row_receipt['checks'] = per.rows; row_receipt['integrity_pass'] = per.passed
            row_receipt['full_pipeline_audit_pass'] = inf_audit['passed']
            row_receipt['full_physics_audit_pass'] = phy_audit['passed']
            path = out / 'scenes' / ('%02d_%s.json' % (index, scene_id))
            save(path, row_receipt)
            receipts.append(dict(scene_id=scene_id, final_status=row['status'], receipt=record(path),
                integrity_pass=per.passed, full_pipeline_audit_pass=inf_audit['passed'],
                full_physics_audit_pass=phy_audit['passed']))
            print(json.dumps(dict(scene_id=scene_id, integrity_pass=per.passed,
                                  final_status=row['status'])), flush=True)
        events = [json.loads(line) for line in (run_dir / 'EVENTS.jsonl').read_text().splitlines() if line.strip()]
        summary['dispatch_events'] = checks.call('event_once_dispatch_and_CPU_concurrency', lambda: event_audit(events, rows, result))
        derived = dict(final_status_counts=dict(collections.Counter(r['status'] for r in rows)),
            finalized_scenes=sum(r['terminal'] for r in rows), original_top1_selected=sum(r.get('final_top1_selected') is True for r in rows),
            physics_invocations=sum(r.get('physics_invocations', 0) for r in rows),
            actual_physical_attempts=sum(r.get('actual_physical_attempts', 0) for r in rows),
            completed_physics_steps=sum(r.get('completed_physics_steps', 0) for r in rows),
            possible_in_flight_steps=sum(r.get('possible_in_flight_steps', 0) for r in rows),
            valid_object_lifted=sum(r['status'] == 'object_lifted' for r in rows),
            observed_lifted=sum(r.get('lift_observed') is True for r in rows),
            not_entered_physics=sum(r['terminal'] and r.get('physics_invocations', 0) == 0 for r in rows),
            active_physics_workers=0, queued_physics_workers=0)
        checks.check('cohort_aggregate_recomputed_exactly', derived == result['aggregate'], recomputed=derived)
        checks.check('all_row_integrity_receipts_pass', len(receipts) == args.expected_scenes and all(r['integrity_pass'] for r in receipts))
        summary['model_cache_audit'] = checks.call('shared_models_and_constructor_RNG_evidence', lambda: cache_audit(run_dir / 'inference'))
        physics_ends = [e for e in events if e['event'] == 'physics_finished']
        inference_ends = [e for e in events if e['event'] == 'inference_finished']
        if inference_ends:
            end = max([e['unix'] for e in inference_ends + physics_ends])
            reconstructed = end - result['pipeline_start_unix']
            recorded = result['timing_seconds']['pipeline_and_last_physics_wall']
            checks.check('pipeline_to_last_completion_wall_crosscheck', abs(reconstructed - recorded) < 1,
                saved_monotonic_wall_seconds=recorded, event_unix_reconstruction_seconds=reconstructed,
                difference_seconds=reconstructed - recorded, tolerance_seconds=1.)
        summary.update(recomputed_aggregate=derived, timing_seconds=result['timing_seconds'],
            batch_timing_seconds=batch.get('timing_seconds'), input_results=record(run_dir / 'RESULTS.json'),
            frozen_inputs=record(run_dir / 'FROZEN_INPUTS.json'), events=record(run_dir / 'EVENTS.jsonl'))
    except Exception as exc:
        checks.check('audit_driver_completed', False, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    # A malformed/missing report or an unexpected driver exception must not hide
    # the other frozen scenes. Preserve an explicit failed receipt for each.
    declared = []
    try:
        declared = read(run_dir / 'FROZEN_INPUTS.json')['scenes']
    except Exception:
        if result is not None:
            declared = result.get('scenes', [])
    covered = {r['scene_id'] for r in receipts}
    for index, declared_row in enumerate(declared):
        if declared_row['scene_id'] in covered:
            continue
        missing = dict(scene_id=declared_row['scene_id'], declared_index=index,
            status='audit_could_not_reach_this_declared_row', integrity_pass=False,
            original_frozen_row=declared_row,
            error='See cohort checks; no missing row was silently removed.',
            additional_network_forwards=0, additional_collision_queries=0,
            additional_renders=0, additional_physical_steps=0)
        path = out / 'scenes' / ('%02d_%s.json' % (index, declared_row['scene_id']))
        save(path, missing)
        receipts.append(dict(scene_id=declared_row['scene_id'], final_status='audit_record_missing',
            receipt=record(path), integrity_pass=False,
            full_pipeline_audit_pass=False, full_physics_audit_pass=False))
    if len(receipts) != args.expected_scenes or not all(r['integrity_pass'] for r in receipts):
        checks.check('every_expected_row_has_a_validated_receipt', False,
                     observed=len(receipts), expected=args.expected_scenes)
    summary.update(checks=checks.rows, scene_receipts=receipts, integrity_pass=checks.passed,
        full_pipeline_rows=sum(r['full_pipeline_audit_pass'] for r in receipts),
        full_physics_rows=sum(r['full_physics_audit_pass'] for r in receipts),
        all_requested_rows_have_receipts=len(receipts) == args.expected_scenes,
        audit_wall_seconds=time.monotonic() - start,
        timing_boundary='Cohort saved monotonic wall covers the original shared-model batch start through the last inference/physics completion. CPU physics clients overlap with inference and each other; summed per-scene CPU wall is not total elapsed wall. Pre-render/decomposition and later video encoding are separate.',
        physical_scope='This hand-only predeclared simulation cohort; no arm, no real hardware, no original-paper benchmark reproduction.')
    summary['status'] = ('cohort_saved_evidence_verified' if summary['integrity_pass'] else 'cohort_saved_evidence_audit_failed')
    save(out / 'RESULTS.json', summary)
    print(json.dumps(dict(status=summary['status'], rows=len(receipts),
        full_pipeline_rows=summary['full_pipeline_rows'], full_physics_rows=summary['full_physics_rows'],
        audit_wall_seconds=summary['audit_wall_seconds'])), flush=True)
    return 0 if summary['integrity_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
