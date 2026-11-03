"""Derive a 12-scene report from two completed, serial original-DH3 cohorts.

This is aggregation only. It does not manufacture one scheduler's event log,
freeze record or five-model cache from two independent processes. The original
ten-scene and two-scene reports and all negative outcomes remain authoritative.
No network, collision query, renderer or physical integration is executed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import time


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


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def aggregate(rows):
    return dict(final_status_counts=dict(collections.Counter(r['status'] for r in rows)),
        finalized_scenes=sum(bool(r['terminal']) for r in rows),
        original_top1_selected=sum(r.get('final_top1_selected') is True for r in rows),
        physics_invocations=sum(r.get('physics_invocations', 0) for r in rows),
        actual_physical_attempts=sum(r.get('actual_physical_attempts', 0) for r in rows),
        completed_physics_steps=sum(r.get('completed_physics_steps', 0) for r in rows),
        possible_in_flight_steps=sum(r.get('possible_in_flight_steps', 0) for r in rows),
        valid_object_lifted=sum(r['status'] == 'object_lifted' for r in rows),
        observed_lifted=sum(r.get('lift_observed') is True for r in rows),
        not_entered_physics=sum(bool(r['terminal']) and r.get('physics_invocations', 0) == 0 for r in rows),
        active_physics_workers=0, queued_physics_workers=0)


def audit_receipt(path, group_dir, expected):
    if path is None:
        return dict(available=False, independently_verified=False,
                    boundary='Group audit has not been supplied; aggregation is not an independent physics audit')
    value = read(path)
    require(Path(value['input_run']).resolve() == group_dir.resolve(), 'Audit belongs to another group')
    require(value.get('expected_scenes') == expected, 'Audit denominator differs from source group')
    require(len(value.get('scene_receipts', [])) == expected and value.get('all_requested_rows_have_receipts'),
            'Audit omitted a declared group row')
    require(value.get('integrity_pass') is True, 'Supplied group audit did not pass integrity checks')
    for item in value['scene_receipts']:
        require(sha(item['receipt']['path']) == item['receipt']['sha256'], 'Per-scene audit receipt changed')
    return dict(available=True, independently_verified=True, receipt=record(path),
                audited_scene_ids=[item['scene_id'] for item in value['scene_receipts']],
                full_pipeline_rows=value.get('full_pipeline_rows'), full_physics_rows=value.get('full_physics_rows'),
                boundary='Independent saved-artifact audit from this group only; original outcomes are retained')


def group_info(root, name, expected_count, supplied_audit, run_override=None):
    directory = run_override.resolve() if run_override is not None else root / name
    result_path = directory / 'RESULTS.json'
    result = read(result_path); frozen_path = directory / 'FROZEN_INPUTS.json'; frozen = read(frozen_path)
    require(result['status'] == 'cohort_finished', 'Group is not finalized: ' + name)
    require(result['requested_scenes'] == len(result['scenes']) == expected_count, 'Wrong group scene count: ' + name)
    require(all(r['terminal'] for r in result['scenes']), 'Group has unfinished rows: ' + name)
    require([r['scene_id'] for r in result['scenes']] == frozen['scene_order'], 'Group order changed: ' + name)
    require(sha(frozen_path) == result['frozen_inputs_sha256'], 'Group frozen-input SHA changed: ' + name)
    require(aggregate(result['scenes']) == result['aggregate'], 'Group aggregate disagrees with retained rows: ' + name)
    require(result['max_physics_workers'] <= 3 and result['peak_concurrent_physics_workers'] <= 3, 'Group CPU concurrency over 3')
    require(all(r.get('physics_invocations', 0) <= 1 and r.get('actual_physical_attempts', 0) <= 1
                for r in result['scenes']), 'Repeated physical attempt in group')
    require(result['same_original_top1_only'] and result['per_scene_physics_invocation_limit'] == 1,
            'Group original top1 policy changed')
    require(result['actual_hz'] == 1920 and result['arm_execution'] is False and result['training_steps'] == 0,
            'Group method boundary changed')
    events_path = directory / 'EVENTS.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    require(events, 'No event evidence for serial group scheduling')
    previous = -float('inf'); active = set(); peak = 0; starts = collections.Counter(); ends = collections.Counter()
    ids = {r['scene_id'] for r in result['scenes']}
    for event in events:
        require(event['scheduler_elapsed_s'] >= previous, 'Group event order changed')
        previous = event['scheduler_elapsed_s']; key = event.get('scene_id')
        if key is not None:
            require(key in ids, 'Event outside group scenes')
        if event['event'] == 'physics_started':
            require(key not in active and starts[key] == 0, 'Repeated physics dispatch')
            active.add(key); starts[key] += 1; peak = max(peak, len(active))
            require(event['concurrency'] == len(active) <= 3, 'Group event CPU worker count')
        elif event['event'] == 'physics_finished':
            require(key in active and ends[key] == 0, 'Unmatched physics completion')
            active.remove(key); ends[key] += 1
    require(not active and peak == result['peak_concurrent_physics_workers'], 'Group final/peak worker count')
    for row in result['scenes']:
        require(starts[row['scene_id']] == ends[row['scene_id']] == row.get('physics_invocations', 0),
                'Group row/event dispatch count differs')
    require(sum(e['event'] == 'inference_started' for e in events) == 1 and
            sum(e['event'] == 'inference_finished' for e in events) == 1, 'Not one completed inference subprocess per group')
    pipeline_start = result['pipeline_start_unix']
    completed_events = [e['unix'] for e in events if e['event'] in ('inference_finished', 'physics_finished')]
    require(completed_events, 'No completion event evidence')
    last_event = max(completed_events)
    pipeline_end = pipeline_start + result['timing_seconds']['pipeline_and_last_physics_wall']
    require(abs(pipeline_end - last_event) < 1, 'Monotonic-wall and event-clock completion differ')
    require(result['cohort_finished_unix'] >= last_event, 'Group final report predates completion')
    cache_path = directory / 'inference/MODEL_CACHE.json'; cache = read(cache_path); c = cache['counts']
    loads = c['representation_checkpoint_file_loads'] + c['decision_checkpoint_file_loads']
    require(loads == cache['model_load_count'] <= 5, 'Model loads are not one per unique process model')
    require(c['representation_model_constructions'] <= 1 and c['decision_model_constructions'] <= 4,
            'Repeated model construction within group')
    audit = audit_receipt(supplied_audit, directory, expected_count)
    if audit['available']:
        require(set(audit['audited_scene_ids']) == ids, 'Audit/source scene IDs differ')
    info = dict(group=name, group_directory=str(directory.resolve()), requested_scenes=expected_count, original_RESULTS=record(result_path),
        original_FROZEN_INPUTS=record(frozen_path), original_EVENTS=record(events_path),
        original_MODEL_CACHE=record(cache_path), independent_audit=audit,
        frozen_source_SHA=frozen.get('sources', {}),
        model_checkpoint_signatures={p: dict(sha256=v['sha256'], bytes=v['bytes'])
            for p, v in cache.get('checkpoint_signatures', {}).items()},
        original_scene_order=[r['scene_id'] for r in result['scenes']],
        event_interval_unix=[min(e['unix'] for e in events), max(e['unix'] for e in events)],
        pipeline_start_unix=pipeline_start, pipeline_last_completion_unix_from_monotonic=pipeline_end,
        actual_last_completion_event_unix=last_event, cohort_finished_unix=result['cohort_finished_unix'],
        timing_seconds=result['timing_seconds'], aggregate=result['aggregate'],
        max_physics_workers=result['max_physics_workers'], peak_concurrent_CPU_physics_workers=peak,
        model_load_count=loads, model_cache_counts=c,
        model_object_ids_process_local_only=cache.get('live_model_object_ids'),
        no_new_forward_or_physics_by_merger=True)
    return result, info


def merge(root, output_dir, nonapple_audit=None, apple_audit=None, nonapple_run=None, apple_run=None):
    root = root.resolve(); protocol_path = root / 'PROTOCOL.json'; protocol = read(protocol_path)
    declared = protocol['scene_names']; require(len(declared) == len(set(declared)) == 12, 'Protocol is not 12 unique scenes')
    expected = [('run_nonapple', 10, nonapple_audit, nonapple_run), ('run_apple', 2, apple_audit, apple_run)]
    sources = []; by_id = {}
    for name, count, audit, run_override in expected:
        result, info = group_info(root, name, count, audit, run_override); sources.append(info)
        wanted = [key for key in declared if key.startswith('apple_yaw') == (name == 'run_apple')]
        require(info['original_scene_order'] == wanted, 'Group is not the intended protocol subsequence: ' + name)
        require(result['seed'] == protocol['inference']['seed'] == 20261103, 'Group seed differs')
        for row in result['scenes']:
            key = row['scene_id']; require(key not in by_id, 'Duplicate scene across groups: ' + key)
            by_id[key] = dict(row, origin_group=name, origin_group_RESULTS=info['original_RESULTS'])
    require(set(by_id) == set(declared), 'Missing or additional scenes in merged denominator')
    first, second = sources
    require(first['frozen_source_SHA'] == second['frozen_source_SHA'], 'Method/adapter source SHA differs between groups')
    require(first['model_checkpoint_signatures'] == second['model_checkpoint_signatures'], 'Checkpoint identity differs between groups')
    # Conservative non-overlap: the entire first scheduler must have finalized
    # before the second inference process started. This also bounds worker overlap.
    require(first['cohort_finished_unix'] <= second['pipeline_start_unix'], 'Groups overlap; cannot claim global peak=max')
    require(first['event_interval_unix'][1] <= second['event_interval_unix'][0], 'Original event streams overlap')
    rows = [by_id[key] for key in declared]; totals = aggregate(rows)
    for key, value in totals.items():
        if key == 'final_status_counts':
            summed = collections.Counter()
            for group in sources:
                summed.update(group['aggregate'][key])
            require(dict(summed) == value, 'Merged status-count sum differs')
        else:
            require(sum(group['aggregate'][key] for group in sources) == value, 'Merged aggregate sum differs: ' + key)
    wall = second['pipeline_last_completion_unix_from_monotonic'] - first['pipeline_start_unix']
    gap = second['pipeline_start_unix'] - first['pipeline_last_completion_unix_from_monotonic']
    require(wall >= 0 and gap >= 0, 'Invalid serial timing interval')
    peak = max(group['peak_concurrent_CPU_physics_workers'] for group in sources)
    audits_complete = all(group['independent_audit']['independently_verified'] for group in sources)
    timings = dict(pipeline_and_last_physics_wall=wall, pipeline_start_to_last_group_completion_wall=wall,
        original_group_pipeline_wall_sum=sum(group['timing_seconds']['pipeline_and_last_physics_wall'] for group in sources),
        inter_group_gap_wall=gap,
        per_group={group['group']: group['timing_seconds'] for group in sources},
        boundary='First group inference start to last group inference/physics completion, including the serial gap waiting for apple assets. Per-group CPU work overlaps within each group. Pre-first-group asset work and later media encoding are outside this wall.')
    report = dict(schema='AnyDex-DH3-two-serial-cohort-derived-aggregation-v1',
        status='merged_completed_cohorts' if audits_complete else 'merged_completed_cohorts_audits_pending',
        source_artifact_kind='derived_aggregation_of_two_serial_cohorts',
        scope='Fixed twelve real-mesh/pose scenes; two serial process batches, original final top1 once per scene; hand-only simulation',
        root=str(root), output_dir=str(output_dir.resolve()), requested_scenes=12, scenes=rows,
        original_protocol=record(protocol_path), source_cohorts=sources,
        original_scene_order=declared, aggregate=totals, seed=20261103,
        max_physics_workers=3, peak_concurrent_physics_workers=peak,
        global_concurrency_derivation='max of per-group event-reconstructed peaks; both full scheduler/event intervals verified non-overlapping',
        same_original_top1_only=True, per_scene_physics_invocation_limit=1, actual_hz=1920,
        arm_execution=False, training_steps=0, all_twelve_retained_in_denominator=True,
        all_failures_and_no_candidate_rows_retained=True, no_scene_reset_regrasp=True,
        independent_group_audits_complete=audits_complete,
        model_loading=dict(independent_inference_processes=2,
            actual_total_checkpoint_file_loads=sum(group['model_load_count'] for group in sources),
            per_group={group['group']: dict(actual_checkpoint_file_loads=group['model_load_count'],
                cache_counts=group['model_cache_counts'], source=group['original_MODEL_CACHE']) for group in sources},
            boundary='Five original networks are reused within each process. Two groups have separate model caches and loading costs; this is not a single twelve-frame five-load batch.'),
        timing_seconds=timings, merger_source=record(__file__),
        derived_at_unix=time.time(), additional_network_forwards=0, additional_collision_queries=0,
        additional_renders=0, additional_physical_steps=0, hardware_connections=0,
        event_log_boundary='No merged EVENTS/FROZEN_INPUTS/cache is fabricated. Inspect each original group artifact and its independent audit.',
        physical_success_rate_claim_scope='Only this fixed twelve-scene hand-only simulation cohort; no arm, hardware or original-paper benchmark success-rate reproduction')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--nonapple-run', type=Path, help='Authoritative completed ten-scene run; default ROOT/run_nonapple')
    parser.add_argument('--apple-run', type=Path, help='Authoritative completed two-scene run; default ROOT/run_apple')
    parser.add_argument('--nonapple-audit', type=Path)
    parser.add_argument('--apple-audit', type=Path)
    parser.add_argument('--require-audits', action='store_true')
    args = parser.parse_args()
    output = (args.output_dir or args.root / 'run').resolve()
    if (output / 'RESULTS.json').exists():
        parser.error('Use a new derived output directory; do not overwrite a previous result')
    if args.require_audits and not (args.nonapple_audit and args.apple_audit):
        parser.error('--require-audits needs both explicit original group audit receipts')
    started = time.monotonic()
    report = merge(args.root, output, args.nonapple_audit, args.apple_audit, args.nonapple_run, args.apple_run)
    report['aggregation_wall_seconds'] = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=True)
    (output / 'RESULTS.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], output=str(output / 'RESULTS.json'),
        aggregate=report['aggregate'], timing_seconds=report['timing_seconds'],
        actual_checkpoint_file_loads=report['model_loading']['actual_total_checkpoint_file_loads'])), flush=True)


if __name__ == '__main__':
    main()
