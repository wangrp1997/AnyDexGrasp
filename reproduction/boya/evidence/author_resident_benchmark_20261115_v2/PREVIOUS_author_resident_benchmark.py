"""Freeze and analyse small resident-collection controls; never dispatch work.

This is an analysis tool, not a label certifier or a collection scheduler.  A
known positive/negative control can measure runtime, but cannot estimate the
population yield.  Whole-pipeline estimates keep fit/internal and all four
gesture quotas separate and include nulls in the scene denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

SCHEMA = 'anydex_boya_resident_benchmark_v1'
GESTURES = (1, 6, 7, 8)
SPLITS = ('fit', 'internal')
MAX_CONTROL_SCENES = 6


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError(f"Bound input changed: {item['path']}")
    return read(item['path'])


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def freeze_benchmark(reference_batch, resident_protocol, output_dir, *,
                     scene_indices=(4,), selection_basis='known_scene_control'):
    """Bind an already-frozen resident loop, without creating/running a queue.

    ``author_resident_collection.freeze_from_batch(..., scene_indices=...)``
    owns the actual input queue and all native entry points.
    """
    if selection_basis not in ('known_scene_control', 'outcome_blind_fixed_prefix'):
        raise ValueError('Selection basis must be explicit')
    indices = list(scene_indices)
    if (not indices or len(indices) > MAX_CONTROL_SCENES
            or len(indices) != len(set(indices))
            or any(type(i) is not int or i < 0 for i in indices)):
        raise ValueError('A unique small scene budget is required')
    root = Path(reference_batch).resolve()
    reference = read(root/'BATCH_PROTOCOL.json')
    resident = read(resident_protocol)
    if resident.get('exclude_training') is not True or resident.get('automatic_training') is not False:
        raise ValueError('Resident benchmark must explicitly exclude training')
    scenes = reference['queue']['scenes']
    selected = []
    for index in indices:
        spec = scenes[index]
        checked(spec['scene_spec'])
        selected.append(dict(reference_queue_index=index, scene=spec))
    # The loop may name its inherited queue differently; compare the actual
    # queued scene descriptors, never infer an equivalent scene from geometry.
    actual = resident['queue']['scenes']
    if actual != [x['scene'] for x in selected]:
        raise ValueError('Resident queue differs from the fixed reference scene/spec')
    if resident['queue']['seed'] != reference['queue']['seed']:
        raise ValueError('Original candidate sampling seed must remain unchanged')
    protocol = dict(schema=SCHEMA, purpose='diagnostic_only', exclude_training=True,
        automatic_training=False, scope='complete_collection', selection_basis=selection_basis,
        population_yield_inference_allowed=selection_basis == 'outcome_blind_fixed_prefix',
        gestures=list(GESTURES), scenes=selected, scene_count=len(selected),
        gesture_attempt_budget=4*len(selected), retry_budget=0,
        candidate_attempts_per_gesture_scene=1, execution_attempts_per_gesture_scene=1,
        source_seed=reference['queue']['seed'],
        inputs=dict(reference_protocol=binding(root/'BATCH_PROTOCOL.json'),
                    reference_ledger=binding(root/'LEDGER.json'),
                    resident_protocol=binding(resident_protocol)),
        sources=dict(analysis=binding(__file__)),
        timing_scope='capture + CGR + preparation + execution + first independent audit + outer bookkeeping',
        excluded_claims=['execution-only multiplier as full throughput',
                         'known controls as random-scene yield', 'automatic large collection'])
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'BENCHMARK_PROTOCOL.json', protocol)
    return protocol


def _ledger(path):
    value = read(path)
    return value['rows'] if isinstance(value, dict) else value


def _raw(row):
    return checked(row['raw_ledger']) if row.get('raw_ledger') else row


def summarize_rows(rows):
    """Count attempted slots, including valid negatives and all null/infra slots."""
    result = {s: {str(g): dict(attempts=0, eligible=0, positive=0, negative=0,
                              statuses={}) for g in GESTURES} for s in SPLITS}
    seen, scene_keys = set(), set()
    for row in rows:
        key = (row['scene_id'], row['gesture_id'])
        if key in seen:
            raise ValueError('Duplicate scene/gesture slot')
        seen.add(key)
        split, g = row['split'], str(row['gesture_id'])
        if split not in result or g not in result[split]:
            raise ValueError('Unknown split/gesture')
        scene_keys.add((split, row['scene_id']))
        slot = result[split][g]
        slot['attempts'] += 1
        status = row['status']
        if status not in ('labelled', 'diagnostic_labelled', 'placement_null',
                          'capture_null', 'no_candidate', 'null_geometry',
                          'infrastructure_failure', 'executed_null',
                          'quota_reached', 'quota_excess_not_admitted'):
            raise ValueError('Nonterminal or unknown outcome cannot estimate throughput')
        slot['statuses'][status] = slot['statuses'].get(status, 0)+1
        label = row.get('diagnostic_label', row.get('label'))
        if label is not None:
            if (type(label) is not int or label not in (0, 1)
                    or status not in ('labelled', 'diagnostic_labelled')):
                raise ValueError('Only completed labelled trials may have binary outcomes')
            if status == 'diagnostic_labelled' and not (
                    row.get('exclude_training') is True
                    and row.get('formal_training_eligible') is False
                    and row.get('label') is None):
                raise ValueError('Control labels must explicitly stay outside training')
            slot['eligible'] += 1
            slot['positive'] += label
            slot['negative'] += 1-label
    # A scene is the shared observation/workload unit.  Missing slots are an
    # incomplete run, not cheap failures or evidence of four attempts.
    for split, scene_id in scene_keys:
        if {(scene_id, g) for g in GESTURES} - seen:
            raise ValueError('Incomplete scene: four terminal slots are required')
    for split in SPLITS:
        for slot in result[split].values():
            n = slot['attempts']
            slot['eligible_fraction'] = slot['eligible']/n if n else None
    return dict(strata=result, scene_counts={s: sum(x[0] == s for x in scene_keys) for s in SPLITS},
                total_scenes=len(scene_keys), total_attempts=len(rows),
                total_eligible=sum(v['eligible'] for r in result.values() for v in r.values()))


def analyze_run(run_dir, *, scene_indices=None, selection_basis='observational_queue'):
    """Read small receipts only. Original numeric audit is never repeated here."""
    root = Path(run_dir).resolve()
    rows = _ledger(root/'LEDGER.json')
    if scene_indices is not None:
        rows = [r for r in rows if r['queue_index'] in scene_indices]
    result = summarize_rows(rows)
    stages, fingerprints, receipts = {}, {}, {}
    steps, unknown_starts = 0, 0
    for row in rows:
        raw = _raw(row)
        artifacts = raw.get('artifact_bindings', {})
        for key, item in artifacts.items():
            if key.endswith('_stage'):
                stage = checked(item)
                stages[str(Path(item['path']).resolve())] = stage
        label = row.get('diagnostic_label', row.get('label'))
        if label is not None:
            item = artifacts.get('runtime_audit')
            if not item:
                raise ValueError('Eligible outcome lacks its first independent audit')
            audit = checked(item)
            if audit.get('status') != 'pass' or audit.get('label') != label:
                raise ValueError('First numeric audit did not pass with this label')
            receipts[row['trial_id']] = item
        trial = raw.get('trial', {})
        fingerprints[row['trial_id']] = dict(scene_id=row['scene_id'], split=row['split'],
            source_object_ids=row.get('source_object_ids'),
            source_object_digests=row.get('source_object_digests'),
            candidate_sha256=trial.get('candidate_sha256', trial.get('candidate', {}).get('sha256')),
            commands_sha256=trial.get('commands', {}).get('sha256'), label=label)
        steps += raw.get('actual_physics_steps', 0) or 0
        unknown_starts += bool(raw.get('execution_steps_unknown'))
    totals, intervals = {}, []
    for path, stage in stages.items():
        category = Path(path).parent.name.split('_g')[0]
        wall = stage.get('wall_s')
        start = stage.get('started_unix_s')
        if wall is not None:
            if not math.isfinite(wall) or wall < 0:
                raise ValueError('Invalid stage clock')
            item = totals.setdefault(category, dict(count=0, wall_sum_s=0.))
            item['count'] += 1
            item['wall_sum_s'] += wall
            if start is not None:
                intervals.append((start, start+wall))
    span = max(b for a, b in intervals)-min(a for a, b in intervals) if intervals else None
    progress = read(root/'PROGRESS.json') if (root/'PROGRESS.json').exists() else {}
    timing = progress.get('timing_seconds', {})
    measured_wall = timing.get('total_wall') if scene_indices is None else None
    wall = measured_wall if measured_wall is not None else span
    result.update(schema=SCHEMA, run_dir=str(root), selection_basis=selection_basis,
        scope='complete_collection',
        numeric_replays_performed=0, training_updates=0,
        analysis_is_label_certification=False,
        inputs=dict(ledger=binding(root/'LEDGER.json')),
        selected_trial_ids=[r['trial_id'] for r in rows],
        first_numeric_audit_receipts=receipts, stage_costs=totals,
        stage_span_s=span, full_pipeline_wall_s=wall,
        wall_scope=('driver total_wall including first audit' if measured_wall is not None
                    else 'first-to-last stage envelope; excludes outside-stage boundary work'),
        wall_is_lower_bound=measured_wall is None,
        stage_wall_sums_may_overlap=True, known_final_execution_physics_steps=steps,
        execution_starts_with_unknown_steps=unknown_starts,
        reported_timing_seconds=timing, trial_fingerprints=fingerprints)
    return result


def combine_yields(summaries):
    """Pool only disjoint observational queues, never outcome-selected controls."""
    if any(x.get('selection_basis') != 'observational_queue' for x in summaries):
        raise ValueError('Known controls must not estimate population yield')
    seen, rows = set(), []
    for summary in summaries:
        for trial, row in summary['trial_fingerprints'].items():
            if trial in seen:
                raise ValueError('Repeated trial across yield samples')
            seen.add(trial)
            # The fingerprints retain status separately below so null outcomes
            # keep their original categories and never become negatives.
        checked(summary['inputs']['ledger'])
        root = Path(summary['run_dir'])
        selected = set(summary['selected_trial_ids'])
        rows.extend(r for r in _ledger(root/'LEDGER.json') if r['trial_id'] in selected)
    result = summarize_rows(rows)
    result.update(schema=SCHEMA, selection_basis='observational_queue',
                  runs=[x['inputs']['ledger'] for x in summaries])
    return result


def estimate_eta(yields, timing, *, fit_per_gesture=1000, internal_per_gesture=200,
                 control_timing_scenario=False):
    """Empirical expected-workload scenario, never a guaranteed completion time.

    All four gestures share each scene: the slowest eligible quota determines
    scene demand. A zero observed yield has no finite point estimate or upper
    bound. No execution-only multiplier, worker multiplier, or fit->internal
    transfer is accepted.
    """
    if yields.get('selection_basis') != 'observational_queue':
        raise ValueError('ETA yield must come from unselected observational queues')
    if timing.get('scope') == 'execution_only':
        raise ValueError('Execution-only timing cannot estimate complete collection')
    if any(type(x) is not int or x < 0 for x in (fit_per_gesture, internal_per_gesture)):
        raise ValueError('Nonnegative integer per-gesture targets required')
    if timing.get('selection_basis') == 'known_scene_control' and not control_timing_scenario:
        raise ValueError('Known-control timing requires an explicitly labelled scenario')
    wall, n = timing.get('full_pipeline_wall_s'), timing['total_scenes']
    per_scene = wall/n if wall is not None and n else None
    splits = {}
    for split, target in (('fit', fit_per_gesture), ('internal', internal_per_gesture)):
        demand, strata = [], {}
        for g in GESTURES:
            source = yields['strata'][split][str(g)]
            rate = source['eligible_fraction']
            required = math.ceil(target/rate) if rate else (0 if target == 0 else None)
            demand.append(required)
            strata[str(g)] = dict(observed_attempts=source['attempts'],
                observed_eligible=source['eligible'], eligible_fraction=rate,
                target=target, scene_demand_point=required,
                reason=('zero_observed_eligible_yield' if source['attempts'] and not rate
                        else 'split_unmeasured' if not source['attempts'] else None))
        count = max(demand) if all(x is not None for x in demand) else None
        splits[split] = dict(gestures=strata, required_scene_count_point=count,
            point_seconds=count*per_scene if count is not None and per_scene is not None else None,
            reliable_upper_bound_seconds=None)
    total = sum(x['point_seconds'] for x in splits.values()) if all(
        x['point_seconds'] is not None for x in splits.values()) else None
    return dict(schema=SCHEMA, scope='empirical_expected_workload_scenario',
        target_fit=4*fit_per_gesture, target_internal=4*internal_per_gesture,
        seconds_per_attempted_scene=per_scene, splits=splits, total_point_seconds=total,
        total_point_days=total/86400 if total is not None else None,
        reliable_upper_bound_seconds=None, automatic_large_collection=False,
        timing_from_known_control=timing.get('selection_basis') == 'known_scene_control',
        assumptions=['same workload mix and observed all-stage wall per attempted scene',
                     'same backend, audit policy, resource contention and concurrency',
                     'empirical yield persists separately for each split and gesture'],
        limitations=['small scene sample; scene outcomes across gestures are correlated',
                     'a point expected workload is not a probability guarantee of filling quotas',
                     'known-control timing may differ from random-scene null/execution mix',
                     'capture/prepare/first audit and null costs are included in measured wall',
                     'stage-envelope timing is a lower bound when driver wall is unavailable'])


def compare_controls(reference, resident):
    a, b = reference['trial_fingerprints'], resident['trial_fingerprints']
    if set(a) != set(b):
        raise ValueError('Paired controls must contain exactly the same scene/gesture slots')
    comparisons = {}
    for key in a:
        if any(a[key][name] != b[key][name] for name in
               ('scene_id', 'split', 'source_object_ids', 'source_object_digests')):
            raise ValueError('Paired controls changed source scene identities')
        comparisons[key] = {name: (a[key][name] == b[key][name]
            if a[key][name] is not None and b[key][name] is not None else None)
            for name in ('candidate_sha256', 'commands_sha256', 'label')}
    old, new = reference.get('full_pipeline_wall_s'), resident.get('full_pipeline_wall_s')
    return dict(schema=SCHEMA, scope='observed_complete_collection_control',
        comparisons=comparisons, reference_wall_s=old, resident_wall_s=new,
        observed_control_wall_ratio=old/new if old and new else None,
        population_yield_inference_allowed=False,
        both_wall_scopes_exact=not reference['wall_is_lower_bound'] and not resident['wall_is_lower_bound'],
        training_labels_added=0)


def host_snapshot():
    """Read Linux resource facts; this never probes or launches a simulator."""
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            mem[key+'_kib'] = int(value.split()[0])
    physical = set()
    for block in Path('/proc/cpuinfo').read_text().split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if 'physical id' in fields and 'core id' in fields:
            physical.add((fields['physical id'], fields['core id']))
    return dict(observed_unix_s=time.time(), logical_cpu_count=os.cpu_count(),
        physical_core_count=len(physical) or None,
        socket_count=len({x[0] for x in physical}) or None,
        allowed_cpus=sorted(os.sched_getaffinity(0)), memory=mem,
        pressure={x: Path('/proc/pressure', x).read_text().strip() for x in ('cpu', 'memory', 'io')},
        gpu='Not queried here; root must confirm GPU0 and leave GPU1 untouched',
        note='Swap allocation is historical usage, not evidence of current paging pressure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('analyze')
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--scene-indices', type=int, nargs='+')
    p.add_argument('--selection-basis', default='observational_queue',
                   choices=('observational_queue', 'known_scene_control'))
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('freeze')
    p.add_argument('--reference-batch', required=True, type=Path)
    p.add_argument('--resident-protocol', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--scene-indices', required=True, type=int, nargs='+')
    p = sub.add_parser('eta')
    p.add_argument('--yield-summary', required=True, type=Path)
    p.add_argument('--timing-summary', required=True, type=Path)
    p.add_argument('--control-timing-scenario', action='store_true')
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('host')
    p.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.operation == 'analyze':
        value = analyze_run(args.run_dir, scene_indices=args.scene_indices,
                            selection_basis=args.selection_basis)
    elif args.operation == 'freeze':
        value = freeze_benchmark(args.reference_batch, args.resident_protocol, args.output_dir,
                                 scene_indices=args.scene_indices)
    elif args.operation == 'eta':
        value = estimate_eta(read(args.yield_summary), read(args.timing_summary),
                             control_timing_scenario=args.control_timing_scenario)
    else:
        value = host_snapshot()
    if hasattr(args, 'output'):
        save(args.output, value)
    print(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    main()
