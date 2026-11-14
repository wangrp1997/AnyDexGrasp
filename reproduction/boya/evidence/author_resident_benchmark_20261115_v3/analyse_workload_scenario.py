"""Read finished small controls and calculate an explicitly conditional workload ETA.

This evidence script never imports a simulator, dispatches work, decodes contacts,
or supplies labels. It does not change the frozen benchmark implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

GESTURES = (1, 6, 7, 8)
BENCHMARK_SHA = 'ed0a29cb988d16025d54f8166855eb7b5b1bd608abbea5a50f768362891a682c'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Reader:
    def __init__(self):
        self.inputs = {}

    def read(self, path):
        path = Path(path).resolve()
        payload = path.read_bytes()
        value = json.loads(payload)
        self.inputs[str(path)] = hashlib.sha256(payload).hexdigest()
        return value

    def verify(self):
        for path, expected in self.inputs.items():
            if sha(path) != expected:
                raise ValueError('Input changed during analysis: ' + path)


def union_seconds(intervals):
    total = 0.
    end = None
    for start, stop in sorted(intervals):
        if not start <= stop:
            raise ValueError('Invalid stage interval')
        if end is None or start >= end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def weighted_components(*, capture_fraction, execution_scene_fraction,
                        execution_fractions, build_capture_cgr_prepare_per_scene,
                        fixed_execution_per_group, variable_execution_by_gesture,
                        observed_parallel_wall_per_serial_work,
                        audit_by_gesture, bookkeeping_per_attempt):
    """Arithmetic scenario; the observed scheduling coefficient is not a speed claim."""
    fractions = [capture_fraction, execution_scene_fraction, *execution_fractions.values()]
    if any(not 0 <= x <= 1 for x in fractions):
        raise ValueError('Invalid workload fraction')
    if not 0 < observed_parallel_wall_per_serial_work <= 1.01:
        raise ValueError('Invalid measured execution scheduling coefficient')
    serial_execution = execution_scene_fraction * fixed_execution_per_group + sum(
        execution_fractions[g] * variable_execution_by_gesture[g] for g in GESTURES)
    parts = dict(
        build_capture_cgr_prepare=capture_fraction * build_capture_cgr_prepare_per_scene,
        execution_observed_two_group_scheduling=serial_execution * observed_parallel_wall_per_serial_work,
        first_numeric_audit=sum(execution_fractions[g] * audit_by_gesture[g] for g in GESTURES),
        bookkeeping=bookkeeping_per_attempt,
    )
    parts['total'] = sum(parts.values())
    return parts, serial_execution


def analyse(run_dir, control_summary, yield_summary, historical_summaries, *, target=1000):
    r = Reader()
    root = Path(run_dir).resolve()
    summary = r.read(control_summary)
    yields = r.read(yield_summary)
    historical = [r.read(path) for path in historical_summaries]
    if summary['selection_basis'] != 'known_scene_control' or summary['run_dir'] != str(root):
        raise ValueError('Wrong control summary')
    if summary['execution_starts_with_unknown_steps'] or summary['numeric_replays_performed']:
        raise ValueError('Unknown execution cost or unexpected replay')
    for b in summary['inputs'].values():
        if sha(b['path']) != b['sha256']:
            raise ValueError('Changed summary input')
        r.inputs[b['path']] = b['sha256']
    for b in summary['first_numeric_audit_receipts'].values():
        audit = r.read(b['path'])
        if r.inputs[b['path']] != b['sha256'] or audit.get('status') != 'pass':
            raise ValueError('Missing or changed first numerical audit')
    progress = r.read(root / 'PROGRESS.json')
    outer = r.read(root / 'OUTER_DRIVER_RESULTS.json')
    protocol = r.read(root / 'RESIDENT_COLLECTION_PROTOCOL.json')
    n = summary['total_scenes']
    if n != 2 or summary['total_eligible'] != n * 4 or progress['status'] != 'complete':
        raise ValueError('This scenario requires the finished two-scene/eight-audit fit control')
    if outer.get('error') or protocol['resident_workers'] != 2:
        raise ValueError('Failed outer loop or changed measured concurrency')
    timing = progress['timing_seconds']
    stages = [(p, r.read(p)) for p in sorted(root.glob('**/stages/*/STAGE.json'))]
    prepare = [x for p, x in stages if p.parent.name.startswith('prepare_g')]
    prepare_wall = union_seconds([(x['started_unix_s'], x['started_unix_s'] + x['wall_s']) for x in prepare])
    audit_by_gesture = {}
    variable_by_gesture = {g: [] for g in GESTURES}
    fixed = []
    group_walls = []
    reset_count = 0
    for directory in sorted((root / 'scenes').iterdir()):
        group = r.read(directory / 'stages/resident_execute/STAGE.json')
        session = r.read(directory / 'resident_session/SESSION_RECEIPT.json')
        if group['status'] != 'complete' or session['status'] != 'closed' or session['Kit_starts'] != 1:
            raise ValueError('Incomplete resident group')
        if session['added_reset_physics_steps'] != 0 or session['execution_starts_with_unknown_steps']:
            raise ValueError('Changed reset cost or unknown steps')
        reset_count += session['state_reset_count']
        group_walls.append(group['wall_s'])
        entries = {g: r.read(directory / f'g{g}/RESIDENT_EXECUTION_ENTRY.json') for g in GESTURES}
        # The directly measured factory and launch are included in first-job wall.
        # Everything outside job walls (imports, verification, final close) stays
        # in the fixed group term rather than disappearing from the estimate.
        group_fixed = group['wall_s'] - sum(e['wall_s'] for e in entries.values())
        for g, e in entries.items():
            if e['status'] != 'complete' or not e['native_and_all_body_initial_gates_completed']:
                raise ValueError('Incomplete job or missing initial gates')
            cold = e.get('native_scene_factory_wall_s', 0.)
            if e['Kit_starts_added']:
                cold += session['Kit_launch_wall_s']
            group_fixed += cold
            variable_by_gesture[g].append(e['wall_s'] - cold)
        fixed.append(group_fixed)
    for g in GESTURES:
        values = [x['wall_s'] for p, x in stages if p.parent.name == f'audit_g{g}']
        if len(values) != n:
            raise ValueError('Missing first audit stage')
        audit_by_gesture[g] = sum(values) / n
    variable = {g: sum(values) / n for g, values in variable_by_gesture.items()}
    if min([*fixed, *variable.values()]) < 0:
        raise ValueError('Timing decomposition is inconsistent')
    phase_cost = timing['build'] + timing['capture'] + timing['representation_batch'] + prepare_wall
    residual = timing['total_wall'] - phase_cost - timing['execution_parallel_wall'] - timing['audit']
    if residual < -.1 or outer['wall_s'] < timing['total_wall']:
        raise ValueError('Overlapping or inconsistent whole-loop timing')
    bookkeeping = (max(residual, 0.) + outer['wall_s'] - timing['total_wall']) / n
    fit = yields['strata']['fit']
    attempts = yields['scene_counts']['fit']
    capture_fraction = 1 - fit['1']['statuses'].get('placement_null', 0) / attempts
    executed_scenes = set()
    for h in historical:
        if h['selection_basis'] != 'observational_queue':
            raise ValueError('Known controls cannot supply random-scene yield')
        for fp in h['trial_fingerprints'].values():
            if fp['split'] == 'fit' and fp['label'] in (0, 1):
                executed_scenes.add(fp['scene_id'])
    fractions = {g: fit[str(g)]['eligible'] / attempts for g in GESTURES}
    components, serial_execution = weighted_components(
        capture_fraction=capture_fraction, execution_scene_fraction=len(executed_scenes) / attempts,
        execution_fractions=fractions, build_capture_cgr_prepare_per_scene=phase_cost / n,
        fixed_execution_per_group=sum(fixed) / n, variable_execution_by_gesture=variable,
        observed_parallel_wall_per_serial_work=timing['execution_parallel_wall'] / sum(group_walls),
        audit_by_gesture=audit_by_gesture, bookkeeping_per_attempt=bookkeeping)
    required = max(math.ceil(target / p) for p in fractions.values())
    internal_zero = [g for g in GESTURES if yields['strata']['internal'][str(g)]['eligible'] == 0]
    r.verify()
    return dict(schema='anydex_boya_resident_workload_scenario_v1',
        numeric_replays_performed=0, physics_steps_added=0, training_updates=0,
        interpretation='Conditional workload-transfer scenario, not a measured random-scene throughput or upper bound',
        assumptions=[
            'Old 11 fit scenes describe future placement and per-gesture executable/eligible fractions; no executed-null fit jobs occurred there',
            'Measured all-ready control build/capture/CGR/prepare costs transfer to captured random scenes, including planning-null scenes',
            'Measured two-group parallel wall/serial-work coefficient transfers to enough ready groups to keep two sessions supplied; no 3-worker extrapolation',
            'Each executed job requires one full first numerical audit; valid negatives count, null/infra never count as zero',
            'All four gestures continue through the bottleneck quota, including raw over-quota execution costs',
            'Unattributed inner and outer overhead is charged per attempted scene at the two-scene control rate',
            'Existing converted assets and prepared hand clouds are reused; future source asset conversion and training time are not measured here'],
        control=dict(scenes=n, outer_wall_s=outer['wall_s'], inner_wall_s=timing['total_wall'],
            prepare_union_s=prepare_wall, phase_sums=timing, fixed_execution_per_group_s=sum(fixed)/n,
            variable_execution_by_gesture_s=variable, first_audit_by_gesture_s=audit_by_gesture,
            scheduling_coefficient=timing['execution_parallel_wall']/sum(group_walls),
            nonoverlapping_bookkeeping_per_attempt_s=bookkeeping, resident_resets=reset_count),
        historical_fit=dict(attempted_scenes=attempts, capture_fraction=capture_fraction,
            execution_scene_fraction=len(executed_scenes)/attempts, execution_fractions=fractions,
            expected_executions_per_attempt=sum(fractions.values())),
        fit=dict(target_per_gesture=target, required_scenes_point_scenario=required,
            required_scenes_by_gesture={g:math.ceil(target/p) for g,p in fractions.items()},
            components_seconds_per_attempt=components,
            expected_raw_eligible_by_gesture={g:required*p for g,p in fractions.items()},
            expected_wall_s=required*components['total'], expected_days=required*components['total']/86400,
            serial_execution_work_seconds_per_attempt=serial_execution,
            same_workload_without_execution_overlap_days=required*(components['total']-components['execution_observed_two_group_scheduling']+serial_execution)/86400),
        internal=dict(target_per_gesture=200,
            observed_eligible_fractions={g:yields['strata']['internal'][str(g)]['eligible_fraction'] for g in GESTURES},
            required_scenes_by_gesture={g:(math.ceil(200/yields['strata']['internal'][str(g)]['eligible_fraction'])
                if yields['strata']['internal'][str(g)]['eligible_fraction'] else None) for g in GESTURES},
            no_observed_eligible_gestures=internal_zero, finite_total_eta=None,
            reason='Zero observed production in these gesture strata; fit rates are not transferred'),
        overall_completion_eta=None, audit_inputs_sha256=r.inputs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--control-summary', required=True)
    p.add_argument('--yield-summary', required=True)
    p.add_argument('--historical-summaries', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    benchmark = Path(__file__).resolve().parents[2] / 'author_resident_benchmark.py'
    if sha(benchmark) != BENCHMARK_SHA:
        raise ValueError('Frozen benchmark source changed')
    result = analyse(a.run_dir, a.control_summary, a.yield_summary, a.historical_summaries)
    result['analysis_source_sha256'] = sha(__file__)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
