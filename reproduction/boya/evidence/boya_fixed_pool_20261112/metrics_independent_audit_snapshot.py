"""Independent post-result arithmetic; does not import the production metric module."""
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
OUT = ROOT / 'outputs/boya_fixed_pool_20261112_v1'
SCORERS = ('preactivation', 'original_mse', 'cgr')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def equal(a, b):
    if type(a) is float or type(b) is float:
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


catalog, scores, result, metrics, protocol = (read(OUT / name) for name in
    ('CATALOG.json', 'SCORES_PREPHYSICS.json', 'RESULTS.json', 'METRICS.json', 'protocol.json'))
assert result['status'] == 'complete' and len(result['results']) == 57
assert sha(OUT / 'SCORES_PREPHYSICS.json') == result['scores_sha256']
assert sha(OUT / 'SCORES_PREPHYSICS.json') == protocol['derived_sha256'][str(OUT / 'SCORES_PREPHYSICS.json')]
for path, expected in scores['input_sha256'].items():
    assert sha(path) == expected, path
by_id = {row['candidate_id']: row for row in catalog['candidates']}
sc = {row['candidate_id']: row for row in scores['candidates']}
executed = {row['candidate_id']: row for row in result['results']}
assert len(by_id) == len(sc) == len(executed) == 57 and set(by_id) == set(sc) == set(executed)
labels, audit_rows = {}, []
recomputed_steps = Counter()
for cid, case in by_id.items():
    row = executed[cid]
    path = Path(row['evaluation'])
    assert sha(path) == row['evaluation_sha256']
    evaluation = read(path)
    assert evaluation['complete'] is True
    assert 0 < evaluation['total_physics_steps'] <= 2100
    m = evaluation['maxima']
    gates = dict(joint_limits=m['joint_limit_violation_rad'] <= .002,
        mimics=m['mimic_error_rad'] <= .002, penetration=m['penetration_m'] <= .001,
        palm_tracking=m['palm_position_error_m'] <= .03 and m['palm_orientation_error_rad'] <= .2617993877991494,
        no_nonhand_object_support=m['illegal_object_contact_force_n'] <= .001,
        fixed_base=m['base_position_drift_m'] <= 2e-5 and m['base_orientation_drift_rad'] <= 2e-5,
        complete_execution=evaluation['total_physics_steps'] == 2100)
    assert gates == evaluation['gates']
    pickup = evaluation['tabletop_pickup']
    final_bearing = False
    if gates['complete_execution']:
        assert evaluation['phases'][-1]['phase'] == 'lift'
        forces = evaluation['phases'][-1]['finger_loads_n']
        final_bearing = forces[0] > .001 and sum(forces[1:]) > .001
        measured_lift = pickup['final_bottom_clearance_m'] >= .005 and pickup['final_reference_rise_m'] >= .005 and final_bearing
        equal(measured_lift, pickup['final_state_still_lifted_and_bearing'])
    success = bool(all(gates.values()) and pickup and pickup['final_bottom_clearance_m'] >= .005
                   and pickup['final_reference_rise_m'] >= .05 and final_bearing)
    equal(success, evaluation['task_success']); equal(success, row['task_success'])
    equal(evaluation['total_physics_steps'], row['physical_steps'])
    equal(evaluation['object_id'], row['recorded_execution_object_id'])
    labels[cid] = int(success)
    recomputed_steps[row['kind']] += evaluation['total_physics_steps']
    audit_rows.append(dict(candidate_id=cid, object_id=case['object_id'], kind=row['kind'],
        evaluation=str(path), evaluation_sha256=sha(path), success=success,
        steps=evaluation['total_physics_steps'], early_stop=bool(evaluation['failure_reason']),
        failure_reason=evaluation['failure_reason'], seven_gates_recomputed=True,
        final_bilateral_load=bool(final_bearing)))

# Independently pull each selected scalar from its historical float32 20-vector.
dirs = dict(preactivation=ROOT / 'outputs/egad_external12_20261107_v1/diagnostic',
            original_mse=ROOT / 'outputs/boya_original_recipe_alignment_20261108_v1/diagnostic')
saved_cache = {}
for oid in sorted({row['object_id'] for row in sc.values()}):
    for scorer, directory in dirs.items():
        with np.load(directory / oid / 'head_outputs.npz', allow_pickle=False) as z:
            data = {}
            for index, (i, g, d, f) in enumerate(zip(z['source_indices'], z['gesture_ids'], z['raw_depth_indices'], z['if_flip'])):
                data[(int(i), int(g), int(d), bool(f))] = z['outputs20'][index].copy()
        saved_cache[(oid, scorer)] = data
for cid, row in sc.items():
    key = (row['source_candidate_index'], row['gesture_id'], row['raw_depth_index'], row['if_flip'])
    target = row['raw_depth_index'] * 4 + row['extra_depth_index']
    equal(target, row['target_index'])
    for scorer in dirs:
        equal(float(saved_cache[(row['object_id'], scorer)][key][target]), row['scores'][scorer])
    with np.load(row['source_npz'], allow_pickle=False) as z:
        equal(float(z['candidates'][row['source_candidate_index'], 0]), row['scores']['cgr'])

saved_objects = {row['object_id']: row for row in metrics['objects']}
saved_ranks = {row['object_id']: row['rankings'] for row in scores['objects']}
objects = []
threshold = float(np.float32(.85))
for oid in sorted({row['object_id'] for row in sc.values()}):
    ids = [cid for cid, row in sc.items() if row['object_id'] == oid]
    count = len(ids); successes = sum(labels[cid] for cid in ids)
    item = dict(object_id=oid, candidates=count, successful_candidates=successes,
        oracle_contains_success=bool(successes), random_top1_success_expectation=successes / count,
        failure_count=count - successes, early_stop_count=sum(r['early_stop'] for r in audit_rows if r['object_id'] == oid),
        new_success_count=sum(r['success'] for r in audit_rows if r['object_id'] == oid and r['kind'] == 'new'),
        reused_success_count=sum(r['success'] for r in audit_rows if r['object_id'] == oid and r['kind'] == 'reused'),
        scorers={})
    equal(count, saved_objects[oid]['candidate_count'])
    equal(successes, saved_objects[oid]['success_count'])
    equal(count - successes, saved_objects[oid]['failure_count'])
    equal(successes / count, saved_objects[oid]['random_top1_success_expectation'])
    equal(bool(successes), saved_objects[oid]['any_success_oracle'])
    for scorer in SCORERS:
        ordered = sorted(ids, key=lambda cid: (-sc[cid]['scores'][scorer], sc[cid]['source_candidate_index'],
            sc[cid]['gesture_id'], sc[cid]['raw_depth_index'], sc[cid]['extra_depth_index'], sc[cid]['if_flip'], cid))
        equal(ordered, saved_ranks[oid][scorer])
        topk = {}
        for k in (1, 3, 5):
            selected = ordered[:k]
            hits = sum(labels[cid] for cid in selected)
            summary = dict(candidate_ids=selected, contains_success=bool(hits), success_count=hits,
                           precision_at_k=hits / len(selected))
            topk[str(k)] = summary
            for key, value in summary.items():
                equal(value, saved_objects[oid]['scorers'][scorer]['topk'][str(k)][key])
        admitted = [cid for cid in ordered if sc[cid]['scores'][scorer] > threshold]
        selected = admitted[0] if admitted else None
        thresh = dict(admitted_count=len(admitted), abstain=not admitted, selected_candidate_id=selected,
                      selected_success=bool(labels[selected]) if selected else None,
                      policy_success=bool(selected and labels[selected]))
        for key, value in thresh.items():
            equal(value, saved_objects[oid]['scorers'][scorer]['strict_threshold'][key])
        successful_ranks = [dict(candidate_id=cid, rank_1based=i + 1, score=sc[cid]['scores'][scorer],
                                source_kind=executed[cid]['kind']) for i, cid in enumerate(ordered) if labels[cid]]
        item['scorers'][scorer] = dict(topk=topk, strict_threshold=thresh,
            successful_candidate_ranks=successful_ranks,
            best_success_rank_1based=successful_ranks[0]['rank_1based'] if successful_ranks else None,
            score_tie_groups={str(value):sum(sc[cid]['scores'][scorer] == value for cid in ids)
                             for value in set(sc[cid]['scores'][scorer] for cid in ids)
                             if sum(sc[cid]['scores'][scorer] == value for cid in ids) > 1})
    objects.append(item)

for key, value in [('new_physics_steps', recomputed_steps['new']), ('reused_physics_steps', recomputed_steps['reused']),
                   ('physical_steps', sum(recomputed_steps.values())), ('success', sum(labels.values()))]:
    equal(value, result[key])
assert metrics['complete'] is True and metrics['aggregate']['candidate_count'] == 57
assert metrics['aggregate']['success_count'] == sum(labels.values())
aggregate = dict(candidates=57, successful_candidates=sum(labels.values()),
    objects_with_any_success=sum(x['oracle_contains_success'] for x in objects),
    random_top1_object_macro_expectation=sum(x['random_top1_success_expectation'] for x in objects) / len(objects),
    physical_steps=dict(recomputed_steps), scorers={})
for scorer in SCORERS:
    aggregate['scorers'][scorer] = {}
    for k in (1, 3, 5):
        values = [o['scorers'][scorer]['topk'][str(k)] for o in objects]
        hits = sum(v['contains_success'] for v in values)
        rate = hits / len(values)
        precision = sum(v['precision_at_k'] for v in values) / len(values)
        saved = metrics['aggregate']['scorers'][scorer]['top' + str(k)]
        equal(hits, saved['objects_with_success']); equal(rate, saved['contains_success_rate'])
        equal(precision, saved['macro_precision_at_k'])
        aggregate['scorers'][scorer]['top' + str(k)] = dict(objects_with_success=hits,
            objects_denominator=len(values), contains_success_rate=rate, macro_precision_at_k=precision)

report = dict(schema='boya_fixed_pool_independent_metrics_audit_v1', complete=True,
    method='Standalone arithmetic, no import/call of fixed_pool_ranking or summarize; recompute seven gates from stored maxima and final success from stored pickup/last forces; all 57 evaluation SHA checked; source-bottom vertex audit is separate.',
    script_path=str(Path(__file__)), script_sha256=sha(__file__),
    input_sha256={str(OUT / name):sha(OUT / name) for name in ('CATALOG.json', 'SCORES_PREPHYSICS.json', 'RESULTS.json', 'METRICS.json', 'protocol.json')},
    old_scores_independently_verified=171, evaluations_verified=57, production_metrics_exactly_matched=True,
    objects=objects, aggregate=aggregate, evaluation_audit=audit_rows,
    limitations=['Known-object diagnostic and score-conditioned two-recipe union; not complete CGR candidate space or independent generalization.',
                 'Oracle is existence within this exact finite pool and unchanged controlled-wrist physics, not actual autonomous selector performance.',
                 'Early physical stops count as failed attempts; infrastructure errors may not be negative labels.'],
    physics_steps=0, new_CGR_forwards=0, new_training_updates=0)
path = OUT / 'METRICS_INDEPENDENT_AUDIT.json'
with path.open('x') as f:
    json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
print(json.dumps(dict(path=str(path), sha256=sha(path), aggregate=aggregate,
    objects=[dict(object_id=x['object_id'], candidates=x['candidates'], success=x['successful_candidates'],
                  best_ranks={s:x['scorers'][s]['best_success_rank_1based'] for s in SCORERS}) for x in objects])))
