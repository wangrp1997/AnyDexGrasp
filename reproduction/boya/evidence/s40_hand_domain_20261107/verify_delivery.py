from pathlib import Path
import hashlib, json, struct, sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')

def load(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def ref(p):
    return dict(path=str(p), sha256=sha(p))

def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def boxes(raw, lo=0, hi=None):
    hi = len(raw) if hi is None else hi
    while lo < hi:
        n, tag = struct.unpack_from('>I4s', raw, lo)
        assert n >= 8 and lo + n <= hi
        yield tag, lo + 8, lo + n
        lo += n
    assert lo == hi

dp = ROOT/'outputs/s40_hand_domain_20261107_v1/RESULTS.json'
qp = ROOT/'outputs/s40_hand_top1_20261107_v1/queue_manifest.json'
ep = ROOT/'outputs/s40_hand_execution_20261107_v1/RESULTS.json'
d, q, e = load(dp), load(qp), load(ep)
assert d['status'] == e['status'] == 'complete' and q['complete']
assert e['inputs_and_sources_unchanged'] and not e['errors'] and not e['media_errors']
bindings = {}
def collect(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if k.startswith('/') and isinstance(v, str) and len(v) == 64:
                assert k not in bindings or bindings[k] == v
                bindings[k] = v
            else:
                collect(v)
    elif isinstance(value, list):
        for v in value:
            collect(v)
for doc in [d, q]:
    collect(doc)
assert all(sha(p) == expected for p, expected in bindings.items())
rows = []
for row in sorted(e['results'], key=lambda r: r['object_id']):
    t = row['trial']; execution = Path(t['execution_dir'])
    native = load(execution/'evaluation.json')
    collect(native)
    collect(load(Path(t['source_plan_dir'])/'plan.json'))
    trace = arrays(execution/'trace.npz')
    command = arrays(Path(t['source_plan_dir'])/'commands.npz')
    names = list(map(str, trace['joint_names']))
    cols = [names.index(str(n)) for n in command['joint_names']]
    assert t['label_eligible'] and all(t['guards'].values()) and t['benchmark_only']
    assert len(trace['time_s']) == native['total_physics_steps'] == t['physics_steps'] == 2100
    assert np.allclose(trace['time_s'], np.arange(1, 2101)*.004, atol=1e-14, rtol=0)
    assert np.array_equal(trace['joint_target'][:, cols], command['joint_target'].astype(np.float32))
    assert np.array_equal(trace['phase'], command['phase'])
    assert np.array_equal(trace['desired_palm'], command['desired_palm'])
    assert sha(execution/'trace.npz') == native['trace_sha256']
    assert sha(Path(t['source_plan_dir'])/'commands.npz') == t['commands_sha256'] == native['commands_sha256']
    assert all(np.isfinite(v).all() for v in trace.values() if v.dtype.kind in 'fiu')
    mesh = Path(native['object_root'])/'mesh/simplified.obj'
    assert str(mesh) in bindings and sha(mesh) == bindings[str(mesh)]
    vertices = np.array([[float(x) for x in line.split()[1:4]]
        for line in mesh.read_text().splitlines() if line.startswith('v ')])*native['object_scale']
    hold = np.flatnonzero(trace['phase'] == 'hold')
    poses = trace['object_pose_xyzw'].astype(float)
    rise = float(poses[-1, 2] - poses[hold[-1], 2])
    bottom = float((vertices @ Rotation.from_quat(poses[-1, 3:]).as_matrix()[2] + poses[-1, 2]).min())
    loads = trace['finger_loads_n'][-1]
    bearing = bool(loads[0] > .001 and loads[1:].sum() > .001)
    held = bool(trace['phase'][-1] == 'lift' and rise >= .005 and bottom >= .005 and bearing)
    assert abs(rise - native['pickup']['final_reference_rise_m']) < 1e-12
    assert abs(bottom - native['pickup']['final_bottom_clearance_m']) < 1e-10
    assert held == native['pickup']['final_state_still_lifted_and_bearing']
    assert len(native['gates']) == 7 and all(native['gates'].values())
    assert bool(all(native['gates'].values()) and held and rise >= .05) == native['task_success'] == bool(t['label'])
    assert native['state_writes_during_execution'] == 0 and not native['external_palm_wrench_used']
    checkpoints = [dict(phase=c['phase'], loads_n=c['loads_n']) for c in native['checkpoints']]
    rows.append(dict(object_id=row['object_id'], label=t['label'], physics_steps=2100,
        final_reference_rise_m=rise, final_bottom_clearance_m=bottom,
        final_finger_loads_n=loads.tolist(), final_bilateral_load=bearing,
        final_lifted_and_bearing=held, all_original_seven_gates=True,
        longest_observed_pickup_run_s=native['pickup']['longest_observed_pickup_run_s'],
        independent_original_command_arrays_and_final_source_mesh_measurement=True,
        checkpoints=checkpoints, evaluation=ref(execution/'evaluation.json'), trace=ref(execution/'trace.npz')))

assert all(sha(p) == expected for p, expected in bindings.items())
media = []
for v in e['videos']:
    p = Path(v['path']); p = p if p.is_absolute() else ROOT/p
    c = load(p.parent/'render_check.json'); raw = p.read_bytes(); found = {}
    def walk(lo, hi):
        for tag, a, b in boxes(raw, lo, hi):
            if tag in [b'moov', b'trak', b'mdia', b'minf', b'stbl']:
                walk(a, b)
            if tag == b'mdhd':
                found['timescale'], found['duration'] = struct.unpack_from('>II', raw, a+12)
            if tag == b'stsz':
                _, found['frames'] = struct.unpack_from('>II', raw, a+4)
    walk(0, len(raw))
    assert found == dict(timescale=25, duration=235, frames=235)
    assert c['physics_steps_during_replay'] == 0
    assert sha(p) == c['video_sha256'] == v['sha256']
    er = next(r for r in rows if r['object_id'] == v['object_id'])
    assert c['source_evaluation_sha256'] == er['evaluation']['sha256']
    assert c['source_trace_sha256'] == er['trace']['sha256']
    media.append(dict(object_id=v['object_id'], label=v['label'], video=ref(p),
        frames=235, fps=25, duration_s=9.4, replay_physics_steps=0,
        source_trace_and_evaluation_bound=True, mp4_box_structure_checked=True,
        independent_pixel_decode_performed=False))

old_q = load(ROOT/'outputs/s40_port_top1_20261107_v2/queue_manifest.json')
old = next(o['selected'] for o in old_q['objects'] if o['object_id'] == 'graspnet_044')['selected_candidate']
new = next(o['selected'] for o in q['objects'] if o['object_id'] == 'graspnet_044')['selected_candidate']
geom = load(dp.parent/'graspnet_044/geometry.json')
match = [c for c in geom['checked'] if all(c[k] == old[k] for k in
    ['source_candidate_index', 'gesture_id', 'if_flip', 'extra_depth_index', 'canonical_CGR480_identity'])]
assert len(match) == 1 and match[0]['geometry']['ok'] and match[0]['learned_score'] == old['learned_score']
assert new['learned_score'] > old['learned_score']
regression = dict(object_id='graspnet_044', old_successful_candidate=old,
    old_candidate_in_new_geometry=match[0], newly_selected_candidate=new,
    old_success_was_not_removed_by_candidate_domain=True,
    note='New top1 failed despite higher learned score. Old success is historical and was not replayed or substituted into this protocol; single trials do not isolate one physical cause.')
diag = dict(source=ref(dp), objects=d['object_count'], head_queries=d['head_queries'],
    eligible_combinations=sum(o['expanded_combinations'] for o in d['objects']),
    score_pass_and_full_geometry_calls=sum(o['geometry_checked'] for o in d['objects']),
    full_geometry_admitted=sum(o['geometry_admitted'] for o in d['objects']),
    objects_with_geometry=sum(o['geometry_admitted'] > 0 for o in d['objects']),
    support_plane_recheck_rejections=sum(o['support_plane_recheck_rejections'] for o in d['objects']),
    wall_s=d['wall_s'], per_object=[{k:o[k] for k in ['object_id', 'head_queries',
        'expanded_combinations', 'shortlist', 'geometry_checked', 'geometry_admitted',
        'geometry_rejection_counts']} for o in d['objects']])
summary = dict(schema='anydex_s40_hand_domain_delivery_v1', status='complete_internal_stage',
    domain=diag, source_and_input_files_rehashed=len(bindings), all_frozen_hashes_match=True,
    preparation=dict(source=ref(qp), ready=len(q['trials']), nulls=len(q['nulls']),
        arm_preparations=sum(o['arm_preparations'] for o in q['objects']), wall_s=q['wall_s'],
        trailing_cli_error='NameError in final print only; complete queue saved first; no repeated preparation',
        future_cli='reproduction/boya/prepare_s40_hand_top1.py'),
    physical=dict(source=ref(ep), objects=8, actual_attempts=5, successes=3,
        eligible_failures=2, no_plan_objects=e['no_plan_objects'], terminal_nulls=0,
        total_steps=10500, rows=rows, benchmark_only=True, outcome_dependent_retry=False,
        training_writeback=False),
    regression_044=regression, videos=media, video_index=e['video_index'],
    execution_and_media_wall_s=e['wall_s'], contract_tests=16,
    new_CGR_forwards=0, new_training_updates=0, EGAD_objects_accessed=0,
    boundary='Fixed internal eight after earlier debugging, not independent external generalization or completed original real-paper reproduction. Current protocol is 3/8 including no-plans; historical 044 success must not be combined into this score.',
    next='Audit ranking and training/deployment candidate coverage before new data or external validation. 009/048 have no full geometry pass in frozen domain; 007 exhausted four arm candidates.')
Path('/tmp/S40_HAND_DOMAIN_DELIVERY_SUMMARY_20261107.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(dict(domain=diag, physical=[{k:r[k] for k in ['object_id','label','final_reference_rise_m','final_bottom_clearance_m','final_finger_loads_n']}for r in rows], source_and_input_files_rehashed=len(bindings), videos=len(media)), indent=2))
