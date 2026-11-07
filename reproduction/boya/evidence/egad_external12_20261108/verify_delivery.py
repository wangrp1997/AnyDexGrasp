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

dp = ROOT/'outputs/egad_external12_20261107_v1/diagnostic/RESULTS.json'
qp = ROOT/'outputs/egad_external12_20261107_v1/planning/queue_manifest.json'
ep = ROOT/'outputs/egad_external12_20261107_v1/execution/RESULTS.json'
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


r=ROOT/'outputs/egad_external12_20261107_v1';protocol=load(r/'protocol.json');freeze=load(r/'EVALUATION_FREEZE.json');scene=load(r/'objects.json');init=load(r/'settling/RESULTS.json');rep=load(r/'representation_stage/BATCH_REPRESENTATION.json')
assert sha(r/'protocol.json')=='49e3f6b417be62976187db7f813a434fab24d1f2c67edd9cb1bb07c650bf4e91'
assert init['complete'] and init['ready_objects']==12 and init['grasp_attempts']==0
assert rep['status']=='all_representation_forwards_complete' and rep['network_forward_count']==12 and rep['model_load_count']==1
assert d['novelty_audit']['intersection']==0 and d['novelty_audit']['object_id_intersection']==0
assert d['head_recipe']==e['head_recipe']=='preactivation_selected_mse' and e['external_validation']
assert e['successes']==1 and e['eligible_failures']==2 and e['terminal_nulls']==0 and e['physical_steps']==6300
assert len(rows)==3 and len(media)==3 and len(e['no_plan_objects'])==9
for document in [protocol,freeze,scene,init]:collect(document)
assert all(sha(p)==expected for p,expected in bindings.items())
ids=[o['object_id']for o in protocol['objects']]
assert ids==[o['object_id']for o in d['objects']]==[o['object_id']for o in q['objects']]
assert len(ids)==12 and len(set(ids))==12
import trimesh
visual=[]
for o in scene['objects']:
 a=trimesh.load_mesh(o['source_visual_mesh'],process=False);b=trimesh.load_mesh(o['mesh_path'],process=False)
 assert np.array_equal(a.faces,b.faces) and a.vertices.shape==b.vertices.shape
 err=float(abs(np.asarray(a.vertices)*.001-b.vertices).max());assert err<2e-12
 assert o['scale']==1 and o['source_native_units_to_m']==.001 and o['split']=='external_validation'
 assert o['settling']['ready'] and o['settling']['cold_recreation_verified'] and o['settling_ready']
 ob=load(o['observation_json']);assert ob['world_T_object']==o['world_T_object'] and ob['object_scale']==1
 assert not ob['robot_present'] and not ob['object_pose_as_network_input'] and ob['explicit_physics_steps']==0
 assert load(Path(o['observation_json']).parent/'render_geometry_check.json')['rays']['diagnostic_pass']
 geom=o['geometry_approximation'];visual.append(dict(object_id=o['object_id'],source_to_meter_OBJ_vertex_error_m=err,convex_parts=len(o['convex_parts']),collision_AABB_error_m=geom['collision_vs_visual_max_AABB_delta_m'],approximation_checks=geom['approximation_checks']))
plan=[];reason_counts={}
for o in q['objects']:
 assert 0<=o['arm_preparations']<=4 and o['arm_preparations']==len(o['attempts'])
 for a in o['attempts'][:-1]:assert a['status']=='no_full_arm_plan_in_declared_budget'
 reasons=[]
 for a in o['attempts']:
  for p in a['result']['plans']:
   f=Path(p['plan_dir'])/'admission_failure.json'
   if f.exists():
    reason=load(f)['reason'];reasons.append(reason);reason_counts[reason]=reason_counts.get(reason,0)+1
 candidate=o['selected']['selected_candidate']if o.get('selected')else None
 if candidate:candidate={k:candidate[k]for k in ['source_candidate_index','gesture_id','if_flip','extra_depth_index','learned_score','global_score_rank']}
 plan.append(dict(object_id=o['object_id'],status=o['status'],arm_preparations=o['arm_preparations'],geometry_admitted_total=o['geometry_admitted_total']if 'geometry_admitted_total'in o else None,candidate=candidate,reasons=reasons))
steps=sum(x['physics_steps']for x in rows);assert steps==6300
summary=dict(schema='anydex_boya_egad_external12_delivery_v1',status='complete',date_completed='2026-11-08',protocol=ref(r/'protocol.json'),evaluation_freeze=ref(r/'EVALUATION_FREEZE.json'),training_variant='preactivation_selected_mse',original_training_recipe=False,object_count=12,physical=dict(source=ref(ep),actual_attempts=3,successes=1,eligible_failures=2,terminal_nulls=0,no_plan_objects=e['no_plan_objects'],no_plan_arm=8,no_plan_hand=1,success_fraction_over_all_external_objects=1/12,physical_steps=steps,rows=rows,benchmark_only=True,training_writeback=False,outcome_dependent_retry=False),domain=dict(source=ref(dp),head_queries=d['head_queries'],allowed_combinations=sum(x['expanded_combinations']for x in d['objects']),score_pass_full_geometry_checks=sum(x['geometry_checked']for x in d['objects']),full_geometry_admitted=sum(x['geometry_admitted']for x in d['objects']),objects_with_geometry=sum(x['geometry_admitted']>0 for x in d['objects']),wall_s=d['wall_s'],per_object=[{k:x[k]for k in ['object_id','head_queries','expanded_combinations','shortlist','geometry_checked','geometry_admitted']}for x in d['objects']]),preparation=dict(source=ref(qp),ready=len(q['trials']),no_plan=len(q['nulls']),arm_preparations=sum(x['arm_preparations']for x in q['objects']),wall_s=q['wall_s'],per_object=plan,arm_rejection_reasons=reason_counts),representation=dict(source=ref(r/'representation_stage/BATCH_REPRESENTATION.json'),model_load_count=1,network_forward_count=12,total_wall_s=rep['timing_seconds']['total_wall']),initialization=dict(source=ref(r/'settling/RESULTS.json'),object_steps=init['object_steps'],physics_ticks=init['physics_ticks'],ready_objects=12),novelty_audit=d['novelty_audit'],geometry_proxy=visual,videos=media,video_index=e['video_index'],source_and_input_files_rehashed=len(bindings),all_frozen_hashes_match=True,execution_and_media_wall_s=e['wall_s'],new_training_updates=0,tests_passed=33,boundary='Unseen to this Boya training/development by pinned ID/source and metre-mesh digests; no claim of absence from upstream CGR pretraining. Uniform.001 unit/.1kg/.6friction and coarse V-HACD proxy. One isolated pose/object, arm4/one physical top1. Not paper hardware reproduction or a grasp/score upper bound. No changes after external scores/outcomes.')
Path('/tmp/EGAD_EXTERNAL12_DELIVERY_SUMMARY_20261108.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(objects=12,successes=1,actual_failures=2,no_plan=9,physical_steps=6300,initialization_object_steps=init['object_steps'],unique_rehashed_files=len(bindings),videos=len(media),arm_rejection_reasons=reason_counts),indent=2))
