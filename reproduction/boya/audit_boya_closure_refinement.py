"""Independent read-only audit of the bounded four-object translation control.

This program imports no simulator or learned model. Source meshes are read only
for outcome measurement, never to choose a refinement. --prepare checks the four
historical bindings; --run checks a completed eight-trial diagnostic.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
SOURCE_JOBS = {
    'egad_A0': ROOT / 'outputs/boya_contact_proxy_alignment_20261108_v2/jobs/egad_A0_coarse.json',
    **{f'egad_{name}': ROOT / f'outputs/egad_floating_three_20261108_v1/jobs/egad_{name}.json'
       for name in ('A3', 'A6', 'E0')},
}
PHASES = [('prehold', 800), ('approach', 500), ('close', 250), ('hold', 50), ('lift', 500)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def check_hash(path, expected):
    actual = sha(path)
    assert actual == expected, ('hash', str(path), expected, actual)
    return actual


def source_binding(source_id):
    job_path = SOURCE_JOBS[source_id]
    job = read(job_path)
    pool = Path(job['candidate_dir'])
    generation = read(pool / 'generation.json')
    assert generation['complete'] and not job.get('collision_refinement')
    check_hash(job['protocol_path'], job['protocol_sha256'])
    check_hash(pool / 'candidates.npz', generation['candidates_sha256'])
    observation_path = Path(job['object_row']['observation_json'])
    observation = read(observation_path)
    frame = Path(generation['source_frame'])
    representation = read(frame / 'RESULTS.json')
    assert Path(generation['source_observation']) == observation_path
    assert Path(representation['scene_binding']['observation_json']) == observation_path
    assert observation['object_id'] == representation['object_id'] == source_id
    check_hash(observation_path, representation['scene_binding']['observation_sha256'])
    for entry in observation['files'].values():
        check_hash(entry['path'], entry['sha256'])
    point_file = frame / 'representation_outputs.npz'
    check_hash(point_file, representation['output_files'][point_file.name]['sha256'])
    # Check the relevant original generation bindings without rerunning old
    # global model/dataset dependency audits or importing CGR.
    for path in (observation_path, point_file):
        check_hash(path, generation['input_sha256'][str(path)])
    frozen = {}
    for field in ('hand_urdf', 'hand_scene', 'package', 'physics_config'):
        path = job[field]
        check_hash(path, generation['input_sha256'][path])
        frozen[path] = sha(path)
    old_protocol = read(job['protocol_path'])
    diagnostic = job['diagnostic_config']
    old_hashes = old_protocol.get('input_sha256', {}) | old_protocol.get('source_sha256', {})
    if diagnostic in old_hashes:
        check_hash(diagnostic, old_hashes[diagnostic])
    frozen[diagnostic] = sha(diagnostic)
    row = job['object_row']
    for path_field, hash_field in [('mesh_path', 'mesh_sha256'), ('urdf_path', 'urdf_sha256'),
                                 ('pose_path', 'pose_sha256'), ('metadata_path', 'metadata_sha256')]:
        check_hash(row[path_field], row[hash_field])
        frozen[row[path_field]] = row[hash_field]
    pool_arrays = arrays(pool / 'candidates.npz')
    np.testing.assert_allclose(pool_arrays['world_T_object'], row['world_T_object'], atol=2e-8, rtol=0)
    np.testing.assert_allclose(pool_arrays['world_T_object'], observation['world_T_object'], atol=2e-8, rtol=0)
    np.testing.assert_array_equal(observation['T_WC'], representation['scene_binding']['T_WC'])
    with np.load(point_file, allow_pickle=False) as z:
        cloud = z['sparse_input_XYZ_features'].copy()
    assert cloud.ndim == 2 and cloud.shape[1] == 3 and np.isfinite(cloud).all()
    execution = Path(job['output_dir'])
    evaluation = read(execution / 'evaluation.json')
    assert evaluation['complete'] and evaluation['total_physics_steps'] == 2100
    check_hash(execution / 'commands.npz', evaluation['commands_sha256'])
    check_hash(execution / 'trace.npz', evaluation['trace_sha256'])
    result = dict(source_object_id=source_id, source_job=ref(job_path), source_pool=ref(pool / 'candidates.npz'),
                  generation=ref(pool / 'generation.json'), candidate_rank=job['candidate_rank'],
                  gesture_id=generation['gesture_id'], observation=ref(observation_path),
                  representation=ref(point_file), point_array='sparse_input_XYZ_features',
                  points=int(len(cloud)), point_array_sha256=hashlib.sha256(cloud.tobytes()).hexdigest(),
                  point_frame='camera; world = XYZ @ T_WC[:3,:3].T + T_WC[:3,3]',
                  point_boundary='Single observed sparse cloud including table; no completed source object surface.',
                  original_evaluation=ref(execution / 'evaluation.json'),
                  original_success=evaluation['task_success'], original_gates=evaluation['gates'],
                  frozen_relevant_files=frozen, collision_refinement=None,
                  mass_kg=row['object_mass_kg'], friction=row['object_friction'])
    return result, job, pool_arrays


def preparation():
    return dict(schema='boya_closure_translation_source_audit_v1', complete=True,
                new_physics_steps=0, new_training_updates=0, new_CGR_forwards=0,
                sources=[source_binding(key)[0] for key in SOURCE_JOBS],
                renderer=dict(source=str(ROOT / 'reproduction/boya/render_egad_floating_three.py'),
                              arbitrary_result_count_supported=True, physics_steps_during_replay=0,
                              caveats=['Old HTML title names A6/A3/E0; provide a new delivery index.',
                                       'Object rise subtitle measures the reference point, not source-bottom clearance.']),
                preserved_success_rule='Seven original gates, final source-bottom >=5mm, final reference rise >=50mm, concurrent thumb/opposing loads >1mN.',
                sustained_load_boundary='Longest and trailing durations are reported separately, not added to the original success gate.')


def runs(mask):
    current = longest = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return dict(longest_s=float(longest * .004), trailing_s=float(current * .004), steps=int(np.sum(mask)))


def measured_pickup(trace, row):
    poses = trace['object_pose_xyzw'].astype(float)
    vertices = np.array([[float(value) for value in line.split()[1:4]]
                         for line in Path(row['mesh_path']).read_text().splitlines()
                         if line.startswith('v ')]) * row['scale']
    rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()
    bottom = np.empty(len(poses))
    for start in range(0, len(poses), 64):
        bottom[start:start + 64] = (vertices @ rotations[start:start + 64, 2, :].T
                                   + poses[start:start + 64, 2]).min(0)
    hold = np.flatnonzero(trace['phase'] == 'hold')
    lift = trace['phase'] == 'lift'
    loads = trace['finger_loads_n']
    bearing = (loads[:, 0] > .001) & (loads[:, 1:].sum(1) > .001)
    if len(hold):
        rise = poses[:, 2] - poses[hold[-1], 2]
        picked = lift & (bottom >= .005) & (rise >= .005) & bearing
    else:
        rise = None
        picked = np.zeros(len(poses), dtype=bool)
    return dict(source_mesh_vertices=len(vertices), final_bottom_clearance_m=float(bottom[-1]),
                final_reference_rise_m=float(rise[-1]) if rise is not None else None,
                final_state_still_lifted_and_bearing=bool(picked[-1]),
                qualifying_pickup=runs(picked), bilateral_load_during_lift=runs(lift & bearing),
                final_finger_loads_n=loads[-1].astype(float).tolist())


def audit_execution(directory, job, source_job, old_pool, delta, declared_result):
    evaluation = read(directory / 'evaluation.json')
    trace = arrays(directory / 'trace.npz')
    commands = arrays(directory / 'commands.npz')
    old_commands = arrays(Path(source_job['output_dir']) / 'commands.npz')
    old_evaluation = read(Path(source_job['output_dir']) / 'evaluation.json')
    native = read(directory / 'native_audit.json')
    init = read(directory / 'initialization_audit.json')
    old_init = read(Path(source_job['output_dir']) / 'initialization_audit.json')
    n = len(trace['time_s'])
    assert evaluation['complete'] and 0 < n <= 2100 and evaluation['total_physics_steps'] == n
    check_hash(directory / 'evaluation.json', declared_result['evaluation_sha256'])
    check_hash(directory / 'commands.npz', evaluation['commands_sha256'])
    check_hash(directory / 'trace.npz', evaluation['trace_sha256'])
    check_hash(Path(job['candidate_dir']) / 'candidates.npz', evaluation['hand_candidate_sha256'])
    for key in ('new_CGR_forwards', 'new_training_updates', 'state_writes_during_execution',
                'root_pose_writes_during_execution', 'object_state_writes_during_execution'):
        assert evaluation[key] == 0, key
    assert not evaluation['training_writeback'] and not evaluation['outcome_dependent_retry']
    assert not evaluation['arm_present'] and evaluation['benchmark_only']
    assert evaluation['dt_s'] == .004 and evaluation['declared_command_steps'] == 2100
    assert evaluation['phase_lengths'] == dict(PHASES)
    for key in ('commanded_gantry', 'object_weight_world_n', 'global_gravity_m_s2'):
        assert evaluation[key] == old_evaluation[key], key
    for key in ('actuation', 'gantry', 'mass_kg', 'native_mimics', 'solver_position_iteration_count'):
        assert evaluation['audit'][key] == old_evaluation['audit'][key], key
    for body in ('robot', 'object'):
        material = evaluation['audit']['materials'][body]
        assert material['native_readback_ok']
        for key in ('static_friction', 'dynamic_friction', 'restitution', 'rest_offset_m', 'contact_offset_m'):
            assert material[key] == old_evaluation['audit']['materials'][body][key], (body, key)
    assert native['within_20um'] and evaluation['native_within_20um'] and native['physics_steps'] == 0
    assert native['max_source_outside_cooked_bound_m'] <= 2e-5
    assert native['max_cooked_outside_source_bound_m'] <= 2e-5
    np.testing.assert_array_equal(init['expected_object_pose_xyzw'], old_init['expected_object_pose_xyzw'])
    wanted_root = np.array(old_init['expected_root_pose_xyzw']); wanted_root[:3] += delta
    np.testing.assert_allclose(init['expected_root_pose_xyzw'], wanted_root, atol=2e-14, rtol=0)
    np.testing.assert_array_equal(commands['joint_names'], old_commands['joint_names'])
    names = list(commands['joint_names']); fingers = [i for i, name in enumerate(names) if not name.startswith('diagnostic_slide_')]
    assert len(fingers) == 20
    for key in ('joint_target', 'velocity_target'):
        np.testing.assert_array_equal(commands[key][:, fingers], old_commands[key][:, fingers])
        np.testing.assert_allclose(commands[key], old_commands[key], atol=2e-8, rtol=0)
    # The world translation can cause only sub-ULP cancellation in relative gantry arithmetic.
    for key in ('phase', 'time_s'):
        np.testing.assert_array_equal(commands[key], old_commands[key])
    expected_palm = old_commands['desired_palm'].copy(); expected_palm[:, :3, 3] += delta
    np.testing.assert_allclose(commands['desired_palm'], expected_palm, atol=2e-14, rtol=0)
    np.testing.assert_array_equal(commands['desired_palm'][:, :3, :3], old_commands['desired_palm'][:, :3, :3])
    for key in commands:
        wanted = commands[key] if key == 'joint_names' else commands[key][:n]
        np.testing.assert_array_equal(trace[key], wanted)
    assert all(np.isfinite(value).all() for value in trace.values() if value.dtype.kind in 'fiu')
    np.testing.assert_array_equal(trace['time_s'], np.arange(1, n + 1) * .004)
    phases = np.concatenate([np.repeat(name, count) for name, count in PHASES])
    np.testing.assert_array_equal(trace['phase'], phases[:n])
    limit = mimic = 0.
    for joint in ET.parse(directory / 'boya_translation_gantry.urdf').getroot().findall('joint'):
        name = joint.get('name')
        if name not in names:
            continue
        q = trace['joint_pos'][:, names.index(name)]
        bounds = joint.find('limit'); relation = joint.find('mimic')
        if bounds is not None and joint.get('type') in ('revolute', 'prismatic'):
            limit = max(limit, float(np.maximum(np.maximum(float(bounds.get('lower')) - q,
                                                           q - float(bounds.get('upper'))), 0).max()))
        if relation is not None:
            expected = trace['joint_pos'][:, names.index(relation.get('joint'))] * float(relation.get('multiplier', '1')) + float(relation.get('offset', '0'))
            mimic = max(mimic, float(abs(q - expected).max()))
    palm_error = np.linalg.norm(trace['palm_pose'][:, :3, 3] - trace['desired_palm'][:, :3, 3], axis=1)
    palm_angle = (Rotation.from_matrix(trace['palm_pose'][:, :3, :3]).inv() * Rotation.from_matrix(trace['desired_palm'][:, :3, :3])).magnitude()
    penetration = float(max(trace['object_penetration_m'].max(), trace['robot_penetration_m'].max()))
    measured = dict(joint_limit_violation_rad=limit, mimic_error_rad=mimic, penetration_m=penetration,
                    palm_position_error_m=float(palm_error.max()), palm_orientation_error_rad=float(palm_angle.max()))
    for key, value in measured.items():
        assert np.isclose(value, evaluation['maxima'][key], atol=2e-7, rtol=0), key
    reconstructed_gates = dict(joint_limits=limit <= .002, mimics=mimic <= .002, penetration=penetration <= .001,
                               palm_tracking=palm_error.max() <= .03 and palm_angle.max() <= .2617993877991494,
                               complete_execution=n == 2100)
    for key, value in reconstructed_gates.items():
        assert bool(value) == evaluation['gates'][key], key
    maxima = evaluation['maxima']
    assert evaluation['gates']['fixed_base'] == bool(maxima['base_position_drift_m'] <= 2e-5 and maxima['base_orientation_drift_rad'] <= 2e-5)
    assert evaluation['gates']['no_nonhand_object_support'] == bool(maxima['illegal_object_contact_force_n'] <= .001)
    measured_outcome = measured_pickup(trace, job['object_row'])
    pickup = evaluation['tabletop_pickup']
    if pickup is not None:
        for key in ('final_bottom_clearance_m', 'final_reference_rise_m'):
            assert np.isclose(measured_outcome[key], pickup[key], atol=2e-12, rtol=0)
        assert measured_outcome['qualifying_pickup']['steps'] == pickup['qualifying_steps']
        assert np.isclose(measured_outcome['qualifying_pickup']['longest_s'], pickup['longest_observed_pickup_run_s'], atol=2e-12, rtol=0)
        assert measured_outcome['final_state_still_lifted_and_bearing'] == pickup['final_state_still_lifted_and_bearing']
    success = bool(all(evaluation['gates'].values()) and measured_outcome['final_state_still_lifted_and_bearing']
                   and measured_outcome['final_reference_rise_m'] >= .05)
    assert success == evaluation['task_success'] == declared_result['task_success']
    assert declared_result.get('physical_steps', n) == n
    return dict(object_id=job['object_id'], evaluation=ref(directory / 'evaluation.json'), physical_steps=n,
                task_success=success, delta_world_m=delta.tolist(), complete_20hand_commands_bitwise_unchanged=True,
                gantry_relative_commands_within_2e_minus8_m=True, desired_palm_only_declared_translation=True,
                object_initial_state_and_physics_preserved=True, native_within_20um=True,
                gates=evaluation['gates'], reconstructed_maxima=measured, pickup=measured_outcome)


def verify_hash_sections(document):
    counts = {}
    for key in ('sources', 'source_sha256', 'input_sha256', 'derived_sha256', 'job_sha256', 'jobs_sha256'):
        if key not in document:
            continue
        section = document[key]
        if not isinstance(section, dict):
            continue
        count = 0
        for path, value in section.items():
            digest = value.get('sha256') if isinstance(value, dict) else value
            actual_path = value.get('path', path) if isinstance(value, dict) else path
            if isinstance(digest, str) and len(digest) == 64:
                check_hash(actual_path, digest); count += 1
        counts[key] = count
    return counts


def audit_run(run):
    protocol = read(run / 'protocol.json'); results = read(run / 'RESULTS.json')
    assert results['status'] == 'complete' and not results.get('errors')
    check_hash(run / 'protocol.json', results['protocol_sha256'])
    counts = verify_hash_sections(protocol)
    prepared = read(run / 'PREPARED.json')
    check_hash(run / 'protocol.json', prepared['protocol_sha256'])
    prepared_counts = verify_hash_sections(prepared)
    assert prepared['physics_steps'] == 0 and prepared['planned_physics_attempts'] == 8
    assert protocol['physics_attempts'] == 8 and protocol['maximum_physics_steps'] == 16800
    expected = {source + '_' + variant for source in SOURCE_JOBS for variant in ('baseline', 'refined')}
    assert len(protocol['cases']) == len(results['results']) == 8
    assert {case['object_id'] for case in protocol['cases']} == expected
    assert {case['object_id'] for case in results['results']} == expected
    assert {path.stem for path in (run / 'jobs').glob('*.json')} == expected
    assert len(list((run / 'execution').glob('*/evaluation.json'))) == 8
    sources = {source: source_binding(source) for source in SOURCE_JOBS}
    audited = []
    for case in protocol['cases']:
        source = case['source_object_id']; source_receipt, source_job, old_pool = sources[source]
        oid = case['object_id']; job = read(run / 'jobs' / (oid + '.json'))
        assert Path(case['source_job']) == SOURCE_JOBS[source]
        assert Path(case['source_pool']) == Path(source_job['candidate_dir'])
        assert oid == source + '_' + case['variant'] and job['object_id'] == oid
        assert job.get('candidate_index', 0) == 0 and case['candidate_rank'] == source_job['candidate_rank']
        assert Path(job['candidate_dir']) == Path(case['candidate_dir'])
        assert job['protocol_path'] == str(run / 'protocol.json') and job['protocol_sha256'] == sha(run / 'protocol.json')
        for field in ('object_row', 'hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config'):
            assert job[field] == source_job[field], (oid, field)
        assert not job.get('collision_refinement')
        pool = arrays(Path(job['candidate_dir']) / 'candidates.npz')
        generation = read(Path(job['candidate_dir']) / 'generation.json')
        original_generation = read(Path(source_job['candidate_dir']) / 'generation.json')
        check_hash(Path(job['candidate_dir']) / 'candidates.npz', generation['candidates_sha256'])
        assert generation['complete'] and generation['gesture_id'] == source_receipt['gesture_id']
        assert generation['source_pool'] == source_job['candidate_dir']
        assert generation['records'] == original_generation['records']
        check_hash(Path(job['candidate_dir']) / 'world_object_pose.json', sha(Path(source_job['candidate_dir']) / 'world_object_pose.json'))
        assert set(pool) == set(old_pool)
        delta = np.array(case['delta_world_m'], dtype=float)
        assert delta.shape == (3,) and np.isfinite(delta).all()
        if case['variant'] == 'baseline':
            np.testing.assert_array_equal(delta, np.zeros(3))
        for key in pool:
            assert pool[key].dtype == old_pool[key].dtype
            if key != 'proposal_palm':
                np.testing.assert_array_equal(pool[key], old_pool[key])
        wanted = old_pool['proposal_palm'].copy(); wanted[0, :3, 3] += delta
        np.testing.assert_array_equal(pool['proposal_palm'], wanted)
        declared = next(row for row in results['results'] if row['object_id'] == oid)
        directory = Path(declared['evaluation']).parent
        assert directory == Path(job['output_dir']) == run / 'execution' / oid
        audited.append(audit_execution(directory, job, source_job, old_pool, delta, declared))
    total = sum(row['physical_steps'] for row in audited)
    assert total <= 16800 and results.get('physical_steps', total) == total
    outcomes = {row['object_id']: row['task_success'] for row in audited}
    promising = (outcomes['egad_A0_baseline'] and outcomes['egad_A0_refined']
                 and any(not outcomes[source + '_baseline'] and outcomes[source + '_refined']
                         for source in SOURCE_JOBS if source != 'egad_A0'))
    decision = 'promising_diagnostic_only' if promising else 'stop_this_rule'
    assert results['decision'] == decision
    assert results['actual_grasps'] == 8 and results['success'] == sum(outcomes.values())
    assert not results['training_writeback'] and not results['default_promotion']
    return dict(schema='boya_closure_translation_independent_audit_v1', complete=True,
                protocol=ref(run / 'protocol.json'), results=ref(run / 'RESULTS.json'),
                frozen_hash_counts=counts, prepared_hash_counts=prepared_counts,
                source_bindings=[value[0] for value in sources.values()], cases=audited,
                physical_steps=total, decision=decision,
                variants={variant: dict(success=sum(row['task_success'] for row in audited if row['object_id'].endswith('_' + variant)), total=4)
                                               for variant in ('baseline', 'refined')},
                new_physics_steps_by_audit=0, new_training_updates_by_audit=0,
                limitations=['Known-object diagnostic; not external generalization, full-arm validation or paper reproduction.',
                             'Five gates are recomputed from every-step traces; nonhand support and fixed-base extrema are bound to saved native reports.',
                             'Native 20um checks preservation of imported convex proxies, not precision against the source CAD.',
                             'Every-step raw contacts/torques are not retained; no independent friction impulse or causal proof.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = preparation() if args.prepare else audit_run(args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(complete=True, output=str(args.output), cases=len(result.get('cases', result.get('sources', []))),
                          physical_steps=result.get('physical_steps', 0)), ensure_ascii=False))


if __name__ == '__main__':
    main()
