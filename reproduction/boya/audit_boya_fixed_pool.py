"""Read-only audit of the frozen existing-pool Boya coverage diagnostic.

No simulator, learned model, training, or candidate generation is imported.
Every physical command is independently reconstructed from frozen endpoints and
URDF mimic relations. The previous trace/source-mesh audit is reused unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from audit_boya_closure_refinement import (
    audit_execution, arrays, check_hash, read, ref, sha, source_binding, PHASES,
)

ROOT = Path(__file__).resolve().parents[2]
EXECUTOR = ROOT / 'reproduction/boya/run_boya_floating_diagnostic.py'
EXECUTOR_SHA = 'f7b581e0428049b4f2bdc239989ae0c54028d0df5d08eb462ab54d74e01e5727'
OBJECTS = ('egad_A3', 'egad_A6', 'egad_E0')
POSE_KEYS = ('proposal_palm', 'proposal_q', 'proposal_q_close',
             'proposal_approach_direction', 'world_T_object')
LOGICAL_KEYS = ('source_candidate_index', 'gesture_id', 'raw_depth_index',
                'extra_depth_index', 'if_flip', 'canonical_CGR480_identity',
                'query_index', 'target_index', 'jaw_assignment')
CONFIG_KEYS = ('hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config')
ROW_KEYS = ('mesh_path', 'mesh_sha256', 'urdf_path', 'urdf_sha256', 'scale',
            'object_mass_kg', 'object_friction', 'convex_parts', 'world_T_object')
PHYSICS_KEYS = ('engine', 'device', 'dt_s', 'declared_command_steps', 'arm_present',
                'object_weight_world_n', 'global_gravity_m_s2', 'commanded_gantry', 'phase_lengths')
GANTRY_NAMES = ('diagnostic_slide_x', 'diagnostic_slide_y', 'diagnostic_slide_z')


def stable_key(record):
    return tuple(int(record[key]) for key in LOGICAL_KEYS[:5])


def stable_id(oid, record):
    source, gesture, depth, extra, flip = stable_key(record)
    return f'{oid}_s{source:04d}_g{gesture:02d}_d{depth}_e{extra}_f{flip}'


def pool_arrays(directory):
    data = arrays(Path(directory) / 'candidates.npz')
    return {key: data[key] for key in POSE_KEYS}


def exact_pose(left, right):
    for key in POSE_KEYS:
        assert left[key].dtype == right[key].dtype, key
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def baseline_job(oid):
    return read(ROOT / f'outputs/egad_floating_three_20261108_v1/jobs/{oid}.json')


def verify_physics(job, oid):
    baseline = baseline_job(oid)
    for key in CONFIG_KEYS:
        assert job[key] == baseline[key], key
    for key in ROW_KEYS:
        assert job['object_row'][key] == baseline['object_row'][key], key
    assert job.get('candidate_index', 0) == 0
    assert not job.get('closure_control') and not job.get('collision_refinement')
    assert job.get('variant') != 'refined'
    protocol = read(job['protocol_path'])
    check_hash(job['protocol_path'], job['protocol_sha256'])
    assert protocol['source_sha256'][str(EXECUTOR)] == EXECUTOR_SHA
    result = read(Path(job['output_dir']) / 'evaluation.json')
    reference = read(Path(baseline['output_dir']) / 'evaluation.json')
    assert result['protocol_ref'] == dict(path=job['protocol_path'], sha256=job['protocol_sha256'])
    for key in PHYSICS_KEYS:
        assert result[key] == reference[key], key
    for key in ('actuation', 'gantry', 'mass_kg', 'native_mimics', 'solver_position_iteration_count'):
        assert result['audit'][key] == reference['audit'][key], key
    for body in ('robot', 'object'):
        for key in ('static_friction', 'dynamic_friction', 'restitution', 'rest_offset_m', 'contact_offset_m'):
            assert result['audit']['materials'][body][key] == reference['audit']['materials'][body][key]
    return result


def reconstruct_commands(job):
    """Independent arithmetic; does not call executor/closure command functions."""
    pool = arrays(Path(job['candidate_dir']) / 'candidates.npz')
    opened, closed = pool['proposal_q'][0], pool['proposal_q_close'][0]
    palm, direction = pool['proposal_palm'][0], pool['proposal_approach_direction'][0]
    pre = palm.copy(); pre[:3, 3] -= .1 * direction
    actual = arrays(Path(job['output_dir']) / 'commands.npz')
    names = actual['joint_names'].tolist()
    motors = pool['independent_motor_names'].tolist()
    assert len(motors) == 16 and len(names) == 23
    assert set(names) == set(GANTRY_NAMES) | set(motors) | {'FFJ1', 'MFJ1', 'RFJ1', 'LFJ1'}
    mimics = {}
    for joint in ET.parse(job['hand_urdf']).getroot().findall('joint'):
        relation = joint.find('mimic')
        if relation is not None:
            mimics[joint.get('name')] = (relation.get('joint'),
                                        float(relation.get('multiplier', '1')),
                                        float(relation.get('offset', '0')))
    targets, desired, phases = [], [], []
    for phase, count in PHASES:
        for step in range(count):
            progress = (step + 1) / count
            u = np.asarray(progress, dtype=float)
            smooth = float(u * u * u * (10. + u * (-15. + 6. * u)))
            pose = pre.copy() if phase in ('prehold', 'approach') else palm.copy()
            if phase == 'approach':
                pose[:3, 3] += smooth * .1 * direction
            elif phase == 'lift':
                pose[:3, 3] += [0., 0., smooth * .1]
            q = ((1. - progress) * opened + progress * closed if phase == 'close'
                 else closed if phase in ('hold', 'lift') else opened)
            state = dict(zip(motors, q))
            for name, (driver, coefficient, offset) in mimics.items():
                state[name] = state[driver] * coefficient + offset
            xyz = pre[:3, :3].T @ (pose[:3, 3] - pre[:3, 3])
            state.update(zip(GANTRY_NAMES, xyz))
            targets.append([state[name] for name in names]); desired.append(pose); phases.append(phase)
    targets = np.asarray(targets, dtype=np.float32)
    velocity = np.zeros_like(targets)
    axes = [names.index(name) for name in GANTRY_NAMES]
    velocity[1:, axes] = np.diff(targets[:, axes], axis=0) / .004
    expected = dict(joint_target=targets, velocity_target=velocity,
                    desired_palm=np.asarray(desired), phase=np.asarray(phases),
                    time_s=np.arange(1, 2101) * .004)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value, err_msg=key)
    init = read(Path(job['output_dir']) / 'initialization_audit.json')
    np.testing.assert_array_equal(init['expected_palm'], pre)
    root = np.asarray(init['expected_root_pose_xyzw'])
    np.testing.assert_array_equal(root[:3], pre[:3, 3])
    np.testing.assert_allclose(Rotation.from_quat(root[3:]).as_matrix(), pre[:3, :3], atol=2e-14, rtol=0)
    object_pose = np.asarray(init['expected_object_pose_xyzw'])
    np.testing.assert_array_equal(object_pose[:3], pool['world_T_object'][:3, 3])
    np.testing.assert_allclose(Rotation.from_quat(object_pose[3:]).as_matrix(),
                               pool['world_T_object'][:3, :3], atol=2e-14, rtol=0)
    assert init['physics_steps'] == 0
    assert init['articulation_root_initial_pose_writes'] == init['object_initial_pose_writes'] == 1
    return dict(all_2100_commands_reconstructed=True, twenty_hand_and_three_gantry_joint_targets_bitwise_equal=True,
                velocity_time_phase_palm_bitwise_equal=True, initial_object_and_palm_verified=True)


def inspect_execution(job_path, candidate, declared=None):
    job_path = Path(job_path); job = read(job_path)
    oid = candidate['object_id']
    exact_pose(pool_arrays(job['candidate_dir']), pool_arrays(candidate['candidate_dir']))
    evaluation = verify_physics(job, oid)
    output = Path(job['output_dir'])
    if declared is None:
        declared = dict(evaluation_sha256=sha(output / 'evaluation.json'),
                        task_success=evaluation['task_success'], physical_steps=evaluation['total_physics_steps'])
    command_audit = reconstruct_commands(job)
    # Self is solely the unmodified helper's reference trajectory. The independent
    # reconstruction above and baseline comparison prevent self-equality shortcuts.
    result = audit_execution(output, job, job, pool_arrays(job['candidate_dir']), np.zeros(3), declared)
    result.update(candidate_id=candidate['candidate_id'], source_object_id=oid, job=ref(job_path),
                  independent_command_reconstruction=command_audit)
    return result


def audit_catalog(path, include_history=True):
    path = Path(path); catalog = read(path)
    check_hash(EXECUTOR, EXECUTOR_SHA)
    assert catalog['complete'] and catalog['total_unique'] == 57
    assert catalog['reusable_unique'] == 15 and catalog['new_attempts'] == 42
    for filename, digest in catalog['input_sha256'].items():
        check_hash(filename, digest)
    candidates = catalog['candidates']; ids = [c['candidate_id'] for c in candidates]
    assert len(ids) == len(set(ids)) == 57
    assert candidates == sorted(candidates, key=lambda c: (c['object_id'], c['stable_order_key']))
    all_pools, history, pose_cache = set(), [], {}
    for candidate in candidates:
        oid = candidate['object_id']; assert oid in OBJECTS
        record = candidate['candidate_record']; cid = candidate['candidate_id']
        assert cid == stable_id(oid, record) and candidate['stable_order_key'] == list(stable_key(record))
        canonical = pool_arrays(candidate['candidate_dir']); pose_cache[cid] = canonical
        for key, metadata in candidate['physical_arrays'].items():
            value = canonical[key]
            assert list(value.shape) == metadata['shape'] and value.dtype.str == metadata['dtype']
            assert hashlib.sha256(value.tobytes()).hexdigest() == metadata['sha256']
        for provenance in candidate['source_provenance']:
            pool = Path(provenance['candidate_dir']); all_pools.add(str(pool))
            check_hash(pool / 'generation.json', provenance['generation_sha256'])
            check_hash(pool / 'candidates.npz', provenance['candidates_sha256'])
            generation = read(pool / 'generation.json')
            assert generation['complete'] and generation['records'][0]['geometry']['ok']
            assert generation['candidates_sha256'] == provenance['candidates_sha256']
            assert generation['records'][0] == provenance['candidate_record']
            assert generation['records'][0]['global_score_rank'] == provenance['global_score_rank']
            exact_pose(canonical, pool_arrays(pool))
            for key in LOGICAL_KEYS:
                assert record[key] == generation['records'][0][key], (cid, key)
        eligible = candidate['eligible_history']
        assert eligible == sorted(eligible, key=lambda h: (h['chronology_order'], h['job_path']))
        assert candidate['selected_history'] == (eligible[0] if eligible else None)
        assert candidate['requires_new_physical_attempt'] == (not eligible)
        assert candidate['duplicate_history_outcome_consistent'] == (len({h['task_success'] for h in eligible}) <= 1)
        for row in eligible:
            check_hash(row['job_path'], row['job']['sha256'])
            check_hash(row['evaluation_path'], row['evaluation']['sha256'])
            if include_history:
                measured = inspect_execution(row['job_path'], candidate,
                    dict(evaluation_sha256=row['evaluation']['sha256'], task_success=row['task_success'],
                         physical_steps=row['physical_steps']))
                history.append(measured)
    expected_pools = set()
    for base in ('egad_external12_20261107_v1', 'boya_original_recipe_alignment_20261108_v1'):
        for oid in OBJECTS:
            for pool in (ROOT / 'outputs' / base / 'diagnostic' / oid).glob('pool_*'):
                generation = read(pool / 'generation.json')
                if generation.get('complete') and generation['records'][0].get('geometry', {}).get('ok'):
                    expected_pools.add(str(pool))
    assert all_pools == expected_pools and len(expected_pools) == 61
    for index, candidate in enumerate(candidates):
        for other in candidates[index + 1:]:
            if candidate['object_id'] == other['object_id']:
                assert not all(np.allclose(pose_cache[candidate['candidate_id']][key],
                                          pose_cache[other['candidate_id']][key], atol=2e-12, rtol=0)
                               for key in POSE_KEYS), 'Unmerged physical duplicate'
    positive_controls = [row for row in catalog['A0_separate_history']
                         if row['role'] == 'independent_positive_control']
    assert len(positive_controls) == 1
    positive = positive_controls[0]
    check_hash(positive['job']['path'], positive['job']['sha256'])
    check_hash(positive['evaluation']['path'], positive['evaluation']['sha256'])
    positive_audit = None
    if include_history:
        binding, control_job, control_pool = source_binding('egad_A0')
        assert positive['job']['path'] == binding['source_job']['path']
        control_commands = reconstruct_commands(control_job)
        positive_audit = audit_execution(Path(control_job['output_dir']), control_job, control_job,
            control_pool, np.zeros(3), dict(evaluation_sha256=positive['evaluation']['sha256'],
                                          task_success=True, physical_steps=2100))
        positive_audit.update(independent_command_reconstruction=control_commands,
                              excluded_from_57_candidate_denominator=True)
    return dict(catalog=ref(path), complete=True, unique_candidates=57, saved_pool_provenances=61,
                logical_aliases_identical=True, all_duplicate_arrays_dtype_and_value_exact=True,
                reusable_unique=15, new_attempts=42, histories_checked=history,
                separate_historical_A0_positive_control=positive_audit,
                source_and_input_hash_count=len(catalog['input_sha256']))


def audit_run(catalog_path, run):
    catalog = read(catalog_path); run = Path(run)
    catalog_audit = audit_catalog(catalog_path)
    protocol = read(run / 'protocol.json'); prepared = read(run / 'PREPARED.json')
    for document in (protocol, prepared):
        for section in ('source_sha256', 'input_sha256', 'derived_sha256', 'jobs_sha256'):
            for filename, digest in document.get(section, {}).items():
                check_hash(filename, digest)
    results = read(run / 'RESULTS.json')
    assert results['status'] == 'complete'
    by_id = {row['candidate_id']: row for row in results['results']}
    assert len(by_id) == len(results['results']) == 57
    assert set(by_id) == {candidate['candidate_id'] for candidate in catalog['candidates']}
    measured = []
    for candidate in catalog['candidates']:
        cid = candidate['candidate_id']; declared = by_id[cid]
        history = candidate['selected_history']
        assert declared['kind'] == ('reused' if history else 'new')
        expected_job = Path(history['job_path']) if history else run / 'jobs' / f'{cid}.json'
        assert Path(declared['job_path']) == expected_job
        assert declared['source_object_id'] == candidate['object_id']
        result = inspect_execution(expected_job, candidate, declared)
        assert declared['gates'] == result['gates']
        measured.append(result)
    new_ids = {c['candidate_id'] for c in catalog['candidates'] if c['selected_history'] is None}
    actual_jobs = {p.stem for p in (run / 'jobs').glob('*.json')}
    assert actual_jobs == new_ids and len(actual_jobs) == 42
    steps = sum(row['physical_steps'] for row in measured if row['candidate_id'] in new_ids)
    assert steps <= 88200
    return dict(schema='boya_fixed_pool_independent_audit_v1', complete=True,
                catalog_audit=catalog_audit, results=ref(run / 'RESULTS.json'), records=measured,
                observed_candidates=57, unique_history_reused=15, newly_executed=42,
                new_actual_physics_steps=steps, audit_physics_steps=0, audit_CGR_forwards=0,
                audit_training_updates=0, independent_source_mesh_bottom_and_bilateral_load_recomputed=True,
                frozen_57_pool_coverage_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--catalog-only', action='store_true')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.catalog_only:
        assert args.run is None
        result = audit_catalog(args.catalog)
    else:
        assert args.run is not None
        result = audit_run(args.catalog, args.run)
    result['audit_source'] = ref(Path(__file__).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(complete=result['complete'], output=str(args.output), sha256=sha(args.output))))


if __name__ == '__main__':
    main()
