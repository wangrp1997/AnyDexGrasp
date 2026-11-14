"""Independent saved-array audit of HOME-only changes; never launches physics.

This compares a new four-type preparation with the frozen 4 cm preparation.
Geometry admission is reported from its separately bound receipt, not inferred
from matching endpoints or confused with a physical grasp result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def check_binding(binding):
    if digest(binding['path']) != binding['sha256']:
        raise ValueError('Changed binding: '+binding['path'])


def compare_commands(reference_dir, proposed_dir):
    """Check actual motor arrays and URDF limits without calling the router."""
    reference_dir, proposed_dir = Path(reference_dir), Path(proposed_dir)
    old_plan, new_plan = read(reference_dir/'plan.json'), read(proposed_dir/'plan.json')
    old, new = arrays(reference_dir/'commands.npz'), arrays(proposed_dir/'commands.npz')
    for root, plan in ((reference_dir, old_plan), (proposed_dir, new_plan)):
        assert digest(root/'commands.npz') == plan['commands_sha256']
    assert set(old) == set(new)
    assert np.array_equal(old['joint_names'], new['joint_names'])
    assert np.array_equal(old['initial_position'], new['initial_position'])
    assert old_plan['arguments']['pregrasp_distance_m'] == new_plan['arguments']['pregrasp_distance_m'] == .04
    assert old_plan['dt_s'] == new_plan['dt_s'] == .004
    phases = ['settle', 'home_to_pregrasp', 'approach']
    for data in (old, new):
        assert list(dict.fromkeys(data['phase'].tolist())) == phases
        assert data['joint_target'].shape == (len(data['phase']), 24)
        for name in ('joint_target', 'desired_palm', 'time_s', 'initial_position'):
            assert np.isfinite(data[name]).all()
        np.testing.assert_allclose(data['time_s'], np.arange(1, len(data['phase'])+1)*.004,
                                   rtol=0, atol=1e-12)
    matches = {}
    for phase in ('settle', 'approach'):
        before, after = old['phase'] == phase, new['phase'] == phase
        for name in ('joint_target', 'desired_palm'):
            assert np.array_equal(old[name][before], new[name][after]), (phase, name)
            matches[phase+'_'+name] = True
    assert int((new['phase'] == 'approach').sum()) == 500
    home_old, home_new = old['phase'] == phases[1], new['phase'] == phases[1]
    for name in ('joint_target', 'desired_palm'):
        np.testing.assert_array_equal(old[name][home_old][-1], new[name][home_new][-1])
    names = list(map(str, new['joint_names']))
    motor_names = list(new_plan['hand_motor_names'])
    motor_cols = [names.index(name) for name in motor_names]
    assert len(motor_cols) == 16
    hand = new['joint_target'][home_new][:, motor_cols]
    np.testing.assert_array_equal(hand, np.broadcast_to(old['joint_target'][home_old][0, motor_cols], hand.shape))
    arm_names = {'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'}
    non_arm = [i for i, name in enumerate(names) if name not in arm_names]
    assert len(non_arm) == 18
    values = new['joint_target'][home_new][:, non_arm]
    np.testing.assert_array_equal(values, np.broadcast_to(old['initial_position'][non_arm], values.shape))
    model = ET.parse(new_plan['arguments']['robot_urdf']).getroot()
    limits = {joint.attrib['name']: joint.find('limit') for joint in model.findall('joint')}
    commands = new['joint_target']
    excess, velocity = {}, {}
    for column, name in enumerate(names):
        limit = limits[name]
        if limit is None:
            continue
        if 'lower' in limit.attrib and 'upper' in limit.attrib:
            violation = max(float(np.max(float(limit.attrib['lower'])-commands[:, column])),
                            float(np.max(commands[:, column]-float(limit.attrib['upper']))), 0.)
            excess[name] = violation
            assert violation <= 1e-7, (name, violation)
        if 'velocity' in limit.attrib:
            # Include the actual initial-to-first-command transition. This
            # checks saved commands rather than planner timing metadata.
            trace = np.r_[new['initial_position'][column], commands[:, column]]
            ratio = float(np.max(np.abs(np.diff(trace)))/.004/float(limit.attrib['velocity']))
            velocity[name] = ratio
            assert ratio <= 1.+1e-7, (name, ratio)
    arm_columns = [i for i, name in enumerate(names) if name in arm_names]
    arm_limits = np.array([float(limits[names[i]].attrib['velocity']) for i in arm_columns])

    def derivatives(values):
        velocity = np.gradient(values, .004, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, .004, axis=0, edge_order=2)
        jerk = np.gradient(acceleration, .004, axis=0, edge_order=2)
        return dict(velocity_limit_ratio=float(np.max(abs(velocity)/arm_limits)),
                    max_acceleration_rad_s2=float(np.max(abs(acceleration))),
                    max_jerk_rad_s3=float(np.max(abs(jerk))))

    home_derivatives = derivatives(np.vstack([new['initial_position'][arm_columns],
                                              commands[home_new][:, arm_columns]]))
    assert home_derivatives['velocity_limit_ratio'] <= 1.+1e-8
    assert home_derivatives['max_acceleration_rad_s2'] <= 3.*(1.+1e-8)
    assert home_derivatives['max_jerk_rad_s3'] <= 30.*(1.+1e-8)
    all_arm_derivatives = derivatives(np.vstack([new['initial_position'][arm_columns],
                                                 commands[:, arm_columns]]))
    return dict(reference_commands_sha256=digest(reference_dir/'commands.npz'),
        proposed_commands_sha256=digest(proposed_dir/'commands.npz'), unchanged_arrays=matches,
        initial_position_exact=True, pregrasp_joint_and_palm_endpoint_exact=True,
        home_hand_16_motors_exact=True, home_non_arm_18_columns_exact=True, approach_steps=500, native_columns=24,
        old_home_steps=int(home_old.sum()), new_home_steps=int(home_new.sum()),
        maximum_joint_limit_violation_rad=max(excess.values(), default=0.),
        maximum_command_velocity_limit_ratio=max(velocity.values(), default=0.),
        home_arm_derivatives=home_derivatives,
        full_prefix_arm_derivatives=all_arm_derivatives,
        proposed_plan_admitted=new_plan['admitted'])


def compare_release_spec(release_spec):
    release_spec = Path(release_spec)
    spec = read(release_spec)
    target = np.asarray(spec['world_T_release_palm'], dtype=float)
    # Independent XML/NumPy FK result from the preceding source review. This
    # does not reuse the release generator's PlanningModel implementation.
    rotation = np.array([[.0007962574874067301, -.013184925338607408, -.9999127580533358],
                         [.9999996829318324, -4.10025819803366e-10, .0007963267134376482],
                         [-1.0499918251862382e-05, -.9999130750939378, .013184921157760834]])
    np.testing.assert_allclose(target[:3, :3], rotation, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(target[:3, 3], [-.4, -.25, .35])
    assert spec['depends_on_candidate_pose'] is False and spec['depends_on_object_pose'] is False
    check_binding(spec['scene'])
    for path, expected in spec['source_sha256'].items():
        assert digest(path) == expected, path
    return dict(release_spec_sha256=digest(release_spec),
        release_rotation_max_error=float(np.max(abs(target[:3, :3]-rotation))),
        fixed_translation_exact=True, candidate_and_object_pose_independent=True,
        source_sha256_verified=len(spec['source_sha256']))


def audit(reference, proposed, release_spec):
    reference, proposed, release_spec = map(Path, (reference, proposed, release_spec))
    release_audit = compare_release_spec(release_spec)
    results = []
    semantic_fields = ('source_row_index', 'source_top500_rank', 'collision_rank', 'gesture_id',
        'source_reference_type', 'score', 'source_if_flip', 'source_row15_camera',
        'actual_row15_camera', 'actual_row17_camera', 'source_feature2261', 'canonical_cgr480',
        'raw_angle_index', 'raw_depth_index', 'raw_width_m', 'expanded_width_m', 'raw_depth_m',
        'extra_depth_m', 'extra_depth_index', 'total_depth_m', 'slot_id', 'slot_sha256',
        'world_T_palm', 'world_T_camera', 'action_contract', 'collision_profile', 'task')
    for gesture in (1, 6, 7, 8):
        old_dir, new_dir = reference/f'prepare_g{gesture}', proposed/f'prepare_g{gesture}'
        old = read(old_dir/'pool/selected_author_candidate.json')
        new = read(new_dir/'pool/selected_author_candidate.json')
        for name in semantic_fields:
            assert old[name] == new[name], (gesture, name)
        protocol = read(new_dir/'PIPELINE_PROTOCOL.json')
        for category in ('sources', 'inputs', 'capture_source_bindings'):
            for binding in protocol.get(category, {}).values():
                check_binding(binding)
        assert protocol['inputs']['fixed_release']['sha256'] == digest(release_spec)
        result = read(new_dir/'PIPELINE_RESULTS.json')
        assert result['physics_steps'] == result['new_CGR_forwards'] == result['training_updates'] == result['new_labels'] == 0
        arm = result['arm_preparation']
        route = arm.get('home_route')
        row = dict(gesture_id=gesture, status=result['status'], stage=result['stage'],
                   semantic_fields_exact=len(semantic_fields), candidate_raw_index=new['source_row_index'],
                   fixed_release_sha256=digest(release_spec), arm_status=arm['status'])
        if route:
            row['home_route_status'] = route['status']
        if arm['status'] == 'ready':
            row['command_audit'] = compare_commands(old_dir/'arm/native', arm['plan_dir'])
        else:
            row['command_audit'] = None
        results.append(row)
    return dict(schema='anydex_boya_author_home_delivery_audit_v1', all_checked_invariants_passed=True,
        reference=str(reference.resolve()), proposed=str(proposed.resolve()), release_spec=str(release_spec.resolve()),
        fixed_release_audit=release_audit, results=results,
        geometry_scope='Bound planner receipts require their separate geometric audit; this script checks command and provenance invariants',
        physics_steps=0, new_CGR_forwards=0, training_updates=0, new_labels=0)


def audit_components(reference, components, release_spec):
    """Audit new transport receipts while retaining the original candidate."""
    reference, components = Path(reference), Path(components)
    release = compare_release_spec(release_spec)
    rows = []
    for gesture in (1, 6, 7):
        directory = components/f'integration_g{gesture}'
        result = read(directory/'COMPONENT_INTEGRATION_RESULTS.json')
        protocol_path = directory/'COMPONENT_INTEGRATION_PROTOCOL.json'
        protocol = read(protocol_path)
        assert result['complete'] is True
        assert result['protocol_sha256'] == digest(protocol_path)
        for category in ('inputs', 'sources', 'capture_source_bindings'):
            for binding in protocol.get(category, {}).values():
                check_binding(binding)
        inputs = protocol['inputs']
        expected_candidate = reference/f'prepare_g{gesture}/pool/selected_author_candidate.json'
        assert Path(inputs['original_candidate']['path']) == expected_candidate.resolve()
        assert inputs['fixed_release']['sha256'] == release['release_spec_sha256']
        for key in ('physics_steps', 'new_labels', 'training_updates', 'new_CGR_forwards',
                    'new_collection_collision_queries', 'new_native_approach_IK', 'new_HOME_search_calls'):
            assert result[key] == 0
        routed = Path(inputs['routed_plan']['path']).parent
        route_audit = compare_commands(reference/f'prepare_g{gesture}/arm/native', routed)
        raw = directory/'transport_raw'
        plan, commands = read(raw/'plan.json'), arrays(raw/'commands.npz')
        route_commands = arrays(routed/'commands.npz')
        assert digest(raw/'commands.npz') == plan['commands_sha256']
        prefix_length = len(route_commands['phase'])
        for key in ('joint_target', 'desired_palm', 'phase', 'time_s'):
            np.testing.assert_array_equal(commands[key][:prefix_length], route_commands[key])
        for key in ('initial_position', 'joint_names'):
            np.testing.assert_array_equal(commands[key], route_commands[key])
        np.testing.assert_array_equal(commands['desired_palm'][-1],
                                      read(release_spec)['world_T_release_palm'])
        for key, filename in (('path_geometry_sha256', 'path_geometry.json'),
                              ('lift_arm_geometry_sha256', 'lift_arm_geometry.json')):
            assert digest(raw/filename) == plan[key]
        open_geometry = read(raw/'path_geometry.json')
        post_geometry = read(raw/'lift_arm_geometry.json')
        assert post_geometry['primary_body_excluded'] is False
        assert post_geometry['initial_pose_proxy_object_ids'] == ['graspnet_000', 'graspnet_002', 'graspnet_005']
        post_phases = [p for p in commands['phase'].tolist()
                       if p not in ('settle', 'home_to_pregrasp', 'approach')]
        counts = Counter((post_phases[v['sample']], v['kind']) for v in post_geometry['violations'])
        endpoint = [v for v in post_geometry['violations'] if v['sample'] == len(post_phases)-1]
        assert len(plan['transport_ik_waypoints']) == 96
        rows.append(dict(gesture_id=gesture, status=result['status'],
            original_candidate_exact=True, new_prefix_exact=True, command_steps=len(commands['phase']),
            route_audit=route_audit, original_transport_IK_returns=96,
            open_geometry_ok=open_geometry['ok'], post_geometry_ok=post_geometry['ok'],
            post_violation_counts=[dict(phase=p, kind=k, count=n) for (p,k),n in sorted(counts.items())],
            release_endpoint_violations=endpoint, fixed_release_desired_endpoint_exact=True,
            protocol_sha256=digest(protocol_path), transport_plan_sha256=digest(raw/'plan.json')))
    return dict(schema='anydex_boya_author_component_array_audit_v1', status='passed',
        results=rows, release_spec_audit=release, source_sha256=digest(__file__),
        scope='Saved command arrays, exact candidate provenance, frozen dependencies and independent geometry receipt aggregation; no physics or replanning',
        physics_steps=0, new_CGR_forwards=0, training_updates=0, new_labels=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'proposed', 'release-spec', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.reference, args.proposed, args.release_spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
