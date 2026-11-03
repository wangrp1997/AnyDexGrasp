"""Read-only checks of reusable DH3 scene executions; no simulator or inference.

Grasp outcome and numerical validity are recomputed independently from saved
per-step states and contacts. This is not a hardware or force-closure proof.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def eq(actual, expected, label, atol=1e-12):
    if isinstance(expected, (str, bool)):
        assert actual == expected, (label, actual, expected)
    else:
        assert np.allclose(actual, expected, rtol=0, atol=atol), label


def contact_summary(rows):
    return dict(rows=len(rows), positive_normal_rows=int((rows[:, 6] > 0).sum()),
                min_signed_distance_m=float(rows[:, 5].min()) if len(rows) else None,
                peak_normal_force_N=float(rows[:, 6].max()) if len(rows) else None)


def audit(run):
    run = Path(run).resolve()
    result = read(run / 'RESULTS.json')
    controller = read(run / 'CONTROLLER.json')
    with np.load(run / 'trace.npz', allow_pickle=False) as saved:
        trace = {k: saved[k].copy() for k in saved.files}
    eq(sha(run / 'trace.npz'), result['trace_sha256'], 'trace SHA')
    eq(sha(run / 'CONTROLLER.json'), result['controller_sha256'], 'controller SHA')
    source = Path(controller['controller_script'])
    eq(sha(source), controller['controller_script_sha256'], 'imported controller SHA')
    for path_key, sha_key in [('adapter_manifest', 'adapter_manifest_sha256'),
                              ('derived_urdf', 'derived_urdf_sha256'),
                              ('scene_manifest', 'scene_manifest_sha256'),
                              ('selection_results', 'selection_sha256'),
                              ('candidate_npz', 'candidate_sha256')]:
        eq(sha(controller[path_key]), controller[sha_key], path_key)
    snapshot = read(run / 'SOURCE_SNAPSHOT.json')
    # The generic executor may freeze itself and the imported mechanism module
    # separately. Bind every recorded file without assuming a single-file model.
    def check_source_record(record):
        if isinstance(record, dict):
            if 'path' in record and 'sha256' in record:
                eq(sha(record['path']), record['sha256'], 'frozen source')
            for value in record.values():
                check_source_record(value)
        elif isinstance(record, list):
            for value in record:
                check_source_record(value)
    check_source_record(snapshot)
    if (run / 'RUN_SOURCES.json').exists():
        dependencies = read(run / 'RUN_SOURCES.json')
        for item in dependencies['runner_and_scene']:
            eq(sha(item['frozen_copy']), item['sha256'], 'preintegration source copy')
        runner_copy = run / 'runner_source_frozen.py'
        tree = ast.parse(runner_copy.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_scene')
        calls = [node.func.attr for node in ast.walk(fn) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == 'p']
        assert calls.count('stepSimulation') == 1
        assert not set(calls).intersection({'resetJointState', 'resetBasePositionAndOrientation',
                                           'resetBaseVelocity', 'resetSimulation', 'createConstraint'})
        guards = next(node.value for node in ast.walk(fn) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'guarded_names' for target in node.targets))
        assert set(ast.literal_eval(guards)) == {'resetJointState', 'resetBasePositionAndOrientation',
                                                'resetBaseVelocity', 'resetSimulation', 'createConstraint'}
        guard = next(node for node in ast.walk(fn) if isinstance(node, ast.FunctionDef) and node.name == 'forbid_reset')
        assert any(isinstance(node, ast.Raise) for node in guard.body)
    a = controller['assumptions_SI_not_hardware_identification']
    baseline = dict(solver_iterations=100, non_contact_inner_iterations=20,
                    position_motor_max_torque_Nm=.6, rotation_motor_max_torque_Nm=.3,
                    position_motor_max_velocity_rad_s=1., rotation_motor_max_velocity_rad_s=.5,
                    gear_max_torque_Nm=100., gear_erp=.2,
                    wrist_max_force_per_row_N_or_Nm=30., wrist_erp=.2,
                    position_gain=.15, velocity_gain=1., contact_lateral_friction=.5,
                    contact_restitution=0., geometric_margin_m=.001)
    for key, value in baseline.items():
        eq(a[key], value, key)
    eq(a['gravity_m_s2'], [0., 0., -9.81], 'gravity')
    dt = a['dt']
    eq(dt, 1 / 1920, 'default integration dt')
    n = len(trace['step'])
    eq(n, 19680, 'full execution steps')
    assert np.array_equal(trace['step'], np.arange(1, n + 1))
    assert np.array_equal(trace['time'], trace['step'] * dt)
    assert all(len(value) == n for value in trace.values())
    finite = all(np.isfinite(value).all() for value in trace.values() if value.dtype.kind in 'fiu')
    names = controller['joint_order']
    ix = {name: i for i, name in enumerate(names)}
    residual = np.column_stack([trace['q'][:, ix[g['dependent']]] -
                               g['factor'] * trace['q'][:, ix[g['master']]] - g['offset']
                               for g in controller['gears']])
    assert residual.tobytes() == trace['gear_residual'].tobytes()
    limits = np.zeros(len(names))
    limits[ix['finger1_joint']] = .6
    limits[ix['finger1_base_joint']] = .3
    motor_ok = bool((abs(trace['motor_torque']) <= limits + 1e-8).all())
    gear_ok = bool((abs(trace['gear_reaction']) <= 100. + 1e-7).all())
    wrist_force_ok = bool((abs(trace['wrist_reaction']) <= 30. + 1e-7).all())
    wrist_ok = bool((trace['wrist_position_error'] < .002).all())
    gearmax = float(abs(residual).max())
    speedmax = float(abs(trace['v']).max())
    stable = bool(finite and gearmax < .01 and speedmax < 1.5 and motor_ok and gear_ok and wrist_ok)
    for key, value in dict(physics_steps=n, physical_seconds=n * dt, finite=bool(finite),
                           max_gear_residual_rad=gearmax, peak_joint_velocity_rad_s=speedmax,
                           motor_torque_bounds_pass=motor_ok, gear_reaction_bounds_pass=gear_ok,
                           max_wrist_COM_error_m=float(trace['wrist_position_error'].max()),
                           port_stability_pass=stable).items():
        eq(result[key], value, key)
    engine = controller['actual_physics_engine_parameters']
    eq(engine['fixedTimeStep'], dt, 'engine actual dt')
    eq(engine['numSubSteps'], 1, 'no hidden integration substeps')
    scene = read(controller['scene_manifest'])
    selection = read(controller['selection_results'])
    eq(scene['scene_signature_sha256'], controller['scene_signature_sha256'], 'scene signature')
    for item in scene['files'] + scene['source_assets']:
        eq(sha(item['path']), item['sha256'], 'observed scene file')
    eq(selection['output']['sha256'], controller['candidate_sha256'], 'selected candidate SHA')
    with np.load(controller['candidate_npz'], allow_pickle=False) as selected:
        hand = selected['top1_DH3_grasp_array']
        two = selected['top1_two_finger_grasp_array']
    H = np.eye(4)
    H[:3, :3], H[:3, 3] = hand[3:12].reshape(3, 3), hand[12:15]
    T_WC = np.array(scene['camera']['T_WC'])
    T_WH = T_WC @ H
    direction = T_WC[:3, :3] @ two[4:13].reshape(3, 3)[:, 0]
    direction /= np.linalg.norm(direction)
    target = T_WH[:3, 3] + (hand[1] - hand[15] / 100 * .005) * direction
    eq(controller['original_hand_datum_T_WH'], T_WH, 'original top1 world hand transform')
    eq(controller['world_target_baseLink_m'], target, 'original depth insertion')
    eq(controller['world_pre_baseLink_m'], target - .07 * direction, 'original 70mm approach')
    start_time = (trace['step'] - 1) * dt
    def smooth(t):
        t = np.clip(t, 0., 1.)
        return t * t * (3 - 2 * t)
    expected_registers = np.column_stack([hand[15] * (1 - smooth((start_time - 3.5) / 2)),
                                         np.full(n, hand[16])])
    eq(trace['register_target'], expected_registers, 'same continuous register script', atol=1e-13)
    expected_q = np.column_stack([row['offset'] + row['position_slope'] * expected_registers[:, 0] +
                                 row['rotation_slope'] * expected_registers[:, 1]
                                 for row in controller['two_register_affine_map'].values()])
    eq(trace['joint_target'], expected_q, 'all commanded affine joint targets', atol=1e-12)
    info = controller['actual_constraint_info']
    hand_id = int(info['wrist'][0])
    constraints = [info['wrist']] + info['gears']
    assert len(constraints) == 10 and info['wrist'][2] == -1
    assert all(row[0] == hand_id and row[2] == hand_id for row in info['gears'])
    ids = controller['body_ids']
    assert ids['hand'] == hand_id
    assert all(ids['object'] not in (row[0], row[2]) for row in constraints)
    contacts = np.load(run / 'contacts.npy', allow_pickle=False)
    assert contacts.ndim == 2 and contacts.shape[1] == 9 and np.isfinite(contacts).all()
    assert ((contacts[:, 0] >= 1) & (contacts[:, 0] <= n)).all()
    def pair(body):
        return (((contacts[:, 1] == hand_id) & (contacts[:, 2] == body)) |
                ((contacts[:, 2] == hand_id) & (contacts[:, 1] == body)))
    obj = contacts[pair(ids['object'])]
    table = contacts[pair(ids['table'])]
    final_positive = obj[(obj[:, 0] > n - round(.5 / dt)) & (obj[:, 6] > 0)]
    count = np.bincount(final_positive[:, 0].astype(int), minlength=n + 1)[n - round(.5 / dt) + 1:]
    lift = float(trace['object_pose'][-1, 2] - scene['actual_state']['object']['position_world_m'][2])
    lowest = float(trace['object_aabb'][-1, 0, 2])
    held = bool(lift > .05 and lowest > .04 and len(final_positive) > 0)
    eq(result['lift_observed'], held, 'lifted with final contact')
    eq(result['object_final_lift_m'], lift, 'final lift')
    eq(result['object_final_lowest_world_z_m'], lowest, 'final lowest point')
    eq(result['status'], 'object_lifted' if held and stable else 'not_lifted_or_port_invalid', 'status')
    for key, value in dict(state_resets_after_first_step=0, object_attachment_constraints=0,
                           object_gravity_retained=True, continuous_same_scene=True).items():
        eq(result[key], value, key)
    frames = read(run / 'frames.json')
    assert frames['actual_simultaneous_render'] and frames['fps'] == 20
    records = frames['frames']
    assert len(records) == 205
    for i, frame in enumerate(records):
        eq(frame['physics_step'], 1 + i * 96, 'frame actual step')
        eq(frame['physical_time_s'], frame['physics_step'] * dt, 'frame actual time')
        with Image.open(run / frame['path']) as image:
            assert image.size == (640, 480)
            image.verify()
    return dict(run=str(run), status='saved_scene_execution_verified',
                physics_steps=n, physical_seconds=n * dt, validated_grasp=bool(held and stable),
                lift_observed=held, port_stability_pass=stable, final_lift_m=lift,
                final_lowest_world_z_m=lowest, max_gear_residual_rad=gearmax,
                peak_joint_velocity_rad_s=speedmax, wrist_reaction_bounds_pass=wrist_force_ok,
                final_positive_contact_steps=int((count > 0).sum()), final_contact_window_steps=len(count),
                final_every_step_contact=bool((count > 0).all()),
                hand_object=contact_summary(obj), hand_table=contact_summary(table),
                artifact_SHA={name: sha(run / name) for name in
                              ['RESULTS.json', 'CONTROLLER.json', 'SOURCE_SNAPSHOT.json',
                               'trace.npz', 'contacts.npy', 'frames.json']},
                checks=dict(per_step_residual_bits=True, finite_limits_and_original_gates=True,
                            original_top1_world_pose_and_depth=True, no_object_constraint_in_inventory=True,
                            same_register_and_joint_target_script=True, actual_dt_no_hidden_substeps=True,
                            all_205_frames_decode=True),
                additional_physics_steps=0, additional_network_forwards=0,
                boundary='Checks saved states, contact outcomes, source hashes and initial constraint inventory; '
                         'does not identify hardware dynamics or establish force closure/general success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Retain previous receipts; use a new output')
    start = time.monotonic()
    records = [audit(run) for run in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(scope=__doc__, runs=records, total_physics_steps_verified=sum(r['physics_steps'] for r in records),
                   additional_physics_steps=0, additional_network_forwards=0,
                   audit_wall_seconds=time.monotonic() - start)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
