"""Deterministically retime an already frozen, collision-free RRT HOME path.

No search, IK, path change, candidate selection or simulator is performed.
The failed first run remains immutable; this output binds it as an input.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import author_home_route as home
import author_joint_home_route as joint
import author_transport as transport

POLICY = dict(schema='anydex_boya_author_home_retime_policy_v1', dt_s=.004,
    max_iterations_including_original=8,
    scale_rule='max(velocity_ratio,sqrt(acceleration_ratio),cbrt(jerk_ratio))*1.01',
    scale_scope='Common duration multiplier for every original connected-path edge',
    path_coordinates_exact=True, search_calls=0, ik_calls=0, physics_steps=0,
    geometry_tolerance_m=1e-7, acceleration_rad_s2=3., jerk_rad_s3=30.,
    require_original_final_geometry_ok=True,
    scope='Same original initial/pregrasp joints and source hull gates; only HOME time parameterization changes')


class TimeScalingRejected(RuntimeError):
    """Expected bounded trajectory-limit rejection, distinct from broken IO."""


def scale_factor(audit):
    ratios = [audit['velocity_ratio'], np.sqrt(audit['acceleration_ratio']), np.cbrt(audit['jerk_ratio'])]
    if not np.isfinite(ratios).all() or min(ratios) < 0:
        raise ValueError('Finite nonnegative whole-HOME derivative ratios required')
    return float(max(ratios)*1.01)


def retime_joint_path(path, original_edge_durations, original_audit, velocity, lower, upper):
    """Reuse the identical joint polyline and original two-knot interpolator."""
    transport._backend_path()
    from canondex.grasp.template_arm_bridge import timed_segment
    path = np.asarray(path, dtype=float); durations = np.asarray(original_edge_durations, dtype=float)
    if path.ndim != 2 or len(path) < 2 or durations.shape != (len(path)-1,) or np.any(durations <= 0):
        raise ValueError('Frozen path and each original positive edge duration required')
    if original_audit.get('ok') is not False:
        raise ValueError('An explicitly failed original whole-HOME derivative audit is required')
    scale = 1.; attempts = [dict(iteration=0, original=True, whole_home=original_audit)]
    previous = original_audit
    for iteration in range(1, POLICY['max_iterations_including_original']):
        scale *= scale_factor(previous)
        pieces, timing = [], []
        for a, b, duration in zip(path[:-1], path[1:], durations):
            try:
                values, edge = timed_segment([a, b], dt=POLICY['dt_s'], velocity_limits=velocity,
                    minimum_s=float(duration*scale), acceleration=POLICY['acceleration_rad_s2'],
                    jerk=POLICY['jerk_rad_s3'], position_lower=lower, position_upper=upper)
            except RuntimeError as error:
                raise TimeScalingRejected(str(error)) from error
            values[-1] = b
            pieces.append(values[1:]); timing.append(edge)
        q = np.concatenate(pieces)
        previous = home.derivative_audit(q, path[0], velocity, lower, upper)
        attempts.append(dict(iteration=iteration, common_duration_scale=scale, whole_home=previous,
                             total_home_steps=len(q), original=False))
        if previous['ok']:
            return q, timing, previous, attempts
    raise TimeScalingRejected('Original whole-HOME derivative limits still fail after bounded common retiming')


def retime_home_route(prepared_route_dir, seed_plan_dir, scene_path, output, *, config_path, archive_sources=None):
    prepared_route_dir, seed_plan_dir, scene_path, output, config_path = (Path(p).resolve()
        for p in (prepared_route_dir, seed_plan_dir, scene_path, output, config_path))
    output.mkdir(parents=True, exist_ok=False); began = time.monotonic()
    receipt = dict(schema='anydex_boya_author_home_retime_preparation_v1', status='preparing',
        admitted=False, complete=False, plan_dir=str(output/'native'), policy=POLICY,
        search_calls=0, ik_calls=0, physics_steps=0, grasp_attempts=0, new_labels=0)
    transport.save(output/'HOME_ROUTE_PREPARATION.json', receipt)
    try:
        transport._backend_path()
        from run_template_arm_positive import bind_hand, object_asset
        seed = transport.read(seed_plan_dir/'plan.json'); args = seed['arguments']
        bind_hand(SimpleNamespace(hand_urdf=Path(args['hand_urdf']), hand_scene=Path(args['hand_scene'])))
        from canondex.planning.model import PlanningModel
        from canondex.robots.reference import load_reference
        from canondex.grasp.admm.arm_only_ik import ARM_JOINTS
        from canondex.grasp.template_arm_bridge import ArmPathGeometry
        old_receipt = transport.read(prepared_route_dir/'HOME_ROUTE_PREPARATION.json')
        original = transport.read(prepared_route_dir/'HOME_ROUTE_PROTOCOL.json')
        final_geometry = transport.read(prepared_route_dir/'full_scene_geometry.json')
        for key, name in (('geometry_sha256', 'full_scene_geometry.json'),
                          ('connected_path_sha256', 'connected_path.npz'),
                          ('candidate_commands_sha256', 'candidate_commands.npz')):
            if transport.sha(prepared_route_dir/name) != old_receipt[key]:
                raise ValueError('Original search output changed: '+name)
        if (old_receipt['status'] != 'null_geometry' or not old_receipt['search']['connected']
                or old_receipt['reason'] != 'first_rrt_connection_failed_final_tick_or_derivative_gate'
                or final_geometry.get('ok') is not True or old_receipt['derivative_audit']['ok'] is not False):
            raise ValueError('Only the frozen connected path with clear final geometry and failed derivative gate may be retimed')
        locks = dict(original['input_sha256'])
        for name, digest in original['source_sha256'].items():
            if name == str(Path(joint.__file__).resolve()):
                snapshot = prepared_route_dir/'AUTHOR_JOINT_HOME_ROUTE_SOURCE.py'
                if transport.sha(snapshot) != digest: raise ValueError('Original RRT source snapshot changed')
                locks[str(snapshot)] = digest
            else: locks[name] = digest
        for name in ('HOME_ROUTE_PREPARATION.json', 'HOME_ROUTE_PROTOCOL.json', 'full_scene_geometry.json',
                     'connected_path.npz', 'timing.json', 'candidate_commands.npz'):
            locks[str(prepared_route_dir/name)] = transport.sha(prepared_route_dir/name)
        if locks.get(str(seed_plan_dir/'plan.json')) != transport.sha(seed_plan_dir/'plan.json'):
            raise ValueError('Original seed plan differs from frozen search')
        if locks.get(str(config_path)) != transport.sha(config_path) or locks.get(str(scene_path)) != transport.sha(scene_path):
            raise ValueError('Scene/backend differs from frozen search')
        scene = transport.validate_scene(scene_path, archive_sources)
        model = PlanningModel(Path(args['robot_urdf'])); reference = load_reference(Path(args['reference_root']))
        arm = [model.active_names.index(n) for n in ARM_JOINTS]
        with np.load(seed_plan_dir/'commands.npz', allow_pickle=False) as data: old = {k:data[k].copy() for k in data.files}
        with np.load(prepared_route_dir/'connected_path.npz', allow_pickle=False) as data: path = data['arm_waypoints'].copy()
        block = home.phase_slice(old['phase'], 'home_to_pregrasp')
        if (not np.array_equal(path[0], old['initial_position'][arm])
                or not np.array_equal(path[-1], old['joint_target'][block.stop-1, arm])):
            raise ValueError('Frozen RRT endpoint changed')
        timings = transport.read(prepared_route_dir/'timing.json')['edges']
        if len(timings) != len(path)-1: raise ValueError('Frozen path/timing correspondence changed')
        parts, owners, part_locks = home._obstacles(scene, object_asset); locks.update(part_locks)
        sources = {str(Path(__file__).resolve()):transport.sha(__file__),
                   str(Path(home.__file__).resolve()):transport.sha(home.__file__),
                   str(Path(joint.__file__).resolve()):transport.sha(joint.__file__),
                   str(Path(transport.__file__).resolve()):transport.sha(transport.__file__)}
        for name, digest in {**locks, **sources}.items():
            if transport.sha(name) != digest: raise ValueError('Frozen retiming dependency changed: '+name)
        protocol = dict(schema='anydex_boya_author_home_retime_protocol_v1', policy=POLICY,
            input_sha256=locks, source_sha256=sources, original_rrt_protocol_sha256=transport.sha(prepared_route_dir/'HOME_ROUTE_PROTOCOL.json'),
            original_connected_path_sha256=transport.sha(prepared_route_dir/'connected_path.npz'),
            original_seed_plan_sha256=transport.sha(seed_plan_dir/'plan.json'),
            original_derivative_audit=old_receipt['derivative_audit'],
            original_geometry_sha256=transport.sha(prepared_route_dir/'full_scene_geometry.json'),
            obstacle_owners=owners, capture_source_bindings=original['capture_source_bindings'])
        transport.save(output/'HOME_ROUTE_PROTOCOL.json', protocol)
        shutil.copyfile(__file__, output/'AUTHOR_HOME_RETIME_SOURCE.py')
        receipt.update(protocol_sha256=transport.sha(output/'HOME_ROUTE_PROTOCOL.json'))
        q, edges, derivatives, attempts = retime_joint_path(path, [t['duration_s'] for t in timings],
            old_receipt['derivative_audit'], model.velocity[arm], model.lower[arm], model.upper[arm])
        all_q = np.broadcast_to(old['initial_position'], (len(q), 24)).copy(); all_q[:, arm] = q
        all_q[-1] = old['joint_target'][block.stop-1]
        state = lambda values:model.full_state(dict(zip(model.active_names, map(float, values))))
        palms = np.asarray([model.fk.transforms(state(values), reference.base_transform)['palm'] for values in all_q])
        candidate = home.replace_home(old, all_q, palms)
        geometry = ArmPathGeometry(model, model.ignored_pairs(), parts, np.eye(4), vectorized_broadphase=True)
        report = geometry.audit([state(values) for values in candidate['joint_target']], tolerance_m=POLICY['geometry_tolerance_m'])
        for row in report['violations']:
            if row['kind'] == 'object': row['obstacle_owner'] = owners[row['other']]
        transport.save(output/'full_scene_geometry.json', report)
        transport.save(output/'timing.json', dict(edges=edges, whole_home=derivatives, attempts=attempts))
        shutil.copyfile(prepared_route_dir/'connected_path.npz', output/'connected_path.npz')
        np.savez_compressed(output/'candidate_commands.npz', **candidate)
        receipt.update(derivative_audit=derivatives, timing_attempts=attempts,
            geometry_sha256=transport.sha(output/'full_scene_geometry.json'),
            connected_path_sha256=transport.sha(output/'connected_path.npz'),
            command_steps=len(candidate['phase']), home_command_steps=len(q))
        if not report['ok']:
            receipt.update(status='null_geometry', complete=True, reason='retimed_full_scene_tick_collision')
        else:
            native = output/'native'; native.mkdir()
            for name in ('lift_arm_geometry.json', 'frozen_input.json'):
                if (seed_plan_dir/name).exists(): shutil.copyfile(seed_plan_dir/name, native/name)
            shutil.copyfile(output/'candidate_commands.npz', native/'commands.npz')
            shutil.copyfile(output/'full_scene_geometry.json', native/'path_geometry.json')
            plan = copy.deepcopy(seed); plan['arguments']['output_dir'] = str(native)
            plan.update(complete=True, admitted=True, total_command_steps=len(candidate['phase']),
                commands_sha256=transport.sha(native/'commands.npz'), path_geometry_sha256=transport.sha(native/'path_geometry.json'),
                author_home_route=dict(strategy='joint_rrt_connect_then_whole_home_retime',
                    original_seed_plan_sha256=transport.sha(seed_plan_dir/'plan.json'),
                    protocol_path=str(output/'HOME_ROUTE_PROTOCOL.json'), protocol_sha256=receipt['protocol_sha256'],
                    original_connected_path_sha256=protocol['original_connected_path_sha256'],
                    connected_path_sha256=receipt['connected_path_sha256'], scope=joint.POLICY['geometry_scope']))
            plan['source_sha256'].update(sources); plan.setdefault('extra_input_sha256', {}).update(locks)
            plan['extra_input_sha256'][str(output/'HOME_ROUTE_PROTOCOL.json')] = receipt['protocol_sha256']
            plan['segment_timing']['home_to_pregrasp'] = dict(steps=len(q), duration_s=len(q)*.004, **derivatives)
            plan['phases'] = []
            for name in dict.fromkeys(candidate['phase'].tolist()):
                part = home.phase_slice(candidate['phase'], name)
                plan['phases'].append(dict(name=name, start_step=part.start+1, end_step=part.stop))
            transport.save(native/'plan.json', plan)
            receipt.update(status='ready', admitted=True, complete=True, plan_sha256=transport.sha(native/'plan.json'))
        for name, digest in {**locks, **sources}.items():
            if transport.sha(name) != digest: raise ValueError('Frozen dependency changed during retiming: '+name)
    except TimeScalingRejected as error:
        receipt.update(status='null_geometry', complete=True, error=str(error))
    except BaseException as error:
        receipt.update(status='infrastructure_failure', admitted=False, complete=False,
                       error=repr(error), traceback=traceback.format_exc())
    receipt['wall_s'] = time.monotonic()-began
    transport.save(output/'HOME_ROUTE_PREPARATION.json', receipt)
    return receipt
