"""Append the original stationary closure and world-Z lift to a fixed prefix.

No object selection, search retry, simulator, label or new holding duration is
introduced. The bin-free multi-object task is explicit and source-bound.
"""
from __future__ import annotations

import argparse
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
import author_transport as common

TASK = 'declared_boya_clutter_lift_v1'
SCENE_SCHEMA = 'anydex_boya_author_lift_scene_v1'
POLICY = dict(schema='anydex_boya_author_lift_plan_policy_v1', task=TASK,
    dt_s=.004, approach_distance_m=.04, close_steps=250, hold_steps=50,
    lift_axis_world=[0., 0., 1.], lift_distance_m=.1,
    lift_waypoints=32, per_waypoint_max_nfev=300, arm_bounds_margin_rad=1e-6,
    lift_minimum_s=2., acceleration_rad_s2=3., jerk_rad_s3=30.,
    geometry_tolerance_m=1e-7, ik_position_tolerance_m=1e-4,
    weighted_orientation_tolerance_m=1e-4, orientation_weight_m=.15,
    fixed_candidate=True, original_prefix_exact=True, additional_lift_hold_steps=0,
    alternate_ik_seeds=0, planning_retries=0, physics_steps=0, labels=0,
    open_geometry='All source objects at their frozen poses; full robot self/floor',
    post_geometry='Rigid arm/wrists/palm self/floor and every source object at its initial pose; moving bodies and all finger contacts remain runtime gates')
CORE_ARRAYS = {'initial_position', 'joint_names', 'time_s', 'joint_target', 'phase', 'desired_palm'}


class LocalPlanningRejected(RuntimeError):
    """Finite original IK/interpolation/timing budget did not yield a plan."""


def _merge(locks, values):
    for name, digest in values.items():
        name = str(Path(name).resolve())
        if name in locks and locks[name] != digest:
            raise ValueError('Conflicting frozen lift dependency: '+name)
        locks[name] = digest


def _verify(locks):
    for name, digest in locks.items():
        if common.sha(name) != digest: raise ValueError('Frozen lift dependency changed: '+name)


def validate_prefix(model, family, seed, old):
    """Guard the exact native 24-coordinate open-hand prefix, without IK."""
    if (not seed.get('complete') or not seed.get('admitted') or seed.get('approach_only') is not True
            or seed.get('author_transport') or seed['arguments'].get('pregrasp_distance_m') != .04):
        raise ValueError('Admitted unchanged source-4cm approach-only plan required')
    if seed.get('handoff_clearance_m', 0.) != 0. or seed.get('solver_close_palm') is not None:
        raise ValueError('Original stationary-palm zero-clearance closure required')
    if set(old) != CORE_ARRAYS or list(map(str, old['joint_names'])) != model.active_names:
        raise ValueError('Exactly the six original native arrays and coordinate order required')
    names = model.active_names
    if (len(names) != 24 or len(model.joints) != 28 or len(family.hand.names) != 16
            or set(names) != set(family.hand.names)|set(('FAJ1', 'FAJ3'))|set(_arm_names())):
        raise ValueError('Original six-arm, two-wrist, sixteen-motor independent state required')
    n = len(old['phase'])
    if (old['joint_target'].shape != (n, 24) or old['desired_palm'].shape != (n, 4, 4)
            or old['initial_position'].shape != (24,) or old['time_s'].shape != (n,)
            or not all(np.isfinite(old[k]).all() for k in ('joint_target', 'desired_palm', 'initial_position', 'time_s'))):
        raise ValueError('Finite aligned native command arrays required')
    if list(dict.fromkeys(old['phase'].tolist())) != ['settle', 'home_to_pregrasp', 'approach']:
        raise ValueError('Transport/closure/lift or duplicate phase blocks cannot enter this adapter')
    settle, route, approach = [home.phase_slice(old['phase'], p) for p in ('settle', 'home_to_pregrasp', 'approach')]
    if (settle.start != 0 or settle.stop != 50 or route.start != 50 or route.stop != approach.start
            or approach.stop != n or approach.stop-approach.start != 500):
        raise ValueError('Original 50 settle / HOME / 500 approach rows required')
    if not np.array_equal(old['time_s'], np.arange(1, n+1)*POLICY['dt_s']):
        raise ValueError('Original native timestep changed')
    if not np.array_equal(old['initial_position'], model.vector(seed['initial'])):
        raise ValueError('Original initial joint state changed')
    hand = [names.index(name) for name in family.hand.names]
    fixed = [i for i, name in enumerate(names) if name not in _arm_names()]
    if (list(family.hand.names) != seed['hand_motor_names']
            or not np.array_equal(old['joint_target'][:, fixed], np.broadcast_to(old['initial_position'][fixed], (n, 18)))
            or not np.array_equal(old['joint_target'][-1, hand], seed['open_motors'])):
        raise ValueError('The original open hand and two wrist commands must remain exact')
    # Native approach writes (palm - distance*axis) + distance*axis. Retain
    # those exact source bytes while allowing its arithmetic roundoff here.
    if not np.allclose(old['desired_palm'][-1], common.transform(seed['palm']), atol=1e-12, rtol=0):
        raise ValueError('Original approach endpoint palm changed')
    return dict(steps=n, independent_coordinates=24, actual_coordinates=28,
                hand_dof=20, independent_hand_motors=16, approach_steps=500, settle_steps=50)


def _arm_names():
    from canondex.grasp.admm.arm_only_ik import ARM_JOINTS
    return ARM_JOINTS


def append_source_closure(model, family, seed, commands, phases, desired):
    """The original 250-step close and 50-step pre-lift hold, unchanged."""
    from canondex.grasp.template_arm_bridge import with_hand_command
    from canondex.grasp.template_gripper_closure import TwoSidedClosure
    closure = TwoSidedClosure.from_template(family, seed['open_motors'], seed['closed_motors'])
    last = model.full_state(dict(zip(model.active_names, map(float, commands[-1]))))
    palm = common.transform(seed['palm'])
    for phase, count in (('close', POLICY['close_steps']), ('hold', POLICY['hold_steps'])):
        for k in range(count):
            progress = (k+1)/count if phase == 'close' else 1.
            state = with_hand_command(model, last, family.hand.names, closure.command(progress, progress))
            commands.append(model.vector(state)); phases.append(phase); desired.append(palm.copy())
        last = state
    return last, closure.audit()


def append_source_lift(model, reference, seed, last, commands, phases, desired, progress):
    """Use native 32 sequential IK waypoints and its original interpolator."""
    from canondex.grasp.admm.arm_only_ik import arm_only_palm_ik
    from canondex.grasp.template_arm_bridge import timed_segment, smooth_fraction, palm_error
    arm_names = _arm_names(); arm = [model.active_names.index(n) for n in arm_names]
    palm = common.transform(seed['palm']); waypoints = [last]
    for index, u in enumerate(np.linspace(0, 1, POLICY['lift_waypoints']+1)[1:], 1):
        target = palm.copy(); target[2, 3] += float(u)*POLICY['lift_distance_m']
        entry = dict(phase='lift', index=index, fraction=float(u), completed=False)
        progress.append(entry)  # Preserve attempted-call count even on failure.
        try:
            solved, audit = arm_only_palm_ik(model, reference, waypoints[-1], target,
                max_nfev=POLICY['per_waypoint_max_nfev'], bounds_margin_rad=POLICY['arm_bounds_margin_rad'])
        except RuntimeError as error:
            entry['error'] = str(error)
            raise LocalPlanningRejected('lift_waypoint_ik_failed:'+str(index)+':'+str(error)) from error
        entry.update(audit, completed=True); waypoints.append(solved)
    try:
        trajectory, timing = timed_segment([[q[n] for n in arm_names] for q in waypoints],
            dt=POLICY['dt_s'], velocity_limits=model.velocity[arm], minimum_s=POLICY['lift_minimum_s'],
            acceleration=POLICY['acceleration_rad_s2'], jerk=POLICY['jerk_rad_s3'],
            position_lower=model.lower[arm], position_upper=model.upper[arm])
    except RuntimeError as error:
        raise LocalPlanningRejected('original_lift_time_scaling_rejected:'+str(error)) from error
    errors = []
    for k, values in enumerate(trajectory[1:], 1):
        state = model.full_state(dict(last, **dict(zip(arm_names, map(float, values)))))
        wanted = palm.copy(); wanted[2, 3] += float(smooth_fraction(k/(len(trajectory)-1)))*POLICY['lift_distance_m']
        errors.append(palm_error(model.fk.transforms(state)['palm'], wanted))
        commands.append(model.vector(state)); phases.append('lift'); desired.append(wanted)
    error = np.max(errors, axis=0)
    if error[0] > POLICY['ik_position_tolerance_m'] or error[1]*POLICY['orientation_weight_m'] > POLICY['weighted_orientation_tolerance_m']:
        raise LocalPlanningRejected('Interpolated lift exceeds original IK acceptance: '
            +repr(dict(position_m=float(error[0]), orientation_rad=float(error[1]), timing=timing)))
    return timing, dict(position_m=float(error[0]), orientation_rad=float(error[1]))


def prepare_lift(seed_plan_dir, scene_path, output_dir, *, archive_sources=None):
    """Prepare one fixed candidate; return ready/null_geometry/infra, 0 physics."""
    seed_plan_dir, scene_path, output_dir = (Path(p).resolve() for p in (seed_plan_dir, scene_path, output_dir))
    output_dir.mkdir(parents=True, exist_ok=False); began = time.monotonic()
    native = output_dir/'native'
    receipt = dict(schema='anydex_boya_author_lift_preparation_v1', task=TASK,
        status='preparing', complete=False, admitted=False, plan_dir=str(native),
        policy=copy.deepcopy(POLICY), ik_waypoints=[], physics_steps=0, grasp_attempts=0,
        labels=0, training_updates=0, source_prefix_steps=None)
    path = output_dir/'LIFT_PREPARATION.json'; common.save(path, receipt)
    try:
        common._backend_path()
        from run_template_arm_positive import bind_hand, object_asset
        from canondex.planning.model import PlanningModel
        from canondex.robots.reference import load_reference
        from canondex.grasp.template_gripper_adapter import TemplateFamily
        from canondex.grasp.template_arm_bridge import ArmPathGeometry
        scene = common.validate_scene(scene_path, archive_sources)
        if scene.get('schema') != SCENE_SCHEMA or scene.get('task') != TASK or scene.get('bin', 'missing') is not None:
            raise ValueError('Explicit bin-free lift scene/task required')
        seed = common.read(seed_plan_dir/'plan.json'); args = seed['arguments']
        bind_hand(SimpleNamespace(hand_urdf=Path(args['hand_urdf']), hand_scene=Path(args['hand_scene'])))
        model = PlanningModel(Path(args['robot_urdf'])); reference = load_reference(Path(args['reference_root']))
        family = TemplateFamily.load(Path(args['package']), Path(args['hand_urdf']), Path(args['hand_scene']), seed['gesture_id'])
        with np.load(seed_plan_dir/'commands.npz', allow_pickle=False) as data: old = {k:data[k].copy() for k in data.files}
        receipt['prefix_validation'] = validate_prefix(model, family, seed, old)
        locks = {str(scene_path):common.sha(scene_path), str(seed_plan_dir/'plan.json'):common.sha(seed_plan_dir/'plan.json')}
        for key, name in (('commands_sha256', 'commands.npz'), ('path_geometry_sha256', 'path_geometry.json'),
                          ('lift_arm_geometry_sha256', 'lift_arm_geometry.json')):
            _merge(locks, {str(seed_plan_dir/name):seed[key]})
        for mapping in (seed['source_sha256'], seed.get('extra_input_sha256', {}), scene.get('input_sha256', {})):
            _merge(locks, mapping)
        for key in ('robot_urdf', 'physics_config', 'diagnostic_config'):
            _merge(locks, {args[key]:seed[key+'_sha256']})
        pool = Path(args['candidates']).resolve(); generation = common.read(pool/'generation.json')
        _merge(locks, {str(pool/'candidates.npz'):seed['candidates_sha256'], str(pool/'generation.json'):seed['generation_sha256']})
        _merge(locks, generation['input_sha256'])
        if not generation.get('complete') or seed['index'] not in generation['selected_indices']:
            raise ValueError('The frozen candidate must remain selected in its original pool')
        with np.load(pool/'candidates.npz', allow_pickle=False) as data:
            if (not np.array_equal(data['proposal_q'][seed['index']], seed['open_motors'])
                    or not np.array_equal(data['proposal_q_close'][seed['index']], seed['closed_motors'])
                    or list(map(str, data['independent_motor_names'])) != list(family.hand.names)):
                raise ValueError('Original open/close endpoints differ from frozen candidate')
        primary = next(row for row in scene['objects'] if row['object_id'] == scene['target_object_id'])
        if (Path(primary['object_root']).resolve() != Path(args['object_root']).resolve()
                or primary['scale'] != seed['source_scale']
                or not np.array_equal(np.asarray(primary['world_T_object']), np.asarray(seed['world_T_object']))
                or Path(scene['physics_config']).resolve() != Path(args['physics_config']).resolve()):
            raise ValueError('Compatibility execution asset/pose/physics differs from bin-free scene')
        repair = primary.get('collision_refinement'); seed_repair = args.get('collision_refinement')
        if bool(repair) != bool(seed_repair) or (repair and Path(repair).resolve() != Path(seed_repair).resolve()):
            raise ValueError('Compatibility source native refinement changed')
        bindings = common.scene_source_bindings(scene, archive_sources)
        for binding in bindings.values(): _merge(locks, {binding['path']:binding['sha256']})
        parts, owners, part_locks = home._obstacles(scene, object_asset); _merge(locks, part_locks)
        arm_repair = seed_plan_dir.parent/'arm_collision_refinements.json'
        if arm_repair.exists(): _merge(locks, {str(arm_repair):common.sha(arm_repair)})
        sources = {str(Path(__file__).resolve()):common.sha(__file__),
                   str(Path(home.__file__).resolve()):common.sha(home.__file__),
                   str(Path(common.__file__).resolve()):common.sha(common.__file__)}
        for name in ('canondex/grasp/template_arm_bridge.py', 'canondex/grasp/admm/arm_only_ik.py',
                     'canondex/grasp/template_gripper_closure.py'):
            source = common.CANON/name; sources[str(source)] = common.sha(source)
        _verify(locks); _verify(sources)
        protocol_path = output_dir/'LIFT_PROTOCOL.json'
        protocol = dict(schema='anydex_boya_author_lift_plan_protocol_v1', task=TASK, policy=POLICY,
            input_sha256=locks, source_sha256=sources, obstacle_owners=owners,
            capture_source_bindings=bindings, original_seed_plan_sha256=common.sha(seed_plan_dir/'plan.json'),
            scene_path=str(scene_path), scene_sha256=common.sha(scene_path),
            scene_archive_sources={str(Path(k).resolve()):str(Path(v).resolve()) for k,v in (archive_sources or {}).items()},
            planning_geometry='Every source CoACD convex hull at its observed pose; native refinement/cooking is separately runtime-audited')
        common.save(protocol_path, protocol)  # Before geometry admission or IK.
        shutil.copyfile(__file__, output_dir/'AUTHOR_LIFT_PLAN_SOURCE.py')
        receipt.update(protocol_path=str(protocol_path), protocol_sha256=common.sha(protocol_path),
            source_prefix_steps=len(old['phase']), source_sha256=sources, input_sha256=locks)
        common.save(path, receipt)
        state = lambda q:model.full_state(dict(zip(model.active_names, map(float, q))))
        geometry = ArmPathGeometry(model, model.ignored_pairs(), parts, np.eye(4), vectorized_broadphase=True)
        open_audit = geometry.audit([state(q) for q in old['joint_target']], tolerance_m=POLICY['geometry_tolerance_m'])
        for row in open_audit['violations']:
            if row['kind'] == 'object': row['obstacle_owner'] = owners[row['other']]
        common.save(output_dir/'path_geometry.json', open_audit)
        receipt['path_geometry_sha256'] = common.sha(output_dir/'path_geometry.json')
        if not open_audit['ok']:
            receipt.update(status='null_geometry', complete=True, reason='frozen_open_prefix_full_scene_collision', ik_skipped=True)
        else:
            commands = list(old['joint_target'].copy()); phases = old['phase'].tolist(); desired = list(old['desired_palm'].copy())
            last, closure = append_source_closure(model, family, seed, commands, phases, desired)
            timing, interpolation = append_source_lift(model, reference, seed, last, commands, phases, desired, receipt['ik_waypoints'])
            arrays = dict(initial_position=old['initial_position'].copy(), joint_names=old['joint_names'].copy(),
                joint_target=np.asarray(commands, dtype=old['joint_target'].dtype),
                phase=np.asarray(phases, dtype=old['phase'].dtype),
                desired_palm=np.asarray(desired, dtype=old['desired_palm'].dtype),
                time_s=np.asarray(np.arange(1, len(commands)+1)*POLICY['dt_s'], dtype=old['time_s'].dtype))
            prefix_exact = all(arrays[k][:len(old['phase'])].dtype == old[k].dtype
                and arrays[k][:len(old['phase'])].tobytes() == old[k].tobytes()
                for k in ('joint_target', 'phase', 'desired_palm', 'time_s'))
            if not prefix_exact: raise ValueError('Original open-hand prefix changed')
            arm = [model.active_names.index(n) for n in _arm_names()]
            derivatives = home.derivative_audit(arrays['joint_target'][:,arm], arrays['initial_position'][arm],
                model.velocity[arm], model.lower[arm], model.upper[arm])
            rigid = [name for name in model.collision_links if not name.startswith(('th', 'ff', 'mf', 'rf', 'lf'))]
            post_audit = ArmPathGeometry(model, model.ignored_pairs(), parts, np.eye(4), collision_links=rigid,
                vectorized_broadphase=True).audit([state(q) for q in arrays['joint_target'][len(old['phase']):]],
                                                tolerance_m=POLICY['geometry_tolerance_m'])
            for row in post_audit['violations']:
                if row['kind'] == 'object': row['obstacle_owner'] = owners[row['other']]
            post_audit.update(scope=POLICY['post_geometry'], initial_pose_proxy_object_ids=[r['object_id'] for r in scene['objects']],
                primary_body_excluded=False, dynamic_object_trajectories_predicted=False)
            common.save(output_dir/'lift_arm_geometry.json', post_audit)
            common.save(output_dir/'timing.json', dict(lift=timing, full_arm_trajectory=derivatives))
            np.savez_compressed(output_dir/'candidate_commands.npz', **arrays)
            receipt.update(closure=closure, lift_timing=timing, interpolation_error=interpolation,
                full_arm_derivative_audit=derivatives, prefix_exact=prefix_exact,
                command_steps=len(phases), independent_coordinates=24, actual_joint_coordinates=28,
                geometry_sha256=common.sha(output_dir/'path_geometry.json'),
                lift_arm_geometry_sha256=common.sha(output_dir/'lift_arm_geometry.json'),
                candidate_commands_sha256=common.sha(output_dir/'candidate_commands.npz'))
            if not post_audit['ok'] or not derivatives['ok']:
                receipt.update(status='null_geometry', complete=True, reason='post_close_rigid_geometry_or_full_arm_derivative_gate')
            else:
                native.mkdir()
                for name in ('frozen_input.json',):
                    if (seed_plan_dir/name).exists(): shutil.copyfile(seed_plan_dir/name, native/name)
                for name in ('path_geometry.json', 'lift_arm_geometry.json'):
                    shutil.copyfile(output_dir/name, native/name)
                shutil.copyfile(output_dir/'candidate_commands.npz', native/'commands.npz')
                if arm_repair.exists(): shutil.copyfile(arm_repair, output_dir/'arm_collision_refinements.json')
                plan = copy.deepcopy(seed); plan['arguments']['output_dir'] = str(native); plan['arguments']['approach_only'] = False
                plan.update(complete=True, admitted=True, approach_only=False, dt_s=POLICY['dt_s'], task=TASK,
                    closure=closure, total_command_steps=len(phases), commands_sha256=common.sha(native/'commands.npz'),
                    path_geometry_sha256=common.sha(native/'path_geometry.json'),
                    lift_arm_geometry_sha256=common.sha(native/'lift_arm_geometry.json'),
                    author_lift=dict(scene_path=str(scene_path), scene_sha256=protocol['scene_sha256'],
                        scene_archive_sources=protocol['scene_archive_sources'], capture_source_bindings=bindings,
                        task=TASK, policy=POLICY, protocol_path=str(protocol_path), protocol_sha256=receipt['protocol_sha256'],
                        original_seed_plan_sha256=protocol['original_seed_plan_sha256'], prefix_commands_exact=True))
                plan['source_sha256'].update(sources); plan.setdefault('extra_input_sha256', {}).update(locks)
                plan['extra_input_sha256'][str(protocol_path)] = receipt['protocol_sha256']
                plan['segment_timing'].update(close=dict(steps=250,duration_s=1.), hold=dict(steps=50,duration_s=.2), lift=timing)
                plan['ik_waypoints'] = list(seed.get('ik_waypoints', []))+receipt['ik_waypoints']
                plan.setdefault('interpolation_errors', {})['lift'] = interpolation
                plan['phases'] = []
                for name in dict.fromkeys(phases):
                    block = home.phase_slice(arrays['phase'], name)
                    plan['phases'].append(dict(name=name, start_step=block.start+1, end_step=block.stop))
                common.save(native/'plan.json', plan)
                receipt.update(status='ready', complete=True, admitted=True,
                    commands_sha256=plan['commands_sha256'], plan_sha256=common.sha(native/'plan.json'),
                    path_geometry_sha256=plan['path_geometry_sha256'])
        _verify(locks); _verify(sources)
    except LocalPlanningRejected as error:
        receipt.update(status='null_geometry', complete=True, reason='original_lift_local_planning_failed', error=str(error))
    except BaseException as error:
        receipt.update(status='infrastructure_failure', complete=False, admitted=False,
            error=repr(error), traceback=traceback.format_exc())
    receipt.update(ik_calls=len(receipt['ik_waypoints']),
        successful_ik_calls=sum(bool(r['completed']) for r in receipt['ik_waypoints']),
        successful_nfev=sum(r.get('nfev', 0) for r in receipt['ik_waypoints']), wall_s=time.monotonic()-began)
    common.save(path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-plan-dir', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-dependency', action='append', default=[])
    args = parser.parse_args()
    receipt = prepare_lift(args.seed_plan_dir, args.scene, args.output,
        archive_sources=dict(v.split('=', 1) for v in args.frozen_dependency))
    print(__import__('json').dumps(receipt, allow_nan=False))
    raise SystemExit(1 if receipt['status']=='infrastructure_failure' else 0)


if __name__ == '__main__': main()
