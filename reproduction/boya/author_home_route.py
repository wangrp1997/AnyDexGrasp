"""Bounded, opt-in HOME detours; keep the frozen grasp/approach unchanged.

Planning imports no simulator. Six common routes are attempted in fixed order;
every emitted command uses the original source-hull/self/floor geometry gate.
This is sampled joint-path admission, not a complete motion planner.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import author_transport as transport

POLICY = dict(schema='anydex_boya_author_home_route_policy_v1', dt_s=.004,
              clearances_m=[.15, .30], world_y_offsets_m=[0., -.15, .15],
              route_order='clearance outer, world_y_offset inner; first admitted',
              max_routes=6, ik_waypoints_per_route=2, max_ik_calls=12,
              per_ik_max_nfev=300, bounds_margin_rad=1e-6,
              minimum_home_s=3., acceleration_rad_s2=3., jerk_rad_s3=30.,
              geometry_tolerance_m=1e-7, preflight_fixed_segments_first=True,
              fixed_candidate=True, fixed_initial=True, fixed_pregrasp_joints=True,
              unchanged_non_home_commands=True, approach_distance_m=.04,
              scope='Source CoACD convex hulls for every scene body, five bin boxes, original robot source convex hulls/self/floor; every emitted tick, sampled not continuous')


def phase_slice(phases, name):
    values = np.asarray(phases)
    indexes = np.flatnonzero(values == name)
    if not len(indexes) or not np.array_equal(indexes, np.arange(indexes[0], indexes[-1]+1)):
        raise ValueError('Exactly one nonempty contiguous phase required: '+name)
    return slice(int(indexes[0]), int(indexes[-1])+1)


def replace_home(arrays, commands, palms):
    """Replace only HOME rows; reject changed hand or pregrasp endpoint."""
    old = {k: np.asarray(v) for k, v in arrays.items()}
    if list(dict.fromkeys(map(str, old['phase']))) != ['settle', 'home_to_pregrasp', 'approach']:
        raise ValueError('Original open-hand three-phase prefix required')
    home = phase_slice(old['phase'], 'home_to_pregrasp')
    q, p = np.asarray(commands), np.array(palms, copy=True)
    if (q.ndim != 2 or q.shape[1] != 24 or not len(q)
            or p.shape != (len(q), 4, 4) or not np.isfinite(q).all()
            or not np.isfinite(p).all() or old['joint_target'].shape[1] != 24):
        raise ValueError('Finite original 24-coordinate commands and palm poses required')
    if not np.array_equal(q[-1], old['joint_target'][home.stop-1]):
        raise ValueError('Frozen pregrasp joint endpoint changed')
    names = list(map(str, old['joint_names']))
    arm = {'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'}
    hand = [i for i, n in enumerate(names) if n not in arm]
    expected = old['initial_position'][hand]
    if not np.array_equal(q[:, hand], np.broadcast_to(expected, (len(q), len(hand)))):
        raise ValueError('Frozen wrist/open-hand command changed during HOME')
    # Preserve the exact old desired endpoint, including its original IK
    # residual convention; the following approach remains byte-identical.
    p[-1] = old['desired_palm'][home.stop-1]
    result = dict(initial_position=old['initial_position'].copy(), joint_names=old['joint_names'].copy())
    result['joint_target'] = np.concatenate([old['joint_target'][:home.start], q, old['joint_target'][home.stop:]])
    result['desired_palm'] = np.concatenate([old['desired_palm'][:home.start], p, old['desired_palm'][home.stop:]])
    result['phase'] = np.concatenate([old['phase'][:home.start], np.full(len(q), 'home_to_pregrasp', dtype='<U16'), old['phase'][home.stop:]])
    result['time_s'] = np.arange(1, len(result['phase'])+1)*POLICY['dt_s']
    return result


def derivative_audit(q, initial, velocity, lower, upper):
    """Recompute the source timed_segment's finite-difference limits."""
    values = np.vstack([initial, q])
    if not np.isfinite(values).all():
        raise ValueError('Finite trajectory required')
    excess = float(np.maximum(lower-values, values-upper).max())
    v = np.gradient(values, POLICY['dt_s'], axis=0, edge_order=2)
    a = np.gradient(v, POLICY['dt_s'], axis=0, edge_order=2)
    j = np.gradient(a, POLICY['dt_s'], axis=0, edge_order=2)
    ratios = [float(np.max(abs(v)/velocity)), float(abs(a).max()/POLICY['acceleration_rad_s2']),
              float(abs(j).max()/POLICY['jerk_rad_s3'])]
    return dict(ok=bool(excess <= 1e-7 and max(ratios) <= 1.+1e-8),
                joint_limit_excess_rad=max(0., excess), velocity_ratio=ratios[0],
                acceleration_ratio=ratios[1], jerk_ratio=ratios[2],
                max_velocity_rad_s=float(abs(v).max()), max_acceleration_rad_s2=float(abs(a).max()),
                max_jerk_rad_s3=float(abs(j).max()))


def _obstacles(scene, object_asset):
    import trimesh
    parts, owners, bindings = [], [], {}
    for row in scene['objects']:
        asset = object_asset(SimpleNamespace(object_root=Path(row['object_root'])),
                             np.asarray(row['world_T_object']), row['scale'])
        urdf = Path(row['object_root'])/'urdf/coacd.urdf'
        bindings[str(urdf)] = transport.sha(urdf)
        for record in asset.collision_provenance:
            bindings[record['path']] = record['sha256']
        for i, source in enumerate(asset.collision_parts):
            mesh = source.copy(); mesh.apply_transform(row['world_T_object'])
            parts.append(mesh); owners.append(dict(object_id=row['object_id'], source_part=i))
    for box in transport.bin_boxes(scene['bin']):
        mesh = trimesh.creation.box(extents=box['extents_m']); mesh.apply_transform(box['world_T_box'])
        parts.append(mesh); owners.append(dict(bin_part=box['name']))
    return parts, owners, bindings


def prepare_home_route(seed_plan_dir, scene_path, output, *, config_path, archive_sources=None):
    """Return ready/null_geometry/infrastructure_failure, with new native plan.

    A complete native admitted=False result is accepted only as an immutable
    seed. No rejected seed is executed. Final readiness requires the full-scene
    geometry and unchanged original non-HOME rows, including all 500 approach.
    """
    seed_plan_dir, scene_path, output, config_path = (Path(p).resolve()
        for p in (seed_plan_dir, scene_path, output, config_path))
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    receipt = dict(schema='anydex_boya_author_home_route_preparation_v1', status='preparing',
        admitted=False, complete=False, plan_dir=str(output/'native'), physics_steps=0,
        grasp_attempts=0, training_updates=0, ik_calls=0, attempts=[], policy=copy.deepcopy(POLICY))
    transport.save(output/'HOME_ROUTE_PREPARATION.json', receipt)
    try:
        transport._backend_path()
        os.environ.setdefault('CANONDEX_COLLISION_PARTITION_CACHE', str(transport.ROOT/'outputs/boya_collision_partition_cache'))
        from run_template_arm_positive import bind_hand, object_asset
        seed = transport.read(seed_plan_dir/'plan.json')
        args = seed['arguments']
        bind_hand(SimpleNamespace(hand_urdf=Path(args['hand_urdf']), hand_scene=Path(args['hand_scene'])))
        from canondex.planning.model import PlanningModel
        from canondex.robots.reference import load_reference
        from canondex.grasp.admm.arm_only_ik import ARM_JOINTS, arm_only_palm_ik
        from canondex.grasp.template_arm_bridge import ArmPathGeometry, timed_segment

        if not seed.get('complete') or seed.get('approach_only') is not True or args.get('pregrasp_distance_m') != .04:
            raise ValueError('Complete original source-4cm approach-only seed required')
        cfg = transport.read(config_path)
        for key in ('robot_urdf', 'reference_root', 'hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config'):
            if Path(cfg[key]).resolve() != Path(args[key]).resolve():
                raise ValueError('Backend configuration differs from frozen seed: '+key)
        scene = transport.validate_scene(scene_path, archive_sources)
        locks = {str(config_path): transport.sha(config_path), str(scene_path): transport.sha(scene_path),
                 str(seed_plan_dir/'plan.json'): transport.sha(seed_plan_dir/'plan.json')}
        for key, name in (('commands_sha256', 'commands.npz'), ('path_geometry_sha256', 'path_geometry.json'),
                          ('lift_arm_geometry_sha256', 'lift_arm_geometry.json')):
            if transport.sha(seed_plan_dir/name) != seed[key]:
                raise ValueError('Native seed artifact changed: '+name)
            locks[str(seed_plan_dir/name)] = seed[key]
        locks.update(seed['source_sha256']); locks.update(seed.get('extra_input_sha256', {}))
        locks.update(scene.get('input_sha256', {}))
        for name, digest in scene['source_sha256'].items():
            resolved = (archive_sources or {}).get(name, name)
            locks[str(Path(resolved).resolve())] = digest
        pool = Path(args['candidates'])
        for name, digest in (('candidates.npz', seed['candidates_sha256']), ('generation.json', seed['generation_sha256'])):
            locks[str(pool/name)] = digest
        generation = transport.read(pool/'generation.json')
        locks.update(generation['input_sha256'])
        model = PlanningModel(Path(args['robot_urdf']))
        reference = load_reference(Path(args['reference_root']))
        with np.load(seed_plan_dir/'commands.npz', allow_pickle=False) as z:
            old = {k: z[k].copy() for k in z.files}
        if list(map(str, old['joint_names'])) != model.active_names or len(model.active_names) != 24:
            raise ValueError('Original active 24-coordinate command order changed')
        home = phase_slice(old['phase'], 'home_to_pregrasp')
        approach = phase_slice(old['phase'], 'approach')
        if home.start != 50 or approach.start != home.stop or approach.stop != len(old['phase']) or approach.stop-approach.start != 500:
            raise ValueError('Original 50 settle / HOME / 500 approach prefix required')
        if not np.array_equal(old['initial_position'], model.vector(seed['initial'])):
            raise ValueError('Frozen HOME initial joint state changed')
        state = lambda q: model.full_state(dict(zip(model.active_names, map(float, q))))
        initial = state(old['initial_position']); endpoint = state(old['joint_target'][home.stop-1])
        target = next(r for r in scene['objects'] if r['object_id'] == scene['target_object_id'])
        if (Path(target['object_root']).resolve() != Path(args['object_root']).resolve()
                or target['scale'] != seed['source_scale']
                or not np.allclose(target['world_T_object'], seed['world_T_object'], atol=1e-12, rtol=0)):
            raise ValueError('Original compatibility asset/observed pose changed')
        parts, owners, part_locks = _obstacles(scene, object_asset)
        locks.update(part_locks)
        # Reuse exactly the source hull-union gate. Native contact partition
        # refinement is NOT substituted for source planning hulls.
        geometry = ArmPathGeometry(model, model.ignored_pairs(), parts, np.eye(4), vectorized_broadphase=True)
        source_locks = {str(Path(__file__).resolve()): transport.sha(__file__),
                        str(Path(transport.__file__).resolve()): transport.sha(transport.__file__)}
        for name, digest in {**locks, **source_locks}.items():
            if transport.sha(name) != digest:
                raise ValueError('Frozen preparation dependency changed: '+name)
        protocol = dict(schema='anydex_boya_author_home_route_protocol_v1', policy=POLICY,
            seed_admitted=seed['admitted'], input_sha256=locks, source_sha256=source_locks,
            capture_source_bindings=transport.scene_source_bindings(scene, archive_sources),
            obstacle_owners=owners, geometry_scope=POLICY['scope'],
            physical_partition_refinements={r['object_id']: r.get('collision_refinement') for r in scene['objects']},
            planning_partition_policy='Original object_asset source convex hull union; runtime split/refinement children stay separately native-audited, no 20um exact-CAD claim')
        transport.save(output/'HOME_ROUTE_PROTOCOL.json', protocol)  # Freeze before any route IK.
        shutil.copyfile(__file__, output/'AUTHOR_HOME_ROUTE_SOURCE.py')
        receipt.update(protocol_sha256=transport.sha(output/'HOME_ROUTE_PROTOCOL.json'),
                       input_sha256=locks, source_sha256=source_locks,
                       capture_source_bindings=protocol['capture_source_bindings'])
        transport.save(output/'HOME_ROUTE_PREPARATION.json', receipt)

        def audit(values, indexes=None):
            report = geometry.audit([state(q) for q in values], tolerance_m=POLICY['geometry_tolerance_m'])
            for violation in report['violations']:
                if indexes is not None:
                    violation['command_index'] = int(indexes[violation['sample']])
                    violation['phase'] = str(old['phase'][violation['command_index']])
                if violation['kind'] == 'object':
                    violation['obstacle_owner'] = owners[violation['other']]
            return report

        fixed_indexes = np.r_[np.arange(home.start), home.stop-1, np.arange(home.stop, len(old['phase']))]
        fixed = audit(old['joint_target'][fixed_indexes], fixed_indexes)
        transport.save(output/'fixed_segments_geometry.json', fixed)
        if not fixed['ok']:
            receipt.update(status='null_geometry', complete=True, reason='unchanged_endpoint_or_non_home_segment_collision',
                           search_skipped=True, fixed_geometry_sha256=transport.sha(output/'fixed_segments_geometry.json'))
        else:
            baseline = audit(old['joint_target'], np.arange(len(old['phase'])))
            transport.save(output/'baseline_full_scene_geometry.json', baseline)
            selected = old if baseline['ok'] else None
            receipt['baseline_admitted'] = baseline['ok']
            start_palm = model.fk.transforms(initial, reference.base_transform)['palm']
            end_palm = model.fk.transforms(endpoint, reference.base_transform)['palm']
            arm_indexes = [model.active_names.index(n) for n in ARM_JOINTS]
            chosen = None
            for clearance in POLICY['clearances_m']:
                if selected is not None: break
                for offset in POLICY['world_y_offsets_m']:
                    attempt = dict(route_index=len(receipt['attempts']), clearance_m=clearance,
                                   world_y_offset_m=offset, ik_calls=0, ik_audits=[], admitted=False)
                    receipt['attempts'].append(attempt)
                    route_dir = output/f"route_{attempt['route_index']:02d}"
                    route_dir.mkdir()
                    route_began = time.monotonic()
                    up, over = start_palm.copy(), end_palm.copy()
                    up[2, 3] = over[2, 3] = max(start_palm[2, 3], end_palm[2, 3])+clearance
                    up[1, 3] += offset; over[1, 3] += offset
                    attempt['cartesian_waypoints'] = [up.tolist(), over.tolist()]
                    try:
                        knots, previous = [initial], initial
                        for goal in (up, over):
                            attempt['ik_calls'] += 1; receipt['ik_calls'] += 1
                            solved, report = arm_only_palm_ik(model, reference, previous, goal,
                                max_nfev=POLICY['per_ik_max_nfev'], bounds_margin_rad=POLICY['bounds_margin_rad'])
                            knots.append(solved); previous = solved; attempt['ik_audits'].append(report)
                        knots.append(endpoint)  # Exact frozen joint endpoint, never re-solve.
                        waypoints = np.array([[s[n] for n in ARM_JOINTS] for s in knots])
                        attempt['arm_waypoints_rad'] = waypoints.tolist()
                        path, timing = timed_segment(waypoints, dt=POLICY['dt_s'],
                            velocity_limits=model.velocity[arm_indexes], minimum_s=POLICY['minimum_home_s'],
                            acceleration=POLICY['acceleration_rad_s2'], jerk=POLICY['jerk_rad_s3'],
                            position_lower=model.lower[arm_indexes], position_upper=model.upper[arm_indexes])
                        new_q = []
                        for values in path[1:]:
                            raw = dict(initial, **dict(zip(ARM_JOINTS, map(float, values))))
                            new_q.append(model.vector(model.full_state(raw)))
                        new_q = np.asarray(new_q)
                        new_q[-1] = old['joint_target'][home.stop-1]  # Cubic endpoint rounding only.
                        palms = np.asarray([model.fk.transforms(state(q), reference.base_transform)['palm'] for q in new_q])
                        candidate = replace_home(old, new_q, palms)
                        limit_audit = derivative_audit(new_q[:, arm_indexes], old['initial_position'][arm_indexes],
                            model.velocity[arm_indexes], model.lower[arm_indexes], model.upper[arm_indexes])
                        attempt.update(timing=timing, derivative_audit=limit_audit)
                        if not limit_audit['ok']:
                            raise RuntimeError('Original finite-difference trajectory limits failed')
                        report = audit(candidate['joint_target'])
                        transport.save(route_dir/'full_scene_geometry.json', report)
                        np.savez_compressed(route_dir/'commands.npz', **candidate)
                        attempt.update(admitted=report['ok'], geometry_sha256=transport.sha(route_dir/'full_scene_geometry.json'),
                                       commands_sha256=transport.sha(route_dir/'commands.npz'))
                        if report['ok']:
                            selected, chosen = candidate, attempt['route_index']
                    except RuntimeError as error:
                        attempt['error'] = str(error)
                    finally:
                        attempt['wall_s'] = time.monotonic()-route_began
                        transport.save(route_dir/'ATTEMPT.json', attempt)
                        transport.save(output/'HOME_ROUTE_PREPARATION.json', receipt)
                    if selected is not None: break
            if selected is None:
                receipt.update(status='null_geometry', complete=True, reason='frozen_home_route_budget_exhausted', search_skipped=False)
            else:
                native = output/'native'; native.mkdir()
                for source in seed_plan_dir.iterdir():
                    if source.is_file() and source.name not in ('plan.json', 'commands.npz', 'path_geometry.json'):
                        shutil.copyfile(source, native/source.name)
                final_geometry = baseline if chosen is None else transport.read(output/f'route_{chosen:02d}/full_scene_geometry.json')
                transport.save(native/'path_geometry.json', final_geometry)
                np.savez_compressed(native/'commands.npz', **selected)
                plan = copy.deepcopy(seed)
                plan['arguments']['output_dir'] = str(native)
                plan.update(admitted=True, complete=True, total_command_steps=len(selected['phase']),
                    commands_sha256=transport.sha(native/'commands.npz'),
                    path_geometry_sha256=transport.sha(native/'path_geometry.json'),
                    author_home_route=dict(protocol_path=str(output/'HOME_ROUTE_PROTOCOL.json'),
                        protocol_sha256=receipt['protocol_sha256'], selected_route_index=chosen,
                        scope=POLICY['scope'], original_seed_plan_sha256=locks[str(seed_plan_dir/'plan.json')]))
                plan['source_sha256'].update(source_locks)
                plan.setdefault('extra_input_sha256', {}).update(locks)
                plan['extra_input_sha256'][str(output/'HOME_ROUTE_PROTOCOL.json')] = receipt['protocol_sha256']
                plan['phases'] = []
                for name in dict.fromkeys(selected['phase'].tolist()):
                    block = phase_slice(selected['phase'], name)
                    plan['phases'].append(dict(name=name, start_step=block.start+1, end_step=block.stop))
                if chosen is not None:
                    plan['segment_timing']['home_to_pregrasp'] = receipt['attempts'][chosen]['timing']
                transport.save(native/'plan.json', plan)
                receipt.update(status='ready', complete=True, admitted=True, selected_route_index=chosen,
                               plan_sha256=transport.sha(native/'plan.json'), command_steps=len(selected['phase']))
        for name, digest in {**locks, **source_locks}.items():
            if transport.sha(name) != digest:
                raise ValueError('Frozen dependency changed during planning: '+name)
    except BaseException as error:
        receipt.update(status='infrastructure_failure', admitted=False, complete=False,
                       error=repr(error), traceback=traceback.format_exc())
    receipt['wall_s'] = time.monotonic()-began
    transport.save(output/'HOME_ROUTE_PREPARATION.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-plan-dir', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--frozen-dependency', action='append', default=[])
    args = parser.parse_args()
    archives = dict(item.split('=', 1) for item in args.frozen_dependency)
    result = prepare_home_route(args.seed_plan_dir, args.scene, args.output,
                                config_path=args.config, archive_sources=archives)
    print(__import__('json').dumps(result, allow_nan=False))
    raise SystemExit(1 if result['status'] == 'infrastructure_failure' else 0)


if __name__ == '__main__':
    main()
