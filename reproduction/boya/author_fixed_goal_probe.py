"""One frozen HOME-seeded IK diagnostic for an unchanged fixed release pose.

This produces no trajectory or physical grasp. A clear endpoint is only an
input for a separately frozen carry planner; local failure is not unreachability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import author_home_route as home
import author_transport as transport

POLICY = dict(schema='anydex_boya_fixed_goal_single_home_seed_v1',
    seed='Original reference HOME six arm joints; other 18 active values from last transport_bin row before release',
    target='Exact explicit fixed release SE3; no candidate/body-relative offset or target adjustment',
    ik_calls=1, max_nfev=300, bounds_margin_rad=1e-6,
    additional_seeds=0, retries=0, geometry_tolerance_m=1e-7,
    geometry='Original ArmPathGeometry, same rigid links/self/floor and all three initial source-body proxies plus five bin boxes',
    triangle_check='Deepest rejected original convex pair, original source collision triangles; triangle surface intersection only, not volume penetration',
    Kit_launches=0, physics_steps=0, training_updates=0, labels=0)


def _binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=transport.sha(path))


def _verify(values):
    for path, digest in values.items():
        if transport.sha(path) != digest:
            raise ValueError('Frozen fixed-goal dependency changed: '+path)


def closed_boundary(arrays):
    phases = np.asarray(arrays['phase'])
    indexes = np.flatnonzero(phases == 'transport_bin')
    if not len(indexes):
        raise ValueError('A completed transport_bin boundary is required')
    index = int(indexes[-1])
    if index+1 >= len(phases) or phases[index+1] != 'release':
        raise ValueError('Closed transport boundary must immediately precede release')
    values = np.asarray(arrays['joint_target'])
    if values.ndim != 2 or values.shape[1] != 24 or not np.isfinite(values[index]).all():
        raise ValueError('Finite original 24-column closed boundary required')
    return index, values[index].copy()


def home_seed(model, reference, boundary, names, arm_names):
    if list(names) != model.active_names or len(names) != 24:
        raise ValueError('Original 24 independent coordinate order required')
    values = dict(zip(names, map(float, boundary)))
    if set(reference.arm_home) != set(arm_names):
        raise ValueError('Exactly the six original reference HOME arm coordinates required')
    values.update(reference.arm_home)
    seed = model.full_state(values)
    non_arm = [name for name in names if name not in arm_names]
    if len(non_arm) != 18 or any(seed[name] != boundary[names.index(name)] for name in non_arm):
        raise AssertionError('HOME seed changed a frozen wrist or finger coordinate')
    return seed


def triangle_intersection(left, right):
    """Coal BVH on unconvexified triangles, without a new mesh approximation."""
    import coal
    def shape(mesh):
        vertices = coal.StdVec_Vec3s()
        for vertex in mesh.vertices:
            vertices.append(np.asarray(vertex, dtype=float))
        triangles = coal.StdVec_Triangle()
        for face in mesh.faces:
            triangles.append(coal.Triangle(*map(int, face)))
        result = coal.BVHModelOBBRSS()
        result.beginModel(len(triangles), len(vertices))
        result.addSubModel(vertices, triangles)
        result.endModel()
        return result
    request = coal.CollisionRequest()
    request.enable_contact = True
    request.num_max_contacts = 1
    result = coal.CollisionResult()
    count = coal.collide(shape(left), coal.Transform3s(), shape(right), coal.Transform3s(), request, result)
    contacts = result.getContacts()
    return dict(surface_intersection=bool(result.isCollision()), collision_count=int(count),
        contact_count=len(contacts), left_vertices=len(left.vertices), left_triangles=len(left.faces),
        right_vertices=len(right.vertices), right_triangles=len(right.faces),
        scope='Original source triangle surface intersection; no signed solid penetration depth or containment claim')


def _imports(args):
    transport._backend_path()
    from run_template_arm_positive import bind_hand, object_asset
    bind_hand(SimpleNamespace(hand_urdf=Path(args['hand_urdf']), hand_scene=Path(args['hand_scene'])))
    from canondex.planning.model import PlanningModel
    from canondex.robots.reference import load_reference
    from canondex.grasp.admm import arm_only_ik
    from canondex.grasp.template_arm_bridge import ArmPathGeometry, palm_error
    return PlanningModel, load_reference, arm_only_ik, ArmPathGeometry, palm_error, object_asset


def freeze(component_dir, release_spec_path, output):
    component_dir, release_spec_path, output = map(lambda p: Path(p).resolve(),
        (component_dir, release_spec_path, output))
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    result = dict(status='freezing', complete=False, policy=POLICY, ik_calls=0, physics_steps=0)
    transport.save(output/'FREEZE_RESULT.json', result)
    try:
        source_protocol_path = component_dir/'COMPONENT_INTEGRATION_PROTOCOL.json'
        source_result_path = component_dir/'COMPONENT_INTEGRATION_RESULTS.json'
        source_protocol, source_result = map(transport.read, (source_protocol_path, source_result_path))
        if (not source_result.get('complete') or source_result.get('protocol_sha256') != transport.sha(source_protocol_path)
                or source_result.get('physics_steps') != 0):
            raise ValueError('Complete unchanged zero-physics component preparation required')
        plan_dir = component_dir/'transport_raw'
        plan = transport.read(plan_dir/'plan.json')
        if (not plan.get('complete') or plan.get('approach_only') is not False
                or source_result['transport_preparation']['plan_sha256'] != transport.sha(plan_dir/'plan.json')):
            raise ValueError('Complete original transport diagnostic plan required')
        locks = {str(path): transport.sha(path) for path in (source_protocol_path, source_result_path, plan_dir/'plan.json')}
        for group in ('inputs', 'sources', 'capture_source_bindings'):
            for binding in source_protocol[group].values():
                locks[binding['path']] = binding['sha256']
        for key, name in (('commands_sha256', 'commands.npz'), ('path_geometry_sha256', 'path_geometry.json'),
                          ('lift_arm_geometry_sha256', 'lift_arm_geometry.json')):
            locks[str(plan_dir/name)] = plan[key]
        locks.update(plan['source_sha256']); locks.update(plan.get('extra_input_sha256', {}))
        scene_path = Path(source_protocol['inputs']['scene']['path'])
        archive = source_protocol.get('scene_archive_sources', {})
        scene = transport.validate_scene(scene_path, archive)
        release = transport.load_release_spec(scene_path, release_spec_path, scene=scene)
        original_release = plan['author_transport']['release_spec']
        if (transport.sha(release_spec_path) != original_release['sha256']
                or not np.array_equal(release['world_T_release_palm'], original_release['world_T_release_palm'])):
            raise ValueError('Probe must retain the exact prior fixed release sidecar and SE3')
        locks[str(release_spec_path)] = transport.sha(release_spec_path)
        locks.update(release['source_sha256'])
        model_type, reference_loader, ik, geometry_type, _, object_asset = _imports(plan['arguments'])
        model = model_type(Path(plan['arguments']['robot_urdf']))
        reference = reference_loader(Path(plan['arguments']['reference_root']))
        with np.load(plan_dir/'commands.npz', allow_pickle=False) as data:
            arrays = {key:data[key].copy() for key in data.files}
        names = list(map(str, arrays['joint_names']))
        index, boundary = closed_boundary(arrays)
        seed = home_seed(model, reference, boundary, names, ik.ARM_JOINTS)
        parts, owners, part_locks = home._obstacles(scene, object_asset)
        locks.update(part_locks)
        rigid = [name for name in model.collision_links if not name.startswith(('th','ff','mf','rf','lf'))]
        sources = {str(Path(path).resolve()): transport.sha(path) for path in
            (__file__, home.__file__, transport.__file__, sys.modules[model_type.__module__].__file__,
             sys.modules[reference_loader.__module__].__file__, ik.__file__, sys.modules[geometry_type.__module__].__file__)}
        _verify(locks); _verify(sources)
        protocol = dict(schema=POLICY['schema'], policy=POLICY, input_sha256=locks, source_sha256=sources,
            source_component=str(component_dir), source_plan=_binding(plan_dir/'plan.json'),
            scene=_binding(scene_path), scene_archive_sources=archive, fixed_release=_binding(release_spec_path),
            target_world_T_palm=release['world_T_release_palm'], closed_boundary_command_index=index,
            boundary_phase='last transport_bin immediately before release', active_names=names,
            closed_boundary_active=boundary.tolist(), initial_raw28=seed,
            reference_HOME_arm={name:seed[name] for name in ik.ARM_JOINTS},
            fixed_non_arm_names=[name for name in names if name not in ik.ARM_JOINTS],
            rigid_collision_links=rigid, obstacle_owners=owners, obstacle_count=len(parts),
            solve_observer='Transparent one-call original least_squares wrapper captures final fit for residuals on local failure; no changed arguments or additional solver calls')
        with (output/'PROTOCOL.json').open('x') as stream:
            json.dump(protocol, stream, indent=2, allow_nan=False); stream.write('\n')
        shutil.copyfile(__file__, output/'SOURCE.py')
        result.update(status='frozen', complete=True, protocol_sha256=transport.sha(output/'PROTOCOL.json'),
            source_sha256=sources, closed_boundary_command_index=index, obstacle_count=len(parts), wall_s=time.monotonic()-began)
        transport.save(output/'FREEZE_RESULT.json', result)
        return result
    except BaseException as error:
        result.update(status='infrastructure_failure', error=repr(error), traceback=traceback.format_exc(), wall_s=time.monotonic()-began)
        transport.save(output/'FREEZE_RESULT.json', result)
        raise


def run(output):
    output = Path(output).resolve()
    freeze_result = transport.read(output/'FREEZE_RESULT.json')
    if freeze_result.get('status') != 'frozen' or freeze_result['protocol_sha256'] != transport.sha(output/'PROTOCOL.json'):
        raise ValueError('An unchanged frozen protocol is required')
    protocol = transport.read(output/'PROTOCOL.json')
    _verify(protocol['input_sha256']); _verify(protocol['source_sha256'])
    with (output/'RUN_STARTED.json').open('x') as stream:
        json.dump(dict(protocol_sha256=transport.sha(output/'PROTOCOL.json'), started_unix_s=time.time(), ik_budget=1), stream)
        stream.write('\n')
    began = time.monotonic()
    result = dict(schema=POLICY['schema'], status='starting', complete=False, ik_calls=0,
        physics_steps=0, Kit_launches=0, labels=0, training_updates=0, protocol_sha256=transport.sha(output/'PROTOCOL.json'))
    transport.save(output/'RESULT.json', result)
    try:
        plan = transport.read(protocol['source_plan']['path'])
        model_type, reference_loader, ik, geometry_type, palm_error, object_asset = _imports(plan['arguments'])
        model = model_type(Path(plan['arguments']['robot_urdf']))
        reference = reference_loader(Path(plan['arguments']['reference_root']))
        scene = transport.validate_scene(protocol['scene']['path'], protocol['scene_archive_sources'])
        release = transport.load_release_spec(protocol['scene']['path'], protocol['fixed_release']['path'], scene=scene)
        target = np.asarray(release['world_T_release_palm'])
        if not np.array_equal(target, protocol['target_world_T_palm']):
            raise ValueError('Frozen fixed release changed')
        parts, owners, _ = home._obstacles(scene, object_asset)
        if owners != protocol['obstacle_owners']:
            raise ValueError('Frozen full-scene obstacle order changed')
        geometry = geometry_type(model, model.ignored_pairs(), parts, np.eye(4),
            collision_links=protocol['rigid_collision_links'], vectorized_broadphase=True)

        def audit(state, name):
            report = geometry.audit([state], tolerance_m=POLICY['geometry_tolerance_m'])
            for violation in report['violations']:
                if violation['kind'] == 'object': violation['obstacle_owner'] = owners[violation['other']]
            deepest = min(report['violations'], key=lambda v:v['distance_m']) if report['violations'] else None
            triangles = None
            if deepest and deepest['kind'] in ('self','object'):
                frames = model.fk.transforms(state)
                left = model.collision_mesh(deepest['link']).copy(); left.apply_transform(frames[deepest['link']])
                if deepest['kind'] == 'self':
                    right = model.collision_mesh(deepest['other']).copy(); right.apply_transform(frames[deepest['other']])
                    right_scope = 'Original URDF source collision triangles'
                else:
                    right = parts[deepest['other']]
                    right_scope = 'Frozen bin box triangles or original source object CoACD-part triangles; not exact object CAD'
                triangles = dict(pair=deepest, right_scope=right_scope, **triangle_intersection(left,right))
            transport.save(output/(name+'_geometry.json'), report)
            return dict(ok=report['ok'], geometry_sha256=transport.sha(output/(name+'_geometry.json')),
                violation_count=len(report['violations']), deepest_violation=deepest, deepest_source_triangle_check=triangles)

        boundary = model.full_state(dict(zip(protocol['active_names'], protocol['closed_boundary_active'])))
        result['old_straight_endpoint'] = audit(boundary, 'old_straight_endpoint')
        initial = protocol['initial_raw28']
        observed = {}
        original_solver = ik.least_squares
        def observer(*args, **kwargs):
            observed['calls'] = observed.get('calls',0)+1
            fit = original_solver(*args, **kwargs)
            observed.update(x=fit.x.tolist(), nfev=int(fit.nfev), status=int(fit.status), success=bool(fit.success), message=str(fit.message))
            return fit
        result['ik_calls'] = 1
        transport.save(output/'RESULT.json', result)
        try:
            with patch.object(ik, 'least_squares', observer):
                solved, ik_audit = ik.arm_only_palm_ik(model, reference, initial, target,
                    max_nfev=POLICY['max_nfev'], bounds_margin_rad=POLICY['bounds_margin_rad'])
            converged = True
            result['original_ik_audit'] = ik_audit
        except RuntimeError as error:
            converged = False
            result['original_ik_error'] = str(error)
            if 'x' not in observed:
                raise
            solved = dict(initial, **dict(zip(ik.ARM_JOINTS, observed['x'])))
        if observed.get('calls') != 1:
            raise AssertionError('Single original IK solve budget violated')
        non_arm = [name for name in model.joints if name not in ik.ARM_JOINTS]
        if any(solved[name] != initial[name] for name in non_arm):
            raise AssertionError('Original IK changed a frozen wrist/finger/mimic value')
        actual = model.fk.transforms(solved, reference.base_transform)['palm']
        position_error, rotation_error = palm_error(actual, target)
        result.update(local_ik_converged=converged, solver_observation=observed,
            residual=dict(position_error_m=position_error, rotation_error_rad=rotation_error, weighted_rotation_error=.15*rotation_error),
            actual_world_T_palm=actual.tolist(), target_world_T_palm=target.tolist(),
            all22_raw_non_arm_values_exact=True, all18_active_non_arm_values_exact=True)
        result['HOME_seed_goal_endpoint'] = audit(solved, 'HOME_seed_goal_endpoint')
        clear = bool(converged and result['HOME_seed_goal_endpoint']['ok'])
        result.update(status='clear_goal_endpoint' if clear else ('null_geometry' if converged else 'null_local_ik'), complete=True,
            solved_raw28=solved, solved_active24=model.vector(solved).tolist(),
            carry_route_ready=False, meaning='Single fixed endpoint diagnostic only; no path, dynamic clearance, physical success or unreachability claim')
        if clear:
            transport.save(output/'FIXED_GOAL_ARM_ENDPOINT.json', dict(schema=POLICY['schema'],
                protocol_sha256=result['protocol_sha256'], arm_joint_names=list(ik.ARM_JOINTS),
                arm_positions_rad=[solved[name] for name in ik.ARM_JOINTS],
                fixed_non_arm_names=protocol['fixed_non_arm_names'],
                fixed_non_arm_values=[solved[name] for name in protocol['fixed_non_arm_names']],
                target_world_T_palm=target.tolist(), endpoint_geometry=result['HOME_seed_goal_endpoint'],
                trajectory_generated=False, execution_allowed=False))
        _verify(protocol['input_sha256']); _verify(protocol['source_sha256'])
        result['wall_s'] = time.monotonic()-began
        transport.save(output/'RESULT.json', result)
        return result
    except BaseException as error:
        result.update(status='infrastructure_failure', error=repr(error), traceback=traceback.format_exc(), wall_s=time.monotonic()-began)
        transport.save(output/'RESULT.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze','run'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--component-dir', type=Path)
    parser.add_argument('--release-spec', type=Path)
    args = parser.parse_args()
    if args.stage == 'freeze':
        if args.component_dir is None or args.release_spec is None:
            parser.error('freeze requires explicit component-dir and release-spec')
        value = freeze(args.component_dir,args.release_spec,args.output_dir)
    else:
        value = run(args.output_dir)
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
