"""Frozen six-arm RRT-Connect HOME adapter, without simulator or IK.

Initial and pregrasp joint states are immutable. All other active coordinates,
settle and approach rows stay exact. The finite search is not a completeness
claim, and its output is admitted only after the original full-scene tick audit.
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
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import author_home_route as home
import author_transport as transport

POLICY = dict(schema='anydex_boya_author_joint_home_policy_v1', algorithm='RRT-Connect',
    random_seed=20261114, step_l2_rad=.25, edge_max_per_joint_rad=.05,
    goal_bias=.2, max_iterations=1200, max_connect_extends=32,
    tree_order='start then goal, alternate each iteration', nearest='Euclidean unwrapped six-arm radians',
    sample='uniform original six-arm hard bounds; goal bias targets opposite tree root',
    solution='first connection, no shortcuts, no smoothing search or second solution retry',
    timing='Each edge is a separate original two-knot quintic timed_segment, concatenate without duplicate start; original limits and final whole-HOME derivative audit',
    minimum_total_home_s=3., dt_s=.004, acceleration_rad_s2=3., jerk_rad_s3=30.,
    geometry_tolerance_m=1e-7, original_approach_distance_m=.04,
    fixed_initial=True, fixed_pregrasp_joint_endpoint=True, fixed_non_arm_coordinates=True,
    original_approach_rows_exact=True, ik_calls=0, physics_steps=0,
    geometry_scope=home.POLICY['scope'])


class TimeScalingRejected(RuntimeError):
    """Expected original trajectory-limit rejection, without search retry."""


def edge_samples(start, end, max_per_joint):
    """All linear edge samples after start, including the exact end."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite([a, b]).all() or max_per_joint <= 0:
        raise ValueError('Finite matching joint vectors and positive edge spacing required')
    count = max(1, int(np.ceil(np.max(abs(b-a))/max_per_joint)))
    result = a+np.arange(1, count+1)[:, None]/count*(b-a)
    result[-1] = b
    return result


def rrt_connect(start, goal, lower, upper, edge_ok, *, policy=None):
    """Deterministic bounded planner; edge_ok checks both original endpoints."""
    p = POLICY if policy is None else policy
    start, goal, lower, upper = map(lambda x: np.asarray(x, dtype=float), (start, goal, lower, upper))
    if (start.ndim != 1 or not all(a.shape == start.shape for a in (goal, lower, upper))
            or not np.isfinite([start, goal, lower, upper]).all() or np.any(lower >= upper)
            or np.any(start < lower) or np.any(start > upper) or np.any(goal < lower) or np.any(goal > upper)):
        raise ValueError('Original finite bounded joint endpoints required')
    rng = np.random.default_rng(p['random_seed'])
    trees = [dict(q=[start.copy()], parent=[-1], is_start=True),
             dict(q=[goal.copy()], parent=[-1], is_start=False)]
    audit = dict(iterations_executed=0, extend_calls=0, connect_extend_calls=0,
                 collision_rejected_edges=0, tree_nodes=[1, 1], connected=False)
    if np.array_equal(start, goal):
        audit.update(connected=True, waypoints=2)
        return np.vstack([start, goal]), audit, trees

    def extend(tree, target):
        audit['extend_calls'] += 1
        nodes = np.asarray(tree['q']); index = int(np.argmin(np.linalg.norm(nodes-target, axis=1)))
        old = nodes[index]; delta = target-old; distance = float(np.linalg.norm(delta))
        if distance < 1e-12: return index, True
        reached = distance <= p['step_l2_rad']
        new = target.copy() if reached else old+delta*(p['step_l2_rad']/distance)
        if not edge_ok(old, new):
            audit['collision_rejected_edges'] += 1
            return None, False
        tree['q'].append(new); tree['parent'].append(index)
        return len(tree['q'])-1, reached

    def chain(tree, index):
        result = []
        while index >= 0:
            result.append(tree['q'][index]); index = tree['parent'][index]
        return result

    for iteration in range(p['max_iterations']):
        audit['iterations_executed'] = iteration+1
        left, right = trees[iteration % 2], trees[1-iteration % 2]
        target = right['q'][0] if rng.random() < p['goal_bias'] else rng.uniform(lower, upper)
        index, _ = extend(left, target)
        if index is None: continue
        connection = left['q'][index]
        for _ in range(p['max_connect_extends']):
            audit['connect_extend_calls'] += 1
            other, reached = extend(right, connection)
            if other is None: break
            if reached:
                a, b = chain(left, index), chain(right, other)
                path = list(reversed(a))+b[1:] if left['is_start'] else list(reversed(b))+a[1:]
                path = np.asarray(path)
                if not np.array_equal(path[0], start) or not np.array_equal(path[-1], goal):
                    raise AssertionError('RRT parent-chain endpoint corruption')
                audit.update(connected=True, tree_nodes=[len(t['q']) for t in trees], waypoints=len(path))
                return path, audit, trees
    audit['tree_nodes'] = [len(t['q']) for t in trees]
    return None, audit, trees


class SourceGeometryOracle:
    """Exact-value cache; no quantization, nearest-neighbour gate or exclusions."""
    def __init__(self, model, geometry, initial, arm_indexes):
        self.model, self.geometry = model, geometry
        self.initial, self.arm_indexes = np.asarray(initial), arm_indexes
        self.cache = {}
        self.calls = self.states = self.queries = self.cache_hits = 0
        self.minimum_floor_m = float('inf')
        self.first_violations = []

    def full(self, q):
        values = self.initial.copy(); values[self.arm_indexes] = q
        return self.model.full_state(dict(zip(self.model.active_names, map(float, values))))

    def check(self, start, end):
        values = edge_samples(start, end, POLICY['edge_max_per_joint_rad'])
        uncached = []
        for q in values:
            key = q.tobytes()
            if key not in self.cache: uncached.append((key, q))
            else: self.cache_hits += 1
        if uncached:
            report = self.geometry.audit([self.full(q) for _, q in uncached], tolerance_m=POLICY['geometry_tolerance_m'])
            self.calls += 1; self.states += len(uncached); self.queries += report['signed_distance_queries']
            self.minimum_floor_m = min(self.minimum_floor_m, report['min_floor_m'])
            failed = {v['sample'] for v in report['violations']}
            for i, (key, _) in enumerate(uncached): self.cache[key] = i not in failed
            if not self.first_violations and report['violations']: self.first_violations = report['violations'][:20]
        return all(self.cache[q.tobytes()] for q in values)

    def receipt(self):
        return dict(audit_calls=self.calls, unique_checked_states=self.states,
            signed_distance_queries=self.queries, exact_cache_hits=self.cache_hits,
            minimum_floor_m=self.minimum_floor_m if np.isfinite(self.minimum_floor_m) else None,
            first_rejected_edge_violations=self.first_violations)


def prepare_home_route(seed_plan_dir, scene_path, output, *, config_path, archive_sources=None):
    seed_plan_dir, scene_path, output, config_path = (Path(p).resolve()
        for p in (seed_plan_dir, scene_path, output, config_path))
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    receipt = dict(schema='anydex_boya_author_joint_home_preparation_v1', status='preparing',
        admitted=False, complete=False, plan_dir=str(output/'native'), physics_steps=0, ik_calls=0,
        grasp_attempts=0, labels=0, training_updates=0, policy=copy.deepcopy(POLICY))
    transport.save(output/'HOME_ROUTE_PREPARATION.json', receipt)
    try:
        transport._backend_path()
        os.environ.setdefault('CANONDEX_COLLISION_PARTITION_CACHE', str(transport.ROOT/'outputs/boya_collision_partition_cache'))
        from run_template_arm_positive import bind_hand, object_asset
        seed = transport.read(seed_plan_dir/'plan.json'); args = seed['arguments']
        bind_hand(SimpleNamespace(hand_urdf=Path(args['hand_urdf']), hand_scene=Path(args['hand_scene'])))
        from canondex.planning.model import PlanningModel
        from canondex.robots.reference import load_reference
        from canondex.grasp.admm.arm_only_ik import ARM_JOINTS
        from canondex.grasp.template_arm_bridge import ArmPathGeometry, timed_segment
        if not seed.get('complete') or seed.get('approach_only') is not True or args.get('pregrasp_distance_m') != .04:
            raise ValueError('Complete unchanged source-4cm open-hand seed required')
        cfg = transport.read(config_path)
        for key in ('robot_urdf', 'reference_root', 'hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config'):
            if Path(cfg[key]).resolve() != Path(args[key]).resolve(): raise ValueError('Backend differs: '+key)
        scene = transport.validate_scene(scene_path, archive_sources)
        locks = {str(config_path): transport.sha(config_path), str(scene_path): transport.sha(scene_path),
                 str(seed_plan_dir/'plan.json'): transport.sha(seed_plan_dir/'plan.json')}
        for key, name in (('commands_sha256', 'commands.npz'), ('path_geometry_sha256', 'path_geometry.json'),
                          ('lift_arm_geometry_sha256', 'lift_arm_geometry.json')):
            if transport.sha(seed_plan_dir/name) != seed[key]: raise ValueError('Native seed artifact changed: '+name)
            locks[str(seed_plan_dir/name)] = seed[key]
        locks.update(seed['source_sha256']); locks.update(seed.get('extra_input_sha256', {})); locks.update(scene.get('input_sha256', {}))
        bindings = transport.scene_source_bindings(scene, archive_sources)
        for binding in bindings.values(): locks[binding['path']] = binding['sha256']
        pool = Path(args['candidates']); generation = transport.read(pool/'generation.json')
        for name, digest in (('candidates.npz', seed['candidates_sha256']), ('generation.json', seed['generation_sha256'])):
            locks[str(pool/name)] = digest
        locks.update(generation['input_sha256'])
        target = next(r for r in scene['objects'] if r['object_id'] == scene['target_object_id'])
        if (Path(target['object_root']).resolve() != Path(args['object_root']).resolve()
                or target['scale'] != seed['source_scale']
                or not np.allclose(target['world_T_object'], seed['world_T_object'], atol=1e-12, rtol=0)):
            raise ValueError('Frozen compatibility execution asset changed')
        model = PlanningModel(Path(args['robot_urdf'])); reference = load_reference(Path(args['reference_root']))
        with np.load(seed_plan_dir/'commands.npz', allow_pickle=False) as data: old = {k: data[k].copy() for k in data.files}
        if list(map(str, old['joint_names'])) != model.active_names or len(model.active_names) != 24:
            raise ValueError('Original 24 active coordinate order required')
        phase = list(dict.fromkeys(old['phase'].tolist()))
        segment = home.phase_slice(old['phase'], 'home_to_pregrasp'); approach = home.phase_slice(old['phase'], 'approach')
        if (phase != ['settle', 'home_to_pregrasp', 'approach'] or segment.start != 50
                or segment.stop != approach.start or approach.stop != len(old['phase'])
                or approach.stop-approach.start != 500 or old['joint_target'].shape[1] != 24):
            raise ValueError('Original 50 settle / HOME / 500 approach rows required')
        if not np.array_equal(old['initial_position'], model.vector(seed['initial'])):
            raise ValueError('Original HOME state changed')
        arm_indexes = [model.active_names.index(n) for n in ARM_JOINTS]
        non_arm = [i for i in range(24) if i not in arm_indexes]
        if not np.array_equal(old['joint_target'][:, non_arm], np.broadcast_to(old['initial_position'][non_arm], (len(old['phase']), 18))):
            raise ValueError('Original prefix must retain the fixed open hand and two wrist joints')
        state = lambda q: model.full_state(dict(zip(model.active_names, map(float, q))))
        parts, owners, part_locks = home._obstacles(scene, object_asset); locks.update(part_locks)
        geometry = ArmPathGeometry(model, model.ignored_pairs(), parts, np.eye(4), vectorized_broadphase=True)
        source_locks = {str(Path(__file__).resolve()): transport.sha(__file__),
                        str(Path(home.__file__).resolve()): transport.sha(home.__file__),
                        str(Path(transport.__file__).resolve()): transport.sha(transport.__file__)}
        for name, digest in {**locks, **source_locks}.items():
            if transport.sha(name) != digest: raise ValueError('Frozen dependency changed: '+name)
        protocol = dict(schema='anydex_boya_author_joint_home_protocol_v1', policy=POLICY,
            input_sha256=locks, source_sha256=source_locks, capture_source_bindings=bindings,
            obstacle_owners=owners, seed_admitted=seed['admitted'],
            start_arm_rad=old['initial_position'][arm_indexes].tolist(),
            goal_arm_rad=old['joint_target'][segment.stop-1, arm_indexes].tolist(),
            lower_rad=model.lower[arm_indexes].tolist(), upper_rad=model.upper[arm_indexes].tolist(),
            physical_partition_refinements={r['object_id']: r.get('collision_refinement') for r in scene['objects']},
            planning_partition_policy='Original source convex hull union; native partition/refinement and cooking remain separately verified, not substituted for planning geometry')
        transport.save(output/'HOME_ROUTE_PROTOCOL.json', protocol)  # Before RNG or search.
        shutil.copyfile(__file__, output/'AUTHOR_JOINT_HOME_ROUTE_SOURCE.py')
        receipt.update(protocol_sha256=transport.sha(output/'HOME_ROUTE_PROTOCOL.json'),
            input_sha256=locks, source_sha256=source_locks, capture_source_bindings=bindings)
        transport.save(output/'HOME_ROUTE_PREPARATION.json', receipt)

        def annotate(report, indexes=None):
            for row in report['violations']:
                if indexes is not None:
                    row['command_index'] = int(indexes[row['sample']]); row['phase'] = str(old['phase'][row['command_index']])
                if row['kind'] == 'object': row['obstacle_owner'] = owners[row['other']]
            return report

        fixed_indexes = np.r_[np.arange(segment.start), segment.stop-1, np.arange(segment.stop, len(old['phase']))]
        fixed = annotate(geometry.audit([state(q) for q in old['joint_target'][fixed_indexes]], tolerance_m=POLICY['geometry_tolerance_m']), fixed_indexes)
        transport.save(output/'fixed_segments_geometry.json', fixed)
        if not fixed['ok']:
            receipt.update(status='null_geometry', complete=True, reason='frozen_endpoint_or_non_home_collision', search_skipped=True)
        else:
            oracle = SourceGeometryOracle(model, geometry, old['initial_position'], arm_indexes)
            start, goal = old['initial_position'][arm_indexes], old['joint_target'][segment.stop-1, arm_indexes]
            search_began = time.monotonic()
            path, search, trees = rrt_connect(start, goal, model.lower[arm_indexes], model.upper[arm_indexes], oracle.check)
            receipt.update(search=search, oracle=oracle.receipt(), search_wall_s=time.monotonic()-search_began)
            np.savez_compressed(output/'search_trees.npz', start_tree_q=trees[0]['q'], start_tree_parent=trees[0]['parent'],
                                goal_tree_q=trees[1]['q'], goal_tree_parent=trees[1]['parent'])
            receipt['search_trees_sha256'] = transport.sha(output/'search_trees.npz')
            if path is None:
                receipt.update(status='null_geometry', complete=True, reason='frozen_rrt_budget_exhausted')
            else:
                np.savez_compressed(output/'connected_path.npz', arm_waypoints=path)
                receipt['connected_path_sha256'] = transport.sha(output/'connected_path.npz')
                path_parts, timings = [], []
                for a, b in zip(path[:-1], path[1:]):
                    try:
                        values, timing = timed_segment([a, b], dt=POLICY['dt_s'], velocity_limits=model.velocity[arm_indexes],
                            minimum_s=POLICY['minimum_total_home_s']/(len(path)-1),
                            acceleration=POLICY['acceleration_rad_s2'], jerk=POLICY['jerk_rad_s3'],
                            position_lower=model.lower[arm_indexes], position_upper=model.upper[arm_indexes])
                    except RuntimeError as error:
                        raise TimeScalingRejected(str(error)) from error
                    values[-1] = b
                    path_parts.append(values[1:]); timings.append(timing)
                arm_q = np.concatenate(path_parts)
                all_q = np.broadcast_to(old['initial_position'], (len(arm_q), 24)).copy(); all_q[:, arm_indexes] = arm_q
                all_q[-1] = old['joint_target'][segment.stop-1]
                palms = np.asarray([model.fk.transforms(state(q), reference.base_transform)['palm'] for q in all_q])
                candidate = home.replace_home(old, all_q, palms)
                derivative = home.derivative_audit(arm_q, start, model.velocity[arm_indexes], model.lower[arm_indexes], model.upper[arm_indexes])
                transport.save(output/'timing.json', dict(edges=timings, whole_home=derivative))
                final_geometry = annotate(geometry.audit([state(q) for q in candidate['joint_target']], tolerance_m=POLICY['geometry_tolerance_m']))
                transport.save(output/'full_scene_geometry.json', final_geometry)
                np.savez_compressed(output/'candidate_commands.npz', **candidate)
                receipt.update(derivative_audit=derivative, geometry_sha256=transport.sha(output/'full_scene_geometry.json'),
                               candidate_commands_sha256=transport.sha(output/'candidate_commands.npz'))
                if not derivative['ok'] or not final_geometry['ok']:
                    receipt.update(status='null_geometry', complete=True, reason='first_rrt_connection_failed_final_tick_or_derivative_gate')
                else:
                    native = output/'native'; native.mkdir()
                    for source in seed_plan_dir.iterdir():
                        if source.is_file() and source.name not in ('plan.json', 'commands.npz', 'path_geometry.json'):
                            shutil.copyfile(source, native/source.name)
                    shutil.copyfile(output/'candidate_commands.npz', native/'commands.npz')
                    shutil.copyfile(output/'full_scene_geometry.json', native/'path_geometry.json')
                    plan = copy.deepcopy(seed); plan['arguments']['output_dir'] = str(native)
                    plan.update(complete=True, admitted=True, total_command_steps=len(candidate['phase']),
                        commands_sha256=transport.sha(native/'commands.npz'), path_geometry_sha256=transport.sha(native/'path_geometry.json'),
                        planner='Frozen six-arm RRT-Connect HOME; original approach/endpoint preserved',
                        author_home_route=dict(strategy='joint_rrt_connect', protocol_path=str(output/'HOME_ROUTE_PROTOCOL.json'),
                            protocol_sha256=receipt['protocol_sha256'], original_seed_plan_sha256=locks[str(seed_plan_dir/'plan.json')],
                            scope=POLICY['geometry_scope'], connected_path_sha256=receipt['connected_path_sha256']))
                    plan['source_sha256'].update(source_locks); plan.setdefault('extra_input_sha256', {}).update(locks)
                    plan['extra_input_sha256'][str(output/'HOME_ROUTE_PROTOCOL.json')] = receipt['protocol_sha256']
                    plan['segment_timing']['home_to_pregrasp'] = dict(steps=len(arm_q), duration_s=len(arm_q)*POLICY['dt_s'], edges=len(timings), **derivative)
                    plan['phases'] = []
                    for name in dict.fromkeys(candidate['phase'].tolist()):
                        block = home.phase_slice(candidate['phase'], name)
                        plan['phases'].append(dict(name=name, start_step=block.start+1, end_step=block.stop))
                    transport.save(native/'plan.json', plan)
                    receipt.update(status='ready', admitted=True, complete=True, plan_sha256=transport.sha(native/'plan.json'),
                                   command_steps=len(candidate['phase']), home_command_steps=len(arm_q))
        for name, digest in {**locks, **source_locks}.items():
            if transport.sha(name) != digest: raise ValueError('Frozen dependency changed during planning: '+name)
    except TimeScalingRejected as error:
        receipt.update(status='null_geometry', complete=True, reason='original_time_scaling_rejected', error=str(error))
    except BaseException as error:
        receipt.update(status='infrastructure_failure', complete=False, admitted=False,
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
    result = prepare_home_route(args.seed_plan_dir, args.scene, args.output, config_path=args.config, archive_sources=archives)
    print(__import__('json').dumps(result, allow_nan=False))
    raise SystemExit(1 if result['status'] == 'infrastructure_failure' else 0)


if __name__ == '__main__': main()
