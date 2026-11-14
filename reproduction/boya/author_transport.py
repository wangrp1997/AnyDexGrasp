"""Opt-in continuous multi-body Boya transport through the native arm executor.

No Kit import occurs at module import or preparation.  The original executor
retains its structure, actuator, FK, material, contact and seven validity gates.
Its old lift-task success is deliberately separate from this module's transport
label.  Source visual meshes are used only for physical assets and measurement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANON = Path('/home/wangrenpeng/CanonDex')
PINS = {
    CANON/'scripts/research/execute_template_arm_positive.py': 'a290fffb110b79235b9dc60bfbc0d2d640b71d6b847e404e6de5a9b540a3fe2e',
    CANON/'scripts/research/run_template_arm_positive.py': 'ac25304f49ab1314a05612f610d5b1b006a59a8b4b019816c25b23bc568fb7e3',
    CANON/'canondex/scene/grasp_object.py': '422324ce9a3047bb98d8e394ee47ceb8b51b5fa8c570676b5aca7b951b82a61e',
    ROOT/'ur_toolbox/ur_toolbox/robot/ur.py': '72125be1412d9039a8a789e9188e2529eacc61ed78c1b595f27b154c7f2a01de',
}
POLICY = dict(dt_s=.004, retreat_first_m=.004, retreat_wait_s=.1,
              retreat_minimum_s=.5, retreat_pregrasp_minimum_s=2.,
              transport_minimum_s=3., release_s=1., release_settle_s=2.,
              final_window_s=.5, source_rise_m=.05, containment_tolerance_m=.002,
              max_final_linear_speed_m_s=.02, max_final_angular_speed_rad_s=.5,
              max_penetration_m=.001, early_stop_penetration_m=.005,
              max_robot_bin_force_n=.001, contact_capacity=262144)


class PhysicalStop(RuntimeError):
    """Observed physical gate failure, distinct from broken infrastructure."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_pins():
    for path, digest in PINS.items():
        if sha(path) != digest:
            raise ValueError('Read-only execution source changed: '+str(path))


def _backend_path():
    verify_pins()
    for path in (ROOT, Path(__file__).resolve().parent, CANON, CANON/'scripts/research'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def transform(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(value[:3, :3].T@value[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-8, rtol=0)):
        raise ValueError('Finite proper homogeneous transform required')
    return value


def scene_source_bindings(scene, archive_sources=None):
    """Resolve historical capture code by exact SHA, without executing it.

    Archives apply only to capture source provenance. Current planner/runtime
    sources are independently bound in the new preparation protocol.
    """
    archives = {str(Path(k).resolve()): Path(v).resolve() for k, v in (archive_sources or {}).items()}
    result = {}
    for name, digest in scene.get('source_sha256', {}).items():
        source = Path(name).resolve()
        resolved = archives.get(str(source), source)
        if sha(resolved) != digest:
            raise ValueError('Scene source or explicit historical archive changed: '+name)
        result[name] = dict(path=str(resolved), sha256=digest, historical_archive=resolved != source)
    if set(archives)-set(result):
        raise ValueError('Archive mapping must name a frozen scene source')
    return result


def validate_scene(path, archive_sources=None):
    """Validate frozen physical inputs; IDs carry no candidate segmentation."""
    scene = read(path)
    if scene.get('schema') != 'anydex_boya_author_scene_v1':
        raise ValueError('Explicit author-scene schema required')
    rows = scene['objects']
    if len(rows) < 2 or len({r['object_id'] for r in rows}) != len(rows):
        raise ValueError('At least two uniquely named dynamic source objects required')
    if scene['target_object_id'] not in {r['object_id'] for r in rows}:
        raise ValueError('Legacy compatibility body must exist')
    for row in rows:
        if not row['object_id'].replace('_', '').isalnum():
            raise ValueError('Safe USD object identifier required')
        transform(row['world_T_object'])
        if row['mass_kg'] != .1 or row['friction'] != .6 or row['scale'] <= 0:
            raise ValueError('Current native mass/friction and positive source scale required')
        source = Path(row['source_mesh'])
        if not source.is_absolute() or sha(source) != row['source_mesh_sha256']:
            raise ValueError('Source visual mesh binding changed')
        if source.resolve() != (Path(row['object_root'])/'mesh/simplified.obj').resolve():
            raise ValueError('Physical and measured source meshes must agree')
        repair = row.get('collision_refinement')
        if repair and sha(repair) != row['collision_refinement_sha256']:
            raise ValueError('Native collision repair binding changed')
    bin_spec = scene['bin']
    transform(bin_spec['world_T_bin'])
    half = np.asarray(bin_spec['inner_half_extents_m'], dtype=float)
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError('Finite positive bin interior required')
    if not all(np.isfinite(bin_spec[k]) and 0 < bin_spec[k] <= .05
               for k in ('floor_thickness_m', 'wall_thickness_m')):
        raise ValueError('Explicit positive bin wall/floor thickness required')
    if bin_spec['friction'] != .6:
        raise ValueError('Frozen bin friction required')
    if 'release_palm_pose' in bin_spec:
        transform(bin_spec['release_palm_pose'])
    for path, digest in scene.get('input_sha256', {}).items():
        if sha(path) != digest:
            raise ValueError('Scene dependency changed: '+path)
    scene_source_bindings(scene, archive_sources)
    return scene


def bin_boxes(bin_spec):
    """Five boxes, each as world transform and full XYZ extents.

    The bin frame is the interior centre, and its interior is +/- half extents.
    No lid or invisible collision wall is introduced above the physical rim.
    """
    frame = transform(bin_spec['world_T_bin'])
    x, y, z = map(float, bin_spec['inner_half_extents_m'])
    floor, wall = bin_spec['floor_thickness_m'], bin_spec['wall_thickness_m']
    definitions = [('floor', [0, 0, -z-floor/2], [2*x+2*wall, 2*y+2*wall, floor]),
                   ('wall_x_minus', [-x-wall/2, 0, 0], [wall, 2*y+2*wall, 2*z]),
                   ('wall_x_plus', [x+wall/2, 0, 0], [wall, 2*y+2*wall, 2*z]),
                   ('wall_y_minus', [0, -y-wall/2, 0], [2*x, wall, 2*z]),
                   ('wall_y_plus', [0, y+wall/2, 0], [2*x, wall, 2*z])]
    result = []
    for name, centre, extents in definitions:
        local = np.eye(4)
        local[:3, 3] = centre
        result.append(dict(name=name, world_T_box=(frame@local).tolist(), extents_m=extents))
    return result


def source_metrics(vertices, pose_xyzw, bin_spec):
    from scipy.spatial.transform import Rotation
    pose = np.asarray(pose_xyzw, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or pose.shape != (7,) or not np.isfinite(vertices).all()
            or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-4):
        raise ValueError('Finite source vertices and native XYZW pose required')
    world = vertices@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
    frame = transform(bin_spec['world_T_bin'])
    local = (world-frame[:3, 3])@frame[:3, :3]
    half = np.asarray(bin_spec['inner_half_extents_m'])
    return dict(source_bottom_m=float(world[:, 2].min()),
                source_top_m=float(world[:, 2].max()),
                bin_min_m=local.min(0).tolist(), bin_max_m=local.max(0).tolist(),
                inside_bin=bool(np.all(local >= -half-POLICY['containment_tolerance_m'])
                                and np.all(local <= half+POLICY['containment_tolerance_m'])))


def object_robot_loads(path, contacts, robot_bodies, legal_hand_owners):
    """Apply the original no-arm-support predicate to one arbitrary body."""
    relevant = [c for c in contacts if c['bodies'][0] == path and c['bodies'][1] in robot_bodies]
    fingers = [sum(abs(c['normal_force_n']) for c in relevant
                   if c['bodies'][1].rsplit('/', 1)[-1].startswith(finger))
               for finger in ('th', 'ff', 'mf', 'rf', 'lf')]
    illegal = max([0.]+[abs(c['normal_force_n']) for c in relevant
                       if c['bodies'][1].rsplit('/', 1)[-1] not in legal_hand_owners])
    return dict(robot_load_n=float(sum(abs(c['normal_force_n']) for c in relevant)),
                finger_loads_n=fingers, illegal_robot_support_force_n=float(illegal))


def preclosure_pushes(states, settled_xy, phase):
    """Original conservative guard applied equally to every source body."""
    if phase not in ('home_to_pregrasp', 'approach'):
        return []
    return [key for key, xy in settled_xy.items()
            if np.linalg.norm(np.asarray(states[key]['pose_xyzw'][:2])-xy) > .03]


def _waypoint_targets(palm, direction, approach_distance, bin_spec, source_pose):
    palm = transform(palm)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isclose(np.linalg.norm(direction), 1., atol=1e-8):
        raise ValueError('Original unit approach direction required')
    first, pre = palm.copy(), palm.copy()
    first[:3, 3] -= POLICY['retreat_first_m']*direction
    pre[:3, 3] -= approach_distance*direction
    if 'release_palm_pose' in bin_spec:
        release = transform(bin_spec['release_palm_pose']).copy()
    else:
        frame = transform(bin_spec['world_T_bin'])
        if not np.allclose(frame[2, :3], [0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError('Default release adaptation supports upright bin only')
        release = palm.copy()
        release[:2, 3] += frame[:2, 3]-np.asarray(source_pose)[:2, 3]
        release[2, 3] = max(palm[2, 3], frame[2, 3]+bin_spec['inner_half_extents_m'][2]+.15)
    return [('retreat_4mm', first, POLICY['retreat_minimum_s']),
            ('retreat_pregrasp', pre, POLICY['retreat_pregrasp_minimum_s']),
            ('transport_bin', release, POLICY['transport_minimum_s'])]


def _append_source_closure(model, family, seed, commands, phases, desired):
    """Replay the native stationary-palm 250-close/50-hold loop exactly."""
    from canondex.grasp.template_arm_bridge import with_hand_command
    from canondex.grasp.template_gripper_closure import TwoSidedClosure

    if seed.get('handoff_clearance_m', 0.) != 0. or seed.get('solver_close_palm') is not None:
        raise ValueError('This action adapter requires the original zero-clearance stationary-palm closure')
    if list(family.hand.names) != seed['hand_motor_names']:
        raise ValueError('Original independent hand motor order changed')
    closure = TwoSidedClosure.from_template(family, seed['open_motors'], seed['closed_motors'])
    last = model.full_state(dict(zip(model.active_names, map(float, commands[-1]))))
    if not np.array_equal([last[n] for n in family.hand.names], seed['open_motors']):
        raise ValueError('Approach must finish at the unchanged original open command')
    palm = transform(seed['palm'])
    for phase, count in (('close', 250), ('hold', 50)):
        for k in range(count):
            progress = (k+1)/count if phase == 'close' else 1.
            state = with_hand_command(model, last, family.hand.names, closure.command(progress, progress))
            commands.append(model.vector(state)); phases.append(phase); desired.append(palm.copy())
        last = state
    return closure.audit()


def prepare_transport(seed_plan_dir, scene_path, output_dir, *, archive_sources=None):
    """Extend one admitted approach/closure plan; local failure becomes null.

    The bin and release frame are inputs.  No retry, target repositioning,
    object-conditioned grasp selection or physical label is produced here.
    """
    _backend_path()
    seed_plan_dir, scene_path, output_dir = map(Path, (seed_plan_dir, scene_path, output_dir))
    seed = read(seed_plan_dir/'plan.json')
    from run_template_arm_positive import object_asset, bind_hand
    bind_hand(SimpleNamespace(hand_urdf=Path(seed['arguments']['hand_urdf']),
                              hand_scene=Path(seed['arguments']['hand_scene'])))
    from canondex.planning.model import PlanningModel
    from canondex.robots.reference import load_reference
    from canondex.grasp.template_gripper_adapter import TemplateFamily
    from canondex.grasp.admm.arm_only_ik import ARM_JOINTS, arm_only_palm_ik
    from canondex.grasp.template_arm_bridge import (ArmPathGeometry, interpolate_palm,
        timed_segment, smooth_fraction, palm_error, with_hand_command)
    import trimesh

    began = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output_dir/'author_transport_source_snapshot.py')
    scene = validate_scene(scene_path, archive_sources)
    receipt = dict(schema='anydex_boya_author_transport_preparation_v1', complete=False,
                   admitted=False, status='preparing', grasp_attempts=0, physics_steps=0,
                   scene_path=str(scene_path.resolve()), scene_sha256=sha(scene_path),
                   seed_plan_path=str((seed_plan_dir/'plan.json').resolve()),
                   seed_plan_sha256=sha(seed_plan_dir/'plan.json'), policy=POLICY,
                   scene_archive_sources={str(Path(k).resolve()): str(Path(v).resolve())
                                          for k, v in (archive_sources or {}).items()},
                   capture_source_bindings=scene_source_bindings(scene, archive_sources),
                   source_sha256={str(p): h for p, h in PINS.items()} |
                                 {str(Path(__file__).resolve()): sha(__file__)})
    save(output_dir/'TRANSPORT_PREPARATION.json', receipt)
    try:
        if not seed.get('complete') or not seed.get('admitted'):
            raise ValueError('Admitted source approach/closure plan required')
        for key, name in (('commands_sha256', 'commands.npz'),
                          ('path_geometry_sha256', 'path_geometry.json')):
            if sha(seed_plan_dir/name) != seed[key]:
                raise ValueError('Seed plan artifact changed: '+name)
        for name, digest in seed['source_sha256'].items():
            if sha(name) != digest:
                raise ValueError('Seed plan source changed: '+name)
        args = seed['arguments']
        bind_hand(SimpleNamespace(hand_urdf=Path(args['hand_urdf']), hand_scene=Path(args['hand_scene'])))
        if sha(Path(args['candidates'])/'candidates.npz') != seed['candidates_sha256']:
            raise ValueError('Seed candidate pool changed before transport preparation')
        for name, digest in seed.get('extra_input_sha256', {}).items():
            if sha(name) != digest:
                raise ValueError('Seed extra binding changed: '+name)
        target = next(r for r in scene['objects'] if r['object_id'] == scene['target_object_id'])
        if (Path(target['object_root']).resolve() != Path(args['object_root']).resolve()
                or target['scale'] != seed['source_scale']
                or not np.allclose(target['world_T_object'], seed['world_T_object'], atol=1e-12, rtol=0)):
            raise ValueError('Compatibility body does not match prepared original execution asset')
        target_repair = target.get('collision_refinement')
        seed_repair = args.get('collision_refinement')
        if bool(target_repair) != bool(seed_repair) or (target_repair
                and Path(target_repair).resolve() != Path(seed_repair).resolve()):
            raise ValueError('Compatibility body native collision repair differs from observed scene')
        if 'physics_config' in scene and Path(scene['physics_config']).resolve() != Path(args['physics_config']).resolve():
            raise ValueError('Observed and executing native physics profiles differ')
        model = PlanningModel(Path(args['robot_urdf']))
        reference = load_reference(Path(args['reference_root']))
        with np.load(seed_plan_dir/'commands.npz') as z:
            old = {k: z[k] for k in z.files}
        phases = old['phase'].tolist()
        approach_only = seed.get('approach_only', False)
        if approach_only:
            if list(dict.fromkeys(phases)) != ['settle', 'home_to_pregrasp', 'approach']:
                raise ValueError('Approach-only seed must contain exactly the original open-hand prefix')
            cut = len(phases)
        else:
            if 'close' not in phases or 'hold' not in phases:
                raise ValueError('Seed must contain complete original closure and hold')
            cut = max(i for i, p in enumerate(phases) if p == 'hold')+1
        if any(p == 'lift' for p in phases[:cut]):
            raise ValueError('Original world-Z lift must not precede transport')
        names = list(old['joint_names'])
        if names != model.active_names:
            raise ValueError('Original independent coordinate order changed')
        commands = list(old['joint_target'][:cut].copy())
        desired = list(old['desired_palm'][:cut].copy())
        phases = phases[:cut]
        closure_audit = seed.get('closure')
        if approach_only:
            family = TemplateFamily.load(Path(args['package']), Path(args['hand_urdf']),
                                         Path(args['hand_scene']), seed['gesture_id'])
            closure_audit = _append_source_closure(model, family, seed, commands, phases, desired)
        receipt.update(seed_approach_only=approach_only,
            unused_world_z_lift_required=not approach_only,
            closure_source='Original TwoSidedClosure, 250 close + 50 hold, original fixed arm and palm')
        last = model.full_state(dict(zip(names, map(float, commands[-1]))))
        opened = model.full_state(dict(zip(names, map(float, commands[0]))))
        hand_names = [n for n in model.active_names if n not in ARM_JOINTS and n not in ('FAJ1', 'FAJ3')]
        arm_indexes = [model.active_names.index(n) for n in ARM_JOINTS]
        palm = desired[-1].copy()
        with np.load(Path(args['candidates'])/'candidates.npz') as pool:
            source_direction = pool['proposal_approach_direction'][seed['index']]
        direction = np.asarray(seed['palm'])[:3, :3]@np.asarray(seed['source_palm'])[:3, :3].T@source_direction
        targets = _waypoint_targets(palm, direction, args.get('pregrasp_distance_m', .1),
                                    scene['bin'], np.asarray(seed['world_T_object']))
        ik_audits, timing, errors = [], {}, {}
        for phase, goal, minimum_s in targets:
            start = model.fk.transforms(last, reference.base_transform)['palm']
            waypoints = [last]
            for u in np.linspace(0, 1, 33)[1:]:
                solved, audit = arm_only_palm_ik(model, reference, waypoints[-1],
                    interpolate_palm(start, goal, float(u)), max_nfev=300, bounds_margin_rad=1e-6)
                waypoints.append(solved)
                ik_audits.append(dict(phase=phase, fraction=float(u), **audit))
            trajectory, timing[phase] = timed_segment(
                [[q[n] for n in ARM_JOINTS] for q in waypoints], dt=POLICY['dt_s'],
                velocity_limits=model.velocity[arm_indexes], minimum_s=minimum_s,
                position_lower=model.lower[arm_indexes], position_upper=model.upper[arm_indexes])
            phase_errors = []
            for k, values in enumerate(trajectory[1:], 1):
                state = model.full_state(dict(last, **dict(zip(ARM_JOINTS, map(float, values)))))
                wanted = interpolate_palm(start, goal, float(smooth_fraction(k/(len(trajectory)-1))))
                actual = model.fk.transforms(state, reference.base_transform)['palm']
                phase_errors.append(palm_error(actual, wanted))
                commands.append(model.vector(state)); phases.append(phase); desired.append(wanted)
            last = state
            errors[phase] = np.max(phase_errors, axis=0).tolist()
            if errors[phase][0] > 1e-4 or errors[phase][1]*.15 > 1e-4:
                raise RuntimeError('Interpolated transport exceeds original IK acceptance')
            if phase == 'retreat_4mm':
                for _ in range(round(POLICY['retreat_wait_s']/POLICY['dt_s'])):
                    commands.append(model.vector(last)); phases.append('retreat_wait'); desired.append(goal.copy())
        closed_motors = np.array([last[n] for n in hand_names])
        open_motors = np.array([opened[n] for n in hand_names])
        count = round(POLICY['release_s']/POLICY['dt_s'])
        for k in range(count):
            u = (k+1)/count
            state = with_hand_command(model, last, hand_names, (1-u)*closed_motors+u*open_motors)
            commands.append(model.vector(state)); phases.append('release'); desired.append(targets[-1][1].copy())
        for _ in range(round(POLICY['release_settle_s']/POLICY['dt_s'])):
            commands.append(model.vector(state)); phases.append('release_settle'); desired.append(targets[-1][1].copy())
        full_states = [model.full_state(dict(zip(names, map(float, q)))) for q in commands]
        rigid = [n for n in model.collision_links if not n.startswith(('th', 'ff', 'mf', 'rf', 'lf'))]
        # Retain the source target admission, and additionally audit stationary
        # distractors/bin for all open-hand states and rigid links afterwards.
        obstacles = []
        for row in scene['objects']:
            if row['object_id'] == scene['target_object_id']:
                continue
            asset = object_asset(SimpleNamespace(object_root=Path(row['object_root'])),
                                 np.asarray(row['world_T_object']), row['scale'])
            for mesh in asset.collision_parts:
                mesh = mesh.copy(); mesh.apply_transform(row['world_T_object']); obstacles.append(mesh)
        for box in bin_boxes(scene['bin']):
            mesh = trimesh.creation.box(extents=box['extents_m'])
            mesh.apply_transform(box['world_T_box']); obstacles.append(mesh)
        open_states = [s for s, p in zip(full_states, phases) if p in ('settle', 'home_to_pregrasp', 'approach')]
        post_states = [s for s, p in zip(full_states, phases) if p not in ('settle', 'home_to_pregrasp', 'approach')]
        open_audit = ArmPathGeometry(model, model.ignored_pairs(), obstacles, np.eye(4),
                                    vectorized_broadphase=True).audit(open_states)
        rigid_audit = ArmPathGeometry(model, model.ignored_pairs(), obstacles, np.eye(4),
            collision_links=rigid, vectorized_broadphase=True).audit(post_states)
        source_geometry = read(seed_plan_dir/'path_geometry.json')
        geometry = dict(ok=bool(source_geometry['ok'] and open_audit['ok']),
                        seed_target_geometry=source_geometry, multi_object_open= open_audit,
                        violations=source_geometry['violations']+open_audit['violations'])
        save(output_dir/'path_geometry.json', geometry)
        save(output_dir/'lift_arm_geometry.json', dict(rigid_audit,
            scope='All post-approach rigid arm/wrists/palm against self/floor/static scene and bin; fingers/actual moving objects remain runtime-audited'))
        np.savez_compressed(output_dir/'commands.npz', initial_position=old['initial_position'],
            joint_names=names, joint_target=commands, phase=phases, desired_palm=desired,
            time_s=np.arange(1, len(commands)+1)*POLICY['dt_s'])
        repair = seed_plan_dir.parent/'arm_collision_refinements.json'
        if repair.exists():
            shutil.copyfile(repair, output_dir.parent/'arm_collision_refinements.json')
        plan = copy.deepcopy(seed)
        plan['arguments']['approach_only'] = False
        plan.update(complete=True, admitted=bool(geometry['ok'] and rigid_audit['ok']),
            approach_only=False, closure=closure_audit,
            commands_sha256=sha(output_dir/'commands.npz'),
            path_geometry_sha256=sha(output_dir/'path_geometry.json'),
            lift_arm_geometry_sha256=sha(output_dir/'lift_arm_geometry.json'),
            total_command_steps=len(commands), author_transport=receipt,
            transport_ik_waypoints=ik_audits, transport_timing=timing,
            transport_interpolation_errors=errors, dt_s=POLICY['dt_s'])
        plan['source_sha256'][str(Path(__file__).resolve())] = sha(__file__)
        for binding in receipt['capture_source_bindings'].values():
            if binding['historical_archive']:
                plan.setdefault('extra_input_sha256', {})[binding['path']] = binding['sha256']
        plan.setdefault('extra_input_sha256', {})[str(scene_path.resolve())] = sha(scene_path)
        plan['phases'] = [dict(name=p, start_step=i+1) for i, p in enumerate(phases)
                          if i == 0 or phases[i-1] != p]
        save(output_dir/'plan.json', plan)
        receipt.update(complete=True, admitted=plan['admitted'], status='ready' if plan['admitted'] else 'null_geometry',
            command_steps=len(commands), independent_coordinates=len(names), actual_joint_coordinates=len(model.joints),
            plan_sha256=sha(output_dir/'plan.json'), wall_s=time.monotonic()-began,
            interpolation_errors=errors, timing=timing)
        save(output_dir/'TRANSPORT_PREPARATION.json', receipt)
        return receipt
    except RuntimeError as error:
        receipt.update(complete=True, status='null_local_planning', error=str(error), wall_s=time.monotonic()-began)
        save(output_dir/'TRANSPORT_PREPARATION.json', receipt)
        return receipt
    except BaseException as error:
        receipt.update(status='infrastructure_error', error=repr(error), wall_s=time.monotonic()-began)
        save(output_dir/'TRANSPORT_PREPARATION.json', receipt)
        raise


def args_for_plan(plan_dir, output_dir):
    """Rebuild only declared backend arguments, preserving exact input paths."""
    plan_dir = Path(plan_dir)
    plan = read(plan_dir/'plan.json')
    values = dict(plan['arguments'])
    path_keys = ('robot_urdf', 'reference_root', 'candidates', 'world_object_pose', 'arm_seed_file',
                 'initial_arm_state', 'hand_urdf', 'hand_scene', 'package', 'object_root',
                 'physics_config', 'diagnostic_config', 'collision_refinement')
    for key in path_keys:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    values.update(plan_dir=plan_dir, output_dir=Path(output_dir), stage='execute', physics_device='cpu',
                  tactile_feedback=False, profile_execution=True, restore_object_initial_state=True)
    return SimpleNamespace(**values)


class MultiObjectRuntime:
    """Native runtime factory and shared-clock participant, with no state resets."""
    def __init__(self, scene_path, output_dir, *, archive_sources=None):
        self.scene_path = Path(scene_path)
        self.scene = validate_scene(scene_path, archive_sources)
        self.output_dir = Path(output_dir)
        self.bodies, self.assets, self.paths = {}, {}, {}
        self.rows, self.contact_rows = [], []
        self.initial_bottom = {}
        self.settled_xy = {}
        self.max_penetration = self.max_robot_bin_force = 0.
        self.max_illegal_object_force = 0.
        self.steps = 0
        self.dt = POLICY['dt_s']
        self.setup_audit = {}

    def factory(self, args, initial, target_asset, build_scene):
        import isaaclab.sim as sim_utils
        from pxr import Gf, UsdGeom
        from scipy.spatial.transform import Rotation
        from run_template_arm_positive import object_asset
        from canondex.grasp.physics import load_physics_config
        from canondex.scene import grasp_object
        from reproduction.boya.settle_s40_scenes import namespaced_spawn

        profile = load_physics_config(args.physics_config)
        self.args = args
        self.target_id = self.scene['target_object_id']
        self.assets[self.target_id] = target_asset
        self.paths[self.target_id] = '/World/Object'
        def before_reset(sim, robot, objects):
            UsdGeom.Xform.Define(sim.stage, '/World/AuthorScene')
            UsdGeom.Xform.Define(sim.stage, '/World/Obstacles')
            for row in self.scene['objects']:
                key = row['object_id']
                if key == self.target_id:
                    continue
                asset = object_asset(SimpleNamespace(object_root=Path(row['object_root'])),
                                     np.asarray(row['world_T_object']), row['scale'])
                self.assets[key] = asset
                path = '/World/AuthorScene/'+key
                self.paths[key] = path
                repair = row.get('collision_refinement')
                self.bodies[key] = namespaced_spawn(grasp_object, path)(sim, asset,
                    physics_profile=profile, collision_refinement=read(repair) if repair else None)
            for box in bin_boxes(self.scene['bin']):
                pose = np.asarray(box['world_T_box'])
                quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
                cfg = sim_utils.CuboidCfg(size=tuple(box['extents_m']),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.6, dynamic_friction=.6,
                        restitution=0., friction_combine_mode='average'),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12, .25, .5), roughness=.6))
                cfg.func('/World/Obstacles/AuthorBin_'+box['name'], cfg,
                         translation=tuple(pose[:3, 3]), orientation=(float(quat[3]), *map(float, quat[:3])))
        result = build_scene(before_reset)
        self.sim, self.robot, _, objects = result
        self.bodies[self.target_id] = objects['object']
        self.dt = self.sim.get_physics_dt()
        if self.dt != POLICY['dt_s']:
            raise ValueError('Frozen native timestep changed')
        # This is the declared initial cold recreation, before ready and before
        # any integration. The original target uses its existing corresponding
        # initialization helper immediately after this factory returns.
        from canondex.grasp.arm_resident_scene import restore_declared_object_state
        self.initial_restores = {}
        for key, body in self.bodies.items():
            if key != self.target_id:
                body.update(self.dt)
                self.initial_restores[key] = restore_declared_object_state(self.sim, body, self.assets[key].pose_xyzw)
        return result

    def ready(self):
        import torch
        import warp as wp
        from pxr import Usd, UsdPhysics
        from template_gripper_native_audit import audit_scene
        from canondex.scene.execute_motion import RobotContactReports
        from canondex.scene.runtime import forward_scene
        from canondex.planning.model import PlanningModel
        from canondex.grasp.template_gripper_adapter import TemplateFamily

        # Original executor already applies target weight and material readback.
        # Every additional body gets exactly the same native mass/friction and
        # persistent world force before the first physical tick.
        material, entry = {}, {}
        for key, body in self.bodies.items():
            body.update(self.dt)
            actual_pose = body.data.root_link_pose_w.torch[0].cpu().numpy().copy()
            actual_velocity = body.data.root_link_vel_w.torch[0].cpu().numpy().copy()
            desired = self.assets[key].pose_xyzw
            error = min(float(abs(actual_pose-desired).max()),
                        float(abs(actual_pose-np.r_[desired[:3], -desired[3:]]).max()))
            if error > 2e-5 or float(abs(actual_velocity).max()) > 2e-5:
                raise RuntimeError('Declared cold scene entry does not match actual native state: '+key)
            entry[key] = dict(pose_xyzw=actual_pose.tolist(), velocity=actual_velocity.tolist(), max_pose_error=error)
            if float(body.root_view.get_masses().numpy().ravel()[0]) != float(np.float32(.1)):
                raise RuntimeError('Actual dynamic object mass changed: '+key)
            if key != self.target_id:
                properties = body.root_view.get_material_properties().numpy().copy()
                properties[..., :2] = .6; properties[..., 2] = 0.
                ids = wp.array(np.arange(len(properties), dtype=np.int32), dtype=wp.int32, device='cpu')
                body.root_view.set_material_properties(wp.array(properties, dtype=wp.float32, device='cpu'), ids)
                actual = body.root_view.get_material_properties().numpy()
                if not np.allclose(actual, properties, atol=1e-7, rtol=0):
                    raise RuntimeError('Native distractor friction readback changed')
                body.permanent_wrench_composer.reset()
                body.permanent_wrench_composer.add_forces_and_torques_index(
                    forces=torch.tensor([[[0., 0., -.98]]], dtype=torch.float32, device=body.device), is_global=True)
                body.write_data_to_sim()
                material[key] = dict(mass_kg=.1, friction=.6, native_readback_ok=True, world_weight_n=[0., 0., -.98])
        forward_scene(self.sim)
        for body in self.bodies.values():
            body.update(self.dt)
        audit_dir = self.output_dir/'multi_object_native'
        audit_dir.mkdir(parents=True, exist_ok=False)
        native = audit_scene(self.sim, audit_dir, ['all', 'object'], roots=tuple(self.paths.values()), fast=True)
        if not native['within_20um']:
            raise RuntimeError('Original native20um gate rejected additional body geometry')
        own = RobotContactReports(self.sim, capacity=POLICY['contact_capacity'], compact=True)
        self.robot_bodies = own.bodies
        plan = read(self.args.plan_dir/'plan.json')
        model = PlanningModel(self.args.robot_urdf)
        family = TemplateFamily.load(self.args.package, self.args.hand_urdf, self.args.hand_scene,
                                     plan['gesture_id'])
        self.legal_hand_owners = {model.fk.fixed_attachment(n, np.eye(4))[0] for n in family.hand.links}
        self.contact_bodies = list(self.paths.values())
        self.contact_targets = self.robot_bodies+list(self.paths.values())+own.targets[len(own.bodies):]
        self.contacts = self.sim.physics_sim_view.create_rigid_contact_view(self.contact_bodies,
            filter_patterns=[self.contact_targets]*len(self.contact_bodies),
            max_contact_data_count=POLICY['contact_capacity'])
        # New bin collision shapes are included by the unmodified reporter
        # under /World/Obstacles, preserving its full robot/self/floor audit.
        self.robot_contacts = own
        self.setup_audit = dict(native=native, additional_materials=material, entry=entry,
            additional_initial_cold_restores=self.initial_restores,
            legal_hand_rigid_owners=sorted(self.legal_hand_owners),
            body_paths=self.paths, contact_targets=self.contact_targets,
            global_gravity_m_s2=[0., 0., 0.], every_dynamic_body_weight_n=[0., 0., -.98],
            physical_state_writes_during_execution=0,
            entry_semantics='Declared independent cold recreation of measured multi-object snapshot; no claim of continuous observation process')
        for key, body in self.bodies.items():
            self.initial_bottom[key] = source_metrics(self.assets[key].mesh.vertices,
                body.data.root_link_pose_w.torch[0].cpu().numpy(), self.scene['bin'])['source_bottom_m']
        save(self.output_dir/'MULTI_OBJECT_SETUP.json', self.setup_audit)

    def before_step(self):
        for key, body in self.bodies.items():
            if key != self.target_id:
                body.write_data_to_sim()

    def after_step(self, phase):
        from canondex.scene.contact_readback import contact_data_to_host
        # The unchanged executor has already updated the compatibility target
        # for this tick before its generator next yields (or finishes).
        for key, body in self.bodies.items():
            if key != self.target_id:
                body.update(self.dt)
        contacts = decode_contacts(self.contact_bodies, self.contact_targets,
            contact_data_to_host(self.contacts, self.dt, POLICY['contact_capacity'], compact=True),
            POLICY['contact_capacity'])
        robot_contacts = self.robot_contacts.read()
        states = {}
        for key, body in self.bodies.items():
            pose = body.data.root_link_pose_w.torch[0].cpu().numpy().copy()
            velocity = body.data.root_link_vel_w.torch[0].cpu().numpy().copy()
            if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
                raise FloatingPointError('Nonfinite dynamic source state: '+key)
            path = self.paths[key]
            loads = object_robot_loads(path, contacts, self.robot_bodies, self.legal_hand_owners)
            self.max_illegal_object_force = max(self.max_illegal_object_force, loads['illegal_robot_support_force_n'])
            states[key] = dict(pose_xyzw=pose.tolist(), velocity=velocity.tolist(), **loads,
                              **source_metrics(self.assets[key].mesh.vertices, pose, self.scene['bin']))
        if phase == 'settle':
            self.settled_xy = {key: np.asarray(state['pose_xyzw'][:2]) for key, state in states.items()}
        depth = max([0.]+[-c['separation_m'] for c in contacts+robot_contacts])
        bin_force = max([0.]+[abs(c['normal_force_n']) for c in robot_contacts
                       if any(p.startswith('/World/Obstacles/AuthorBin_') for p in c['bodies'])])
        self.max_penetration = max(self.max_penetration, depth)
        self.max_robot_bin_force = max(self.max_robot_bin_force, bin_force)
        self.rows.append(dict(step=self.steps, time_s=self.steps*self.dt, phase=phase, objects=states))
        self.contact_rows.append(dict(step=self.steps, object_contacts=contacts, robot_contacts=robot_contacts))
        if depth > POLICY['early_stop_penetration_m']:
            raise PhysicalStop('multi_object_deep_collision_stop')
        pushed = preclosure_pushes(states, self.settled_xy, phase)
        if pushed:
            raise PhysicalStop('source_object_pushed_before_closure:'+','.join(pushed))

    def finish(self, original, error=None, physical_stop=None):
        result = evaluate_transport(self.rows, self.initial_bottom, self.dt)
        complete = bool(original and original.get('gates', {}).get('complete_execution'))
        gates = dict(original.get('gates', {}) if original else {})
        gates.update(all_dynamic_contacts=self.max_penetration <= POLICY['max_penetration_m'],
                     all_dynamic_objects_no_arm_support=self.max_illegal_object_force <= .001,
                     author_phase_order=result['phase_order_valid'],
                     no_robot_bin_support=self.max_robot_bin_force <= POLICY['max_robot_bin_force_n'],
                     complete_execution=complete, no_runtime_error=error is None)
        phases = list(dict.fromkeys(row['phase'] for row in self.rows))
        normalized = []
        for phase in phases:
            name = ('retreat' if phase.startswith('retreat_') else
                    'transport' if phase == 'transport_bin' else
                    'post_release' if phase == 'release_settle' else phase)
            if name not in normalized:
                normalized.append(name)
        binding = getattr(self, 'collection_binding', {})
        result.update(schema='anydex_boya_author_transport_results_v1', complete=bool(original and original.get('complete')),
            task_success=bool(gates and all(gates.values()) and result['delivered_object_ids']),
            status='infrastructure_error' if error else ('success' if all(gates.values()) and result['delivered_object_ids'] else 'physical_failure'),
            gates=gates, policy=POLICY, scene_path=str(self.scene_path), scene_sha256=sha(self.scene_path),
            physics_steps=self.steps, dynamic_object_steps=self.steps*len(self.bodies),
            total_physics_steps=self.steps, completed_phases=normalized, actual_phases=phases,
            label_eligible=bool(gates and all(gates.values()) and binding.get('collection_candidate_sha256')),
            collection_candidate_sha256=binding.get('collection_candidate_sha256'),
            task=binding.get('task'),
            commands_sha256=getattr(self, 'commands_sha256', None),
            error=error, physical_stop=physical_stop, max_penetration_m=self.max_penetration,
            max_robot_bin_force_n=self.max_robot_bin_force,
            max_all_object_illegal_support_force_n=self.max_illegal_object_force,
            legacy_target_id=self.scene['target_object_id'],
            label_scope='Any originally tabletop dynamic source body lifted with concurrent bilateral finger load during transport, still held at final transport step, then released and continuously contained in declared bin; all-body original no-arm-support gate retained',
            preclosure_guard_scope='Original conservative >3cm XY push stop, extended to every source body; no candidate target association',
            legacy_lift_evaluation_is_transport_label=False,
            coarse_convex_collision_is_exact_source_mesh=False, state_writes_during_execution=0)
        save(self.output_dir/'TRANSPORT_EVALUATION.json', result)
        save(self.output_dir/'MULTI_OBJECT_TRACE.json', dict(rows=self.rows))
        save(self.output_dir/'MULTI_OBJECT_CONTACTS.json', dict(rows=self.contact_rows))
        result['trace_sha256'] = sha(self.output_dir/'MULTI_OBJECT_TRACE.json')
        result['contacts_sha256'] = sha(self.output_dir/'MULTI_OBJECT_CONTACTS.json')
        save(self.output_dir/'TRANSPORT_EVALUATION.json', result)
        return result


def decode_contacts(bodies, targets, data, capacity):
    """Decode every native object/robot/object/floor/bin pair, refusing truncation."""
    if len(data) != 6:
        raise RuntimeError('Native contact buffer count mismatch')
    force, point, normal, separation, count, start = map(np.asarray, data)
    if (count.shape != (len(bodies), len(targets)) or start.shape != count.shape
            or count.dtype.kind not in 'iu' or start.dtype.kind not in 'iu'
            or np.any(count < 0) or int(count.sum()) >= capacity):
        raise RuntimeError('Native contact pair table invalid or exhausted')
    if (force.ndim not in (1, 2) or separation.ndim not in (1, 2)
            or (force.ndim == 2 and force.shape[1:] != (1,))
            or (separation.ndim == 2 and separation.shape[1:] != (1,))
            or point.ndim != 2 or point.shape[1:] != (3,) or normal.shape != point.shape
            or len({len(x) for x in (force, point, normal, separation)}) != 1):
        raise RuntimeError('Native contact buffer geometry shapes invalid')
    result = []
    for a, b in np.argwhere(count > 0):
        first, last = int(start[a, b]), int(start[a, b]+count[a, b])
        if first < 0 or last > min(map(len, (force, point, normal, separation))):
            raise RuntimeError('Native contact interval invalid')
        if not all(np.isfinite(v[first:last]).all() for v in (force, point, normal, separation)):
            raise FloatingPointError('Nonfinite native contact')
        for i in range(first, last):
            result.append(dict(bodies=[bodies[a], targets[b]], position_world_m=point[i].tolist(),
                raw_normal_world=normal[i].tolist(), separation_m=float(separation[i].item()),
                normal_force_n=float(force[i].item())))
    return result


def evaluate_transport(rows, initial_bottom, dt):
    window = round(POLICY['final_window_s']/dt)
    blocks = [r['phase'] for i, r in enumerate(rows) if i == 0 or rows[i-1]['phase'] != r['phase']]
    required = ['approach', 'close', 'hold', 'retreat_4mm', 'retreat_wait',
                'retreat_pregrasp', 'transport_bin', 'release', 'release_settle']
    phase_order_valid = [phase for phase in blocks if phase in required] == required
    delivered, objects = [], {}
    for key in initial_bottom:
        samples = [r['objects'][key] for r in rows]
        final = samples[-window:]
        max_rise = max((r['source_bottom_m']-initial_bottom[key] for r in samples), default=0.)
        carrying = [r['objects'][key] for r in rows
                    if r['phase'] in ('retreat_4mm', 'retreat_wait', 'retreat_pregrasp', 'transport_bin')]
        def held(sample):
            fingers = np.asarray(sample.get('finger_loads_n', [0.]*5), dtype=float)
            return bool(sample['source_bottom_m']-initial_bottom[key] >= POLICY['source_rise_m']
                        and fingers.shape == (5,) and fingers[0] > .001 and fingers[1:].sum() > .001
                        and sample.get('illegal_robot_support_force_n', 0.) <= .001)
        qualified_carry = any(held(sample) for sample in carrying)
        transport_samples = [r['objects'][key] for r in rows if r['phase'] == 'transport_bin']
        held_at_release = bool(transport_samples and held(transport_samples[-1]))
        valid_window = len(final) == window and all(r['inside_bin'] and r['robot_load_n'] <= .001
            and np.linalg.norm(r['velocity'][:3]) <= POLICY['max_final_linear_speed_m_s']
            and np.linalg.norm(r['velocity'][3:]) <= POLICY['max_final_angular_speed_rad_s'] for r in final)
        release_window = len(rows) >= window and all(r['phase'] == 'release_settle' for r in rows[-window:])
        passed = bool(phase_order_valid and valid_window and release_window and qualified_carry and held_at_release)
        objects[key] = dict(delivered=passed, maximum_source_bottom_rise_m=float(max_rise),
                            carried_with_bilateral_finger_load=bool(qualified_carry),
                            held_above_table_at_release=held_at_release,
                            final_contained_unloaded_stationary=bool(valid_window), release_settle_window=release_window,
                            final=samples[-1] if samples else None)
        if passed:
            delivered.append(key)
    return dict(delivered_object_ids=delivered, objects=objects, final_window_steps=window,
                required_phases=required, actual_phase_blocks=blocks, phase_order_valid=phase_order_valid)


def execution_steps(args, scene_path):
    """Yield ready/step; caller performs exactly one sim.step per step event."""
    _backend_path()
    import execute_template_arm_positive as original_executor
    from run_template_arm_positive import bind_hand
    bind_hand(args)
    plan = read(args.plan_dir/'plan.json')
    binding = plan.get('author_transport')
    if not binding or binding['scene_sha256'] != sha(scene_path):
        raise ValueError('Execution requires the prepared unchanged multi-object transport scene')
    if getattr(args, 'tactile_feedback', False):
        raise ValueError('Feedback changes the frozen transport trajectory')
    runtime = MultiObjectRuntime(scene_path, args.output_dir,
                                 archive_sources=binding.get('scene_archive_sources'))
    runtime.collection_binding = plan.get('author_collection_binding', {})
    runtime.commands_sha256 = plan['commands_sha256']
    with np.load(args.plan_dir/'commands.npz') as z:
        phases = z['phase'].tolist()
    generator = original_executor.execution_steps(args, runtime_factory=runtime.factory)
    error = None
    physical_stop = None
    try:
        event = next(generator)
        while event is not None:
            if event['kind'] == 'ready':
                runtime.ready()
                yield dict(event, multi_object_runtime=runtime)
                event = next(generator)
            elif event['kind'] == 'step':
                runtime.before_step()
                before = event['sim'].get_physics_step_count()
                yield event
                if event['sim'].get_physics_step_count() != before+1:
                    raise RuntimeError('Caller must advance the shared physics clock exactly once')
                runtime.steps += 1
                # Complete the original current-tick update/audit first. Its
                # next command can be written before our observation, but no
                # additional integration occurs until our next yielded event.
                try:
                    following = next(generator)
                except StopIteration:
                    following = None
                except BaseException:
                    # The physical tick already happened even if the original
                    # post-step audit failed. Preserve its cost and attempt the
                    # multi-body observation before propagating the failure.
                    runtime.after_step(phases[event['step']-1])
                    raise
                runtime.after_step(phases[event['step']-1])
                event = following
            else:
                raise RuntimeError('Unexpected native execution event')
    except PhysicalStop as exception:
        physical_stop = str(exception)
    except BaseException as exception:
        error = repr(exception)
        raise
    finally:
        generator.close()
        original_path = args.output_dir/'evaluation.json'
        original = read(original_path) if original_path.exists() else {}
        runtime.finish(original, error, physical_stop)


def run(args, scene_path):
    for event in execution_steps(args, scene_path):
        if event['kind'] == 'step':
            event['sim'].step(render=False)
    return read(args.output_dir/'TRANSPORT_EVALUATION.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'execute'))
    parser.add_argument('--seed-plan-dir', type=Path)
    parser.add_argument('--plan-dir', type=Path)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--archive-sources', type=Path, help='Explicit original source path -> exact historical capture snapshot path JSON')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if not args.seed_plan_dir:
            parser.error('--seed-plan-dir required')
        print(json.dumps(prepare_transport(args.seed_plan_dir, args.scene, args.output_dir,
            archive_sources=read(args.archive_sources) if args.archive_sources else None), indent=2))
    else:
        if not args.plan_dir:
            parser.error('--plan-dir required')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True, device='cpu', visualizer=None, kit_args='--/log/file=/dev/null')
        code = 1
        try:
            result = run(args_for_plan(args.plan_dir, args.output_dir), args.scene)
            print(json.dumps(result, indent=2)); code = 0
        finally:
            launcher.app.close(exit_code=code)


if __name__ == '__main__':
    main()
