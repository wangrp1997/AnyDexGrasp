"""Explicit Boya action contract and read-only, source-based semantic audit.

This module imports no CanonDex controller or simulator. It binds the existing
computed pad reference to the actual 16 independent / 4 mimic joint commands.
Numerical agreement is separate from contact, transport and hardware validity.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from audit_reference_geometry import URDFFK, angle, axis_rotation, sha, unit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json'
DEFAULT_SURFACES = ROOT/'outputs/boya_full_alignment_20261114_v2/hand_clouds/MANIFEST.json'
DEFAULT_CATALOG = ROOT/'outputs/boya_fixed_pool_20261112_v1/CATALOG.json'
SOURCE_FLIPS = {
    'dh3_collection': ('collect_dh3_grasp_data.py', '1bcb655fad63a7c2b4f65e29b426b6aa614785d3cea514e0cd290fd05eb8bbcb', [0, 1]),
    'dh3_deployment': ('robot_dh3.py', '7d227451761a429fff87f689e5dcb9c9e215da5ec3c5c594be1e16f3333fa797', [1, 2]),
}
CGR_R_REFERENCE = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_flip(rows17, profile):
    """Execute only the pinned source flip function, preserving paired row order."""
    if profile not in SOURCE_FLIPS:
        raise ValueError('An explicit DH3 collection or deployment profile is required')
    rows = np.asarray(rows17)
    if rows.ndim != 2 or rows.shape[1] != 17 or not np.isfinite(rows).all():
        raise ValueError('Finite Nx17 rows required')
    name, digest, _ = SOURCE_FLIPS[profile]
    path = ROOT/name
    if sha(path) != digest:
        raise ValueError('Changed author flip source: '+name)
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == 'flip_ggarray']
    if len(nodes) != 1:
        raise ValueError('Unique original flip function required')
    scope = dict(np=np, copy=copy)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    result, flips = scope['flip_ggarray'](rows.copy())
    return result, np.asarray(flips, dtype=bool)


def _rigid(value):
    a = np.asarray(value, dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=2e-6, rtol=0)
            or not np.allclose(a[:3, :3].T@a[:3, :3], np.eye(3), atol=2e-6, rtol=0)
            or abs(np.linalg.det(a[:3, :3])-1) > 2e-6):
        raise ValueError('Finite proper rigid transform required')
    return a


class ActionContract:
    """An inspectable geometric/command contract, never a physical-success gate."""
    def __init__(self, calibration=DEFAULT_CALIBRATION, urdf=None):
        self.calibration_path = None if isinstance(calibration, dict) else Path(calibration).resolve()
        self.calibration = copy.deepcopy(calibration) if isinstance(calibration, dict) else json.loads(self.calibration_path.read_text())
        lock = self.calibration['source_lock']
        self.sources = {p: item['sha256'] for p, item in lock['source_files'].items()}
        for path, digest in self.sources.items():
            if sha(path) != digest:
                raise ValueError('Changed Boya source asset: '+path)
        source_urdf = next(Path(p) for p in self.sources if p.endswith('/boya.urdf'))
        self.urdf = Path(urdf).resolve() if urdf else source_urdf
        if sha(self.urdf) != sha(source_urdf):
            raise ValueError('Action URDF differs from computed reference source')
        self.robot = ET.parse(self.urdf).getroot()
        self.fk = URDFFK(self.robot)
        self.names = tuple(lock['independent_motor_names'])
        self.mimics = {name: tuple(values) for name, values in lock['mimics'].items()}
        self.hand_names = self.names+tuple(self.mimics)
        actuated = {name for name, joint in self.fk.joints.items()
                    if joint['kind'] != 'fixed' and name.startswith(('TH', 'FF', 'MF', 'RF', 'LF'))}
        if len(self.names) != 16 or len(self.mimics) != 4 or set(self.hand_names) != actuated:
            raise ValueError('Expected all 16 independent and 4 mimic source finger joints')
        for name, (driver, multiplier, offset) in self.mimics.items():
            if self.fk.joints[name]['mimic'] != (driver, multiplier, offset):
                raise ValueError('Mimic declaration differs from source URDF')
        self.limits = {}
        for element in self.robot.findall('joint'):
            if element.get('name') in self.hand_names:
                limit = element.find('limit')
                self.limits[element.get('name')] = (float(limit.get('lower')), float(limit.get('upper')))
        self.slots = {s['slot_id']: s for s in self.calibration['slots'] if s['gesture_id'] in (1, 6, 7, 8)}
        package_path = next(Path(p) for p in self.sources if p.endswith('/package.json'))
        pads_path = next(Path(p) for p in self.sources if p.endswith('/templates/five_finger_wrap.json'))
        self.entries = {e['gesture_id']: e for e in json.loads(package_path.read_text())['entries']}
        self.pads = {c['group']: c for c in json.loads(pads_path.read_text())['contacts']}
        self.active = {g: tuple(c['group'] for c in self.entries[g]['template']['contacts']
                               if c['group'] in ('th', 'ff', 'mf', 'rf', 'lf')) for g in (1, 6, 7, 8)}
        self.link_names = tuple(lock['collision_links'])
        if len(self.link_names) != 21:
            raise ValueError('All 21 original collision links must be represented')
        for slot in self.slots.values():
            self._validate_slot(slot)

    def _validate_slot(self, slot):
        gesture = slot['gesture_id']
        target = self.entries[gesture]['adaptation']['pose_targets_rad']
        opened, closed = np.asarray(slot['independent_q']), np.asarray(slot['close_q'])
        nominal = np.array([target[n] for n in self.names])
        if opened.shape != (16,) or not np.array_equal(closed, nominal):
            raise ValueError('Closing endpoint differs from the corresponding company pose')
        expected = nominal.copy()
        for i, name in enumerate(self.names):
            group = name[:2].lower()
            if group in self.active[gesture] and (name in ('THJ1', 'THJ2') or (group != 'th' and name.endswith(('J2', 'J3')))):
                expected[i] *= slot['fraction']
        if not np.array_equal(opened, expected):
            raise ValueError('Opening changes source non-flexion or inactive-finger coordinates')
        if slot['wrist'] != {n: target[n] for n in ('FAJ1', 'FAJ3')}:
            raise ValueError('Wrist substitution differs from source company pose')
        for q in (opened, closed):
            self._check_joint_limits(self._expand(q))
        saved = _rigid(slot['palm_T_reference'])
        coordinates = dict(zip(self.names, opened)); coordinates.update(slot['wrist'])
        frames = self.fk.evaluate(coordinates)
        points, normals = {}, {}
        for group in ('th', slot['opposing_group']):
            if group not in self.active[gesture]:
                raise ValueError('Reference opponent is not an active source finger')
            pad = self.pads[group]; frame = frames[pad['collision_link']]
            points[group] = frame[:3, :3]@pad['point_link_m']+frame[:3, 3]
            normals[group] = unit(frame[:3, :3]@pad['normal_link'])
        delta = points[slot['opposing_group']]-points['th']; x = unit(delta)
        tangent = np.array([0., 1., 0.]); tangent -= (tangent@normals['th'])*normals['th']
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.eye(3)[np.argmin(abs(normals['th']))]
            tangent -= (tangent@normals['th'])*normals['th']
        z = unit(tangent-(tangent@x)*x)
        rebuilt = np.eye(4); rebuilt[:3, :3] = np.column_stack((x, np.cross(z, x), z))
        rebuilt[:3, 3] = (points['th']+points[slot['opposing_group']])/2
        if (np.max(abs(saved-rebuilt)) > 1e-9
                or abs(np.linalg.norm(delta)-slot['reference_width_m']) > 1e-9):
            raise ValueError('Reference axis/midpoint/width differs from independent source-pad FK')

    def _expand(self, motors):
        q = np.asarray(motors, dtype=float)
        if q.shape[-1:] != (16,) or not np.isfinite(q).all():
            raise ValueError('Finite original 16 independent motor targets required')
        values = {n: q[..., i] for i, n in enumerate(self.names)}
        for name, (driver, coefficient, offset) in self.mimics.items():
            values[name] = coefficient*values[driver]+offset
        return np.stack([values[n] for n in self.hand_names], axis=-1)

    def _check_joint_limits(self, targets):
        for i, name in enumerate(self.hand_names):
            low, high = self.limits[name]
            if np.any(targets[..., i] < low-1e-12) or np.any(targets[..., i] > high+1e-12):
                raise ValueError('Command exceeds original source limit: '+name)

    def for_slot(self, slot_id):
        slot = self.slots[slot_id]
        opened, closed = np.asarray(slot['independent_q']), np.asarray(slot['close_q'])
        result = dict(schema='anydex_boya_action_contract_v1', slot_id=slot_id,
            gesture_id=slot['gesture_id'], opposing_group=slot['opposing_group'],
            independent_motor_names=list(self.names), hand_joint_names=list(self.hand_names),
            opened_independent_rad=opened.tolist(), closed_independent_rad=closed.tolist(),
            opened_hand_rad=self._expand(opened).tolist(), closed_hand_rad=self._expand(closed).tolist(),
            wrist_coordinates=slot['wrist'], mimics={n: list(v) for n, v in self.mimics.items()},
            active_groups=list(self.active[slot['gesture_id']]),
            inactive_groups=[g for g in ('th', 'ff', 'mf', 'rf', 'lf') if g not in self.active[slot['gesture_id']]],
            retained_collision_links=list(self.link_names), palm_T_reference=slot['palm_T_reference'],
            reference_width_m=slot['reference_width_m'],
            reference_semantics='Thumb-to-single-opponent annotated distal pad midpoint; width is pad separation',
            approach_semantics='Reference +Z equals mapped actual CGR +X; source thumb tangent projected off pair axis',
            closure_semantics='Two progress variables interpolate source endpoints; inactive/non-flexion coordinates retained; native mimics unchanged',
            source_asset_sha256=self.sources, calibration_sha256=digest_json(self.calibration),
            numeric_contract_validated=True, physical_semantics_validated=False,
            pending_physical_gates=['Observation-to-execution all-body initial-state binding',
                'Object/table and inactive-finger collision during the executed closing path',
                'Loaded grasp, approach-axis retreat, transport and release into the declared bin'],
            hardware_semantics_validated=False, physics_steps=0)
        result['contract_sha256'] = digest_json(result)
        return result

    def closure_targets(self, slot_id, thumb_progress, fingers_progress=None):
        thumb = np.atleast_1d(np.asarray(thumb_progress, dtype=float))
        fingers = thumb if fingers_progress is None else np.atleast_1d(np.asarray(fingers_progress, dtype=float))
        if (thumb.ndim != 1 or fingers.shape != thumb.shape or not np.isfinite(thumb).all()
                or not np.isfinite(fingers).all() or (thumb < 0).any() or (thumb > 1).any()
                or (fingers < 0).any() or (fingers > 1).any()):
            raise ValueError('Paired finite progress vectors in [0,1] required')
        slot = self.slots[slot_id]
        per_motor = np.where(np.array([n.startswith('TH') for n in self.names])[None], thumb[:, None], fingers[:, None])
        motors = (1.-per_motor)*np.asarray(slot['independent_q'])+per_motor*np.asarray(slot['close_q'])
        full = self._expand(motors)
        self._check_joint_limits(full)
        return dict(joint_names=list(self.hand_names), joint_target=full,
                    independent_motor_names=list(self.names), independent_target=motors,
                    contract_sha256=self.for_slot(slot_id)['contract_sha256'])

    def mapped_palm(self, slot_id, candidate_row15, world_T_camera, *, extra_depth_m=0.):
        row = np.asarray(candidate_row15, dtype=float)
        if row.shape != (15,) or not np.isfinite(row).all() or row[1] <= 0 or row[2] < 0 or not np.isfinite(extra_depth_m):
            raise ValueError('Finite actual post-flip source row15 required')
        cgr = np.eye(4); cgr[:3, :3] = row[3:12].reshape(3, 3); cgr[:3, 3] = row[12:15]
        _rigid(cgr)
        u, _, vh = np.linalg.svd(cgr[:3, :3]); cgr[:3, :3] = u@vh
        reference = np.eye(4); reference[:3, :3] = CGR_R_REFERENCE; reference[0, 3] = row[2]+extra_depth_m
        return _rigid(world_T_camera)@cgr@reference@np.linalg.inv(_rigid(self.slots[slot_id]['palm_T_reference']))

    def validate_command_arrays(self, arrays, *, slot_id, motor_targets=None,
                                candidate_row15=None, world_T_camera=None, extra_depth_m=0.):
        names = [str(n) for n in arrays['joint_names']]
        targets = np.asarray(arrays['joint_target'])
        phase = np.asarray(arrays['phase'])
        if (len(names) != len(set(names)) or targets.ndim != 2 or targets.shape[1] != len(names)
                or phase.shape != (len(targets),) or not np.isfinite(targets).all()
                or not set(self.hand_names).issubset(names)):
            raise ValueError('Finite trajectory with each original hand joint exactly once required')
        if motor_targets is None:
            progress = np.empty(len(targets))
            opened = {'prehold', 'settle', 'home', 'approach', 'open', 'postrelease'}
            closed = {'hold', 'lift', 'retreat', 'transfer'}
            for name in np.unique(phase):
                mask = phase == name
                if name in opened: progress[mask] = 0.
                elif name in closed: progress[mask] = 1.
                elif name in ('close', 'release'):
                    steps = np.flatnonzero(mask)
                    if len(steps) > 1 and not np.all(np.diff(steps) == 1):
                        raise ValueError('Each closing/release segment must be contiguous')
                    fraction = np.arange(1, len(steps)+1, dtype=float)/len(steps)
                    progress[mask] = fraction if name == 'close' else 1-fraction
                else: raise ValueError('Supply explicit motor_targets for unknown phase: '+str(name))
            expected = self.closure_targets(slot_id, progress)['joint_target']
        else:
            motors = np.asarray(motor_targets)
            if motors.shape != (len(targets), 16):
                raise ValueError('Explicit original independent Nx16 targets required')
            opened = np.asarray(self.slots[slot_id]['independent_q'])
            closed = np.asarray(self.slots[slot_id]['close_q'])
            delta = closed-opened
            changed = abs(delta) > 1e-12
            if np.any(abs(motors[:, ~changed]-opened[~changed]) > 1e-12):
                raise ValueError('Explicit targets change inactive or non-flexion coordinates')
            for thumb_side in (True, False):
                selected = changed & np.array([n.startswith('TH') == thumb_side for n in self.names])
                if not selected.any(): continue
                progress = (motors[:, selected]-opened[selected])/delta[selected]
                if (progress < -1e-12).any() or (progress > 1+1e-12).any() or (np.ptp(progress, axis=1) > 1e-10).any():
                    raise ValueError('Explicit targets leave the original two-sided closure path')
            expected = self._expand(motors)
            self._check_joint_limits(expected)
        actual = targets[:, [names.index(n) for n in self.hand_names]]
        equal = np.array_equal(actual, expected.astype(targets.dtype))
        if not equal:
            raise ValueError('Actual 20 hand-DOF commands differ from bound source action contract')
        palm_error = None
        if candidate_row15 is not None:
            if world_T_camera is None: raise ValueError('Camera transform required with actual row')
            desired = np.asarray(arrays['desired_palm'])
            if desired.shape != (len(targets), 4, 4): raise ValueError('One desired palm per command required')
            mapped = self.mapped_palm(slot_id, candidate_row15, world_T_camera, extra_depth_m=extra_depth_m)
            mask = np.isin(phase, ['close', 'hold'])
            if not mask.any(): raise ValueError('At least one grasp-pose closing/holding command required')
            palm_error = float(abs(desired[mask]-mapped).max())
            if palm_error > 2e-6: raise ValueError('Desired grasp palm differs from actual row/reference/depth binding')
        return dict(numeric_contract_validated=True, hand_dof=20, steps=len(targets),
                    all_hand_commands_exact=True, max_hand_command_error_rad=0.,
                    max_grasp_palm_matrix_error=palm_error, slot_id=slot_id,
                    contract_sha256=self.for_slot(slot_id)['contract_sha256'],
                    physical_semantics_validated=False, physics_steps=0)


def _mesh_data(contract):
    import trimesh
    result = {}
    for name in contract.link_names:
        collision = contract.robot.find(f"link[@name='{name}']/collision")
        geometry = collision.find('geometry/mesh')
        path = (contract.urdf.parent/geometry.get('filename')).resolve()
        mesh = trimesh.load(path, force='mesh')
        origin = collision.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        rotation = axis_rotation([0, 0, 1], rpy[2])@axis_rotation([0, 1, 0], rpy[1])@axis_rotation([1, 0, 0], rpy[0])
        mesh.vertices = np.asarray(mesh.vertices)*np.fromstring(geometry.get('scale', '1 1 1'), sep=' ')
        result[name] = dict(mesh=mesh, vertices=np.asarray(mesh.vertices)@rotation.T+xyz,
                            collision_rotation=rotation, collision_origin=xyz)
    return result


def _contact_points(contract, slot, motors):
    coordinates = dict(zip(contract.names, motors)); coordinates.update(slot['wrist'])
    frames = contract.fk.evaluate(coordinates)
    points = {}
    for group in contract.active[slot['gesture_id']]:
        pad = contract.pads[group]; frame = frames[pad['collision_link']]
        points[group] = frame[:3, :3]@pad['point_link_m']+frame[:3, 3]
    return points, frames


def audit(calibration=DEFAULT_CALIBRATION, surfaces=DEFAULT_SURFACES, catalog=DEFAULT_CATALOG, *, output):
    import trimesh
    started = time.monotonic()
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a fresh audit directory; preserve earlier attempts')
    output.mkdir(parents=True)
    contract = ActionContract(calibration)
    sources = dict(contract.sources)
    for path in (Path(__file__), Path(calibration), Path(surfaces), Path(catalog), Path(__file__).with_name('audit_reference_geometry.py')):
        sources[str(path.resolve())] = sha(path)
    meshes = _mesh_data(contract)
    pad_surface = []
    for group, pad in contract.pads.items():
        mesh = meshes[pad['collision_link']]['mesh']
        point = np.asarray(pad['point_link_m'])
        nearest, distance, triangle = trimesh.proximity.closest_point_naive(mesh, point[None])
        normal = unit(pad['normal_link'])
        pad_surface.append(dict(group=group, link=pad['collision_link'],
            nearest_source_triangle=int(triangle[0]), nearest_source_surface_distance_m=float(distance[0]),
            annotated_normal_dot_nearest_triangle=float(normal@mesh.face_normals[int(triangle[0])]),
            source_status=pad.get('status'), source_pad_point_m=point.tolist()))
    if max(p['nearest_source_surface_distance_m'] for p in pad_surface) > 1e-7:
        raise ValueError('Source pad does not lie on its exact original distal mesh')
    summaries = {}
    contracts = {}
    for gesture in (1, 6, 7, 8):
        slots = sorted((s for s in contract.slots.values() if s['gesture_id'] == gesture and s['opposing_group'] == 'ff'), key=lambda s:s['sample'])
        opening = slots[0]; closed = slots[-1]
        first_points, _ = _contact_points(contract, opening, opening['independent_q'])
        last_points, _ = _contact_points(contract, opening, opening['close_q'])
        frame = np.asarray(opening['palm_T_reference']); x, z = frame[:3, 0], frame[:3, 2]
        trajectories = {}
        samples = np.linspace(0, 1, 101)
        coordinates = contract.closure_targets(opening['slot_id'], samples)['independent_target']
        pair = []
        for motors in coordinates:
            points, _ = _contact_points(contract, opening, motors)
            pair.append([points['th'], points['ff']])
        pair = np.asarray(pair)
        for group, sign in [('th', 1), ('ff', -1)]:
            delta = last_points[group]-first_points[group]
            trajectories[group] = dict(displacement_palm_m=delta.tolist(),
                motion_toward_opponent_along_initial_pair_axis_m=float(sign*(delta@x)),
                motion_along_initial_approach_m=float(delta@z),
                motion_orthogonal_to_initial_pair_axis_m=float(np.linalg.norm(delta-(delta@x)*x)))
        widths = np.linalg.norm(pair[:, 1]-pair[:, 0], axis=1)
        changed = np.abs(np.asarray(opening['close_q'])-opening['independent_q']) > 1e-12
        inactive = [g for g in ('th', 'ff', 'mf', 'rf', 'lf') if g not in contract.active[gesture]]
        summaries[str(gesture)] = dict(active_groups=list(contract.active[gesture]), inactive_groups=inactive,
            changed_independent_motors=[n for n, yes in zip(contract.names, changed) if yes],
            inactive_motors_unchanged=all(not yes for n, yes in zip(contract.names, changed) if n[:2].lower() in inactive),
            opening_width_m=float(widths[0]), closed_width_m=float(widths[-1]),
            pair_separation_increases_at_101_command_samples=int((np.diff(widths) > 1e-9).sum()),
            pair_midpoint_open_to_closed_m=float(np.linalg.norm(pair[-1].mean(0)-pair[0].mean(0))),
            reference_approach_axis_change_deg=angle(frame[:3, 2], np.asarray(closed['palm_T_reference'])[:3, 2]),
            pad_motion=trajectories,
            interpretation='Free FK trajectories, not contact-force balance, root/palm motion, failure cause or a parallel-jaw approximation')
        for slot in slots: contracts[slot['slot_id']] = contract.for_slot(slot['slot_id'])
    library = json.loads(Path(surfaces).read_text())
    if library['calibration']['sha256'] != sha(calibration): raise ValueError('Surface/reference binding differs')
    if set(library['slot_to_state']) != set(contract.slots): raise ValueError('Full surface library misses action slots')
    retained = all([p['link'] for p in s['parts']] == list(contract.link_names) for s in library['states'])
    if not retained: raise ValueError('Inactive hand collision links lost from surface assets')
    catalog_data = json.loads(Path(catalog).read_text())
    historical = []; raw_cache = {}; observation_cache = {}
    for entry in catalog_data['candidates']:
        candidate_id = entry['candidate_id']; generation_path = Path(entry['generation_json'])
        generation = json.loads(generation_path.read_text()); record = entry['candidate_record']
        pool_path = Path(entry['candidate_dir'])/'candidates.npz'
        with np.load(pool_path, allow_pickle=False) as data: pool = {k:data[k].copy() for k in data.files}
        slot_id = record['source_slot']; slot = contract.slots[slot_id]
        if (list(pool['independent_motor_names']) != list(contract.names)
                or not np.array_equal(pool['proposal_q'][0], slot['independent_q'])
                or not np.array_equal(pool['proposal_q_close'][0], slot['close_q'])):
            raise ValueError('Historical candidate endpoints differ from action contract')
        selected = entry['selected_history']
        execution = Path(selected['output_dir']) if selected else Path(catalog).parent/'execution'/candidate_id
        commands_path, trace_path, evaluation_path = [execution/n for n in ('commands.npz', 'trace.npz', 'evaluation.json')]
        evaluation = json.loads(evaluation_path.read_text())
        if sha(commands_path) != evaluation['commands_sha256'] or sha(trace_path) != evaluation['trace_sha256']:
            raise ValueError('Historical executed command/trace bytes differ from recorded evaluation')
        with np.load(commands_path, allow_pickle=False) as data: arrays = {k:data[k].copy() for k in data.files}
        with np.load(trace_path, allow_pickle=False) as data: trace = {k:data[k].copy() for k in data.files}
        source_frame = Path(generation['source_frame'])/'raw_camera_candidates.npz'
        if source_frame not in raw_cache:
            with np.load(source_frame, allow_pickle=False) as data: raw_cache[source_frame] = data['candidates'].copy()
            sources[str(source_frame)] = sha(source_frame)
        observation_path = Path(generation['source_observation'])
        if observation_path not in observation_cache:
            observation_cache[observation_path] = json.loads(observation_path.read_text())
            sources[str(observation_path)] = sha(observation_path)
        row = raw_cache[source_frame][record['source_candidate_index']].copy()
        # These saved old candidates explicitly bound deployment-style flip,
        # including forced flips; they are not silently called collection flips.
        if record['if_flip']: row[3:12].reshape(3, 3)[:, 1:3] *= -1
        T_WC = np.asarray(observation_cache[observation_path]['T_WC'])
        checked = contract.validate_command_arrays(arrays, slot_id=slot_id, candidate_row15=row,
            world_T_camera=T_WC, extra_depth_m=record['additional_depth_m'])
        grasp_palm = np.asarray(pool['proposal_palm'][0]); direction = np.asarray(pool['proposal_approach_direction'][0])
        recovered_approach = (grasp_palm@np.asarray(slot['palm_T_reference']))[:3, 2]
        if not np.allclose(direction, recovered_approach, atol=1e-8, rtol=0):
            raise ValueError('Saved physical approach is not mapped actual CGR +X')
        n = int(evaluation['total_physics_steps'])
        for key in ('joint_target', 'desired_palm', 'phase', 'time_s'):
            if not np.array_equal(trace[key], arrays[key][:n]): raise ValueError('Historical trace did not execute bound commands')
        names = list(trace['joint_names']); indices = [names.index(k) for k in contract.hand_names]
        observed = trace['joint_pos'][:, indices]
        actual_mimic_error = max(float(np.max(abs(observed[:, contract.hand_names.index(name)]
            -coefficient*observed[:, contract.hand_names.index(driver)]-offset)))
            for name, (driver, coefficient, offset) in contract.mimics.items())
        static_minimum = []
        for motors in contract.closure_targets(slot_id, np.linspace(0, 1, 11))['independent_target']:
            _, frames = _contact_points(contract, slot, motors)
            low = np.inf
            for name, mesh in meshes.items():
                world = grasp_palm@frames[name]
                low = min(low, float(np.min(mesh['vertices']@world[2, :3]+world[2, 3])))
            static_minimum.append(low)
        for path in (generation_path, pool_path, commands_path, trace_path, evaluation_path): sources[str(path)] = sha(path)
        historical.append(dict(candidate_id=candidate_id, gesture_id=slot['gesture_id'], slot_id=slot_id,
            execution_dir=str(execution), actual_physical_steps=n, originally_complete=evaluation['complete'],
            originally_success=evaluation['task_success'], legacy_flip_profile='dh3_deployment_or_explicit_override',
            command_audit=checked, actual_mimic_error_rad=actual_mimic_error,
            approach_direction_max_abs_error=float(np.max(abs(direction-recovered_approach))),
            source_mesh_min_table_height_at_11_free_closure_waypoints_m=static_minimum,
            static_closed_source_hand_intersects_table=bool(min(static_minimum) < 0),
            static_geometry_boundary='Exact source triangle vertices at 11 nominal free-closing waypoints; not continuous clearance, contacts, new physics or new outcomes'))
    flip_rows = np.zeros((3, 17)); flip_rows[:, 4:13] = np.eye(3).ravel(); flip_rows[:, 1] = .05
    flip_rows[1, 4:13] = axis_rotation([0, 0, 1], np.pi).ravel()
    flip_rows[2, 4:13] = axis_rotation([1, 0, 0], np.pi/2).ravel()
    flipped = {}; flips_report = {}
    for profile, (name, source_sha, columns) in SOURCE_FLIPS.items():
        transformed, did_flip = source_flip(flip_rows, profile); flipped[profile] = transformed
        flips_report[profile] = dict(source=name, source_sha256=source_sha, negated_rotation_columns=columns,
            input_gate='R[1,1] > 0', if_flip=did_flip.tolist(),
            approach_preserved_on_flip=columns != [0, 1],
            actual_approach_vectors=transformed[:, 4:13].reshape(-1, 3, 3)[:, :, 0].tolist())
        sources[str(ROOT/name)] = source_sha
    for path, digest in sources.items():
        if sha(path) != digest: raise ValueError('Input changed during audit: '+path)
    result = dict(schema='anydex_boya_author_action_semantics_audit_v1', status='complete',
        numerical_action_contract_pass=True, physical_semantics_validated=False, hardware_semantics_validated=False,
        source_pad_geometry=pad_surface, gesture_semantics=summaries, surface_library=dict(path=str(Path(surfaces).resolve()),
            sha256=sha(surfaces), states=len(library['states']), slot_bindings=len(library['slot_to_state']),
            all_21_links_including_inactive_retained=retained),
        source_flip_profiles=flips_report, historical_executions=historical,
        historical_summary=dict(executions=len(historical), hand_dof=20,
            bound_hand_target_scalars_checked=sum(r['command_audit']['steps']*20 for r in historical),
            recorded_physical_steps_read=sum(r['actual_physical_steps'] for r in historical),
            static_closure_table_intersections=sum(r['static_closed_source_hand_intersects_table'] for r in historical),
            actual_mimic_error_rad_max=max(r['actual_mimic_error_rad'] for r in historical)),
        pending_gates=['New collection profile integrated end-to-end with actual scene and canonical feature binding',
            'Dynamic full-body closure/contact and table admission under the new task',
            'Continuous observation-to-execution multi-object state binding',
            'Approach-axis retreat, transfer, release and all-object bin outcome'],
        changes_to_mapping_or_endpoints=False, old_labels_rewritten=False,
        new_physics_steps=0, new_CGR_forwards=0, real_training_updates=0,
        source_and_input_sha256=sources, audit_wall_s=time.monotonic()-started)
    (output/'ACTION_CONTRACTS.json').write_text(json.dumps(dict(schema='anydex_boya_action_contracts_v1', contracts=contracts), indent=2, allow_nan=False)+'\n')
    result['action_contracts'] = dict(path=str(output/'ACTION_CONTRACTS.json'), sha256=sha(output/'ACTION_CONTRACTS.json'))
    (output/'AUDIT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status', 'numerical_action_contract_pass', 'physical_semantics_validated', 'historical_summary', 'audit_wall_s')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--surfaces', type=Path, default=DEFAULT_SURFACES)
    parser.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.calibration, arguments.surfaces, arguments.catalog, output=arguments.output)
