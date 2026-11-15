"""Opt-in CPU audit for explicitly recorded GPU vector lift diagnostics.

Importing this module launches no simulator. Raw contact intervals and actor IDs
are decoded independently; no aggregate force or legacy primary success is used.
This diagnostic schema deliberately cannot produce a training-eligible record.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

import audit_author_collection_runtime as A

AUDIT_SCHEMA = 'anydex_boya_gpu_vector_audit_v1'
RAW_SCHEMA = 'anydex_boya_gpu_raw_contacts_v1'
DT = .004
EXECUTION_SCHEMA = 'anydex_boya_author_gpu_vector_execution_v1'


def need(value, reason):
    if not value:
        raise ValueError(reason)


def array(value, shape=None):
    result = np.asarray(value, dtype=float)
    need((shape is None or result.shape == shape) and np.isfinite(result).all(), 'Nonfinite or misaligned numeric array')
    return result


def local_poses(value, origin):
    result = array(value).copy()
    need(result.shape[-1:] == (7,), 'XYZW poses required')
    need(np.all(abs(np.linalg.norm(result[..., 3:], axis=-1)-1) <= 1e-4), 'Quaternion is not unit length')
    result[..., :3] -= array(origin, (3,))
    return result


def source_bottoms(vertices, poses_world, origin):
    """Every original source vertex; translation-only environment origins."""
    vertices = array(vertices)
    need(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices), 'Full source vertices required')
    poses = local_poses(poses_world, origin)
    need(poses.ndim == 2, 'A sequence of body poses required')
    axes = Rotation.from_quat(poses[:, 3:]).as_matrix()[:, 2, :]
    return np.asarray([np.min(vertices@axis)+pose[2] for axis, pose in zip(axes, poses)])


def decode_raw_contacts(frame, sensor_paths, actor_id_paths, actors, capacity):
    """Decode PhysX get_raw_contact_data(dt)'s original seven buffers.

    Starts index the native frame buffer and can contain unused gaps. Only
    referenced entries must be finite. Gaps are never repacked or reindexed.
    """
    need(type(capacity) is int and capacity > 0, 'Positive raw contact capacity required')
    need(frame.get('schema') == RAW_SCHEMA and frame.get('overflow') is False, 'Explicit complete raw-contact frame required')
    need(len(sensor_paths) == len(set(sensor_paths)) and set(sensor_paths) <= set(actors), 'Unique bound sensor owners required')
    force, point, normal, separation = [np.asarray(frame[key]) for key in ('force', 'point', 'normal', 'separation')]
    count, start, other = [np.asarray(frame[key]) for key in ('count', 'start', 'other_actor_ids')]
    if not len(force):
        # JSON cannot retain the trailing dimension of an empty (0, 3) array.
        if point.size == 0: point = point.reshape(0, 3)
        if normal.size == 0: normal = normal.reshape(0, 3)
        if other.size == 0: other = other.astype(np.uint64)
    need(count.shape == start.shape == (len(sensor_paths),) and count.dtype.kind in 'iu'
         and start.dtype.kind in 'iu' and np.all(count >= 0), 'Invalid native raw count/start table')
    total = sum(map(int, count))
    need(total < capacity, 'Raw contact capacity exhausted')
    need(force.ndim in (1, 2) and separation.ndim in (1, 2)
         and (force.ndim == 1 or force.shape[1:] == (1,))
         and (separation.ndim == 1 or separation.shape[1:] == (1,))
         and point.ndim == 2 and point.shape[1:] == (3,) and normal.shape == point.shape
         and other.ndim == 1 and other.dtype.kind in 'iu'
         and len({len(value) for value in (force, point, normal, separation, other)}) == 1,
         'Original seven raw buffer shapes required')
    result = []
    for sensor, number, first in zip(sensor_paths, count, start):
        first, number = int(first), int(number)
        if not number:
            continue
        need(first >= 0 and first+number <= len(force), 'Raw native interval outside saved buffer')
        for i in range(first, first+number):
            need(all(np.isfinite(value[i]).all() for value in (force, point, normal, separation)), 'Nonfinite referenced raw contact')
            key = str(int(other[i])); need(key in actor_id_paths, 'Unmapped raw other_actor_id')
            target = actor_id_paths[key]; need(target in actors, 'Unbound native actor path')
            left, right = actors[sensor], actors[target]
            need(left.get('env_index') is None or right.get('env_index') is None
                 or left['env_index'] == right['env_index'], 'Cross-environment physical contact')
            result.append(dict(bodies=[sensor, target], normal_force_n=float(force[i].item()),
                separation_m=float(separation[i].item()), position_world_m=point[i].tolist(),
                raw_normal_world=normal[i].tolist(), raw_buffer_index=i))
    need(len(result) == total, 'Raw contact accounting differs')
    return result


def normalize_path(path, environment_root):
    """Translate one explicitly bound lane, rejecting every other lane."""
    need(isinstance(path, str) and path.startswith('/World/'), 'Absolute native USD contact path required')
    if path.startswith(environment_root+'/'):
        return '/World/'+path[len(environment_root)+1:]
    need(not path.startswith('/World/envs/'), 'Cross-environment path in complete native contact stream')
    need(path.lower() == '/world/ground' or path.lower().startswith('/world/ground/'),
         'Unnamespaced non-ground native contact path')
    return path


def _normalized_setup(value, root, origin):
    result = copy.deepcopy(value)
    result['body_paths'] = {key: normalize_path(path, root) for key, path in value['body_paths'].items()}
    result['contact_targets'] = [normalize_path(path, root) for path in value['contact_targets']]
    need(any(path.startswith('/World/Robot/') for path in result['contact_targets']),
         'Namespaced full robot contact filters missing')
    for key, entry in result['entry'].items():
        entry['pose_xyzw'] = local_poses(entry['pose_xyzw'], origin).tolist()
    return result


def _projected_runtime(scene, plan, arrays, execution, evaluation, bindings, root, origin):
    """Reuse the frozen independent audit; raw artifacts never change.

    A private module instance prevents concurrent audits changing each other's
    globals. The only projections are entry XYZ and contact paths/positions;
    the persisted command, native/all-body trace, force and normal stay intact.
    """
    spec = importlib.util.spec_from_file_location('_gpu_lane_runtime_audit', A.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original_read, original_contacts = module.read, module.iter_contact_rows
    setup_path = (execution/'MULTI_OBJECT_SETUP.json').resolve()
    def read(path):
        result = original_read(path)
        return _normalized_setup(result, root, origin) if Path(path).resolve() == setup_path else result
    def contacts(path, ledger=None):
        for row in original_contacts(path, ledger):
            value = copy.deepcopy(row)
            for group in ('object_contacts', 'robot_contacts'):
                for contact in value[group]:
                    contact['bodies'] = [normalize_path(path, root) for path in contact['bodies']]
                    contact['position_world_m'] = (array(contact['position_world_m'], (3,))-origin).tolist()
            yield value
    module.read, module.iter_contact_rows = read, contacts
    # Symlinks preserve the original audit's fixed filenames. Bindings.add()
    # resolves them to the real files, so no temporary names escape the audit.
    with tempfile.TemporaryDirectory(prefix='anydex_gpu_lane_audit_') as tmp:
        view = Path(tmp)
        for name in ('MULTI_OBJECT_SETUP.json', 'MULTI_OBJECT_TRACE.json',
                     'MULTI_OBJECT_CONTACTS.json', 'evaluation.json', 'trace.npz'):
            source = (execution/name).resolve(); need(source.is_file(), 'Missing original native GPU artifact: '+name)
            (view/name).symlink_to(source)
        return module.audit_runtime(scene, plan, arrays, view, evaluation, bindings)


def _verify_lane_state(execution, plan, arrays, origin, bindings):
    """Additional original native state/material/force evidence, no integration."""
    native = A.read(execution/'evaluation.json'); setup = A.read(execution/'MULTI_OBJECT_SETUP.json')
    structure_path = execution/'native_structure.json'; bindings.add(structure_path); structure = A.read(structure_path)
    need(structure.get('is_fixed_base') is True and structure.get('num_base_dofs') == 0,
         'Actual native fixed-base structure missing')
    need(native.get('device') in ('cuda:0', 'cuda') and native.get('dt_s') == DT
         and native.get('fixed_base') is True and native.get('independent_coordinates') == 24
         and native.get('passive_mimics') == 4, 'Actual GPU native structure/device/timestep changed')
    need(native.get('object_mass_kg') == float(np.float32(.1)) and native.get('object_friction') == .6
         and native.get('global_gravity_m_s2') == [0., 0., 0.] and native.get('robot_gravity_disabled') is True
         and native.get('object_weight_world_n') == [0., 0., -.98], 'Original primary material/weight evidence missing')
    need(np.array_equal(array(native.get('environment_origin_m'), (3,)), origin), 'Native origin echo changed')
    primary = setup['body_paths']; target = A.read(execution/'LIFT_EVALUATION.json')['legacy_target_id']
    need(target in primary and set(setup.get('additional_materials', {})) == set(primary)-{target},
         'Every additional body material/weight readback required')
    for material in setup['additional_materials'].values():
        need(material.get('mass_kg') == .1 and material.get('friction') == .6
             and material.get('native_readback_ok') is True and material.get('world_weight_n') == [0., 0., -.98],
             'Additional native body material/weight readback changed')
    with np.load(execution/'trace.npz', allow_pickle=False) as data:
        trace = {key: data[key].copy() for key in data.files}
    n = len(arrays['phase']); names = list(map(str, trace['body_names']))
    poses = array(trace['body_pose_xyzw'], (n, len(names), 7))
    need(np.all(abs(np.linalg.norm(poses[..., 3:], axis=-1)-1) <= 1e-4), 'Actual native body quaternion invalid')
    array(trace['joint_velocity'], (n, 28)); velocity = array(trace['joint_velocity_target'], (n, 28))
    feedforward = native['arm_velocity_feedforward']; arm_names = list(map(str, arrays['joint_names'][:6]))
    need(feedforward.get('enabled') is True and feedforward.get('hand_velocity_targets_zero') is True
         and len(set(feedforward['joints'])) == 6 and set(feedforward['joints']) == set(arm_names),
         'Original six-arm-only velocity feedforward changed')
    columns = [list(map(str, trace['joint_names'])).index(name) for name in arm_names]
    expected_velocity = np.zeros_like(trace['joint_target'], dtype=np.float32)
    expected_velocity[:, columns] = np.gradient(np.asarray(trace['joint_target'], float)[:, columns], DT, axis=0, edge_order=2)
    need(np.array_equal(velocity, expected_velocity), 'Actual native arm/hand velocity targets changed')
    offset = array(native['palm_owner_to_frame'], (4, 4))
    # Independently reconstruct the fixed palm attachment from source XML.
    fixed = {j.find('child').get('link'): j for j in ET.parse(plan['arguments']['robot_urdf']).getroot().findall('joint')
             if j.get('type') == 'fixed'}
    owner = 'palm'; expected = np.eye(4); seen = set()
    while owner in fixed:
        need(owner not in seen, 'Cyclic fixed palm attachment'); seen.add(owner); joint = fixed[owner]
        node = joint.find('origin'); transform = np.eye(4)
        if node is not None:
            transform[:3, 3] = [float(v) for v in node.get('xyz', '0 0 0').split()]
            transform[:3, :3] = Rotation.from_euler('xyz', [float(v) for v in node.get('rpy', '0 0 0').split()]).as_matrix()
        expected = transform@expected; owner = joint.find('parent').get('link')
    need(native.get('palm_rigid_owner') == owner and owner in names and np.max(abs(expected-offset)) <= 1e-12,
         'Native palm fixed-frame readback differs from source XML')
    palm = poses[:, names.index(owner)]; rotations = Rotation.from_quat(palm[:, 3:]).as_matrix()
    actual_palm_position = np.einsum('nij,j->ni', rotations, offset[:3, 3])+palm[:, :3]
    actual_palm_rotation = rotations@offset[:3, :3]
    wanted = array(arrays['desired_palm'], (n, 4, 4))
    position_error = float(np.linalg.norm(actual_palm_position-wanted[:, :3, 3], axis=1).max())
    angle_error = float(Rotation.from_matrix(actual_palm_rotation@wanted[:, :3, :3].transpose(0, 2, 1)).magnitude().max())
    base = poses[:, 0]; base_position = float(np.linalg.norm(base[:, :3]-base[0, :3], axis=1).max())
    base_angle = float((Rotation.from_quat(base[:, 3:])*Rotation.from_quat(base[0, 3:]).inv()).magnitude().max())
    need(position_error <= .03 and angle_error <= np.deg2rad(15) and base_position <= 2e-5 and base_angle <= 2e-5,
         'Measured GPU palm tracking/base stability gate failed')
    return dict(max_palm_position_error_m=position_error, max_palm_angle_error_rad=angle_error,
        max_base_drift_from_first_tick_m=base_position, max_base_angle_from_first_tick_rad=base_angle,
        full_all_body_material_and_weight_readbacks=True,
        force_scope='Native applied external-force commands/readback; no independent force sensor',
        base_scope='Independent all-tick stability from first recorded tick plus original native cold-entry/fixed-base gate')


def _probe_evidence(group, bindings, roots):
    """Verify the two saved native snapshots, retaining interval gaps."""
    result = {}
    for key in ('initial_gpu_probe', 'first_tick_gpu_probe'):
        probe = group[key]; bindings.tree(probe)
        need(probe.get('status') == 'live_api_snapshot_pass' and probe.get('errors') == []
             and probe.get('physics_steps_added') == 0 and probe.get('disable_contact_processing') is False
             and probe.get('simulation_view_device_ordinal') == 0
             and probe.get('physics_clock_before') == probe.get('physics_clock_after'), 'Actual GPU contact/state snapshot failed')
        expected = {f'env_{i}_{role}' for i in range(len(roots)) for role in ('objects', 'robot')}
        need(set(probe['contacts']) == expected, 'Every environment object/robot contact snapshot required')
        with np.load(probe['snapshot']['path'], allow_pickle=False) as z:
            arrays = {name: z[name].copy() for name in z.files}
        total = 0
        for name, meta in probe['contacts'].items():
            i = int(name.split('_')[1]); root = roots[i]
            need(meta.get('status') == 'pass' and meta.get('mode') == 'filtered' and meta.get('dt_s') == DT
                 and meta.get('contact_fields_dropped') is False and meta.get('saturation_detected') is False,
                 'Original lossless filtered6 CUDA snapshot required')
            need(len(meta['tensors']) == 6 and all('cuda' in t['device'].lower() for t in meta['tensors']),
                 'Contact snapshot is not native CUDA')
            for path in meta['sensor_paths']: normalize_path(path, root)
            def paths(value):
                if isinstance(value, str): normalize_path(value, root)
                else:
                    need(isinstance(value, list), 'Saved native filter path table malformed')
                    for item in value: paths(item)
            paths(meta['filter_paths'])
            payload = [arrays[f'contact_{name}_{j}'] for j in range(6)]
            counts, starts = payload[4:]; capacity = meta['capacity']
            need(counts.shape == starts.shape == (meta['sensor_count'], meta['filter_count'])
                 and counts.dtype.kind in 'iu' and starts.dtype.kind in 'iu'
                 and np.all(counts >= 0) and np.all(starts >= 0), 'Saved filtered count/start changed')
            count = sum(map(int, counts.flat)); need(count < capacity and count == meta['active_contact_records'], 'Saved contact capacity/count changed')
            for values, width in zip(payload, (1, 3, 3, 1)):
                need(values.shape == (capacity, width), 'Saved native contact payload shape changed')
            endpoint = 0
            for count_i, start_i in zip(counts.flat, starts.flat):
                start_i, count_i = int(start_i), int(count_i)
                if not count_i: continue
                endpoint = max(endpoint, start_i+count_i)
                need(start_i+count_i <= capacity, 'Saved native contact interval exceeds capacity')
                for values, width in zip(payload, (1, 3, 3, 1)):
                    need(values.shape == (capacity, width) and np.isfinite(values[start_i:start_i+count_i]).all(),
                         'Saved native contact payload shape/finite gate failed')
            need(endpoint == meta['largest_referenced_endpoint'], 'Saved contact endpoint changed')
            total += count
        result[key] = dict(active_contact_records=total, native_cuda_views=len(expected), buffers_sha256=probe['snapshot']['sha256'])
    return result


def audit_execution(group_path, split_manifest=None):
    """Audit the actual adapter's filtered6 output, with no new runtime schema."""
    began = time.monotonic(); path = Path(group_path).resolve(); bindings = A.Bindings()
    group = A.read(path); bindings.add(path); bindings.add(__file__); bindings.add(A.__file__)
    need(group.get('schema') == EXECUTION_SCHEMA and group.get('exclude_training') is True
         and group.get('label') is None and group.get('formal_training_eligible') is False
         and group.get('automatic_training') is False and group.get('retry_count') == 0,
         'Explicit diagnostic-only GPU execution required')
    bindings.mapping(group['source_sha256']); bindings.tree(group.get('protocol'))
    need(group.get('status') == 'complete' and group.get('complete') is True, 'GPU execution incomplete; no certifiable binary label')
    jobs = group['jobs']; roots = group['environment_roots']; origins = array(group['origins_m'], (len(jobs), 3))
    need(len(jobs) in (1, 2) and len(roots) == len(jobs) and len(set(roots)) == len(jobs)
         and roots == ['/World/envs/env_'+str(i) for i in range(len(jobs))]
         and np.array_equal(origins[0], np.zeros(3)) and np.all(origins[:, 2] == 0)
         and (len(jobs) == 1 or np.linalg.norm(origins[1, :2]) >= 4.), 'Frozen one/two environment layout changed')
    native = group['native_scene']; attrs = native['native_scene']
    need(group.get('device') == 'cuda:0' and native.get('actual_gpu_validated') is True
         and native.get('asset_setup_count') == native.get('physics_scene_count') == 1
         and native.get('cross_environment_contacts') == 0
         and attrs.get('physxScene:enableGPUDynamics') is True and attrs.get('physxScene:broadphaseType') == 'GPU'
         and attrs.get('physxScene:enableCCD') is False and attrs.get('physxScene:timeStepsPerSecond') == 250,
         'Actual shared GPU scene/isolation evidence failed')
    probes = _probe_evidence(group, bindings, roots)
    bindings.tree(group.get('tail_observations', {}))
    for tail in group.get('tail_observations', {}).values():
        pointer = A.read(tail['contacts']['path']); bindings.tree(pointer)
        need(pointer.get('lossless') is True and pointer.get('contact_records_omitted') == 0
             and Path(pointer['data']['path']).stat().st_size == pointer['data']['bytes'], 'Tail lossless contact data changed')
    reports = []; lengths = []
    seen_sources = set(); scene_sha = None
    for i, job in enumerate(jobs):
        origin = origins[i]; root = roots[i]
        need(job.get('environment') == i and job.get('environment_root') == root
             and np.array_equal(array(job['origin_m'], (3,)), origin)
             and job.get('trace_frame') == 'Original plan world frame'
             and job.get('contact_position_frame') == 'Physical translated environment world'
             and np.array_equal(array(job['source_pose_translation_removed_m'], (3,)), origin),
             'Per-lane frame metadata changed')
        need(job.get('exclude_training') is True and job.get('formal_training_eligible') is False
             and job.get('label') is None, 'Diagnostic lane accidentally admitted to training')
        bindings.tree(job['input_bindings']); bindings.tree(job['evaluation']); bindings.tree(job['diagnostic_record'])
        source = Path(job['source_prepared']).resolve(); ready = A.read(source/'READY.json')
        need(source not in seen_sources, 'A source preparation was executed twice in this GPU batch'); seen_sources.add(source)
        if scene_sha is None: scene_sha = ready['scene']['sha256']
        need(ready['scene']['sha256'] == scene_sha, 'GPU diagnostic source cohort must use one identical measured scene')
        need(job['input_bindings']['ready'] == dict(path=str(source/'READY.json'), sha256=A.sha(source/'READY.json')),
             'Source READY path/SHA changed')
        for name in ('protocol', 'scene', 'candidate', 'plan', 'commands', 'command_binding'):
            need(job['input_bindings'][name] == ready[name], 'Actual source binding differs: '+name)
        preparation = A.read(job['input_bindings']['preparation']['path'])
        need(preparation.get('status') == 'ready' and preparation['ready_sha256'] == A.sha(source/'READY.json'),
             'Original preparation changed')
        protocol = A.read(ready['protocol']['path']); scene = A.read(ready['scene']['path'])
        need(protocol.get('task') == A.TASK and protocol.get('bin', 'missing') is None
             and protocol.get('candidate_budget') == 1 and protocol.get('fallback_candidate') is False
             and protocol.get('primary_is_candidate_target') is False, 'Frozen first-candidate protocol changed')
        for field in ('inputs', 'sources', 'capture_source_bindings'): bindings.tree(protocol.get(field, {}))
        for name, digest in scene.get('source_sha256', {}).items():
            bindings.add(protocol.get('scene_archive_sources', {}).get(str(Path(name).resolve()), name), digest)
        bindings.mapping(scene.get('input_sha256', {}))
        ids = [o['object_id'] for o in scene['objects']]
        need(len(ids) == len(set(ids)) == 3 and scene.get('original_object_ids') == ids
             and scene.get('task') == A.TASK and scene.get('bin', 'missing') is None and scene.get('dt_s') == DT,
             'Original three-body bin-free scene changed')
        identity = []
        for obj in scene['objects']:
            bindings.add(obj['source_mesh'], obj['source_mesh_sha256'])
            bindings.add(obj['source_visual_mesh'], obj['source_visual_mesh_sha256'])
            need(obj.get('mass_kg') == .1 and obj.get('friction') == .6 and np.isfinite(obj['scale']) and obj['scale'] > 0,
                 'Source body material/scale changed')
            identity.append(obj['source_visual_mesh_sha256'])
        need(scene['all_source_mesh_sha256'] == identity, 'Whole-scene original identity changed')
        split = None
        if split_manifest is not None: _, _, _, split = A._identity(scene, split_manifest, bindings)
        plan = A.read(ready['plan']['path']); candidate = A.read(ready['candidate']['path']); command = A.read(ready['command_binding']['path'])
        for field in ('source_sha256', 'input_sha256', 'extra_input_sha256'): bindings.mapping(plan.get(field, {}))
        arrays, _, _, counts = A._commands(source, ready, plan, candidate, command, bindings)
        n = len(arrays['phase']); lengths.append(n)
        need(command['execution_spec'].get('original_object_ids') == ids, 'Command omitted original scene body')
        evaluation = A.read(job['evaluation']['path']); execution = Path(job['evaluation']['path']).resolve().parent
        need(execution == path.parent/f'env_{i}'/'execution', 'Lane evaluation escaped its new GPU output directory')
        need(evaluation.get('task') == A.TASK and evaluation.get('complete') is True and evaluation.get('label_eligible') is True
             and not evaluation.get('error') and not evaluation.get('physical_stop')
             and all(evaluation.get('gates', {}).get(key) is True for key in A.GATES)
             and evaluation.get('original_object_ids') == ids and evaluation.get('completed_phases') == A.PHASES
             and evaluation.get('legacy_primary_task_success_used') is False and evaluation.get('state_writes_during_execution') == 0
             and evaluation.get('bin_geometry_count') == 0 and evaluation.get('scene_sha256') == ready['scene']['sha256']
             and evaluation.get('commands_sha256') == ready['commands']['sha256']
             and evaluation.get('collection_candidate_sha256') == candidate['candidate_sha256'], 'Actual GPU evaluation binding/validity changed')
        record = A.read(job['diagnostic_record']['path'])
        need(record.get('candidate') == candidate and record.get('command_binding') == command and record.get('task') == A.TASK
             and record.get('evaluation_binding') == job['evaluation'] and record.get('label') is None
             and record.get('formal_training_eligible') is False and record.get('exclude_training') is True,
             'Diagnostic record differs from executed source/evaluation')
        bindings.tree(record['evaluation_binding'])
        numeric = _projected_runtime(scene, plan, arrays, execution, evaluation, bindings, root, origin)
        state = _verify_lane_state(execution, plan, arrays, origin, bindings)
        label = int(any(numeric['object_success'].values()))
        need(type(job.get('raw_label')) is int and job['raw_label'] == label
             and type(record.get('diagnostic_raw_label')) is int and record['diagnostic_raw_label'] == label
             and job.get('complete') is True and job.get('gates') == evaluation['gates']
             and job.get('command_steps') == job.get('completion_tick') == n,
             'Raw diagnostic label/completion differs from independent all-body audit')
        reports.append(dict(environment=i, environment_root=root, origin_m=origin.tolist(), diagnostic_label=label,
            label=None, formal_training_eligible=False, exclude_training=True, original_object_ids=ids,
            all_scene_source_mesh_sha256=identity, split=split, candidate_sha256=candidate['candidate_sha256'],
            commands_sha256=ready['commands']['sha256'], phase_counts=counts, runtime=numeric, native_state=state))
    ticks = max(lengths)
    need(group['physical_clock_steps'] == ticks and group['recorded_command_environment_steps'] == sum(lengths)
         and group['actual_integrated_environment_steps'] == ticks*len(jobs)
         and [j['tail_integrated_steps'] for j in jobs] == [ticks-n for n in lengths],
         'Shared clock/tail integration cost accounting changed')
    bindings.verify()
    return dict(schema=AUDIT_SCHEMA, status='pass', label=None, formal_training_eligible=False, exclude_training=True,
        environments=reports, physical_clock_steps=ticks, recorded_command_environment_steps=sum(lengths),
        actual_integrated_environment_steps=ticks*len(jobs), gpu_snapshots=probes, audit_inputs_sha256=bindings.files,
        audit_source=dict(path=str(Path(__file__).resolve()), sha256=A.sha(__file__)), wall_s=time.monotonic()-began,
        audit_Kit_launches=0, audit_physics_steps=0, audit_model_forwards=0,
        normalization_scope='Read-only in-memory lane path/entry-XYZ/contact-XYZ projection; original files and all other fields preserved',
        contact_scope='Every original lossless filtered6 contact decoded once per lane; separate live cross-environment view is bound runtime evidence; raw7 full-tick capture is not claimed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    need(not args.output.exists(), 'Fresh audit output required')
    try:
        report = audit_execution(args.execution, args.split_manifest)
    except Exception:
        report = dict(schema=AUDIT_SCHEMA, status='failed', label=None, formal_training_eligible=False,
            error=traceback.format_exc(), audit_Kit_launches=0, audit_physics_steps=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ('status', 'label', 'formal_training_eligible')}))
    if report['status'] != 'pass': raise SystemExit(1)


if __name__ == '__main__':
    main()
