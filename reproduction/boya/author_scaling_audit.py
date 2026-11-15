"""Read-only audit of the explicitly bounded CPU/GPU scaling comparison.

No simulator or model is imported. Prefix observations certify data integrity
and measured throughput, never completed lifts, labels or collection ETA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

import audit_author_collection_runtime as A
import author_gpu_vector_audit as G

SCHEMA = 'anydex_boya_author_scaling_audit_v1'
WORKER_SCHEMA = 'anydex_boya_author_scaling_execution_v1'
CONTROL_SCHEMA = 'anydex_boya_author_scaling_control_v1'
GROUPS = [('gpu', n) for n in (1, 4, 8)]+[('cpu', n) for n in (1, 4, 8)]
DT = .004
need = A.need


class Bindings(A.Bindings):
    """Hash shared dependencies once, and every unique input again at the end."""
    def add(self, path, expected=None):
        path = str(Path(path).resolve())
        if path in self.files:
            need(expected is None or expected == self.files[path], 'Conflicting bound SHA: '+path)
            return dict(path=path, sha256=self.files[path])
        return super().add(path, expected)


def interval_stats(intervals):
    """CPU worker windows may overlap; their durations cannot be summed."""
    values = [tuple(map(float, value)) for value in intervals]
    need(values and all(np.isfinite(v).all() and v[1] > v[0] for v in values),
         'Finite positive prefix intervals required')
    events = sorted((x, delta) for start, end in values for x, delta in ((start, 1), (end, -1)))
    previous = events[0][0]; active = peak = 0; union = all_overlap = 0.
    for position, delta in events:
        width = position-previous
        if active: union += width
        if active == len(values): all_overlap += width
        active += delta; peak = max(peak, active); previous = position
    span = max(end for _, end in values)-min(start for start, _ in values)
    return dict(span_s=span, union_s=union, idle_s=span-union, peak_concurrency=peak,
                all_worker_overlap_s=all_overlap, worker_duration_sum_s=sum(e-s for s, e in values))


def comparison_metrics(mode, count, workers, group_wall_s):
    need((mode, count) in GROUPS and len(workers) == (1 if mode == 'gpu' else count),
         'Frozen GPU lanes / CPU worker count changed')
    steps = sum(w['environment_steps'] for w in workers)
    need(steps == count*100 and np.isfinite(group_wall_s) and group_wall_s > 0,
         'Exactly 100 observed ticks per lane and a positive outer group wall required')
    windows = [(w['prefix_started_unix_s'], w['prefix_finished_unix_s']) for w in workers]
    intervals = interval_stats(windows)
    # GPU's monotonic duration avoids subtracting large Unix timestamps. CPU
    # spans combine processes on the same host; Unix endpoint span is explicit.
    prefix_wall = workers[0]['prefix_wall_s'] if mode == 'gpu' else intervals['span_s']
    return dict(mode=mode, count=count, environment_steps=steps,
        physics_clock_steps=sum(w['physics_clock_steps'] for w in workers),
        prefix_wall_s=prefix_wall, prefix_envsteps_per_s=steps/prefix_wall,
        outer_group_wall_s=group_wall_s, startup_inclusive_envsteps_per_s=steps/group_wall_s,
        prefix_intervals=intervals,
        scope='Bounded execution prefix including generator close; group wall includes process startup/close. No lift, full collection or ETA claim.')


def resource_summary(samples, group, workers, gpu_uuid):
    """Bounded host samples supplement execution intervals, never replace them."""
    selected = [r for r in samples if group['started_unix_s'] <= r['unix_s'] <= group['finished_unix_s']]
    pids = {w['worker_pid'] for w in workers}; histories = {pid: [] for pid in pids}
    peak = rss_peak = 0; utilization = []; memory = []; power = []; errors = 0
    for row in selected:
        need(np.isfinite(row['unix_s']) and row['cpu_clock_ticks_per_second'] > 0, 'Invalid host sampling clock')
        owned = [v for v in row['owned_workers'] if v['process']['pid'] in pids]
        need(len({v['process']['pid'] for v in owned}) == len(owned), 'Duplicate sampled worker identity')
        peak = max(peak, len(owned)); rss_peak = max(rss_peak, sum(v['rss_bytes'] for v in owned))
        for value in owned:
            need(value['rss_bytes'] >= 0 and value['utime_ticks'] >= 0 and value['stime_ticks'] >= 0,
                 'Invalid host process resource counter')
            histories[value['process']['pid']].append((value['process'], value['utime_ticks']+value['stime_ticks'], row['cpu_clock_ticks_per_second']))
        if 'gpu0_nvml_csv' in row:
            values = [x.strip() for x in row['gpu0_nvml_csv'].split(',')]
            need(len(values) == 4 and values[0] == gpu_uuid, 'Host sample escaped the frozen physical GPU')
            for target, value in zip((utilization, memory, power), values[1:]):
                try: number = float(value)
                except ValueError: continue  # NVML can explicitly report N/A.
                need(np.isfinite(number) and number >= 0, 'Invalid GPU resource sample')
                target.append(number)
        else: errors += 1
    captured_cpu_s = 0.
    for history in histories.values():
        if len(history) < 2: continue
        need(all(v[0] == history[0][0] and v[2] == history[0][2] for v in history), 'Sampled PID identity or clock changed')
        need(all(b[1] >= a[1] for a, b in zip(history, history[1:])), 'Process CPU counter regressed')
        captured_cpu_s += (history[-1][1]-history[0][1])/history[0][2]
    def stats(values):
        return None if not values else dict(min=min(values), mean=sum(values)/len(values), max=max(values))
    return dict(sample_count=len(selected), sampled_peak_live_owned_workers=peak,
        sampled_peak_sum_worker_rss_bytes=rss_peak, captured_worker_cpu_s=captured_cpu_s,
        gpu_utilization_percent=stats(utilization), gpu_memory_used_mib=stats(memory), gpu_power_w=stats(power),
        gpu_sampling_error_count=errors,
        scope='Two-second host snapshots. CPU seconds cover only each PID first-to-last captured counter; startup/exit edges are excluded. NVML covers the whole device and cannot attribute utilization to these workers alone. Sampled occupancy can miss short overlaps.')


def _snapshot_probes(group, bindings, mode, roots):
    reports = {}
    for key in ('initial_gpu_probe', 'first_tick_gpu_probe'):
        probe = group[key]; bindings.tree(probe)
        need(probe['status'] == 'live_api_snapshot_pass' and probe['errors'] == []
             and probe['physics_steps_added'] == 0 and probe['disable_contact_processing'] is False
             and probe['simulation_view_device_ordinal'] == (0 if mode == 'gpu' else -1)
             and probe['physics_clock_before'] == probe['physics_clock_after'], 'Native backend probe failed')
        expected = {f'env_{i}_{role}' for i in range(len(roots)) for role in ('objects', 'robot')}
        need(set(probe['contacts']) == expected, 'Every lane full object/robot snapshot required')
        with np.load(probe['snapshot']['path'], allow_pickle=False) as z:
            saved = {k: z[k].copy() for k in z.files}
        total = 0
        for name, meta in probe['contacts'].items():
            root = roots[int(name.split('_')[1])]
            need(meta['status'] == 'pass' and meta['mode'] == 'filtered' and meta['dt_s'] == DT
                 and meta['contact_fields_dropped'] is False and meta['saturation_detected'] is False,
                 'Original lossless filtered6 probe required')
            devices = [str(x['device']).lower() for x in meta['tensors']]
            need(len(devices) == 6 and all(('cuda' in d) if mode == 'gpu' else d == 'cpu' for d in devices),
                 'Requested actual native contact device changed')
            def paths(value):
                if isinstance(value, str): G.normalize_path(value, root)
                else:
                    need(isinstance(value, list), 'Native path table malformed')
                    for item in value: paths(item)
            paths(meta['sensor_paths']); paths(meta['filter_paths'])
            payload = [saved[f'contact_{name}_{i}'] for i in range(6)]
            counts, starts = payload[4:]; capacity = meta['capacity']
            need(counts.shape == starts.shape == (meta['sensor_count'], meta['filter_count'])
                 and counts.dtype.kind in 'iu' and starts.dtype.kind in 'iu'
                 and (counts >= 0).all() and (starts >= 0).all(), 'Native count/start invalid')
            number = sum(map(int, counts.flat)); endpoint = 0
            need(number < capacity and number == meta['active_contact_records'], 'Contact capacity/count changed')
            for values, width in zip(payload, (1, 3, 3, 1)):
                need(values.shape == (capacity, width), 'Saved native payload shape changed')
            for count, start in zip(counts.flat, starts.flat):
                if not count: continue
                end = int(start)+int(count); endpoint = max(endpoint, end)
                need(end <= capacity and all(np.isfinite(v[int(start):end]).all() for v in payload[:4]),
                     'Referenced native interval missing/nonfinite')
            need(endpoint == meta['largest_referenced_endpoint'], 'Saved native endpoint changed')
            total += number
        reports[key] = dict(active_contact_records=total, native_views=len(expected),
                            snapshot_sha256=probe['snapshot']['sha256'], actual_device=mode)
    return reports


def _source_commands(job, bindings):
    source = Path(job['source_prepared']); ready = A.read(source/'READY.json')
    bindings.add(source/'READY.json'); bindings.tree(job['input_bindings'])
    for key in ('protocol', 'scene', 'candidate', 'plan', 'commands', 'command_binding'):
        need(ready[key] == job['input_bindings'][key], 'Original READY input changed: '+key)
    protocol = A.read(ready['protocol']['path'])
    for key in ('inputs', 'sources', 'capture_source_bindings'): bindings.tree(protocol.get(key, {}))
    scene, plan, candidate, command = [A.read(ready[k]['path']) for k in ('scene', 'plan', 'candidate', 'command_binding')]
    for key in ('source_sha256', 'input_sha256', 'extra_input_sha256'): bindings.mapping(plan.get(key, {}))
    for path, digest in scene.get('source_sha256', {}).items():
        bindings.add(protocol.get('scene_archive_sources', {}).get(str(Path(path).resolve()), path), digest)
    bindings.mapping(scene.get('input_sha256', {}))
    for obj in scene['objects']:
        bindings.add(obj['source_mesh'], obj['source_mesh_sha256'])
        bindings.add(obj['source_visual_mesh'], obj['source_visual_mesh_sha256'])
    arrays, _, _, _ = A._commands(source, ready, plan, candidate, command, bindings)
    return ready, scene, plan, candidate, command, arrays


def _lane(job, group_root, bindings, mesh_cache, ticks, mode):
    ready, scene, plan, candidate, command, arrays = _source_commands(job, bindings)
    execution = group_root/f"env_{job['environment']}/execution"
    bindings.tree(job['evaluation']); bindings.tree(job['diagnostic_record'])
    evaluation = A.read(job['evaluation']['path']); record = A.read(job['diagnostic_record']['path'])
    need(evaluation['total_physics_steps'] == ticks and evaluation['dynamic_object_steps'] == 3*ticks
         and evaluation['complete'] is False and evaluation['label_eligible'] is False
         and evaluation['collection_candidate_sha256'] == candidate['candidate_sha256']
         and evaluation['commands_sha256'] == ready['commands']['sha256'], 'Partial lane counts/eligibility/command echo changed')
    need(record['label'] is None and record['diagnostic_raw_label'] is None
         and record['formal_training_eligible'] is False and record['exclude_training'] is True,
         'An incomplete prefix cannot have a binary label')
    need(record['candidate'] == candidate and record['command_binding'] == command
         and record['evaluation_binding'] == job['evaluation'], 'Diagnostic candidate/command binding changed')
    need(not (execution.parent/'COLLECTION_RECORD.json').exists(), 'Diagnostic produced a training record')
    ids = [o['object_id'] for o in scene['objects']]
    need(len(ids) == len(set(ids)) == 3 and evaluation['original_object_ids'] == ids, 'All original bodies required')
    setup_path = execution/'MULTI_OBJECT_SETUP.json'; bindings.add(setup_path)
    setup = G._normalized_setup(A.read(setup_path), job['environment_root'], np.asarray(job['origin_m']))
    need(setup['native']['within_20um'] is True and setup['physical_state_writes_during_execution'] == 0
         and setup['global_gravity_m_s2'] == [0., 0., 0.]
         and setup['every_dynamic_body_weight_n'] == [0., 0., -.98], 'Original native geometry/force/no-reset evidence changed')
    joints, legal = A._robot_xml(plan, bindings)
    need(set(setup['legal_hand_rigid_owners']) == legal, 'Original legal hand-owner set changed')
    vertices = {}
    for obj in scene['objects']:
        pose = np.asarray(setup['entry'][obj['object_id']]['pose_xyzw']); desired = np.asarray(obj['world_T_object'])
        need(np.max(abs(pose[:3]-desired[:3, 3])) <= 2e-5
             and np.max(abs(Rotation.from_quat(pose[3:]).as_matrix()-desired[:3, :3])) <= 1e-4
             and np.max(abs(np.asarray(setup['entry'][obj['object_id']]['velocity']))) <= 2e-5,
             'Cold-entry pose/velocity differs from source observation after origin projection')
        key = (obj['source_mesh_sha256'], obj['scale'])
        if key not in mesh_cache:
            import trimesh
            mesh_cache[key] = A._array(trimesh.load_mesh(obj['source_mesh'], process=False).vertices)*obj['scale']
        vertices[obj['object_id']] = mesh_cache[key]
    trace_path = execution/'MULTI_OBJECT_TRACE.json'; contacts_path = execution/'MULTI_OBJECT_CONTACTS.json'
    bindings.add(trace_path, evaluation['trace_sha256']); bindings.add(contacts_path, evaluation['contacts_sha256'])
    rows = A.read(trace_path)['rows']; need(len(rows) == ticks, 'All-body prefix trace length changed')
    paths = setup['body_paths']; reverse = {v: k for k, v in paths.items()}
    robot = {p for p in setup['contact_targets'] if p.startswith('/World/Robot/')}
    point_count = 0; max_bottom_error = max_load_error = 0.
    stream = A.iter_contact_rows(contacts_path, bindings)
    for index, row in enumerate(rows):
        need(row['step'] == index+1 and row['phase'] == str(arrays['phase'][index])
             and abs(row['time_s']-(index+1)*DT) <= 1e-12 and set(row['objects']) == set(ids), 'All-body prefix clock/identity changed')
        contacts = next(stream, None); need(isinstance(contacts, dict) and contacts['step'] == index+1, 'One complete contact row per observed tick required')
        loads = {k: np.zeros(5) for k in ids}; totals = dict.fromkeys(ids, 0.); illegal = dict.fromkeys(ids, 0.)
        for category in ('object_contacts', 'robot_contacts'):
            need(isinstance(contacts.get(category), list), 'Both complete contact streams required')
            for contact in contacts[category]:
                pair = [G.normalize_path(p, job['environment_root']) for p in contact['bodies']]
                need(len(pair) == 2, 'Two physical body paths required')
                A._array(contact['position_world_m'], (3,))
                if category == 'object_contacts': A._array(contact['raw_normal_world'], (3,))
                force, separation = contact['normal_force_n'], contact['separation_m']
                need(np.isfinite(force) and np.isfinite(separation), 'Nonfinite contact data')
                point_count += 1
                if category == 'object_contacts':
                    need(pair[0] in reverse and pair[1] in setup['contact_targets'], 'Unbound contact actor')
                    if pair[1] in robot:
                        key = reverse[pair[0]]; owner = pair[1].rsplit('/', 1)[-1]; magnitude = abs(force)
                        totals[key] += magnitude
                        for f, prefix in enumerate(('th', 'ff', 'mf', 'rf', 'lf')):
                            if owner.startswith(prefix): loads[key][f] += magnitude
                        if owner not in legal: illegal[key] = max(illegal[key], magnitude)
        for key in ids:
            state = row['objects'][key]; pose = A._array(state['pose_xyzw'], (7,)); A._array(state['velocity'], (6,))
            need(abs(np.linalg.norm(pose[3:])-1) < 1e-4, 'Invalid original body quaternion')
            bottom = float((vertices[key]@Rotation.from_quat(pose[3:]).as_matrix()[2]+pose[2]).min())
            error = abs(bottom-state['source_bottom_m']); max_bottom_error = max(max_bottom_error, error)
            need(error <= 1e-9, 'Full source vertex bottom differs from saved original body metric')
            error = float(np.max(abs(A._array(state['finger_loads_n'], (5,))-loads[key])))
            max_load_error = max(max_load_error, error)
            need(error <= 1e-10 and abs(state['robot_load_n']-totals[key]) <= 1e-10
                 and abs(state['illegal_robot_support_force_n']-illegal[key]) <= 1e-10, 'Per-body loads differ from full contact stream')
    need(next(stream, None) is None, 'Extra contact ticks')
    native_path = execution/'evaluation.json'; bindings.add(native_path); native = A.read(native_path)
    for key in ('source_sha256', 'input_sha256'): bindings.mapping(native.get(key, {}))
    bindings.add(plan['arguments']['physics_config'], native['physics_config_sha256'])
    bindings.add(execution/'trace.npz', native['trace_sha256']); bindings.add(execution/'native_structure.json')
    structure = A.read(execution/'native_structure.json')
    need(structure['is_fixed_base'] is True and structure['num_base_dofs'] == 0, 'Fixed-base native structure changed')
    need(native['complete'] is False and native['dt_s'] == DT and native['device'] == ('cuda:0' if mode == 'gpu' else 'cpu')
         and native['independent_coordinates'] == 24 and native['passive_mimics'] == 4
         and native['object_mass_kg'] == float(np.float32(.1)) and native['object_friction'] == .6
         and native['global_gravity_m_s2'] == [0., 0., 0.] and native['robot_gravity_disabled'] is True
         and native['object_weight_world_n'] == [0., 0., -.98], 'Original native physics/material/force changed')
    need(np.array_equal(A._array(native['environment_origin_m'], (3,)), job['origin_m']), 'Native environment origin echo changed')
    need(set(setup['additional_materials']) == set(ids)-{scene['target_object_id']}, 'All nonprimary material evidence required')
    for material in setup['additional_materials'].values():
        need(material['mass_kg'] == .1 and material['friction'] == .6 and material['native_readback_ok'] is True
             and material['world_weight_n'] == [0., 0., -.98], 'Nonprimary material or force changed')
    with np.load(execution/'trace.npz', allow_pickle=False) as z: actual = {k: z[k].copy() for k in z.files}
    names = list(map(str, actual['joint_names'])); need(len(names) == len(set(names)) == 28 and set(names) == set(joints), '28 actual native coordinates required')
    expected = A._full_target(list(map(str, arrays['joint_names'])), arrays['joint_target'], joints, names)
    need(np.array_equal(actual['joint_target'], expected[:ticks]), 'Original 24-to-28 command targets changed')
    for key in ('desired_palm', 'phase', 'time_s'):
        need(np.array_equal(actual[key], arrays[key][:ticks]), 'Original prefix command changed: '+key)
    for key in ('joint_pos', 'joint_target', 'joint_velocity', 'joint_velocity_target'): A._array(actual[key], (ticks, 28))
    arm_names = list(map(str, arrays['joint_names'][:6])); feedforward = native['arm_velocity_feedforward']
    need(feedforward['enabled'] is True and feedforward['hand_velocity_targets_zero'] is True
         and len(feedforward['joints']) == 6 and set(feedforward['joints']) == set(arm_names),
         'Original six-arm-only velocity feedforward changed')
    columns = [names.index(name) for name in arm_names]
    expected_velocity = np.zeros_like(expected, dtype=np.float32)
    expected_velocity[:, columns] = np.gradient(np.asarray(expected, float)[:, columns], DT, axis=0, edge_order=2)
    need(np.array_equal(actual['joint_velocity_target'], expected_velocity[:ticks]), 'Original native arm/hand velocity targets changed')
    body_names = list(map(str, actual['body_names']))
    poses = A._array(actual['body_pose_xyzw'], (ticks, len(body_names), 7))
    need(np.all(abs(np.linalg.norm(poses[..., 3:], axis=-1)-1) <= 1e-4), 'Native body state quaternion changed')
    need(np.max(abs(actual['object_pose_xyzw']-np.asarray([r['objects'][scene['target_object_id']]['pose_xyzw'] for r in rows]))) <= 1e-10,
         'Native primary trace differs from all-body plan-frame trace')
    return dict(environment=job['environment'], original_object_ids=ids, candidate_sha256=candidate['candidate_sha256'],
        commands_sha256=ready['commands']['sha256'], scene_sha256=ready['scene']['sha256'], trace_rows=ticks,
        contact_rows=ticks, contact_points=point_count, max_source_bottom_error_m=max_bottom_error,
        max_finger_load_error_n=max_load_error, targets24_to_28_prefix_exact=True,
        original_arm_velocity_targets_exact=True,
        full_source_vertices=True, label=None, raw_label=None, full_gates_certified=False)


def audit_worker(path, bindings=None, mesh_cache=None):
    bindings = bindings or Bindings(); mesh_cache = {} if mesh_cache is None else mesh_cache
    path = Path(path).resolve(); bindings.add(path); worker = A.read(path); bindings.tree(worker)
    need(worker['schema'] == WORKER_SCHEMA and worker['status'] == 'partial_scaling'
         and worker['complete'] is False and worker['label'] is None and worker['raw_label'] is None
         and worker['formal_training_eligible'] is False and worker['exclude_training'] is True
         and worker['automatic_training'] is False, 'Diagnostic-only bounded worker required')
    mode, lanes, ticks = worker['mode'], worker['lanes'], worker['prefix_ticks']
    need(mode in ('cpu', 'gpu') and lanes in ((1, 4, 8) if mode == 'gpu' else (1,))
         and ticks == 100 and worker['no_new_commands'] is True and worker['identical_template_repetition'] is True,
         'Frozen 100-tick repeated-template budget changed')
    need(worker['device'] == ('cuda:0' if mode == 'gpu' else 'cpu'), 'Requested worker backend echo changed')
    bindings.mapping(worker['source_sha256']); group_path = Path(worker['group_receipt']['path'])
    group = A.read(group_path); profile = A.read(worker['profile']['path']); bindings.mapping(group['source_sha256'])
    bindings.mapping(profile['source_sha256']); bindings.tree(group.get('protocol')); bindings.tree(group.get('reference_assets', {}))
    need(group['status'] == profile['status'] == 'partial_profile' and group['complete'] is False
         and profile['complete'] is False and group['label'] is None and profile['label'] is None
         and profile['raw_label'] is None and profile['error'] is None, 'Only complete bounded prefixes are auditable here')
    need(worker['physical_clock_steps'] == group['physical_clock_steps'] == profile['actual_physics_steps'] == ticks
         and worker['actual_integrated_environment_steps'] == worker['recorded_command_environment_steps']
         == group['actual_integrated_environment_steps'] == group['recorded_command_environment_steps'] == ticks*lanes,
         'Shared clock versus environment-step accounting changed')
    need(worker['returned_physical_steps'] == worker['actual_simulation_step_counter']
         == worker['physical_steps_lower_bound'] == worker['physical_steps_upper_bound'] == ticks
         and worker['observations_completed_steps_per_lane'] == [ticks]*lanes
         and worker['integrated_environment_steps_lower_bound']
         == worker['integrated_environment_steps_upper_bound'] == ticks*lanes
         and worker['physical_steps_unknown'] is False and worker['inflight_step_unconfirmed'] is False
         and worker['native_counter_read_error'] is None,
         'Native attempted clock, returned step, observed step or in-flight bound differs')
    clock = group['clock']; need(clock['prefix_limited'] is True and clock['requested_prefix_ticks'] == ticks
         and clock['commands_added'] == 0 and clock['lane_steps'] == [ticks]*lanes
         and clock['completed_lane_tail_integrated_steps'] == [0]*lanes and not group.get('tail_observations')
         and not group.get('journal_finalize_errors') and group['retry_count'] == 0, 'Exact bounded prefix/no-retry evidence missing')
    roots = [f'/World/envs/env_{i}' for i in range(lanes)]; origins = [[4.*i, 0., 0.] for i in range(lanes)]
    need(group['environment_roots'] == worker['environment_roots'] == roots and group['origins_m'] == worker['origins_m'] == origins,
         'Frozen namespaces and environment origins changed')
    native = group['native_scene']; attrs = native['native_scene']
    need(native['cross_environment_contacts'] == 0 and attrs['physxScene:enableGPUDynamics'] is (mode == 'gpu')
         and attrs['physxScene:broadphaseType'] == ('GPU' if mode == 'gpu' else 'MBP')
         and attrs['physxScene:enableCCD'] is False and attrs['physxScene:timeStepsPerSecond'] == 250,
         'Actual physical backend/frequency/isolation changed')
    probes = _snapshot_probes(group, bindings, mode, roots)
    need(len(group['jobs']) == lanes, 'Every requested lane required')
    lane_reports = []
    template = worker['prepared_template']; bindings.add(Path(template['path'])/'READY.json', template['ready_sha256'])
    for index, job in enumerate(group['jobs']):
        need(job['environment'] == index and job['environment_root'] == roots[index] and job['origin_m'] == origins[index]
             and str(Path(job['source_prepared']).resolve()) == str(Path(template['path']).resolve())
             and job['label'] is None and job['raw_label'] is None and job['complete'] is False
             and job['completion_tick'] == ticks and job['tail_integrated_steps'] == 0
             and job['formal_training_eligible'] is False and job['exclude_training'] is True,
             'Repeated source/frames/diagnostic lane changed')
        lane_reports.append(_lane(job, group_path.parent, bindings, mesh_cache, ticks, mode))
    need(all(r['commands_sha256'] == template['commands_sha256'] and r['scene_sha256'] == template['scene_sha256'] for r in lane_reports), 'One identical source scene and command sequence required')
    start, end = worker['execution_prefix_started_unix_s'], worker['execution_prefix_finished_unix_s']
    duration = worker['execution_prefix_finished_monotonic_s']-worker['execution_prefix_started_monotonic_s']
    need(end > start and duration > 0 and abs((end-start)-duration) <= .05
         and abs(duration-worker['execution_wall_s']) <= 1e-9, 'Prefix wall-clock interval inconsistent')
    before_close_wall = worker['worker_finished_before_close_unix_s']-worker['worker_started_unix_s']
    need(worker['worker_started_unix_s'] <= start < end <= worker['worker_finished_before_close_unix_s']
         and abs(start-worker['worker_started_unix_s']-worker['setup_wall_s']) <= .05,
         'Setup/prefix/finalization clock ordering changed')
    timing = profile['timing']; prefix = timing['execution_prefix']
    for section in timing.values():
        for value in section.values():
            need(type(value['calls']) is int and value['calls'] >= 0
                 and 0 <= value['exclusive_wall_s'] <= value['inclusive_wall_s']+1e-12, 'Invalid nested timing bucket')
    for name in ('physx.simulate', 'physx.fetch_results', 'sim.step'):
        need(prefix[name]['calls'] == ticks, 'Exactly one simulate/fetch per tick required')
    return dict(path=str(path), mode=mode, lanes=lanes, physics_clock_steps=ticks,
        environment_steps=ticks*lanes, dynamic_object_steps=3*ticks*lanes,
        prefix_started_unix_s=start, prefix_finished_unix_s=end, prefix_wall_s=duration,
        setup_wall_s=worker['setup_wall_s'], worker_wall_before_close_s=before_close_wall,
        finalize_after_prefix_before_close_s=worker['worker_finished_before_close_unix_s']-end,
        worker_pid=worker['worker_pid'],
        template=template, probes=probes, lanes_audit=lane_reports, timing=timing,
        label=None, raw_label=None, full_gates_certified=False)


def audit_control(results_path):
    began = time.monotonic(); bindings = Bindings(); mesh_cache = {}
    for source in (__file__, A.__file__, G.__file__): bindings.add(source)
    results_path = Path(results_path).resolve(); bindings.add(results_path); results = A.read(results_path)
    protocol_path = results_path.with_name('SCALING_CONTROL_PROTOCOL.json'); bindings.add(protocol_path)
    protocol = A.read(protocol_path); bindings.tree(protocol); bindings.tree(results)
    need(protocol['schema'] == CONTROL_SCHEMA and results['schema'] == CONTROL_SCHEMA,
         'Frozen scaling-control schema required')
    need(results['label'] is None and results['exclude_training'] is True
         and results['raw_label'] is None and results['formal_training_eligible'] is False
         and results['status'] == 'complete' and results['retries'] == 0
         and results['new_training_labels'] == results['training_updates'] == 0,
         'Complete outer diagnostic exclusion/no-retry required')
    need(protocol['prefix_ticks'] == 100 and protocol['retry_budget'] == 0
         and protocol['total_worker_attempt_budget'] == 16
         and protocol['total_integrated_environment_steps'] == 2600
         and protocol['total_worker_physical_clock_steps'] == 1600
         and protocol['groups_are_sequential'] is True, 'Frozen complete comparison budget changed')
    need(len(protocol['groups']) == len(results['groups']) == 6, 'All six fixed comparisons must be terminal')
    def invalid(value): raise ValueError('Nonfinite host sample JSON: '+value)
    with Path(results['resource_samples']['path']).open() as stream:
        samples = [json.loads(line, parse_constant=invalid) for line in stream]
    need(all(b['unix_s'] >= a['unix_s'] for a, b in zip(samples, samples[1:])), 'Host sample timestamps regressed')
    reports = []; seen_template = None
    for config, group, (mode, count) in zip(protocol['groups'], results['groups'], GROUPS):
        need(config['mode'] == group['mode'] == mode and config['count'] == group['count'] == count
             and config['name'] == group['name'] == f'{mode}_{count}'
             and group['status'] == 'partial_scaling_complete' and group['label'] is None
             and group['raw_label'] is None and group['exclude_training'] is True, 'Frozen ordered diagnostic comparison changed')
        need(len(group['workers']) == len(config['workers']) == (1 if mode == 'gpu' else count), 'Worker budget changed')
        workers = []
        for expected, observed in zip(config['workers'], group['workers']):
            need(expected['worker_id'] == observed['worker_id'], 'Worker order changed')
            stage = A.read(observed['stage']['path'])
            need(stage['returncode'] == 0 and stage['status'] == 'partial_scaling'
                 and stage['complete'] is True and stage['retry_count'] == 0
                 and stage['physical_steps_unknown'] is False, 'Failed/unknown stage cannot be a completed scaling comparison')
            worker = audit_worker(observed['receipt']['path'], bindings, mesh_cache)
            need(worker['mode'] == mode and worker['lanes'] == expected['lanes'], 'Actual worker layout changed')
            need(stage['process']['pid'] == worker['worker_pid'], 'Actual worker PID differs from stage identity')
            current = worker['template']
            if seen_template is None: seen_template = current
            need(current == seen_template, 'CPU/GPU comparisons must use one identical frozen preparation')
            workers.append(worker)
        metrics = comparison_metrics(mode, count, workers, group['wall_s'])
        need(abs(group['finished_unix_s']-group['started_unix_s']-group['wall_s']) <= .05,
             'Outer group wall must include stage startup and real process close')
        for worker in workers:
            need(group['started_unix_s'] <= worker['prefix_started_unix_s'] <= worker['prefix_finished_unix_s'] <= group['finished_unix_s'],
                 'Worker prefix falls outside measured group wall')
        if 'prefix_interval_stats' in group:
            for key in ('span_s', 'union_s', 'peak_concurrency', 'all_worker_overlap_s'):
                need(abs(group['prefix_interval_stats'][key]-metrics['prefix_intervals'][key]) <= 1e-5,
                     'Controller overlap summary differs from independent interval sweep')
        reports.append(dict(name=config['name'], metrics=metrics, workers=workers,
            resources=resource_summary(samples, group, workers, protocol['gpu_uuid'])))
    pairwise = []
    for count in (1, 4, 8):
        gpu = next(g['metrics'] for g in reports if g['name'] == f'gpu_{count}')
        cpu = next(g['metrics'] for g in reports if g['name'] == f'cpu_{count}')
        pairwise.append(dict(count=count,
            gpu_over_cpu_prefix_envstep_rate=gpu['prefix_envsteps_per_s']/cpu['prefix_envsteps_per_s'],
            gpu_over_cpu_startup_inclusive_envstep_rate=gpu['startup_inclusive_envsteps_per_s']/cpu['startup_inclusive_envsteps_per_s']))
    bindings.verify()
    return dict(schema=SCHEMA, status='prefix_comparison_integrity_pass', groups=reports, pairwise_comparisons=pairwise,
        environment_steps=sum(g['metrics']['environment_steps'] for g in reports),
        physics_clock_steps=sum(g['metrics']['physics_clock_steps'] for g in reports),
        trace_rows=sum(r['trace_rows'] for g in reports for w in g['workers'] for r in w['lanes_audit']),
        contact_rows=sum(r['contact_rows'] for g in reports for w in g['workers'] for r in w['lanes_audit']),
        contact_points=sum(r['contact_points'] for g in reports for w in g['workers'] for r in w['lanes_audit']),
        label=None, raw_label=None, formal_training_eligible=False, exclude_training=True,
        full_gates_certified=False, full_lift_certified=False, full_collection_ETA=None,
        audit_physics_steps=0, audit_Kit_launches=0, audit_wall_s=time.monotonic()-began,
        audit_inputs_sha256=bindings.files,
        timing_scope='Prefix includes generator close; outer includes process startup/close. Nested inclusive timings overlap; only exclusive buckets may be summed. No added CUDA synchronization or GPU kernel timing. Resource samples establish observed occupancy, not total device utilization attribution.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'Unique audit output required; preserve any first failure')
    try:
        result = audit_control(args.control_results)
    except Exception:
        result = dict(schema=SCHEMA, status='failed', error=traceback.format_exc(), label=None, raw_label=None,
            formal_training_eligible=False, exclude_training=True, full_gates_certified=False,
            audit_physics_steps=0, audit_Kit_launches=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('status', 'label', 'formal_training_eligible')}))
    return 0 if result['status'] == 'prefix_comparison_integrity_pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
