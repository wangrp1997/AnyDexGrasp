"""GPU audit production-shaped CPU fixtures; no simulator/model/planner."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from reproduction.boya import author_gpu_vector_audit as gpu
from reproduction.boya import audit_author_collection_runtime as legacy
from reproduction.boya.test_audit_author_collection_runtime import Fixture, save, bind


class GPUFixture:
    def __init__(self, root, success=True, lanes=1):
        self.f = Fixture(root, success=success)
        self.out = Path(root)/'gpu'; self.out.mkdir(); self.jobs = []
        self.origins = [[0., 0., 0.], [4., 0., 0.]][:lanes]
        self.roots = ['/World/envs/env_'+str(i) for i in range(lanes)]
        self.group_path = self.out/'GPU_VECTOR_EXECUTION.json'
        for i, (root, origin) in enumerate(zip(self.roots, self.origins)):
            origin = np.asarray(origin)
            source = self.f.trial
            if i:
                source = self.f.root/'source_second'
                shutil.copytree(self.f.trial, source)
            lane = self.out/f'env_{i}'; execution = lane/'execution'
            shutil.copytree(self.f.execution, execution)
            setup = legacy.read(execution/'MULTI_OBJECT_SETUP.json')
            prefix = lambda value: root+value[len('/World'): ] if value.startswith('/World/') and value != '/World/Ground' else value
            setup['body_paths'] = {key: prefix(path) for key, path in setup['body_paths'].items()}
            setup['contact_targets'] = [prefix(path) for path in setup['contact_targets']]
            for entry in setup['entry'].values(): entry['pose_xyzw'][:3] = (np.asarray(entry['pose_xyzw'][:3])+origin).tolist()
            setup['additional_materials'] = {key: dict(mass_kg=.1, friction=.6, native_readback_ok=True,
                world_weight_n=[0., 0., -.98]) for key in self.f.ids if key != 'primary'}
            save(execution/'MULTI_OBJECT_SETUP.json', setup)
            contacts = copy.deepcopy(self.f.contacts)
            for row in contacts:
                for group in ('object_contacts', 'robot_contacts'):
                    for contact in row[group]:
                        contact['bodies'] = list(map(prefix, contact['bodies']))
                        contact['position_world_m'] = (np.asarray(contact['position_world_m'])+origin).tolist()
            data = execution/'MULTI_OBJECT_CONTACTS.jsonl.gz'
            with gzip.open(data, 'wt') as stream:
                for row in contacts: stream.write(json.dumps(row)+'\n')
            pointer = legacy.read(execution/'MULTI_OBJECT_CONTACTS.json')
            pointer['data'] = dict(path=str(data), sha256=legacy.sha(data), bytes=data.stat().st_size)
            save(execution/'MULTI_OBJECT_CONTACTS.json', pointer)
            arrays = copy.deepcopy(self.f.native_arrays)
            poses = np.zeros((self.f.n, 2, 7)); poses[..., 6] = 1.
            arrays.update(body_names=np.array(['base', 'pmbase']), body_pose_xyzw=poses,
                joint_velocity=np.zeros((self.f.n, 28)), joint_velocity_target=np.zeros((self.f.n, 28)))
            np.savez_compressed(execution/'trace.npz', **arrays)
            native = legacy.read(execution/'evaluation.json')
            native.update(trace_sha256=legacy.sha(execution/'trace.npz'), device='cuda:0', dt_s=.004,
                fixed_base=True, independent_coordinates=24, passive_mimics=4,
                object_mass_kg=float(np.float32(.1)), object_friction=.6, global_gravity_m_s2=[0., 0., 0.],
                robot_gravity_disabled=True, object_weight_world_n=[0., 0., -.98], environment_origin_m=origin.tolist(),
                arm_velocity_feedforward=dict(enabled=True, hand_velocity_targets_zero=True, joints=self.f.names[:6]),
                palm_rigid_owner='pmbase', palm_owner_to_frame=np.eye(4).tolist())
            save(execution/'evaluation.json', native)
            save(execution/'native_structure.json', dict(is_fixed_base=True, num_base_dofs=0))
            evaluation = copy.deepcopy(self.f.evaluation)
            evaluation.update(legacy_target_id='primary', contacts_sha256=legacy.sha(execution/'MULTI_OBJECT_CONTACTS.json'))
            save(execution/'LIFT_EVALUATION.json', evaluation)
            record = legacy.read(self.f.trial/'COLLECTION_RECORD.json')
            record.update(diagnostic_raw_label=int(success), label=None, formal_training_eligible=False,
                exclude_training=True, evaluation_binding=bind(execution/'LIFT_EVALUATION.json'))
            save(lane/'DIAGNOSTIC_RECORD.json', record)
            ready = legacy.read(source/'READY.json'); bindings = {key: ready[key] for key in
                ('protocol', 'scene', 'candidate', 'plan', 'commands', 'command_binding')}
            bindings.update(ready=bind(source/'READY.json'), preparation=bind(source/'LIFT_PIPELINE_RESULTS.json'))
            self.jobs.append(dict(environment=i, environment_root=root, origin_m=origin.tolist(), source_prepared=str(source),
                input_bindings=bindings, command_steps=self.f.n, completion_tick=self.f.n, tail_integrated_steps=0,
                evaluation=bind(execution/'LIFT_EVALUATION.json'), diagnostic_record=bind(lane/'DIAGNOSTIC_RECORD.json'),
                complete=True, raw_label=int(success), label=None, gates=evaluation['gates'], formal_training_eligible=False,
                exclude_training=True, trace_frame='Original plan world frame',
                contact_position_frame='Physical translated environment world', source_pose_translation_removed_m=origin.tolist()))
        self.group = dict(schema=gpu.EXECUTION_SCHEMA, status='complete', complete=True, device='cuda:0',
            exclude_training=True, formal_training_eligible=False, label=None, automatic_training=False, retry_count=0,
            source_sha256={}, protocol=None, jobs=self.jobs, environment_roots=self.roots, origins_m=self.origins,
            physical_clock_steps=self.f.n, recorded_command_environment_steps=self.f.n*lanes,
            actual_integrated_environment_steps=self.f.n*lanes,
            native_scene=dict(actual_gpu_validated=True, asset_setup_count=1, physics_scene_count=1, cross_environment_contacts=0,
                native_scene={'physxScene:enableGPUDynamics': True, 'physxScene:broadphaseType': 'GPU',
                    'physxScene:enableCCD': False, 'physxScene:timeStepsPerSecond': 250}))
        for key in ('initial_gpu_probe', 'first_tick_gpu_probe'):
            directory = self.out/key; directory.mkdir(); contacts = {}; arrays = {}
            for i, root in enumerate(self.roots):
                for role in ('objects', 'robot'):
                    name = f'env_{i}_{role}'
                    values = [np.zeros((4, width)) for width in (1, 3, 3, 1)] + [np.zeros((1, 1), dtype=np.int32)]*2
                    arrays.update({f'contact_{name}_{j}': value for j, value in enumerate(values)})
                    contacts[name] = dict(status='pass', mode='filtered', dt_s=.004, contact_fields_dropped=False,
                        saturation_detected=False, capacity=4, sensor_count=1, filter_count=1,
                        active_contact_records=0, largest_referenced_endpoint=0,
                        sensor_paths=[root+'/Object'], filter_paths=[['/World/Ground']], tensors=[dict(device='cuda:0')]*6)
            snapshot = directory/'TENSOR_SNAPSHOT.npz'; np.savez(snapshot, **arrays)
            self.group[key] = dict(status='live_api_snapshot_pass', errors=[], physics_steps_added=0,
                disable_contact_processing=False, simulation_view_device_ordinal=0, physics_clock_before=0,
                physics_clock_after=0, contacts=contacts, snapshot=bind(snapshot))
        self.save()

    def save(self): save(self.group_path, self.group)
    def run(self): return gpu.audit_execution(self.group_path, self.f.split)
    def rebind_evaluation(self, i):
        job = self.jobs[i]; path = Path(job['evaluation']['path']); job['evaluation'] = bind(path)
        record_path = Path(job['diagnostic_record']['path']); record = legacy.read(record_path)
        record['evaluation_binding'] = job['evaluation']; save(record_path, record)
        job['diagnostic_record'] = bind(record_path); self.save()


class AuditTests(unittest.TestCase):
    def test_complete_two_lane_translation_and_all_body_positive_no_source_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = GPUFixture(tmp, lanes=2)
            files = {str(p): legacy.sha(p) for p in Path(tmp).rglob('*') if p.is_file()}
            result = f.run()
            self.assertEqual(result['status'], 'pass'); self.assertIsNone(result['label'])
            self.assertEqual([r['diagnostic_label'] for r in result['environments']], [1, 1])
            self.assertEqual(result['environments'][1]['runtime']['object_success'], dict(primary=False, other=True, third=False))
            self.assertEqual(result['audit_physics_steps'], 0)
            self.assertEqual(files, {str(p): legacy.sha(p) for p in Path(tmp).rglob('*') if p.is_file()})
            self.assertFalse(any('anydex_gpu_lane_audit_' in path for path in result['audit_inputs_sha256']))

    def test_complete_physical_negative_is_certified_only_as_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = GPUFixture(tmp, success=False); result = f.run()
            self.assertEqual(result['environments'][0]['diagnostic_label'], 0)
            self.assertFalse(result['formal_training_eligible']); self.assertIsNone(result['label'])

    def test_cross_environment_contact_and_missing_namespace_rejected(self):
        for wrong in ('cross', 'legacy'):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as tmp:
                f = GPUFixture(tmp); execution = f.out/'env_0/execution'
                rows = list(legacy.iter_contact_rows(execution/'MULTI_OBJECT_CONTACTS.json'))
                rows[-1]['object_contacts'][0]['bodies'][1] = '/World/envs/env_1/Robot/thbody0' if wrong == 'cross' else '/World/Robot/thbody0'
                path = execution/'MULTI_OBJECT_CONTACTS.json'; save(path, dict(rows=rows))
                evaluation = legacy.read(execution/'LIFT_EVALUATION.json'); evaluation['contacts_sha256'] = legacy.sha(path)
                save(execution/'LIFT_EVALUATION.json', evaluation); f.rebind_evaluation(0)
                with self.assertRaisesRegex(ValueError, 'environment|Unnamespaced'): f.run()

    def test_nonprimary_source_bottom_and_material_corruption_rejected(self):
        for wrong in ('bottom', 'material'):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as tmp:
                f = GPUFixture(tmp); execution = f.out/'env_0/execution'
                if wrong == 'bottom':
                    path = execution/'MULTI_OBJECT_TRACE.json'; trace = legacy.read(path)
                    trace['rows'][-1]['objects']['other']['source_bottom_m'] += .002; save(path, trace)
                    evaluation = legacy.read(execution/'LIFT_EVALUATION.json'); evaluation['trace_sha256'] = legacy.sha(path)
                    save(execution/'LIFT_EVALUATION.json', evaluation); f.rebind_evaluation(0)
                else:
                    path = execution/'MULTI_OBJECT_SETUP.json'; setup = legacy.read(path)
                    setup['additional_materials']['third']['world_weight_n'][-1] = 0.; save(path, setup)
                with self.assertRaisesRegex(ValueError, 'Source mesh|material/weight'): f.run()

    def test_native_28_target_or_palm_corruption_rejected_despite_true_gates(self):
        for wrong in ('target', 'palm'):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as tmp:
                f = GPUFixture(tmp); execution = f.out/'env_0/execution'; path = execution/'trace.npz'
                with np.load(path, allow_pickle=False) as data: arrays = {key: data[key].copy() for key in data.files}
                if wrong == 'target': arrays['joint_target'][-1, 0] += .1
                else: arrays['body_pose_xyzw'][-1, 1, 0] += .1
                np.savez_compressed(path, **arrays)
                native = legacy.read(execution/'evaluation.json'); native['trace_sha256'] = legacy.sha(path)
                save(execution/'evaluation.json', native)
                with self.assertRaisesRegex(ValueError, 'command target|palm tracking'): f.run()

    def test_partial_infra_and_raw_label_never_certified(self):
        for wrong in ('infra', 'label', 'tail'):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as tmp:
                f = GPUFixture(tmp)
                if wrong == 'infra': f.group['complete'] = False; f.group['status'] = 'infrastructure_failure'
                elif wrong == 'label': f.jobs[0]['raw_label'] = 0
                else: f.jobs[0]['tail_integrated_steps'] += 1
                f.save()
                with self.assertRaisesRegex(ValueError, 'incomplete|diagnostic label|cost accounting'): f.run()

    def test_raw7_native_gaps_ids_capacity_crossenv_and_overlap_scope(self):
        actors = {'/sensor': dict(env_index=0), '/object': dict(env_index=0), '/other_env': dict(env_index=1)}
        frame = dict(schema=gpu.RAW_SCHEMA, overflow=False, force=[[float('nan')], [2.]], point=[[float('nan')]*3, [4., 0., 0.]],
            normal=[[float('nan')]*3, [0., 0., 1.]], separation=[[float('nan')], [-.0001]], count=[1], start=[1], other_actor_ids=[0, 9])
        rows = gpu.decode_raw_contacts(frame, ['/sensor'], {'9': '/object'}, actors, 10)
        self.assertEqual(rows[0]['raw_buffer_index'], 1); self.assertEqual(rows[0]['normal_force_n'], 2.)
        with self.assertRaisesRegex(ValueError, 'capacity'): gpu.decode_raw_contacts(frame, ['/sensor'], {'9': '/object'}, actors, 1)
        with self.assertRaisesRegex(ValueError, 'Cross-environment'): gpu.decode_raw_contacts(frame, ['/sensor'], {'9': '/other_env'}, actors, 10)
        with self.assertRaisesRegex(ValueError, 'Unmapped'): gpu.decode_raw_contacts(frame, ['/sensor'], {}, actors, 10)
        # The public API does not promise disjoint sensor intervals. Preserve
        # both references and never silently deduplicate a repeated interval.
        actors['/sensor2'] = dict(env_index=0); frame.update(count=[1, 1], start=[1, 1])
        self.assertEqual(len(gpu.decode_raw_contacts(frame, ['/sensor', '/sensor2'], {'9': '/object'}, actors, 10)), 2)

    def test_saved_native_snapshot_device_or_capacity_cannot_be_overridden_by_true_gates(self):
        for wrong in ('device', 'capacity'):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as tmp:
                f = GPUFixture(tmp); probe = f.group['first_tick_gpu_probe']
                if wrong == 'device': probe['contacts']['env_0_robot']['tensors'][0]['device'] = 'cpu'
                else:
                    path = Path(probe['snapshot']['path'])
                    with np.load(path) as z: values = {key: z[key].copy() for key in z.files}
                    values['contact_env_0_objects_4'][0, 0] = 4
                    np.savez(path, **values); probe['snapshot'] = bind(path)
                f.save()
                with self.assertRaisesRegex(ValueError, 'CUDA|capacity'): f.run()


if __name__ == '__main__': unittest.main()
