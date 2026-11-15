"""Production-shaped CPU-only checks of prefix evidence and comparison clocks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from reproduction.boya import author_scaling_audit as S
from reproduction.boya import audit_author_collection_runtime as A
from reproduction.boya.test_author_gpu_vector_audit import GPUFixture
from reproduction.boya.test_audit_author_collection_runtime import save, bind


class PrefixFixture:
    def __init__(self, root, mode='gpu'):
        self.f = GPUFixture(root); self.root = self.f.out; self.mode = mode
        job = self.f.jobs[0]; self.job = job; execution = self.root/'env_0/execution'
        self.execution = execution
        source = Path(job['source_prepared']); ready = A.read(source/'READY.json')
        physics = Path(root)/'physics.json'; save(physics, dict(dt_s=.004, scope='Synthetic physics source pin'))
        plan_path = Path(ready['plan']['path']); plan = A.read(plan_path)
        plan['arguments']['physics_config'] = str(physics); save(plan_path, plan)
        ready['plan'] = bind(plan_path); save(source/'READY.json', ready)
        job['input_bindings']['plan'] = ready['plan']; job['input_bindings']['ready'] = bind(source/'READY.json')
        rows_path = execution/'MULTI_OBJECT_TRACE.json'; rows = A.read(rows_path)['rows'][:100]
        save(rows_path, dict(rows=rows))
        contacts = list(A.iter_contact_rows(execution/'MULTI_OBJECT_CONTACTS.json'))[:100]
        # Legacy lossless contact documents remain supported; corruptions below
        # change the full stream and rebind it, so tests exercise numeric gates.
        save(execution/'MULTI_OBJECT_CONTACTS.json', dict(rows=contacts))
        with np.load(execution/'trace.npz', allow_pickle=False) as z:
            data = {k: z[k][:100] if z[k].ndim and len(z[k]) == self.f.f.n else z[k] for k in z.files}
        np.savez_compressed(execution/'trace.npz', **data)
        native = A.read(execution/'evaluation.json')
        native.update(complete=False, total_physics_steps=100, trace_sha256=A.sha(execution/'trace.npz'),
                      device='cuda:0' if mode == 'gpu' else 'cpu', physics_config_sha256=A.sha(physics))
        save(execution/'evaluation.json', native)
        evaluation = A.read(execution/'LIFT_EVALUATION.json')
        evaluation.update(complete=False, label_eligible=False, total_physics_steps=100,
            dynamic_object_steps=300, trace_sha256=A.sha(rows_path), contacts_sha256=A.sha(execution/'MULTI_OBJECT_CONTACTS.json'))
        save(execution/'LIFT_EVALUATION.json', evaluation)
        job.update(complete=False, raw_label=None, completion_tick=100, evaluation=bind(execution/'LIFT_EVALUATION.json'))
        record_path = Path(job['diagnostic_record']['path']); record = A.read(record_path)
        record.update(diagnostic_raw_label=None, evaluation_binding=job['evaluation']); save(record_path, record)
        job['diagnostic_record'] = bind(record_path)
        self.f.group.update(status='partial_profile', complete=False, physical_clock_steps=100,
            actual_integrated_environment_steps=100, recorded_command_environment_steps=100,
            retry_count=0, device='cuda:0' if mode == 'gpu' else 'cpu',
            clock=dict(prefix_limited=True, requested_prefix_ticks=100, commands_added=0,
                lane_steps=[100], completed_lane_tail_integrated_steps=[0]))
        self.f.group['native_scene']['native_scene']['physxScene:enableGPUDynamics'] = mode == 'gpu'
        self.f.group['native_scene']['native_scene']['physxScene:broadphaseType'] = 'GPU' if mode == 'gpu' else 'MBP'
        for key in ('initial_gpu_probe', 'first_tick_gpu_probe'):
            probe = self.f.group[key]
            probe['simulation_view_device_ordinal'] = 0 if mode == 'gpu' else -1
            for value in probe['contacts'].values(): value['tensors'] = [dict(device='cuda:0' if mode == 'gpu' else 'cpu')]*6
        self.f.save()
        self.timing = dict(execution_prefix={name: dict(calls=100, inclusive_wall_s=value, exclusive_wall_s=value)
            for name, value in [('physx.simulate', .1), ('physx.fetch_results', .7), ('sim.step', 1.)]})
        profile_path = self.root/'GPU_VECTOR_PROFILE.json'
        save(profile_path, dict(status='partial_profile', complete=False, label=None, raw_label=None,
            formal_training_eligible=False, error=None, actual_physics_steps=100, source_sha256={}, timing=self.timing))
        source = Path(job['source_prepared']); ready = A.read(source/'READY.json')
        self.worker_path = self.root/'SCALING_EXECUTION.json'
        self.worker = dict(schema=S.WORKER_SCHEMA, status='partial_scaling', complete=False,
            label=None, raw_label=None, formal_training_eligible=False, exclude_training=True, automatic_training=False,
            mode=mode, lanes=1, prefix_ticks=100, no_new_commands=True, identical_template_repetition=True,
            device='cuda:0' if mode == 'gpu' else 'cpu',
            physical_clock_steps=100, actual_integrated_environment_steps=100, recorded_command_environment_steps=100,
            returned_physical_steps=100, actual_simulation_step_counter=100,
            physical_steps_lower_bound=100, physical_steps_upper_bound=100,
            observations_completed_steps_per_lane=[100],
            integrated_environment_steps_lower_bound=100, integrated_environment_steps_upper_bound=100,
            physical_steps_unknown=False, inflight_step_unconfirmed=False, native_counter_read_error=None,
            environment_roots=self.f.roots, origins_m=self.f.origins, source_sha256={},
            group_receipt=bind(self.f.group_path), profile=bind(profile_path),
            prepared_template=dict(path=str(source), ready_sha256=A.sha(source/'READY.json'),
                commands_sha256=ready['commands']['sha256'], scene_sha256=ready['scene']['sha256']),
            execution_prefix_started_unix_s=20., execution_prefix_finished_unix_s=22.,
            execution_prefix_started_monotonic_s=10., execution_prefix_finished_monotonic_s=12.,
            execution_wall_s=2., setup_wall_s=5., worker_pid=123,
            worker_started_unix_s=15., worker_finished_before_close_unix_s=24.)
        self.persist()

    def persist(self):
        self.f.save(); self.worker['group_receipt'] = bind(self.f.group_path)
        save(self.worker_path, self.worker)

    def rebind_lane(self):
        path = self.execution/'LIFT_EVALUATION.json'; value = A.read(path)
        value.update(trace_sha256=A.sha(self.execution/'MULTI_OBJECT_TRACE.json'),
                     contacts_sha256=A.sha(self.execution/'MULTI_OBJECT_CONTACTS.json'))
        save(path, value); self.job['evaluation'] = bind(path)
        record_path = Path(self.job['diagnostic_record']['path']); record = A.read(record_path)
        record['evaluation_binding'] = self.job['evaluation']; save(record_path, record)
        self.job['diagnostic_record'] = bind(record_path); self.persist()

    def run(self): return S.audit_worker(self.worker_path)


class ScalingAuditTests(unittest.TestCase):
    def test_cpu_and_gpu_prefix_original_targets_full_source_and_contacts(self):
        for mode in ('cpu', 'gpu'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                f = PrefixFixture(tmp, mode)
                before = {str(p): A.sha(p) for p in Path(tmp).rglob('*') if p.is_file()}
                result = f.run()
                self.assertEqual(result['environment_steps'], 100)
                self.assertEqual(result['lanes_audit'][0]['contact_rows'], 100)
                self.assertLessEqual(result['lanes_audit'][0]['max_source_bottom_error_m'], 1e-9)
                self.assertIsNone(result['label']); self.assertFalse(result['full_gates_certified'])
                self.assertEqual(before, {str(p): A.sha(p) for p in Path(tmp).rglob('*') if p.is_file()})

    def test_nonprimary_source_bottom_and_contact_force_corruptions_rejected(self):
        for kind in ('bottom', 'force'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                f = PrefixFixture(tmp)
                if kind == 'bottom':
                    path = f.execution/'MULTI_OBJECT_TRACE.json'; rows = A.read(path)
                    rows['rows'][70]['objects']['third']['source_bottom_m'] += .001
                else:
                    path = f.execution/'MULTI_OBJECT_CONTACTS.json'; rows = A.read(path)
                    root = f.job['environment_root']
                    rows['rows'][70]['object_contacts'].append(dict(bodies=[root+'/Objects/other', root+'/Robot/thbody0'],
                        normal_force_n=.01, separation_m=0., position_world_m=[0., 0., 0.], raw_normal_world=[0., 0., 1.]))
                save(path, rows); f.rebind_lane()
                with self.assertRaisesRegex(ValueError, 'bottom|loads'): f.run()

    def test_cross_environment_contact_and_binary_prefix_label_rejected(self):
        for kind in ('cross', 'label'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                f = PrefixFixture(tmp)
                if kind == 'cross':
                    path = f.execution/'MULTI_OBJECT_CONTACTS.json'; rows = A.read(path)
                    rows['rows'][3]['robot_contacts'][0]['bodies'][0] = '/World/envs/env_7/Robot/armbody0'
                    save(path, rows); f.rebind_lane()
                else:
                    f.worker['raw_label'] = 0; f.persist()
                with self.assertRaises(ValueError): f.run()

    def test_target_changed_after_rebinding_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = PrefixFixture(tmp); path = f.execution/'trace.npz'
            with np.load(path, allow_pickle=False) as z: data = {k: z[k].copy() for k in z.files}
            data['joint_target'][75, -1] += .01; np.savez_compressed(path, **data)
            native_path = f.execution/'evaluation.json'; native = A.read(native_path)
            native['trace_sha256'] = A.sha(path); save(native_path, native)
            with self.assertRaisesRegex(ValueError, '24-to-28'): f.run()

    def test_hand_velocity_target_cannot_be_added_to_original_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = PrefixFixture(tmp); path = f.execution/'trace.npz'
            with np.load(path, allow_pickle=False) as z: data = {k: z[k].copy() for k in z.files}
            data['joint_velocity_target'][75, -1] = .1; np.savez_compressed(path, **data)
            native_path = f.execution/'evaluation.json'; native = A.read(native_path)
            native['trace_sha256'] = A.sha(path); save(native_path, native)
            with self.assertRaisesRegex(ValueError, 'velocity targets changed'): f.run()

    def test_cpu_probe_cannot_claim_cuda_and_capacity_saturation_rejected(self):
        for kind in ('device', 'capacity'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                f = PrefixFixture(tmp, 'cpu')
                meta = f.f.group['first_tick_gpu_probe']['contacts']['env_0_objects']
                if kind == 'device': meta['tensors'][0] = dict(device='cuda:0')
                else: meta['active_contact_records'] = meta['capacity']
                f.persist()
                with self.assertRaisesRegex(ValueError, 'device|capacity'): f.run()

    def test_inflight_native_step_is_not_an_exact_completed_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = PrefixFixture(tmp)
            f.worker.update(actual_simulation_step_counter=101, physical_steps_unknown=True,
                inflight_step_unconfirmed=True, physical_steps_upper_bound=101,
                integrated_environment_steps_upper_bound=101)
            f.persist()
            with self.assertRaisesRegex(ValueError, 'in-flight bound'): f.run()

    def test_interval_sweep_uses_span_and_reports_real_concurrency(self):
        workers = [dict(environment_steps=100, physics_clock_steps=100, prefix_started_unix_s=s,
            prefix_finished_unix_s=e, prefix_wall_s=e-s) for s, e in [(0., 4.), (1., 5.), (3., 7.), (4., 8.)]]
        metrics = S.comparison_metrics('cpu', 4, workers, 20.)
        self.assertEqual(metrics['prefix_envsteps_per_s'], 50.)
        self.assertEqual(metrics['startup_inclusive_envsteps_per_s'], 20.)
        self.assertEqual(metrics['prefix_intervals']['peak_concurrency'], 3)
        self.assertEqual(metrics['prefix_intervals']['worker_duration_sum_s'], 16.)
        self.assertEqual(metrics['prefix_intervals']['all_worker_overlap_s'], 0.)
        gpu = S.comparison_metrics('gpu', 4, [dict(workers[0], environment_steps=400, prefix_wall_s=4.)], 20.)
        self.assertEqual(gpu['prefix_envsteps_per_s'], 100.)

    def test_dead_time_and_simultaneous_endpoints_not_false_parallelism(self):
        result = S.interval_stats([(0., 1.), (1., 2.), (4., 5.)])
        self.assertEqual(result['span_s'], 5.); self.assertEqual(result['union_s'], 3.)
        self.assertEqual(result['peak_concurrency'], 1); self.assertEqual(result['idle_s'], 2.)

    def test_sampled_resources_keep_cpu_counter_edges_and_device_scope_explicit(self):
        identity = dict(pid=7, pgid=7, start_ticks=10)
        samples = [dict(unix_s=t, cpu_clock_ticks_per_second=100,
            owned_workers=[dict(process=identity, utime_ticks=10+t, stime_ticks=t, rss_bytes=500)],
            gpu0_nvml_csv='GPU-frozen, 20, 200, 100') for t in (1, 3)]
        result = S.resource_summary(samples, dict(started_unix_s=0, finished_unix_s=4), [dict(worker_pid=7)], 'GPU-frozen')
        self.assertEqual(result['sampled_peak_live_owned_workers'], 1)
        self.assertAlmostEqual(result['captured_worker_cpu_s'], .04)
        self.assertEqual(result['gpu_memory_used_mib']['max'], 200.)
        samples[-1]['gpu0_nvml_csv'] = 'GPU-other, 20, 200, 100'
        with self.assertRaisesRegex(ValueError, 'physical GPU'):
            S.resource_summary(samples, dict(started_unix_s=0, finished_unix_s=4), [dict(worker_pid=7)], 'GPU-frozen')

    def test_six_group_controller_accounting_and_final_source_recheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); configs = []; groups = []; reports = {}
            for index, (mode, count) in enumerate(S.GROUPS):
                name = f'{mode}_{count}'; configs.append(dict(name=name, mode=mode, count=count, workers=[]))
                group = dict(name=name, mode=mode, count=count, status='partial_scaling_complete',
                    label=None, raw_label=None, exclude_training=True, workers=[],
                    started_unix_s=100.*index, finished_unix_s=100.*index+50., wall_s=50.)
                for i in range(1 if mode == 'gpu' else count):
                    worker_id = f'worker_{i:02d}'; lanes = count if mode == 'gpu' else 1; pid = 100+20*index+i
                    base = root/name/worker_id; stage = base/'STAGE.json'; receipt = base/'SCALING_EXECUTION.json'
                    save(stage, dict(returncode=0, status='partial_scaling', complete=True,
                        retry_count=0, physical_steps_unknown=False, process=dict(pid=pid)))
                    save(receipt, dict(test='A real worker is separately covered by numeric fixtures'))
                    configs[-1]['workers'].append(dict(worker_id=worker_id, lanes=lanes))
                    group['workers'].append(dict(worker_id=worker_id, stage=bind(stage), receipt=bind(receipt)))
                    reports[str(receipt)] = dict(mode=mode, lanes=lanes, environment_steps=100*lanes,
                        physics_clock_steps=100, prefix_started_unix_s=100.*index+10+i,
                        prefix_finished_unix_s=100.*index+20+i, prefix_wall_s=10., worker_pid=pid,
                        template=dict(path='/same/frozen/template'), lanes_audit=[dict(trace_rows=100,
                            contact_rows=100, contact_points=0)]*lanes)
                groups.append(group)
            protocol = dict(schema=S.CONTROL_SCHEMA, groups=configs, prefix_ticks=100,
                retry_budget=0, total_worker_attempt_budget=16, total_integrated_environment_steps=2600,
                total_worker_physical_clock_steps=1600, groups_are_sequential=True, gpu_uuid='GPU-frozen')
            save(root/'SCALING_CONTROL_PROTOCOL.json', protocol)
            samples = root/'RESOURCE_SAMPLES.jsonl'; samples.write_text('')
            path = root/'SCALING_CONTROL_RESULTS.json'
            result = dict(schema=S.CONTROL_SCHEMA, status='complete', groups=groups, label=None, raw_label=None,
                formal_training_eligible=False, exclude_training=True, retries=0, new_training_labels=0,
                training_updates=0, resource_samples=bind(samples))
            save(path, result)
            with mock.patch.object(S, 'audit_worker', side_effect=lambda p, *args: copy.deepcopy(reports[str(p)])):
                audit = S.audit_control(path)
            self.assertEqual(audit['environment_steps'], 2600)
            self.assertEqual(audit['physics_clock_steps'], 1600)
            self.assertEqual(audit['trace_rows'], 2600)
            self.assertIsNone(audit['full_collection_ETA'])
            self.assertEqual(audit['groups'][-1]['metrics']['prefix_intervals']['peak_concurrency'], 8)
            samples.write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'SHA changed'):
                S.audit_control(path)


if __name__ == '__main__': unittest.main()
