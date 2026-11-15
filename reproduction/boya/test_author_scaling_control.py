"""CPU tests for bounded scheduling, accounting and owned-child cancellation.

No Kit or model imports, physical integration, real child launch, or signals.
"""
import contextlib
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import author_scaling_control as M


class FakePool:
    def __init__(self, *args, **kwargs):
        pass


class FakeSampler:
    def __init__(self, root, protocol):
        self.path = Path(root)/'RESOURCE_SAMPLES.jsonl'

    def start(self):
        self.path.write_text('{"synthetic":true,"Kit_launches":0}\n')

    def close(self):
        pass


def protocol_fixture(root):
    template = root/'template'
    template.mkdir()
    for name in ('READY.json', 'commands.npz'):
        (template/name).write_text('synthetic '+name)
    groups = []
    for mode in ('gpu', 'cpu'):
        for count in M.COUNTS:
            name = f'{mode}_{count}'
            groups.append(dict(name=name, mode=mode, count=count, workers=[
                dict(worker_id=f'worker_{i:02d}', prepared_dir=str(template),
                    stage_dir=str(root/name/f'worker_{i:02d}'),
                    output_dir=str(root/name/f'worker_{i:02d}'/'result'),
                    lanes=count if mode == 'gpu' else 1)
                for i in range(1 if mode == 'gpu' else count)]))
    return dict(schema=M.SCHEMA, diagnostic_only=True, exclude_training=True,
        prefix_ticks=100, retry_budget=0, stage_timeout_s=900.,
        resource_limits={'kit':8}, total_worker_attempt_budget=16,
        total_integrated_environment_steps=2600,
        prepared_template=dict(path=str(template), ready=M.B.binding(template/'READY.json'),
            commands=M.B.binding(template/'commands.npz')), groups=groups,
        gpu_uuid='GPU-synthetic', gpu_identity={'test':'no hardware query'},
        env={'CUDA_VISIBLE_DEVICES':'GPU-synthetic','NVIDIA_VISIBLE_DEVICES':'0'},
        inherited_env={'PYTHONPATH':''},
        inputs={'nvidia_smi':M.B.binding(template/'READY.json')}, sources={},
        source_archive=M.archive_sources(root,{}),
        worker_source=M.B.binding(template/'READY.json'))


def receipt_fixture(root, protocol, group, worker):
    out = Path(worker['output_dir'])
    out.mkdir(exist_ok=True,parents=True)
    lanes = worker['lanes']
    native = dict(status='partial_profile', complete=False, physical_clock_steps=100,
        actual_integrated_environment_steps=100*lanes,
        recorded_command_environment_steps=100*lanes, clock={'lane_steps':[100]*lanes},
        jobs=[dict(completion_tick=100,label=None,raw_label=None) for _ in range(lanes)],
        initial_gpu_probe={'status':'live_api_snapshot_pass'},
        first_tick_gpu_probe={'status':'live_api_snapshot_pass'})
    profile = dict(status='partial_profile', actual_physics_steps=100,
        label=None,raw_label=None,exclude_training=True)
    M.B.save(out/'GPU_VECTOR_EXECUTION.json', native)
    M.B.save(out/'GPU_VECTOR_PROFILE.json', profile)
    return dict(status='partial_scaling', complete=False, label=None, raw_label=None,
        exclude_training=True,mode=group['mode'],lanes=lanes,prefix_ticks=100,
        physical_clock_steps=100,actual_integrated_environment_steps=100*lanes,
        recorded_command_environment_steps=100*lanes,
        prepared_template=copy.deepcopy(protocol['prepared_template']),
        group_receipt=M.B.binding(out/'GPU_VECTOR_EXECUTION.json'),
        profile=M.B.binding(out/'GPU_VECTOR_PROFILE.json'),
        source_sha256={},setup_wall_s=2.,execution_wall_s=4.,
        execution_prefix_started_unix_s=100.,execution_prefix_finished_unix_s=104.,
        device='cuda:0' if group['mode']=='gpu' else 'cpu',worker_pid=12345,
        returned_physical_steps=100,actual_simulation_step_counter=100,
        observations_completed_steps_per_lane=[100]*lanes,physical_steps_unknown=False,
        inflight_step_unconfirmed=False,physical_steps_lower_bound=100,physical_steps_upper_bound=100,
        integrated_environment_steps_lower_bound=100*lanes,integrated_environment_steps_upper_bound=100*lanes)


class TestScalingControl(unittest.TestCase):
    def test_interval_union_and_real_overlap(self):
        r = M.interval_stats([(0.,3.),(2.,5.),(6.,7.)])
        self.assertEqual((r['span_s'],r['union_s'],r['peak_concurrency'],r['all_worker_overlap_s']),
            (7.,6.,2,0.))
        self.assertEqual(M.interval_stats([(0.,1.),(1.,2.)])['peak_concurrency'],1)
        with self.assertRaises(ValueError):
            M.interval_stats([(2.,1.)])

    def test_admission_rejects_accounting_labels_backend_and_probe_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=protocol_fixture(root); g=p['groups'][1]; w=g['workers'][0]
            base=receipt_fixture(root,p,g,w)
            M.admit(base,w,g,p)
            variants=[('physical_clock_steps',99),('actual_integrated_environment_steps',399),
                ('recorded_command_environment_steps',399),('label',0),('raw_label',1),
                ('exclude_training',False),('device','cpu'),('actual_simulation_step_counter',101),
                ('physical_steps_unknown',True),('observations_completed_steps_per_lane',[99]*4)]
            for key,value in variants:
                with self.subTest(key=key):
                    r=copy.deepcopy(base); r[key]=value
                    with self.assertRaises(ValueError):M.admit(r,w,g,p)
            original_native=M.B.read(base['group_receipt']['path'])
            for field in ('lane_steps','completion_tick','probe'):
                with self.subTest(native=field):
                    r=copy.deepcopy(base); n=copy.deepcopy(original_native)
                    if field=='lane_steps':n['clock']['lane_steps'][0]=99
                    if field=='completion_tick':n['jobs'][0]['completion_tick']=99
                    if field=='probe':n['first_tick_gpu_probe']['status']='failed'
                    path=Path(r['group_receipt']['path']);M.B.save(path,n);r['group_receipt']=M.B.binding(path)
                    with self.assertRaises(ValueError):M.admit(r,w,g,p)

    def test_verify_bounds_inputs_and_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=protocol_fixture(root);M.B.save(root/M.PROTOCOL,p)
            M.verify(root)
            bad=copy.deepcopy(p);bad['groups'][-1]['workers'][0]['output_dir']='/elsewhere'
            M.B.save(root/M.PROTOCOL,bad)
            with self.assertRaises(ValueError):M.verify(root)
            M.B.save(root/M.PROTOCOL,p)
            Path(p['prepared_template']['commands']['path']).write_text('changed')
            with self.assertRaises(ValueError):M.verify(root)

    def test_failure_stops_global_dispatch_and_start_marker_blocks_rerun(self):
        for fail_name in ('gpu_1','cpu_1',None):
            with self.subTest(fail_name=fail_name), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);p=protocol_fixture(root);M.B.save(root/M.PROTOCOL,p)
                calls=[]
                def group(runroot,g,protocol,pool,sampler):
                    calls.append(g['name'])
                    return dict(name=g['name'],status='stopped_after_failure' if g['name']==fail_name
                        else 'partial_scaling_complete',workers=[])
                with patch.object(M.G,'LOCK',root/'gpu.lock'), \
                     patch.object(M.G,'REGISTRY',root/'registry.json'), \
                     patch.object(M,'gpu_identity',return_value=p['gpu_identity']), \
                     patch.object(M,'ScalingPool',FakePool), \
                     patch.object(M,'ResourceSampler',FakeSampler), \
                     patch.object(M,'run_group',side_effect=group):
                    result=M.run(root)
                    self.assertEqual(len(calls),1 if fail_name=='gpu_1' else 4 if fail_name=='cpu_1' else 6)
                    self.assertEqual(result['status'],'complete' if fail_name is None else 'stopped_after_failure')
                    with self.assertRaises(ValueError):M.run(root)
                    self.assertEqual(M.B.read(root/'registry.json')['state'],'released')
                    with M.R.lease(root/'DRIVER.lock'),M.R.lease(root/'gpu.lock'):pass
                    self.assertIsNone(result['label'])

    def test_cpu_eight_workers_really_enter_same_group_concurrently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=protocol_fixture(root);g=p['groups'][-1]
            barrier=threading.Barrier(8,timeout=5.)
            live=0;peak=0;lock=threading.Lock()
            def stage(w,group,protocol,pool,sampler,stop):
                nonlocal live,peak
                with lock:live+=1;peak=max(peak,live)
                barrier.wait()
                r=Path(w['stage_dir'])/'receipt.json'
                M.B.save(r,dict(setup_wall_s=1.,execution_wall_s=2.,
                    execution_prefix_started_unix_s=10.,execution_prefix_finished_unix_s=12.))
                with lock:live-=1
                return dict(worker_id=w['worker_id'],status='partial_scaling',receipt=M.B.binding(r))
            with patch.object(M,'run_stage',side_effect=stage):
                result=M.run_group(root,g,p,None,None)
            self.assertEqual(peak,8)
            self.assertEqual(result['prefix_interval_stats']['peak_concurrency'],8)
            self.assertEqual(result['environment_steps'],800)
            self.assertEqual(result['prefix_environment_steps_per_s'],400.)
            self.assertEqual(len(result['workers']),8)

    def test_independent_pool_reserves_eight_without_mutating_original_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pool=M.ScalingPool(root,0,{})
            original=copy.deepcopy(M.R.LIMITS)
            slots=[]
            try:
                slots=[pool._acquire('kit',{'stage_dir':'synthetic'}) for _ in range(8)]
                self.assertEqual([s['state']['slot'] for s in slots],list(range(8)))
                self.assertEqual(M.R.LIMITS,original)
                self.assertEqual(M.R.LIMITS['kit'],3)
            finally:
                for slot in slots:os.close(slot['fd'])

    def test_cancellation_signals_only_owned_identity_and_prefers_sigint(self):
        identity=dict(pid=12345,pgid=12345,start_ticks=987)
        class Process:
            pid=12345
            returncode=None
            def poll(self):return self.returncode
            def wait(self,timeout):
                if len(signals)==1:raise subprocess.TimeoutExpired('synthetic',timeout)
                self.returncode=-signal.SIGTERM
        process=Process();signals=[]
        with patch.object(M.R,'process_identity',return_value=dict(identity,start_ticks=999)), \
                patch.object(M.os,'killpg') as kill:
            with self.assertRaises(M.R.Quarantined):M.signal_owned(process,identity,signal.SIGINT)
            kill.assert_not_called()
        with patch.object(M.R,'process_identity',return_value=identity), \
                patch.object(M.os,'killpg',side_effect=lambda pid,sig:signals.append(sig)):
            events=M.finish_owned(process,identity,dict(cancel_sigint_grace_s=30.,
                cancel_sigterm_grace_s=5.,cancel_sigkill_grace_s=5.))
        self.assertEqual(signals,[signal.SIGINT,signal.SIGTERM])
        self.assertEqual([e['signal'] for e in events],['SIGINT','SIGTERM'])

    def test_failed_worker_keeps_partial_count_bounds_unknown_and_stops_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=protocol_fixture(root);g=p['groups'][0];w=g['workers'][0]
            p['_protocol_path']=str(root/M.PROTOCOL)
            identity=dict(pid=12345,pgid=12345,start_ticks=987)
            class Process:
                pid=12345
                returncode=1
                def poll(self):return self.returncode
            class Pool:
                @contextlib.contextmanager
                def stage(self,*args,**kwargs):yield
                def popen(self,*args,**kwargs):
                    cost=dict(returned_physical_steps=37,actual_simulation_step_counter=38,
                        observations_completed_steps_per_lane=[37],physical_steps_unknown=True,
                        inflight_step_unconfirmed=True,physical_steps_lower_bound=37,
                        physical_steps_upper_bound=38,integrated_environment_steps_lower_bound=37,
                        integrated_environment_steps_upper_bound=38)
                    M.B.save(Path(w['output_dir'])/M.WORKER_RECEIPT,
                        dict(status='infrastructure_failure',physical_clock_steps=37,**cost))
                    return Process()
            class Sampler:
                def add(self,value):pass
            stop=threading.Event()
            with patch.object(M.R,'process_identity',return_value=identity), \
                    patch.object(M.os,'killpg') as kill:
                result=M.run_stage(w,g,p,Pool(),Sampler(),stop)
            kill.assert_not_called()
            self.assertTrue(stop.is_set())
            self.assertEqual(result['status'],'infrastructure_failure')
            self.assertTrue(result['physical_steps_unknown'])
            self.assertEqual(result['physics_steps_known'],37)
            self.assertEqual(result['physical_cost']['physical_steps_upper_bound'],38)
            self.assertIsNotNone(result['partial_receipt'])

    def test_source_archive_deduplicates_aliases_and_preserves_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.py';source.write_bytes(b'# frozen source\n')
            binding=M.B.binding(source)
            result=M.archive_sources(root,{'first':binding,'alias':dict(binding)})
            data=M.B.read(result['path'])
            self.assertEqual(len(data['sources']),1)
            row=data['sources'][0]
            self.assertEqual(row['binding_names'],['first','alias'])
            self.assertEqual(row['archive']['sha256'],binding['sha256'])
            source.write_bytes(b'# source changed\n')
            self.assertEqual(Path(row['archive']['path']).read_bytes(),b'# frozen source\n')
            with self.assertRaises(ValueError):M.B.verify({'original':binding})


if __name__=='__main__':
    unittest.main()
