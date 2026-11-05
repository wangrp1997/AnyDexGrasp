"""Zero-physics collection-queue contract checks using saved bound artifacts.

No saved trial is executed. The full-trace guard is exercised against an
already-finished pilot trace, then against deliberately damaged temporary copies.
"""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

SOURCE=Path(__file__).with_name('execute_collection_queue.py')
if not SOURCE.is_file():SOURCE=Path('/tmp/execute_collection_queue.py')
spec=importlib.util.spec_from_file_location('anydex_queue_contract',SOURCE)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
ANYDEX=Path('/home/wangrenpeng/AnyDexGrasp')
q.ROOT=ANYDEX
PILOT=ANYDEX/'outputs/boya_pilot8_20261105_v2'


class TestQueueContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=next(PILOT.glob('trials/*/round_*/training_record.json'))
        cls.record=q.load(cls.source);cls.round=cls.source.parent
        cls.plan=next(p.parent for p in cls.round.glob('plan_*/commands.npz')
                      if q.sha(p)==cls.record['execution_artifacts']['commands_sha256'])
        plan=q.load(cls.plan/'plan.json')
        cls.row=dict(trial_id='saved_contract_only',output_identity='saved_contract_only',
            plan_dir=str(cls.plan),binding_json=str(cls.round/'trial_binding.json'),
            commands_sha256=plan['commands_sha256'],trial_binding_sha256=plan['trial_binding_sha256'],
            object_id=Path(plan['arguments']['object_root']).name,gesture_id=plan['gesture_id'],
            extra_depth_index=cls.record['binding']['multifinger_pose_depth_type'])

    def test_real_bound_plan_is_readonly_accepted(self):
        frozen={};row=q.check_trial(self.row,frozen)
        self.assertIn(self.record['binding']['source_npz'],frozen)
        self.assertIn(str(self.plan/'commands.npz'),frozen)
        self.assertEqual(row['output_identity'],self.row['trial_id'])

    def test_wrong_source_sha_rejected_before_simulation(self):
        row=dict(self.row,commands_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'command SHA'):q.check_trial(row,{})

    def test_wrong_gesture_and_depth_rejected(self):
        for kwargs in [dict(gesture_id=99),dict(extra_depth_index=4),dict(object_id='wrong')]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError,'object/gesture/depth'):q.check_trial(dict(self.row,**kwargs),{})

    def test_safe_output_identity(self):
        with self.assertRaisesRegex(ValueError,'Unsafe'):q.check_trial(dict(self.row,output_identity='../escape'),{})

    def test_grouping_parallel_never_replays_or_crosses_assets(self):
        rows=[dict(cohort_key='a' if i%2 else 'b',arm_collision_refinement_sha256='same') for i in range(9)]
        waves=q.make_cohorts(rows,'parallel',4)
        self.assertEqual(sorted(sum(waves,[])),list(range(9)))
        self.assertEqual([len(x) for x in waves],[4,1,4])
        for wave in waves:self.assertEqual(len({rows[i]['cohort_key'] for i in wave}),1)

    def test_arm_native_repair_prevents_unsafe_residency(self):
        rows=[dict(cohort_key='a',arm_collision_refinement_sha256=x) for x in ['r1','r2','r1']]
        self.assertEqual(q.make_cohorts(rows,'serial',2),[[0,2],[1]])

    def test_serial_reuses_within_wave_and_reports_cold_wave_boundary(self):
        rows=[dict(cohort_key='a',arm_collision_refinement_sha256=None) for _ in range(19)]
        waves=q.make_cohorts(rows,'serial',2)
        self.assertEqual(sorted(sum(waves,[])),list(range(19)))
        self.assertEqual([len(x) for x in waves],[2]*9+[1])
        self.assertTrue(all(len(x)<=16 for x in q.make_cohorts(rows,'parallel',16)))
        for size in [0,17,True]:
            with self.assertRaises(ValueError):q.make_cohorts(rows,'parallel',size)

    def fixture(self,root):
        (root/'cohort/plans/p000').mkdir(parents=True)
        (root/'cohort/execution_000').mkdir()
        shutil.copyfile(self.plan/'commands.npz',root/'cohort/plans/p000/commands.npz')
        shutil.copyfile(self.round/'execution/evaluation.json',root/'cohort/execution_000/evaluation.json')
        shutil.copyfile(self.round/'execution/trace.npz',root/'cohort/execution_000/trace.npz')
        return dict(device='cpu',native_scene={'physxScene:enableGPUDynamics':False},
                    inputs_unchanged=True,sources_unchanged=True,benchmark_only=True,paired_benchmark_id='contract_only')

    def test_existing_full_trace_label_and_benchmark_identity_preserved(self):
        with tempfile.TemporaryDirectory(prefix='anydex_queue_contract_') as temp:
            root=Path(temp);receipt=self.fixture(root)
            result=q.finish_trial(self.row,root/'cohort',0,root/'trial',receipt,0)
            self.assertTrue(result['label_eligible'],result['guards'])
            self.assertEqual(result['label'],self.record['result'])
            self.assertEqual(result['physics_steps'],2100)
            saved=q.load(root/'trial/training_record.json')
            self.assertTrue(saved['benchmark_only'])
            self.assertEqual(saved['binding'],self.record['binding'])

    def test_partial_execution_is_null_even_native_complete(self):
        with tempfile.TemporaryDirectory(prefix='anydex_queue_contract_') as temp:
            root=Path(temp);receipt=self.fixture(root)
            path=root/'cohort/execution_000/evaluation.json';ev=q.load(path);ev['total_physics_steps']=2099;q.save(path,ev)
            result=q.finish_trial(self.row,root/'cohort',0,root/'trial',receipt,0)
            self.assertIsNone(result['label']);self.assertFalse(result['label_eligible'])

    def test_backend_not_enabled_is_null(self):
        with tempfile.TemporaryDirectory(prefix='anydex_queue_contract_') as temp:
            root=Path(temp);receipt=self.fixture(root);receipt['device']='cuda:0'
            result=q.finish_trial(self.row,root/'cohort',0,root/'trial',receipt,0)
            self.assertIsNone(result['label'])
            self.assertFalse(result['guards']['native_scene_backend_confirmed'])

    def test_modified_recorded_command_is_null(self):
        with tempfile.TemporaryDirectory(prefix='anydex_queue_contract_') as temp:
            root=Path(temp);receipt=self.fixture(root)
            path=root/'cohort/execution_000/trace.npz'
            with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            arrays['joint_target'][10,0]+=.001;np.savez_compressed(path,**arrays)
            result=q.finish_trial(self.row,root/'cohort',0,root/'trial',receipt,0)
            self.assertIsNone(result['label']);self.assertFalse(result['guards']['original_target_arrays'])

    def test_torch_style_cuda_host_conversion_is_explicit(self):
        class Buffer:
            def __init__(self):self.calls=[]
            def detach(self):self.calls.append('detach');return self
            def cpu(self):self.calls.append('cpu');return self
            def numpy(self):self.calls.append('numpy');return np.array([0],dtype=np.int32)
        buf=Buffer();self.assertEqual(q.host_numpy(buf).tolist(),[0]);self.assertEqual(buf.calls,['detach','cpu','numpy'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestQueueContract))
    receipt=dict(all_passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        new_physics_steps=0,new_network_forwards=0,new_training_steps=0,source_sha256=q.sha(SOURCE),
        saved_trace_scope='Contract-only read of already-completed pilot trace; not another trial or new performance result')
    Path('/tmp/anydex_queue_contract_tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
