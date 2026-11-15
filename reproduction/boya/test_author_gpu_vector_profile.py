"""CPU-only checks of bounded profiling and faithful forwarding."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import author_gpu_vector_profile as M


class ProfileTests(unittest.TestCase):
    def test_nested_wall_buckets_do_not_add_inclusive_children_twice(self):
        book = M.Timings(clock=iter([0., 1., 3., 5.]).__next__)
        with book.measure('outer'):
            with book.measure('inner'):
                pass
        self.assertEqual(book.values['initialization']['outer'],
            dict(calls=1, inclusive_wall_s=5., exclusive_wall_s=3.))
        self.assertEqual(book.values['initialization']['inner']['exclusive_wall_s'], 2.)

    def test_native_interface_forwards_arguments_results_and_failure_once(self):
        calls = []
        class Interface:
            marker = object()
            def simulate(self, *a, **kw):
                calls.append((a, kw))
                return self.marker
            def fetch_results(self):
                calls.append('fetch')
                raise RuntimeError('native fetch failure')
        raw = Interface()
        timings = M.Timings()
        proxy = M.InterfaceTimer(raw, timings)
        self.assertIs(proxy.simulate(.004, 0., opt=1), raw.marker)
        with self.assertRaisesRegex(RuntimeError, 'native fetch'):
            proxy.fetch_results()
        self.assertEqual(calls, [((.004, 0.), {'opt': 1}), 'fetch'])
        self.assertEqual(timings.values['initialization']['physx.fetch_results']['calls'], 1)

    def test_actual_native_step_limit_observes_every_integrated_tick_then_closes(self):
        path = M.V.CANON/'canondex/grasp/arm_parallel.py'
        spec = importlib.util.spec_from_file_location('original_profile_clock_test', path)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        sim = types.SimpleNamespace(tick=0)
        actions, observed, closed, completed = [], [], [], []
        def stream(lane, length):
            yield dict(kind='ready',sim=sim,steps=length)
            try:
                for tick in range(1, length+1):
                    actions.append((lane,tick))
                    yield dict(kind='step',sim=sim,step=tick)
                    observed.append((lane,tick,sim.tick))
                completed.append(lane)
            finally:
                closed.append(lane)
        def step(actual):
            self.assertIs(actual,sim)
            sim.tick += 1
        with patch.dict(sys.modules, {'canondex.grasp.arm_parallel':native}):
            receipt = M.prefix_clock([stream(0,5),stream(1,6)],step,on_step=None,
                prefix_ticks=2,timings=M.Timings())
        self.assertEqual(sim.tick,2)
        self.assertEqual(receipt['lane_steps'],[2,2])
        self.assertFalse(receipt['all_streams_complete'])
        self.assertEqual(receipt['actual_integrated_environment_steps'],4)
        self.assertEqual(observed,[(0,1,1),(1,1,1),(0,2,2),(1,2,2)])
        self.assertEqual(actions,[(0,1),(1,1),(0,2),(1,2),(0,3),(1,3)])
        self.assertEqual(sorted(closed),[0,1])
        self.assertEqual(completed,[])

    def test_partial_cannot_contain_any_label_or_completed_trial(self):
        value = dict(clock={'prefix_limited':True},physical_clock_steps=20,
            complete=False,jobs=[{'raw_label':None,'label':None}])
        self.assertEqual(M.profile_receipt(value,20)['status'],'partial_profile')
        for changed in (dict(value,complete=True),dict(value,physical_clock_steps=21),
                        dict(value,jobs=[{'raw_label':0,'label':None}])):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                M.profile_receipt(changed,20)

    def test_patches_restore_on_exception_and_bound_source_is_required(self):
        obj = types.SimpleNamespace(value='original')
        patches = M.Patches()
        try:
            patches.set(obj,'value','profile')
            raise RuntimeError('stop')
        except RuntimeError:
            patches.restore()
        self.assertEqual(obj.value,'original')
        with patch.object(M,'sha',return_value='changed'), self.assertRaisesRegex(ValueError,'entry changed'):
            M.run([],Path('/unused'),env_roots=[],origins_m=[],prefix_ticks=20)

    def test_profile_is_saved_before_original_app_close_can_exit(self):
        script = r'''
import json,os,sys,types
from pathlib import Path
from unittest.mock import patch
import author_gpu_vector_profile as M
out=Path(sys.argv[1])
def save(path,value):Path(path).write_text(json.dumps(value))
storage=types.SimpleNamespace(save=save,sha=M.sha)
lift=types.SimpleNamespace(args_for_plan=lambda *a,**kw:None)
def run_group(*args,**kw):
 out.mkdir()
 storage.save(out/'GPU_VECTOR_EXECUTION.json',dict(clock={'prefix_limited':True},physical_clock_steps=20,
  actual_integrated_environment_steps=20,complete=False,jobs=[{'label':None,'raw_label':None,'evaluation':None}]))
 os._exit(0)
with patch.object(M.V,'load_jobs',return_value=[{'command_steps':100}]), \
 patch.object(M.V,'_modules',return_value=(storage,lift,None,None)), \
 patch.object(M.V,'source_bindings',return_value={}), \
 patch.object(M.V,'verify_sources'),patch.object(M.V,'run_group',run_group):
 M.run([],out,env_roots=[],origins_m=[],prefix_ticks=20)
'''
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh'
            env=dict(os.environ,PYTHONPATH=str(M.V.HERE)+os.pathsep+os.environ.get('PYTHONPATH',''))
            process=subprocess.run([sys.executable,'-B','-c',script,str(out)],env=env,
                text=True,capture_output=True)
            self.assertEqual(process.returncode,0,process.stderr)
            result=json.loads((out/'GPU_VECTOR_PROFILE.json').read_text())
            self.assertEqual(result['status'],'partial_profile')
            self.assertEqual(result['actual_physics_steps'],20)
            self.assertIsNone(result['raw_label'])
            self.assertFalse(result['formal_training_eligible'])


if __name__ == '__main__':
    unittest.main()
