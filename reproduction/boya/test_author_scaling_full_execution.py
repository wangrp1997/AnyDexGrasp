"""CPU-only complete-clock/diagnostic-label checks, with no Kit dispatch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import author_scaling_full_execution as M


class FullTests(unittest.TestCase):
    def test_changed_scaling_source_is_rejected_before_launch(self):
        with patch.object(M.P,'sha',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'scaling adapter changed'):
                M.run('/unused','/unused')

    def test_complete_original_clock_and_raw_label_are_saved_before_immediate_close(self):
        script=r'''
import importlib.util,json,os,sys,types
from pathlib import Path
from unittest.mock import patch
import author_scaling_full_execution as M
prepared,out=Path(sys.argv[1]),Path(sys.argv[2])
def save(path,value):Path(path).write_text(json.dumps(value))
storage=types.SimpleNamespace(save=save,sha=M.P.sha)
lift=types.SimpleNamespace(args_for_plan=lambda *a,**kw:None)
spec=importlib.util.spec_from_file_location('native_full_clock',M.V.CANON/'canondex/grasp/arm_parallel.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
def vector_run(prepared_dirs,output_dir,*,device,**kw):
 assert device=='cpu' and len(prepared_dirs)==1
 scene=M.V.VectorScene([],[],[],output_dir)
 sim=types.SimpleNamespace(tick=0)
 sim.get_physics_step_count=lambda:sim.tick
 scene.sim=sim
 output_dir.mkdir()
 observed=[]
 def stream():
  yield dict(kind='ready',sim=sim,steps=5)
  for tick in range(1,6):
   yield dict(kind='step',sim=sim,step=tick)
   observed.append(tick)
 def step(actual):actual.tick+=1
 clock=M.V.shared_clock([stream()],step)
 assert observed==[1,2,3,4,5] and clock['all_streams_complete']
 storage.save(output_dir/'GPU_VECTOR_EXECUTION.json',dict(clock=clock,status='complete',
  physical_clock_steps=5,actual_integrated_environment_steps=5,recorded_command_environment_steps=5,
  complete=True,jobs=[{'label':None,'raw_label':1,'complete':True,'completion_tick':5,'gates':{'original_gate':True}}]))
 os._exit(0)
with patch.object(M.S,'compile_entries',return_value=(vector_run,None)), \
 patch.object(M.V,'load_jobs',return_value=[{'command_steps':5}]), \
 patch.object(M.V,'_modules',return_value=(storage,lift,None,None)), \
 patch.object(M.V,'source_bindings',return_value={}),patch.object(M.V,'verify_sources'), \
 patch.dict(sys.modules,{'canondex.grasp.arm_parallel':native}):
 M.run(prepared,out)
'''
        with tempfile.TemporaryDirectory() as tmp:
            prepared,out=Path(tmp)/'prepared',Path(tmp)/'fresh';prepared.mkdir()
            for name in ('commands.npz','scene.json'):(prepared/name).write_text('frozen')
            ready={key:{'path':str(prepared/name),'sha256':M.P.sha(prepared/name)} for key,name in
                (('commands','commands.npz'),('scene','scene.json'))}
            (prepared/'READY.json').write_text(json.dumps(ready))
            env=dict(os.environ,PYTHONPATH=str(M.V.HERE)+os.pathsep+os.environ.get('PYTHONPATH',''))
            process=subprocess.run([sys.executable,'-B','-c',script,str(prepared),str(out)],env=env,text=True,capture_output=True)
            self.assertEqual(process.returncode,0,process.stderr)
            result=json.loads((out/'SCALING_EXECUTION.json').read_text())
            self.assertEqual(result['status'],'diagnostic_full')
            self.assertTrue(result['complete'])
            self.assertTrue(result['full_trial'])
            self.assertEqual(result['raw_label'],1)
            self.assertIsNone(result['label'])
            self.assertFalse(result['formal_training_eligible'])
            self.assertEqual(result['original_gates'],{'original_gate':True})
            self.assertEqual(result['physical_clock_steps'],5)
            self.assertEqual(result['observations_completed_steps_per_lane'],[5])
            self.assertFalse(result['physical_steps_unknown'])
            self.assertIn('execution_started_unix_s',result)
            self.assertEqual(result['execution_started_unix_s'],result['execution_prefix_started_unix_s'])
            self.assertEqual(result['group_receipt']['sha256'],M.P.sha(out/'GPU_VECTOR_EXECUTION.json'))
            self.assertEqual(result['profile']['sha256'],M.P.sha(out/'GPU_VECTOR_PROFILE.json'))


if __name__=='__main__':unittest.main()
