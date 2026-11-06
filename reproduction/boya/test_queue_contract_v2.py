"""Zero-physics contracts for unchanged source-plan adaptive approach counts."""
from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile,unittest
import numpy as np
SOURCE=Path(__file__).with_name('execute_collection_queue_v2.py')
spec=importlib.util.spec_from_file_location('queuev2',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RUN=Path('/home/wangrenpeng/AnyDexGrasp/outputs/collection40_20261106_v2/run_v2')
class Arrays(dict):
 @property
 def files(self):return list(self)
class Contract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.queue=m.load(RUN/'planning/queue_manifest.json');cls.rows=cls.queue['trials'];cls.long=next(r for r in cls.rows if r['trial_id'].endswith('__g07__d0') and 'Snowman' in r['object_id']);cls.normal=next(r for r in cls.rows if m.load(Path(r['plan_dir'])/'plan.json')['total_command_steps']==2100)
 def fixture(self,row=None):
  row=row or self.long;p=Path(row['plan_dir']);plan=m.load(p/'plan.json')
  with np.load(p/'commands.npz',allow_pickle=False)as z:a=Arrays({k:z[k].copy()for k in z.files})
  return plan,a
 def test_real_2100_and_all_three_extended_plans(self):
  found=[]
  for row in self.rows:
   plan=m.load(Path(row['plan_dir'])/'plan.json')
   if plan['total_command_steps']!=2100 or row is self.normal:
    plan,a=self.fixture(row);d=m.command_contract(plan,a);found.append(d['expected_command_steps']);self.assertEqual(d['expected_phase_counts']['close'],250);self.assertEqual(d['expected_phase_counts']['hold'],50);self.assertEqual(d['expected_phase_counts']['lift'],500)
  self.assertEqual(sorted(found),[2100,2124,2126,2142])
 def test_all_original_431_plans_valid_without_rewriting(self):
  for row in self.rows:
   plan,a=self.fixture(row);m.command_contract(plan,a)
 def test_truncated_or_shortened_plan_rejected(self):
  p,a=self.fixture();a['phase']=a['phase'][:-1]
  with self.assertRaises(ValueError):m.command_contract(p,a)
  p,a=self.fixture();p['total_command_steps']=2099
  with self.assertRaises(ValueError):m.command_contract(p,a)
 def test_only_approach_may_grow(self):
  p,a=self.fixture();a['phase'][1325]='close'
  with self.assertRaises(ValueError):m.command_contract(p,a)
 def test_phase_boundary_mismatch_rejected(self):
  p,a=self.fixture();p['phases'][1]['end_step']-=1
  with self.assertRaises(ValueError):m.command_contract(p,a)
 def test_plan_timing_mismatch_rejected(self):
  p,a=self.fixture();p['segment_timing']['approach']['steps']-=1
  with self.assertRaises(ValueError):m.command_contract(p,a)
 def test_original_grid_and_target_shapes_required(self):
  p,a=self.fixture();a['time_s'][500]+=.001
  with self.assertRaises(ValueError):m.command_contract(p,a)
  p,a=self.fixture();a['joint_target']=a['joint_target'][:-1]
  with self.assertRaises(ValueError):m.command_contract(p,a)
  p,a=self.fixture();a['desired_palm'][0,0,0]=np.nan
  with self.assertRaises(ValueError):m.command_contract(p,a)
 def test_original_physics_timestep_required(self):
  p,a=self.fixture();p['dt_s']=.002
  with self.assertRaises(ValueError):m.command_contract(p,a)
 def test_saved_positive_and_negative_labels_retained(self):
  result=m.load(RUN/'execution/RESULTS.json')
  for label in (0,1):
   old=next(r for r in result['trials']if r.get('label')==label);row=next(r for r in self.rows if r['trial_id']==old['trial_id']);plan,a=self.fixture(row);row=dict(row,output_identity=old['output_identity'],**m.command_contract(plan,a));execution=Path(old['execution_dir']);cohort=m.load(execution.parent/'result.json');cohort.update(device='cpu',benchmark_only=False,paired_benchmark_id=None)
   with tempfile.TemporaryDirectory()as t:
    replay=m.finish_trial(row,execution.parent,int(execution.name.split('_')[-1]),Path(t)/'audit_only',cohort,0);self.assertEqual(replay['label'],label);self.assertTrue(replay['label_eligible']);self.assertEqual(replay['physics_steps'],2100)
 def test_original_executor_bytes_unchanged(self):
  self.assertEqual(m.sha(m.ROOT/'reproduction/boya/execute_collection_queue.py'),m.ORIGINAL_EXECUTOR_SHA256)
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
 out=dict(all_passed=r.wasSuccessful(),tests=r.testsRun,errors=len(r.errors),failures=len(r.failures),source_sha256=m.sha(SOURCE),original_executor_sha256=m.sha(m.ROOT/'reproduction/boya/execute_collection_queue.py'),real_plans_checked=431,new_physics_steps=0,new_network_forwards=0,scope='Read-only command validation and saved-trace label replay only; no new physical trials')
 Path('/tmp/anydex_queue_v2_contract.json').write_text(json.dumps(out,indent=2)+'\n');raise SystemExit(not r.wasSuccessful())
