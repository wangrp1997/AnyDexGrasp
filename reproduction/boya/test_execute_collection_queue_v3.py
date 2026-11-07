from pathlib import Path
import sys,importlib.util,unittest,json,copy
import numpy as np
BOYA=Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya');sys.path.insert(0,str(BOYA))
p=Path(__file__).with_name('execute_collection_queue_v3.py');s=importlib.util.spec_from_file_location('queue_v3_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Arrays(dict):
 @property
 def files(self):return list(self)
class CommandContractTests(unittest.TestCase):
 def fixture(self):
  q=json.loads((BOYA.parents[1]/'outputs/s40_stable_supplement_20261107_v2/collection/planning/queue_manifest.json').read_text());r=next(x for x in q['trials']if x['trial_id']=='graspnet_060_g6_d1');d=Path(r['plan_dir']);plan=json.loads((d/'plan.json').read_text())
  with np.load(d/'commands.npz',allow_pickle=False)as z:a=Arrays({k:z[k].copy()for k in z.files})
  return plan,a
 def test_real_native_extended_home_keeps_all_bytes(self):
  p,a=self.fixture();before={k:v.tobytes()for k,v in a.items()};x=m.command_contract(p,a);self.assertEqual(x['expected_command_steps'],2166);self.assertEqual(x['expected_phase_counts']['home_to_pregrasp'],816);self.assertEqual(x['approach_extension_steps'],0);self.assertEqual(before,{k:v.tobytes()for k,v in a.items()})
 def test_reject_phase_reordering_fixed_closure_changes_and_time_shift(self):
  for kind in ['phase','close','time']:
   p,a=self.fixture()
   if kind=='phase':a['phase'][800]='approach'
   elif kind=='close':p['phases'][2]['end_step']+=1
   else:a['time_s'][0]+=.004
   with self.subTest(kind=kind),self.assertRaises(ValueError):m.command_contract(p,a)
 def test_original_limits_and_metadata_cannot_be_relaxed(self):
  for field,value in [('max_acceleration_rad_s2',3.1),('max_jerk_rad_s3',31.),('velocity_ratio',1.1),('duration_s',99.),('steps',True),('acceleration_limit_rad_s2',4.)]:
   p,a=self.fixture();p['segment_timing']['home_to_pregrasp'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.command_contract(p,a)
 def test_all195_prepared_native_command_arrays_accepted(self):
  q=json.loads((BOYA.parents[1]/'outputs/s40_stable_supplement_20261107_v2/collection/planning/queue_manifest.json').read_text())
  for r in q['trials']:
   d=Path(r['plan_dir']);p=json.loads((d/'plan.json').read_text())
   with np.load(d/'commands.npz',allow_pickle=False)as z:self.assertEqual(m.command_contract(p,z)['expected_command_steps'],p['total_command_steps'])
 def test_certificate_uses_same_explicit_contract(self):
  import terminal_collection_outcome
  self.assertIs(terminal_collection_outcome.command_contract,m.command_contract)
if __name__=='__main__':unittest.main()
