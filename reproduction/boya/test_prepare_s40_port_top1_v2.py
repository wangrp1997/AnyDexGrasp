import argparse,copy,importlib.util,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('porttop1',str(Path(__file__).resolve().with_name('prepare_s40_port_top1_v2.py')));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Top1Contract(unittest.TestCase):
 def item(self,n=6):
  return dict(object_id='graspnet_044',geometry_admitted_total=n,candidates=[dict(row=dict(global_score_rank=i),job=dict(trial_id=f'graspnet_044_port_rank_{i:05d}'))for i in range(n)])
 def stub(self,status):
  return lambda job,*unused:dict(trial_id=job['trial_id'],object_id='graspnet_044',status=status,actual_physics_steps=0,training_label=None,inputs_unchanged=True)
 def test_sort_and_fail_closed_on_conflicting_score_rank(self):
  self.assertEqual([r['global_score_rank']for r in m.rank_candidates([dict(global_score_rank=5,learned_score=.9),dict(global_score_rank=1,learned_score=1.2)])],[1,5])
  for rows in ([dict(global_score_rank=1,learned_score=.9)]*2,[dict(global_score_rank=0,learned_score=.9),dict(global_score_rank=1,learned_score=1.2)]):
   with self.assertRaises(ValueError):m.rank_candidates(rows)
 def test_stop_immediately_at_first_ready(self):
  calls=[];verified=[]
  def prepare(job,*unused):
   calls.append(job['trial_id']);return self.stub('ready'if len(calls)==2 else'no_full_arm_plan_in_declared_budget')(job)
  r=m.prepare_object(self.item(),Path('/tmp'),Path('/tmp/config'),prepare,lambda c,r:verified.append(r['trial_id']))
  self.assertEqual(len(calls),2);self.assertEqual(verified,['graspnet_044_port_rank_00001']);self.assertEqual(r['arm_preparations'],2)
 def test_all_rejections_respect_shared_four_budget_and_null_preserved(self):
  r=m.prepare_object(self.item(),Path('/tmp'),Path('/tmp/config'),self.stub('no_full_arm_plan_in_declared_budget'),lambda*x:self.fail())
  self.assertEqual(r['arm_preparations'],4);self.assertIsNone(r['selected'])
  r=m.prepare_object(self.item(0),Path('/tmp'),Path('/tmp/config'),lambda*x:self.fail(),lambda*x:self.fail())
  self.assertEqual(r['status'],'no_geometry_admitted_candidate');self.assertEqual(r['arm_preparations'],0)
 def test_unknown_fault_or_physical_return_not_hidden_as_null(self):
  for status,steps in [('unknown',0),('ready',1)]:
   def bad(job,*unused):return dict(self.stub(status)(job),actual_physics_steps=steps)
   with self.assertRaises(ValueError):m.prepare_object(self.item(),Path('/tmp'),Path('/tmp/cfg'),bad,lambda*x:None)
 def test_old_output_reuse_rejected_before_read_or_mutation(self):
  with tempfile.TemporaryDirectory()as td:
   output=Path(td);sentinel=output/'old.txt';sentinel.write_text('unchanged')
   a=argparse.Namespace(diagnostic='/missing',backend_config='/missing',calibration='/missing',output_dir=output,workers=4)
   with self.assertRaises(ValueError):m.run(a,prepare_fn=lambda*x:self.fail())
   self.assertEqual(sentinel.read_text(),'unchanged');self.assertEqual(list(output.iterdir()),[sentinel])
if __name__=='__main__':unittest.main(verbosity=2)
