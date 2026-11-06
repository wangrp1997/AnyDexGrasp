import importlib.util,json,os,pathlib,tempfile,unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('delivery',str(pathlib.Path(__file__).with_name('run_boya_training_delivery_v2.py')));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Contracts(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='boya_delivery_contract_');self.root=pathlib.Path(self.tmp.name);self.collection=self.root/'collection';self.collection.mkdir();self.output=self.root/'delivery';self.output.mkdir();self.plan=dict(collection_run=str(self.collection),source_sha256={},max_wait_hours=1,poll_seconds=.001)
 def tearDown(self):self.tmp.cleanup()
 def save(self,p,v):p.write_text(json.dumps(v))
 def make(self):
  self.save(self.output/'DELIVERY_PLAN.json',self.plan)
  return m.Delivery(self.output,self.plan)
 def test_completed_producer_unblocks_without_simulation(self):
  self.save(self.collection/'RESULTS.json',dict(complete=True,unknown_execution_trial_ids=[],actual_grasps=431,positives=300,negatives=131));d=self.make();d.wait_collection();self.assertEqual(d.status['status'],'collection_complete');self.assertEqual(d.status['stages']['collection']['actual_grasps'],431)
 def test_incomplete_producer_never_trains(self):
  self.save(self.collection/'RESULTS.json',dict(complete=False));d=self.make()
  with self.assertRaises(RuntimeError):d.wait_collection()
 def test_unknown_producer_never_trains(self):
  self.save(self.collection/'RESULTS.json',dict(complete=True,unknown_execution_trial_ids=['possibly_executed']));d=self.make()
  with self.assertRaises(RuntimeError):d.wait_collection()
 def test_error_status_never_waits_silently(self):
  self.save(self.collection/'PROGRESS.json',dict(status='infrastructure_stopped',error='control exit'));self.save(self.collection/'LAUNCH.json',dict(pid=os.getpid()));d=self.make()
  with self.assertRaisesRegex(RuntimeError,'infrastructure'):d.wait_collection()
 def test_dead_producer_never_waits_silently(self):
  self.save(self.collection/'PROGRESS.json',dict(status='running'));self.save(self.collection/'LAUNCH.json',dict(pid=123));d=self.make()
  with patch.object(m,'alive',return_value=False):
   with self.assertRaisesRegex(RuntimeError,'exited'):d.wait_collection()
 def test_frozen_sources_reject_changes(self):
  f=self.root/'source.py';f.write_text('old');h=m.sha(f);f.write_text('changed')
  with self.assertRaises(ValueError):m.verify({str(f):h})
 def test_certified_task_stop_not_an_infrastructure_error(self):
  row=dict(trial_id='candidate',object_id='obj',label=None,label_eligible=False,physics_steps=187)
  source=self.root/'policy.json';self.save(source,dict(benchmark_only=True,trials=[row]));self.plan.update(policies=[dict(name='baseline',manifest=str(source)),dict(name='learned',manifest=str(source))],validation_object_ids=['obj'])
  d=self.make()
  def fake_command(name,argv):
   p=pathlib.Path(argv[argv.index('--output-dir')+1]);p.mkdir();self.save(p/'RESULTS.json',dict(status='complete_with_null',trials=[row],physical_steps=187,actual_grasps=1));return dict(returncode=0)
  with patch.object(d,'command',side_effect=fake_command),patch.object(m,'certify_known_task_stop',return_value=dict(reason='certified')):
   result=d.physical_comparison()
  self.assertTrue(result['complete']);self.assertEqual(result['metrics']['baseline']['known_task_terminated_nulls'],1);self.assertEqual(result['metrics']['baseline']['successes'],0)
 def test_uncertified_partial_still_blocks_physical_comparison(self):
  row=dict(trial_id='candidate',object_id='obj',label=None,label_eligible=False,physics_steps=187)
  source=self.root/'policy.json';self.save(source,dict(benchmark_only=True,trials=[row]));self.plan.update(policies=[dict(name='baseline',manifest=str(source))],validation_object_ids=['obj'])
  d=self.make()
  def fake_command(name,argv):
   p=pathlib.Path(argv[argv.index('--output-dir')+1]);p.mkdir();self.save(p/'RESULTS.json',dict(status='complete_with_null',trials=[row],physical_steps=187,actual_grasps=1));return dict(returncode=0)
  with patch.object(d,'command',side_effect=fake_command),patch.object(m,'certify_known_task_stop',side_effect=ValueError('unproven')):
   with self.assertRaises(RuntimeError):d.physical_comparison()
 def test_changed_plan_rejected_before_stage(self):
  d=self.make();self.save(self.output/'DELIVERY_PLAN.json',dict(self.plan,poll_seconds=42))
  with self.assertRaisesRegex(ValueError,'plan changed'):d.check()
 def test_all_failure_records_remain_real_results(self):
  self.save(self.collection/'RESULTS.json',dict(complete=True,unknown_execution_trial_ids=[],actual_grasps=4,positives=0,negatives=4));d=self.make();d.wait_collection();self.assertEqual(d.status['status'],'collection_complete');self.assertEqual(d.status['stages']['collection']['negatives'],4)
 def test_unexpected_phase_failure_published(self):
  d=self.make()
  with patch.object(d,'wait_collection',side_effect=RuntimeError('source error')):
   with self.assertRaises(RuntimeError):d.run()
  status=json.loads((self.output/'DELIVERY_STATUS.json').read_text());self.assertEqual(status['status'],'blocked');self.assertIn('source error',status['error'])

if __name__=='__main__':unittest.main(verbosity=2)
