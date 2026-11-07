"""Bounded S40 continuation proof tests; no subprocess, Kit or physics."""
from pathlib import Path
import argparse,copy,importlib.util,json,tempfile,unittest
P=Path(__file__).with_name('resume_s40_settled_execution_v3.py');S=importlib.util.spec_from_file_location('resume_s40_test',P);m=importlib.util.module_from_spec(S);S.loader.exec_module(m)
RUN=m.ROOT/'outputs/s40_stable_supplement_20261107_v2/collection'

class ResumeProofs(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.real=m.inspect_original(RUN)
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

 def test_actual157_consumed_and_only38_proven_unexecuted_ready(self):
  p,q,old,actual,records,remaining,proofs,refs=self.real
  self.assertEqual(len(p['slots']),640);self.assertEqual((len(q['trials']),len(q['nulls'])),(195,445))
  self.assertEqual((len(actual),sum(r['physics_steps']for r in actual)),(157,329700))
  self.assertEqual(len(records),157);self.assertEqual(len(remaining),38)
  self.assertFalse({r['trial_id']for r in actual}&{r['trial_id']for r in remaining})
  self.assertEqual(len(proofs['graspnet_060']['trial_ids']),3)
  self.assertTrue(all(r['new_physics_steps']==0 for r in proofs.values()))
  self.assertEqual(set(old['unknown_execution_trial_ids']),set(proofs['graspnet_060']['trial_ids']))

 def test_failed_prepare_proof_rejects_any_worker_or_changed_log(self):
  oid='graspnet_060';ids=self.real[6][oid]['trial_ids'];base=self.root/'original';execution=base/'execution';(execution/'object_queues').mkdir(parents=True);(base/'logs').mkdir()
  directory=execution/oid;directory.mkdir();(directory/'.queue.lock').touch()
  sub=execution/'object_queues'/(oid+'.json');m.save(sub,dict(trials=[dict(trial_id=t)for t in ids]))
  marker=execution/'object_queues'/(oid+'.SUBMITTING.json');m.save(marker,dict(trial_ids=ids,manifest=str(sub),manifest_sha256=m.sha(sub)))
  log=base/'logs'/('execution_'+oid+'.log');log.write_text((RUN/'logs'/log.name).read_text())
  proof=m.zero_prepare_proof(base,oid,ids);self.assertEqual(proof['new_physics_steps'],0)
  (directory/'PREPARED.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'possible execution artifacts'):m.zero_prepare_proof(base,oid,ids)
  (directory/'PREPARED.json').unlink();log.write_text('ambiguous failure')
  with self.assertRaisesRegex(ValueError,'exact zero-Kit'):m.zero_prepare_proof(base,oid,ids)

 def test_freeze_is_exporter_compatible_and_never_replays_existing_output(self):
  args=argparse.Namespace(original_run=RUN,output_dir=self.root/'resume',executor=m.BOYA/'execute_collection_queue_v3.py',workers=4,extra_source=[])
  p=m.freeze(args)
  self.assertEqual(p['schema'],'anydex_boya_s40_collection_v1');self.assertEqual(p['collection_variant'],'settled_execution_resume_v3')
  self.assertEqual((p['inherited_actual_trials'],p['new_slot_count']),(157,38));self.assertEqual(p['inherited_physics_steps'],329700)
  self.assertEqual(m.sha(args.output_dir/'objects.json'),m.sha(RUN/'objects.json'))
  self.assertEqual(m.sha(args.output_dir/'planning/queue_manifest.json'),m.sha(RUN/'planning/queue_manifest.json'))
  self.assertIn(str(args.executor),p['source_sha256']);self.assertIn(str(m.BOYA/'prepare_s40_training_data_v2.py'),p['source_sha256'])
  self.assertIn(str(m.BOYA/'watch_s40_supplement_training_v2.py'),p['source_sha256'])
  self.assertEqual(m.freeze(args),p)
  (args.output_dir/'execution').mkdir()
  with self.assertRaisesRegex(ValueError,'already dispatched'):m.freeze(args)

 def test_inherited_actuals_are_exported_once_without_subprocess(self):
  original,_,_,actual,records,_,_,_=self.real;run=self.root/'no_new';run.mkdir();(run/'planning').mkdir()
  consumed=actual[:2];ids={r['trial_id']for r in consumed};inherited=[r for r in records if r['collection_context']['trial_id']in ids]
  slots=original['slots'];nulls=[r for r in slots if r['trial_id']not in ids]
  m.save(run/'objects.json',m.read(RUN/'objects.json'));m.save(run/'planning/queue_manifest.json',dict(complete=True,trials=consumed,nulls=nulls))
  m.save(run/'INHERITED_EXECUTION.json',dict(rows=consumed,records=inherited,original_protocol_sha256=m.sha(RUN/'protocol.json')))
  p=dict(config=dict(execution_workers=2,executor='forbidden'),slot_count=640,slots=slots,input_sha256={},source_sha256={},manifest_sha256={});m.save(run/'protocol.json',p)
  pipeline=m.Pipeline(run,p)
  def forbidden(*a,**k):raise AssertionError('Consumed source trials must never run again')
  pipeline.command=forbidden;pipeline.publish=lambda s:None;summary,_=pipeline.execution();final=m.read(run/'execution/RESULTS.json')
  self.assertTrue(final['complete']);self.assertEqual(final['actual_grasps'],2);self.assertEqual(final['new_actual_grasps'],0);self.assertEqual(final['new_physical_steps'],0)
  self.assertEqual(final['eligible_labels'],2);self.assertEqual(summary['new_physical_steps'],0)
  exported=[json.loads(x)for x in(run/'execution/all_records.jsonl').read_text().splitlines()]
  for value,source in zip(exported,inherited):
   value.pop('collection_context');core=copy.deepcopy(source);core.pop('collection_context');self.assertEqual(value,core)

if __name__=='__main__':unittest.main(verbosity=2)
