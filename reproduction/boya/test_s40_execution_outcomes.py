import unittest
from s40_execution_outcomes import partition_execution_rows as partition

def row(tid,label=1,n=2100,eligible=True):return dict(trial_id=tid,object_id='objectA',label=label,physics_steps=n,label_eligible=eligible)
def valid(r):
 if not r.get('artifact_valid',True):raise ValueError('command SHA differs')
def cert(r):
 if not r.get('artifact_valid',True):raise ValueError('wrong prefix')
 return dict(certified=True,label=None,label_eligible=False,outcome='known_task_terminated_null',trial_id=r['trial_id'],object_id=r['object_id'],physics_steps=r['physics_steps'],expected_command_steps=2100,reason=r.get('stop_reason','object_pushed_before_closure'))
def run(rows,ids,**kw):return partition(rows,ids,validate_full_label=valid,certify_terminal=cert,**kw)

class ReconciliationTests(unittest.TestCase):
 def test_complete_negative_is_real_label_and_continues(self):
  x=run([row('a',0)],['a']);self.assertEqual((x['eligible_labels'],x['negatives'],x['complete']),(1,1,True));self.assertFalse(x['should_stop_new_dispatch'])
 def test_two_certified_stop_reasons_are_consumed_null(self):
  for reason in ['object_pushed_before_closure','deep_collision_stop']:
   with self.subTest(reason=reason):
    a=row('a',None,1100,False);a['stop_reason']=reason;x=run([a,row('b')],['a','b']);self.assertEqual((x['terminal_nulls'],x['eligible_labels'],x['known_recorded_physics_steps']),(1,1,3200));self.assertTrue(x['complete']);self.assertFalse(x['should_stop_new_dispatch']);self.assertEqual(x['negatives'],0)
 def test_unknown_stop_reason_must_stop_and_not_replay(self):
  a=row('a',None,1100,False);a['stop_reason']='unknown_fault';x=run([a],['a']);self.assertTrue(x['should_stop_new_dispatch']);self.assertIn('a',x['automatic_replay_forbidden_trial_ids']);self.assertEqual(x['terminal_nulls'],0)
 def test_tampered_prefix_cannot_be_consumed(self):
  a=row('a',None,1100,False);a['artifact_valid']=False;x=run([a],['a']);self.assertEqual(x['unknown'],1);self.assertEqual(x['consumed_trial_ids'],[])
 def test_submitted_missing_lane_is_unknown(self):
  x=run([row('a')],['a','b']);self.assertEqual(x['unknown_rows'][0]['trial_id'],'b');self.assertTrue(x['should_stop_new_dispatch']);self.assertFalse(x['physical_cost_complete'])
 def test_zero_step_placeholder_alone_is_unknown(self):
  a=row('a',None,0,False);a['null_reason']='queue_not_dispatched_after_infrastructure_error';x=run([a],['a']);self.assertEqual(x['unknown'],1)
 def test_unsubmitted_lane_needs_explicit_dispatch_proof(self):
  x=run([row('a')],['a','b'],proven_unexecuted_ids=['b']);self.assertEqual(x['unknown'],0);self.assertFalse(x['should_stop_new_dispatch']);self.assertFalse(x['complete']);self.assertEqual(x['proven_unexecuted_trial_ids'],['b']);self.assertNotIn('b',x['automatic_replay_forbidden_trial_ids'])
 def test_proof_conflicting_with_actual_steps_is_error(self):
  with self.assertRaises(ValueError):run([row('a')],['a'],proven_unexecuted_ids=['a'])
 def test_duplicate_and_foreign_ids_rejected(self):
  for rows,ids in [([row('a'),row('a')],['a']),([row('b')],['a'])]:
   with self.subTest(rows=rows):
    with self.assertRaises(ValueError):run(rows,ids)
 def test_no_null_to_zero_and_no_bool_label(self):
  for a in [row('a',None,2100,False),row('a',True),row('a',False),row('a',0,0,True)]:
   with self.subTest(row=a):
    x=run([a],['a']);self.assertEqual(x['eligible_labels'],0);self.assertEqual(x['negatives'],0);self.assertTrue(x['should_stop_new_dispatch'])
 def test_full_label_artifacts_required(self):
  a=row('a');a['artifact_valid']=False;x=run([a],['a']);self.assertEqual(x['unknown'],1);self.assertEqual(x['positives'],0)
 def test_first_failure_preserves_sibling_complete_and_costs(self):
  a=row('a',None,197,False);a['artifact_valid']=False;x=run([a,row('b',0)],['a','b','c']);self.assertTrue(x['should_stop_new_dispatch']);self.assertEqual((x['negatives'],x['known_recorded_physics_steps'],x['unknown']),(1,2297,2));self.assertIn('b',x['consumed_trial_ids'])

if __name__=='__main__':unittest.main(verbosity=2)
