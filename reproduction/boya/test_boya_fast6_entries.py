import argparse,ast,copy,importlib.util,json,tempfile,unittest
from pathlib import Path

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
HERE=Path(__file__).resolve().parent
p=module('prep_hand_v3',HERE/'prepare_boya_fast6_top1.py')
r=module('runner_hand_v2',HERE/'run_boya_fast6_trials.py')
DEFINITIONS=dict(geometry_calls_before_freeze='full_cloud_collision_calls',
 support_plane_queries_before_freeze='open_hand_table_support_function_queries_not_full_collision_admission')

def frozen():return dict(labels_used_for_ranking=False,old_admitted_queue_read=False,
 geometry_calls_before_freeze=0,arm_calls_before_freeze=0,support_plane_queries_before_freeze=64,
 query_count_definitions=DEFINITIONS,support_plane_is_full_collision_admission=False,
 outputs_sha256={},ranking_inputs_sha256={})
def write(path,value):path.write_text(json.dumps(value))

class HandPortContract(unittest.TestCase):
 def fixture(self,root):
  diag=root/'diagnostic';diag.mkdir();cal=root/'cal.json';cal.write_text('{}')
  protocol=root/'protocol.json';base_protocol=dict(schema=p.PROTOCOL_SCHEMA,objects=[dict(object_id=x)for x in sorted(p.OBJECT_IDS)],object_count=6,arm_budget_per_object=4,physical_attempts_per_object=1,outcome_dependent_retry=False,benchmark_only=True,training_writeback=False,head_recipe=p.HEAD_RECIPE)
  source=root/'source.py';source.write_text('frozen_source')
  inputs={str(cal.resolve()):p.sha(cal)};sources={str(source.resolve()):p.sha(source)}
  from settle_s40_scenes import POLICY as SETTLING
  from planning_recovery import POLICY as PLANNING
  base_protocol.update(source_unit_to_m=.001,object_mass_kg=.1,object_friction=.6,settling_policy=SETTLING,planning_recovery_policy=PLANNING,input_sha256=inputs,source_sha256=sources)
  write(protocol,base_protocol)
  objs=[]
  for oid in sorted(p.OBJECT_IDS):
   local=diag/oid;local.mkdir()
   summary=dict(object_id=oid,arm_checked=0,selected=None,shortlist=0,geometry_admitted=0,geometry_checked=0)
   objs.append(summary);write(local/'RESULTS.json',summary);write(local/'RANKING_FREEZE.json',frozen())
   write(local/'shortlist.json',dict(flip_policy='both',candidates=[]))
  report=dict(schema=p.DIAGNOSTIC_SCHEMA,status='complete',flip_policy='both',head_recipe=p.HEAD_RECIPE,
   object_count=6,external_protocol_ref=dict(path=str(protocol),sha256=p.sha(protocol)),new_physics_steps=0,labels_used_for_ranking=False,old_ready_queue_read=False,
   no_threshold_tuning=True,input_sha256=inputs,source_sha256=sources,objects=objs)
  write(diag/'RESULTS.json',report)
  native=root/'native.json';write(native,dict(complete=True,new_physics_steps=0,
   results=[dict(object_id=o['object_id'])for o in objs]))
  return diag,cal,native,report,inputs,sources
 def test_catalog_accepts_support_domain_and_no_cloud_admission(self):
  with tempfile.TemporaryDirectory()as tmp:
   diag,cal,native,report,inputs,sources=self.fixture(Path(tmp));freeze=p.Inputs()
   out,items=p.diagnostic_catalog(diag,cal,freeze,native,diag.parent/'protocol.json')
   self.assertEqual(out,report);self.assertEqual(len(items),6)
   self.assertTrue(all(not x['candidates']for x in items))
   self.assertTrue(all(x['ranking_query_contract']['support_plane_queries_before_freeze']==64 for x in items))
   self.assertTrue(all(x['native_preflight']['summary_sha256']==p.sha(native)for x in items))
 def test_freeze_contract_rejects_collision_conflation_and_bad_counts(self):
  for changed in [dict(support_plane_queries_before_freeze=-1),dict(support_plane_queries_before_freeze=True),
   dict(geometry_calls_before_freeze=1),dict(geometry_calls_before_freeze=False),
   dict(support_plane_is_full_collision_admission=True),dict(query_count_definitions={})]:
   with self.assertRaises(ValueError):p.validate_ranking_freeze(dict(frozen(),**changed))
 def test_v2_paper20_and_oneflip_are_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   diag,cal,native,report,*unused=self.fixture(Path(tmp))
   for changed in [dict(schema='anydex_s40_boya_port_v2_internal_diagnostic'),dict(head_recipe='paper20'),dict(flip_policy='single')]:
    write(diag/'RESULTS.json',dict(report,**changed))
    with self.assertRaises(ValueError):p.diagnostic_catalog(diag,cal,p.Inputs(),native,diag.parent/'protocol.json')
 def test_preserved_global_rank_and_shared_arm4_first_ready(self):
  rows=[dict(global_score_rank=9,learned_score=.9),dict(global_score_rank=2,learned_score=1.2)]
  self.assertEqual([x['global_score_rank']for x in p.rank_candidates(rows)],[2,9])
  with self.assertRaises(ValueError):p.rank_candidates([dict(global_score_rank=1,learned_score=.9),dict(global_score_rank=2,learned_score=1.2)])
  item=dict(object_id='egad_A0',geometry_admitted_total=8,
   candidates=[dict(row=dict(global_score_rank=i,if_flip=bool(i%2)),job=dict(trial_id=str(i)))for i in range(8)])
  for success_at,expected in [(2,2),(5,4)]:
   calls=[]
   def prepare(job,*unused):
    calls.append(job['trial_id']);return dict(trial_id=job['trial_id'],object_id='egad_A0',
     status='ready'if len(calls)==success_at else'no_full_arm_plan_in_declared_budget',
     actual_physics_steps=0,training_label=None,inputs_unchanged=True)
   result=p.prepare_object(item,Path('/tmp'),Path('/tmp'),prepare,lambda*x:None)
   self.assertEqual(len(calls),expected);self.assertEqual(result['arm_preparations'],expected)
   self.assertEqual(result['selected']is not None,success_at==2)
 def queue_fixture(self,root):
  diag,cal,native,report,inputs,sources=self.fixture(root)
  objs=[];trials=[]
  for i,oid in enumerate(sorted(p.OBJECT_IDS)):
   ready=dict(object_id=oid,trial_id=oid+'_rank1',status='ready')if i==0 else None
   attempts=[dict(status='ready',result=ready)]if ready else[]
   objs.append(dict(object_id=oid,arm_budget=4,arm_preparations=len(attempts),attempts=attempts,selected=ready))
   if ready:trials.append(ready)
  q=root/'queue.json';queue=dict(schema=p.QUEUE_SCHEMA,external_protocol_ref=report['external_protocol_ref'],object_count=6,training_writeback=False,outcome_dependent_retry=False,physical_attempts_per_object=1,arm_budget_per_object=4,complete=True,errors=[],
   benchmark_only=True,independent_training_eligible=False,inputs_unchanged=True,physical_steps=0,
   head_recipe=p.HEAD_RECIPE,diagnostic_schema=p.DIAGNOSTIC_SCHEMA,
   support_plane_filter_is_full_collision_admission=False,source_diagnostic=str(diag),
   source_diagnostic_sha256=p.sha(diag/'RESULTS.json'),objects=objs,trials=trials,
   input_sha256=inputs,source_sha256=sources)
  queue['nulls']=[dict(object_id=o['object_id'],status='no_geometry_admitted_candidate')for o in objs if o['selected']is None]
  write(q,queue);a=argparse.Namespace(queue=q,diagnostic=diag,output=root/'new_execution',workers=3,render=True)
  return a,queue
 def test_runner_accepts_exactly_one_first_ready_and_rejects_retry_or_foreign_source(self):
  with tempfile.TemporaryDirectory()as tmp:
   a,q=self.queue_fixture(Path(tmp));self.assertEqual(len(r.validate_execution_inputs(a)[0]['trials']),1)
   for alter in ['budget','first_ready','source','flip']:
    changed=copy.deepcopy(q)
    if alter=='budget':changed['objects'][0]['arm_preparations']=5
    if alter=='first_ready':changed['objects'][0]['attempts'].append(dict(status='ready',result=changed['trials'][0]));changed['objects'][0]['arm_preparations']=2
    if alter=='source':changed['source_diagnostic_sha256']='0'*64
    if alter=='flip':
     d=json.loads((a.diagnostic/'RESULTS.json').read_text());d['flip_policy']='single';write(a.diagnostic/'RESULTS.json',d)
    write(a.queue,changed)
    with self.assertRaises(ValueError):r.validate_execution_inputs(a)
 def test_original_native_process_function_reused(self):
  self.assertIs(r.process,r.prior.process)
  self.assertIs(p.prepare_object,p.prior.prepare_object)
  self.assertIs(p.verify_selected,p.prior.verify_selected)
 def test_actual_external_evaluator_emits_matching_freeze_contract(self):
  tree=ast.parse((HERE/'evaluate_boya_external.py').read_text())
  freeze_calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='save'and n.args and isinstance(n.args[0],ast.BinOp)and isinstance(n.args[0].right,ast.Constant)and n.args[0].right.value=='RANKING_FREEZE.json']
  self.assertEqual(len(freeze_calls),1)
  keywords={k.arg:k.value for k in freeze_calls[0].args[1].keywords}
  definition_node=keywords['query_count_definitions']
  self.assertIsInstance(definition_node,ast.Call)
  definitions={k.arg:ast.literal_eval(k.value)for k in definition_node.keywords}
  contract=p.validate_ranking_freeze(dict(frozen(),query_count_definitions=definitions))
  self.assertEqual(contract['query_count_definitions'],DEFINITIONS)
  self.assertEqual(ast.literal_eval(keywords['labels_used_for_ranking']),False)
  self.assertEqual(ast.literal_eval(keywords['geometry_calls_before_freeze']),0)
  self.assertEqual(ast.literal_eval(keywords['arm_calls_before_freeze']),0)
 def test_zero_support_queries_are_valid_nulls_not_geometry_admission(self):
  contract=p.validate_ranking_freeze(dict(frozen(),support_plane_queries_before_freeze=0))
  self.assertEqual(contract['support_plane_queries_before_freeze'],0)
 def test_protocol_cannot_change_ids_or_attempt_policy(self):
  with tempfile.TemporaryDirectory()as tmp:
   diag,cal,native,report,*unused=self.fixture(Path(tmp));protocol=diag.parent/'protocol.json';original=json.loads(protocol.read_text())
   for alter in [dict(physical_attempts_per_object=2),dict(outcome_dependent_retry=True),dict(head_recipe='code100')]:
    write(protocol,dict(original,**alter))
    with self.assertRaises(ValueError):p.validate_protocol(protocol,p.Inputs())
 def test_all_no_plan_objects_remain_in_fixed_external_denominator(self):
  with tempfile.TemporaryDirectory()as tmp:
   a,q=self.queue_fixture(Path(tmp));q['trials']=[]
   q['nulls']=[dict(object_id=o['object_id'],status='no_geometry_admitted_candidate')for o in q['objects']]
   for o in q['objects']:o.update(selected=None,attempts=[],arm_preparations=0)
   write(a.queue,q);a.render=False;report=r.run(a)
   self.assertEqual(report['objects'],6);self.assertEqual(len(report['no_plan_objects']),6)
   self.assertEqual(report['actual_attempts'],0);self.assertEqual(report['success_fraction_over_external_objects'],0)
   self.assertTrue(report['external_validation']);self.assertFalse(report['training_writeback'])
 def test_output_exists_rejected_before_input_reads(self):
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp);sentinel=out/'old';sentinel.write_text('unchanged')
   a=argparse.Namespace(diagnostic=Path('/missing'),backend_config=Path('/missing'),calibration=Path('/missing'),native_preflight=Path('/missing'),output_dir=out,workers=4)
   with self.assertRaises(ValueError):p.run(a,prepare_fn=lambda*x:self.fail())
   with self.assertRaises(ValueError):r.validate_execution_inputs(argparse.Namespace(output=out,workers=3))
   self.assertEqual(sentinel.read_text(),'unchanged')


class StageRecoveryContract(unittest.TestCase):
 def test_completed_stage_skips_and_failure_log_blocks_replay(self):
  import run_boya_fast6 as driver
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);protocol=dict(source_sha256={},input_sha256={})
   write(root/'protocol.json',protocol)
   write(root/'EVALUATION_FREEZE.json',dict(protocol_sha256=driver.sha(root/'protocol.json')))
   write(root/'EVALUATION_PROGRESS.json',dict(stages=[dict(stage='initialize',returncode=0)]))
   with patch.object(driver.subprocess,'run',side_effect=AssertionError('must not execute')):
    driver.stage_run(root,'initialize')
    (root/'observe.log').write_text('failed startup retained')
    with self.assertRaises(ValueError):driver.stage_run(root,'observe')
 def test_incomplete_development_cannot_freeze_validation(self):
  import run_boya_fast6 as driver
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);write(root/'planning.json',dict(complete=False,errors=[],physics_steps=0,results=[]))
   write(root/'candidate.json',dict(status='infrastructure_error'))
   with self.assertRaises(ValueError):driver.validate_development(root/'planning.json',root/'candidate.json')
if __name__=='__main__':unittest.main(verbosity=2)
