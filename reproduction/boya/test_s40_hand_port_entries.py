import argparse,ast,copy,importlib.util,json,tempfile,unittest
from pathlib import Path

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
HERE=Path(__file__).resolve().parent
p=module('prep_hand_v3',HERE/'prepare_s40_port_top1_v3.py')
r=module('runner_hand_v2',HERE/'run_s40_port_trials_v2.py')
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
  source=root/'source.py';source.write_text('frozen_source')
  inputs={str(cal.resolve()):p.sha(cal)};sources={str(source.resolve()):p.sha(source)}
  objs=[]
  for oid in sorted(p.OBJECT_IDS):
   local=diag/oid;local.mkdir()
   summary=dict(object_id=oid,arm_checked=0,selected=None,shortlist=0,geometry_admitted=0,geometry_checked=0)
   objs.append(summary);write(local/'RESULTS.json',summary);write(local/'RANKING_FREEZE.json',frozen())
   write(local/'shortlist.json',dict(flip_policy='both',candidates=[]))
  report=dict(schema=p.DIAGNOSTIC_SCHEMA,status='complete',flip_policy='both',head_recipe='code100',
   object_count=8,new_physics_steps=0,labels_used_for_ranking=False,old_ready_queue_read=False,
   no_threshold_tuning=True,input_sha256=inputs,source_sha256=sources,objects=objs)
  write(diag/'RESULTS.json',report)
  native=root/'native.json';write(native,dict(complete=True,new_physics_steps=0,
   results=[dict(object_id=o['object_id'])for o in objs]))
  return diag,cal,native,report,inputs,sources
 def test_catalog_accepts_support_domain_and_no_cloud_admission(self):
  with tempfile.TemporaryDirectory()as tmp:
   diag,cal,native,report,inputs,sources=self.fixture(Path(tmp));freeze=p.Inputs()
   out,items=p.diagnostic_catalog(diag,cal,freeze,native)
   self.assertEqual(out,report);self.assertEqual(len(items),8)
   self.assertTrue(all(not x['candidates']for x in items))
   self.assertTrue(all(x['ranking_query_contract']['support_plane_queries_before_freeze']==64 for x in items))
   self.assertTrue(all(x['native_preflight']['summary_sha256']==p.sha(native)for x in items))
 def test_freeze_contract_rejects_collision_conflation_and_bad_counts(self):
  for changed in [dict(support_plane_queries_before_freeze=0),dict(support_plane_queries_before_freeze=True),
   dict(geometry_calls_before_freeze=1),dict(geometry_calls_before_freeze=False),
   dict(support_plane_is_full_collision_admission=True),dict(query_count_definitions={})]:
   with self.assertRaises(ValueError):p.validate_ranking_freeze(dict(frozen(),**changed))
 def test_v2_paper20_and_oneflip_are_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   diag,cal,native,report,*unused=self.fixture(Path(tmp))
   for changed in [dict(schema='anydex_s40_boya_port_v2_internal_diagnostic'),dict(head_recipe='paper20'),dict(flip_policy='single')]:
    write(diag/'RESULTS.json',dict(report,**changed))
    with self.assertRaises(ValueError):p.diagnostic_catalog(diag,cal,p.Inputs(),native)
 def test_preserved_global_rank_and_shared_arm4_first_ready(self):
  rows=[dict(global_score_rank=9,learned_score=.9),dict(global_score_rank=2,learned_score=1.2)]
  self.assertEqual([x['global_score_rank']for x in p.rank_candidates(rows)],[2,9])
  with self.assertRaises(ValueError):p.rank_candidates([dict(global_score_rank=1,learned_score=.9),dict(global_score_rank=2,learned_score=1.2)])
  item=dict(object_id='graspnet_044',geometry_admitted_total=8,
   candidates=[dict(row=dict(global_score_rank=i,if_flip=bool(i%2)),job=dict(trial_id=str(i)))for i in range(8)])
  for success_at,expected in [(2,2),(5,4)]:
   calls=[]
   def prepare(job,*unused):
    calls.append(job['trial_id']);return dict(trial_id=job['trial_id'],object_id='graspnet_044',
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
  q=root/'queue.json';queue=dict(schema='anydex_boya_collection_queue_v1',complete=True,errors=[],
   benchmark_only=True,independent_training_eligible=False,inputs_unchanged=True,physical_steps=0,
   head_recipe='code100',diagnostic_schema=p.DIAGNOSTIC_SCHEMA,
   support_plane_filter_is_full_collision_admission=False,source_diagnostic=str(diag),
   source_diagnostic_sha256=p.sha(diag/'RESULTS.json'),objects=objs,trials=trials,
   input_sha256=inputs,source_sha256=sources)
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
 def test_original_execution_and_media_control_functions_reused_verbatim(self):
  old=ast.parse(r.PRIOR_EXECUTION_SOURCE.read_text());new=ast.parse(Path(r.__file__).read_text())
  def function(tree,name):return next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name==name)
  for name in ['process','one']:
   self.assertEqual(ast.dump(function(old,name),include_attributes=False),ast.dump(function(new,name),include_attributes=False))
  oldtext=r.PRIOR_EXECUTION_SOURCE.read_text();newtext=Path(r.__file__).read_text()
  self.assertEqual(oldtext[oldtext.index(' if a.render:'):oldtext.index(" print(json.dumps")],newtext[newtext.index(' if a.render:'):newtext.index(" print(json.dumps")])
 def test_output_exists_rejected_before_input_reads(self):
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp);sentinel=out/'old';sentinel.write_text('unchanged')
   a=argparse.Namespace(diagnostic=Path('/missing'),backend_config=Path('/missing'),calibration=Path('/missing'),native_preflight=Path('/missing'),output_dir=out,workers=4)
   with self.assertRaises(ValueError):p.run(a,prepare_fn=lambda*x:self.fail())
   with self.assertRaises(ValueError):r.validate_execution_inputs(argparse.Namespace(output=out,workers=3))
   self.assertEqual(sentinel.read_text(),'unchanged')
if __name__=='__main__':unittest.main(verbosity=2)
