"""Prepare frozen EGAD external top1 plans with unchanged native arm helpers.

This is an explicit external protocol adapter, not a synthetic S40 interface.
Shared four-arm-candidate budget and first-ready selection are inherited verbatim.
No physical execution, labels or training update occur in this entry.
"""
from __future__ import annotations
import argparse,copy,json,math,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
BOYA=Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya')
sys.path.insert(0,str(BOYA))
import prepare_s40_port_top1_v2 as prior
from prepare_s40_port_top1_v2 import (
    ROOT,BUDGET,IDENTITY,Inputs,sha,save,read,rank_candidates,
    prepare_object,verify_selected)
PRIOR_SOURCE_SHA256='77357f60df7217016f8eccac09a5ed61908052a4905bcd2b1ae437954c287db7'
DIAGNOSTIC_SCHEMA='anydex_boya_external_hand_domain_diagnostic_v1'
PROTOCOL_SCHEMA='anydex_boya_external_protocol_v1'
QUEUE_SCHEMA='anydex_boya_external_top1_queue_v1'
OBJECT_IDS={'egad_'+letter+str(index) for letter in ('A','C','E','G') for index in (0,3,6)}
HEAD_RECIPE='preactivation_selected_mse'


def validate_protocol(path,inputs):
 path=Path(path).resolve();doc=inputs.read(path)
 if (doc.get('schema')!=PROTOCOL_SCHEMA or doc.get('object_count')!=12 or doc.get('head_recipe')!=HEAD_RECIPE
     or doc.get('arm_budget_per_object')!=4 or doc.get('physical_attempts_per_object')!=1
     or doc.get('outcome_dependent_retry')is not False or doc.get('benchmark_only')is not True
     or doc.get('training_writeback')is not False):
  raise ValueError('Frozen external12 one-attempt/four-arm-plan benchmark protocol required')
 ids=[r['object_id']for r in doc.get('objects',[])]
 if len(ids)!=12 or len(set(ids))!=12 or set(ids)!=OBJECT_IDS:
  raise ValueError('Frozen external12 EGAD identities required')
 return doc,dict(path=str(path),sha256=inputs.files[str(path)])


def validate_prior_source():
 if sha(prior.__file__)!=PRIOR_SOURCE_SHA256:raise ValueError('Frozen v2 preparation helper changed')

def validate_ranking_freeze(frozen):
 if (frozen.get('labels_used_for_ranking')is not False or frozen.get('old_admitted_queue_read')is not False
     or type(frozen.get('geometry_calls_before_freeze'))is not int or frozen['geometry_calls_before_freeze']!=0
     or type(frozen.get('arm_calls_before_freeze'))is not int or frozen['arm_calls_before_freeze']!=0):
  raise ValueError('Ranking must precede full-cloud collision and arm checks')
 count=frozen.get('support_plane_queries_before_freeze')
 if type(count)is not int or count<0:raise ValueError('Explicit nonnegative support-plane query count required')
 definitions=frozen.get('query_count_definitions',{})
 if (definitions.get('geometry_calls_before_freeze')!='full_cloud_collision_calls'
     or definitions.get('support_plane_queries_before_freeze')!='open_hand_table_support_function_queries_not_full_collision_admission'):
  raise ValueError('Explicit full-cloud versus support-plane query definitions required')
 if frozen.get('support_plane_is_full_collision_admission')is not False:
  raise ValueError('Support-plane domain filtering must not claim full collision admission')
 return dict(geometry_calls_before_freeze=0,support_plane_queries_before_freeze=count,
             query_count_definitions=definitions,support_plane_is_full_collision_admission=False)

def diagnostic_catalog(diagnostic,calibration,inputs,native_preflight,protocol):
 protocol_doc,protocol_ref=validate_protocol(protocol,inputs)
 diagnostic=Path(diagnostic).resolve();report=inputs.read(diagnostic/'RESULTS.json')
 if report.get('external_protocol_ref')!=protocol_ref:raise ValueError('Diagnostic external protocol binding differs')
 if (report.get('schema')!=DIAGNOSTIC_SCHEMA or report.get('status')!='complete'
     or report.get('flip_policy')!='both'or report.get('head_recipe')!=HEAD_RECIPE
     or report.get('object_count')!=12 or report.get('new_physics_steps')!=0
     or report.get('labels_used_for_ranking')is not False or report.get('old_ready_queue_read')is not False
     or report.get('no_threshold_tuning')is not True):raise ValueError('Completed label-blind external12 dual-flip diagnostic required')
 objects=report.get('objects',[])
 if len(objects)!=12 or {r.get('object_id')for r in objects}!=OBJECT_IDS:raise ValueError('Frozen external12 object identities required')
 if not report.get('input_sha256')or not report.get('source_sha256'):raise ValueError('Frozen diagnostic inputs/sources required')
 for section in ('input_sha256','source_sha256'):
  for p,h in report[section].items():inputs.freeze(p,h)
 calibration=Path(calibration).resolve();calsha=inputs.freeze(calibration)
 if report['input_sha256'].get(str(calibration))!=calsha:raise ValueError('Calibration differs from frozen diagnostic')
 catalog=[]
 for summary in objects:
  oid=summary['object_id'];directory=diagnostic/oid;local=inputs.read(directory/'RESULTS.json')
  if local!=summary or summary.get('arm_checked')!=0 or summary.get('selected')is not None:raise ValueError('Diagnostic summary/local result mismatch or prior arm selection')
  freeze_path=directory/'RANKING_FREEZE.json';frozen=inputs.read(freeze_path)
  query_contract=validate_ranking_freeze(frozen)
  for section in ('outputs_sha256','ranking_inputs_sha256'):
   for p,h in frozen.get(section,{}).items():inputs.freeze(p,h)
  shortlist=inputs.read(directory/'shortlist.json')
  if shortlist.get('flip_policy')!='both':raise ValueError('Both-flip shortlist required')
  ranked=rank_candidates(shortlist['candidates']);byrank={r['global_score_rank']:r for r in ranked}
  if len(ranked)!=summary['shortlist']:raise ValueError('Shortlist count differs')
  gp=directory/'geometry.json'
  if gp.exists():geometry=inputs.read(gp)
  elif not ranked and summary['geometry_admitted']==0:geometry=dict(checked=[],admitted=[])
  else:raise ValueError('Missing geometry evidence for nonempty shortlist')
  admitted=geometry.get('admitted',[])
  if len(admitted)!=summary['geometry_admitted']or len(geometry.get('checked',[]))!=summary['geometry_checked']:raise ValueError('Geometry summary/count mismatch')
  checked={r['global_score_rank']:r for r in geometry.get('checked',[])};candidates=[];seen=set()
  for item in admitted:
   rank=item['global_score_rank'];row=item['candidate']
   if rank in seen or rank not in byrank or rank!=row['global_score_rank']:raise ValueError('Foreign or duplicate geometry admission')
   seen.add(rank)
   if any(row.get(k)!=byrank[rank].get(k)for k in IDENTITY)or checked.get(rank)!=row or row.get('geometry',{}).get('ok')is not True:raise ValueError('Geometry identity differs from frozen ranking')
   pool=Path(item['candidate_dir']).resolve()
   if pool.parent!=directory or pool.name!=f'pool_{rank:05d}':raise ValueError('Pool is not the ranked diagnostic output')
   gen=inputs.read(pool/'generation.json');inputs.freeze(pool/'candidates.npz',gen['candidates_sha256']);inputs.freeze(pool/'world_object_pose.json')
   if (gen.get('complete')is not True or gen.get('kind')!='anydex_boya_score_first_single_candidate_v1'
       or gen.get('physics_steps')!=0 or gen.get('selected_indices')!=[0]or gen.get('records')!=[row]
       or gen.get('global_ranking_freeze_sha256')!=sha(freeze_path)):
    raise ValueError('Unbound single-candidate generation')
   for p,h in gen['input_sha256'].items():inputs.freeze(p,h)
   if gen['input_sha256'].get(str(calibration))!=calsha:raise ValueError('Pool calibration mismatch')
   obs=Path(gen['source_observation']).resolve();observed=inputs.read(obs)
   if observed.get('object_id')!=oid:raise ValueError('Foreign pool observation')
   source=Path(gen['source_frame']).resolve()/'raw_camera_candidates.npz';inputs.freeze(source)
   job=dict(trial_id=f'{oid}_port_rank_{rank:05d}',object_id=oid,split='external_deployment_benchmark',
      gesture_id=row['gesture_id'],extra_depth_index=row['extra_depth_index'],candidate_dir=str(pool),observation_json=str(obs),source_npz=str(source),calibration=str(calibration),candidate_indices=[0],max_arm_plans=1)
   candidates.append(dict(row=row,job=job))
  order=rank_candidates([x['row']for x in candidates]);byrank={x['row']['global_score_rank']:x for x in candidates}
  catalog.append(dict(object_id=oid,candidates=[byrank[x['global_score_rank']]for x in order][:BUDGET],geometry_admitted_total=len(candidates),diagnostic_object_dir=str(directory),ranking_query_contract=query_contract))
 if native_preflight is None:raise ValueError('Explicit static native preflight summary required')
 preflight_path=Path(native_preflight).resolve();preflight=inputs.read(preflight_path)
 if preflight.get('complete')is not True or preflight.get('new_physics_steps')!=0 or not isinstance(preflight.get('results'),list):raise ValueError('Complete zero-step static native preflight required')
 native={}
 for row in preflight['results']:
  oid=row.get('object_id')
  if oid in native:raise ValueError('Duplicate static native preflight object')
  native[oid]=row
 if not OBJECT_IDS<=set(native):raise ValueError('Static native preflight does not cover all external objects')
 for item in catalog:
  record=native[item['object_id']];repair=record.get('repair_path');digest=record.get('repair_sha256')
  if bool(repair)!=bool(digest):raise ValueError('Static repair path/SHA must be paired')
  if record.get('result_json'):inputs.freeze(record['result_json'])
  for artifact in record.get('artifacts',[]):inputs.freeze(artifact)
  if repair:inputs.freeze(repair,digest)
  item['native_preflight']=dict(summary_json=str(preflight_path),summary_sha256=inputs.files[str(preflight_path)],repair_path=repair,repair_sha256=digest)
  for candidate in item['candidates']:
   if repair:candidate['job'].update(collision_refinement=str(Path(repair).resolve()),collision_refinement_sha256=digest)
 inputs.verify();return report,catalog

def run(args,prepare_fn=None,selected_verifier=verify_selected):
 validate_prior_source()
 diagnostic=Path(args.diagnostic).resolve();output=Path(args.output_dir).resolve();backend=Path(args.backend_config).resolve();calibration=Path(args.calibration).resolve()
 if type(args.workers)is not int or not 1<=args.workers<=4:raise ValueError('Cross-object workers must be1..4')
 if output.exists():raise ValueError('Use a new output directory; existing results never reused')
 inputs=Inputs();report,catalog=diagnostic_catalog(diagnostic,calibration,inputs,args.native_preflight,args.protocol);inputs.freeze(backend)
 if output==diagnostic or diagnostic in output.parents:raise ValueError('Do not write into frozen diagnostic')
 if prepare_fn is None:
  if str(BOYA)not in sys.path:sys.path.insert(0,str(BOYA))
  from prepare_collection_queue import prepare_one
  prepare_fn=prepare_one
 sources={str(p):sha(p)for p in [Path(__file__).resolve(),Path(prior.__file__).resolve(),BOYA/'prepare_s40_port_top1_v3.py',BOYA/'prepare_collection_queue.py',BOYA/'trial_record.py',BOYA/'asset_adapter.py',BOYA/'native_arm.py']}
 output.mkdir(parents=True,exist_ok=False);plans=output/'plans';plans.mkdir()
 jobs_doc=dict(schema='anydex_boya_external_top1_jobs_v1',diagnostic=str(diagnostic),diagnostic_results_sha256=sha(diagnostic/'RESULTS.json'),backend_config=str(backend),calibration=str(calibration),objects=catalog,jobs=[x['job']for obj in catalog for x in obj['candidates']],workers=args.workers,arm_budget_per_object=BUDGET,benchmark_only=True,source_sha256=sources,input_sha256=inputs.files.copy())
 jobs_path=output/'jobs.json';save(jobs_path,jobs_doc)
 manifest=dict(schema=QUEUE_SCHEMA,complete=False,external_protocol_ref=report['external_protocol_ref'],object_count=12,training_writeback=False,outcome_dependent_retry=False,physical_attempts_per_object=1,arm_budget_per_object=4,trials=[],nulls=[],errors=[],objects=[],jobs_json=str(jobs_path),jobs_sha256=sha(jobs_path),source_sha256=sources,input_sha256=inputs.files.copy(),physical_steps=0,learned_Boya_head_loaded=False,learned_Boya_ranking_used=True,benchmark_only=True,independent_training_eligible=False,inputs_unchanged=False,scope='One final scored and arm-admitted top1 per object; no physical selection or old outcome ranking; finite4 arm candidates/object',source_diagnostic=str(diagnostic),source_diagnostic_sha256=sha(diagnostic/'RESULTS.json'),head_recipe=report['head_recipe'],diagnostic_schema=DIAGNOSTIC_SCHEMA,support_plane_filter_is_full_collision_admission=False,native_preflight=str(Path(args.native_preflight).resolve()),native_preflight_sha256=inputs.files[str(Path(args.native_preflight).resolve())])
 began=time.monotonic();results={}
 with ThreadPoolExecutor(max_workers=args.workers)as pool:
  pending={pool.submit(prepare_object,item,plans,backend,prepare_fn,selected_verifier):item for item in catalog}
  for future in as_completed(pending):
   item=pending[future]
   try:results[item['object_id']]=future.result()
   except Exception as exc:manifest['errors'].append(dict(object_id=item['object_id'],error=repr(exc),traceback=traceback.format_exc()))
 for item in catalog:
  obj=results.get(item['object_id'])
  if obj is None:continue
  manifest['objects'].append(obj)
  if obj['selected']:manifest['trials'].append(obj['selected'])
  else:manifest['nulls'].append(dict(trial_id=item['object_id']+'_port_final_null',object_id=item['object_id'],status=obj['status'],actual_physics_steps=0,training_label=None,arm_preparations=obj['arm_preparations']))
 try:inputs.verify()
 except Exception as exc:manifest['errors'].append(dict(stage='final_input_verification',error=repr(exc)))
 manifest['inputs_unchanged']=not any(e.get('stage')=='final_input_verification'for e in manifest['errors'])
 manifest['complete']=not manifest['errors'];manifest['wall_s']=time.monotonic()-began
 save(output/'queue_manifest.json',manifest)
 if not manifest['complete']:raise RuntimeError('Preparation failed; partial evidence retained, nothing dispatched')
 return manifest

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('protocol','diagnostic','backend-config','calibration','native-preflight','output-dir'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--workers',type=int,default=4);a=p.parse_args();result=run(a)
 print(json.dumps(dict(complete=result['complete'],objects=len(result['objects']),top1=len(result['trials']),nulls=len(result['nulls']),physical_steps=0)))
if __name__=='__main__':main()
