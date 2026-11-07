"""Prepare at most four ranked arm candidates/object, selecting one top1.

V2 binds the existing static native object repair metadata by object identity.
Consumes an immutable completed eight-object Boya dual-flip geometry diagnostic.
No heads, old outcome labels, Kit execution or physical trials are used here.
Native prepare_one performs only the existing read-only geometry/IK preparation.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,re,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
OBJECT_IDS={'graspnet_'+x for x in ('007','008','009','021','044','048','066','069')}
BUDGET=4
IDENTITY=('source_candidate_index','gesture_id','if_flip','extra_depth_index','target_index','canonical_CGR480_identity','learned_score','global_score_rank')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def save(path,doc):
 with Path(path).open('x')as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
def read(path):return json.loads(Path(path).read_text())
class Inputs:
 def __init__(self):self.files={}
 def freeze(self,path,expected=None):
  path=Path(path).resolve();digest=sha(path)
  if expected is not None and digest!=expected:raise ValueError('SHA mismatch: '+str(path))
  if str(path)in self.files and self.files[str(path)]!=digest:raise ValueError('Input changed: '+str(path))
  self.files[str(path)]=digest;return digest
 def read(self,path,expected=None):self.freeze(path,expected);return read(path)
 def verify(self):
  for p,digest in self.files.items():
   if sha(p)!=digest:raise ValueError('Input changed during preparation: '+p)
def rank_candidates(rows):
 """Honor frozen learned rank, retaining gaps caused by the score gate."""
 result=sorted(rows,key=lambda r:r['global_score_rank']);seen=set()
 for r in result:
  rank=r.get('global_score_rank');score=r.get('learned_score')
  if type(rank)is not int or rank<0 or rank in seen or type(score)not in(int,float)or not math.isfinite(score):raise ValueError('Invalid/duplicate frozen rank or score')
  seen.add(rank)
 if any(a['learned_score']<b['learned_score']for a,b in zip(result,result[1:])):raise ValueError('Global ranks disagree with frozen learned ordering')
 return result

def diagnostic_catalog(diagnostic,calibration,inputs,native_preflight):
 diagnostic=Path(diagnostic).resolve();report=inputs.read(diagnostic/'RESULTS.json')
 if (report.get('schema')!='anydex_s40_boya_port_v2_internal_diagnostic'or report.get('status')!='complete'
     or report.get('flip_policy')!='both'or report.get('head_recipe')not in('paper20','code100')
     or report.get('object_count')!=8 or report.get('new_physics_steps')!=0
     or report.get('labels_used_for_ranking')is not False or report.get('old_ready_queue_read')is not False
     or report.get('no_threshold_tuning')is not True):raise ValueError('Completed label-blind eight-object dual-flip port diagnostic required')
 objects=report.get('objects',[])
 if len(objects)!=8 or {r.get('object_id')for r in objects}!=OBJECT_IDS:raise ValueError('Frozen internal eight object identities required')
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
  if (frozen.get('labels_used_for_ranking')is not False or frozen.get('old_admitted_queue_read')is not False
      or frozen.get('geometry_calls_before_freeze')!=0 or frozen.get('arm_calls_before_freeze')!=0):raise ValueError('Ranking was not frozen before geometry/arm')
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
   job=dict(trial_id=f'{oid}_port_rank_{rank:05d}',object_id=oid,split='internal_deployment_benchmark',
      gesture_id=row['gesture_id'],extra_depth_index=row['extra_depth_index'],candidate_dir=str(pool),observation_json=str(obs),source_npz=str(source),calibration=str(calibration),candidate_indices=[0],max_arm_plans=1)
   candidates.append(dict(row=row,job=job))
  order=rank_candidates([x['row']for x in candidates]);byrank={x['row']['global_score_rank']:x for x in candidates}
  catalog.append(dict(object_id=oid,candidates=[byrank[x['global_score_rank']]for x in order][:BUDGET],geometry_admitted_total=len(candidates),diagnostic_object_dir=str(directory)))
 if native_preflight is None:raise ValueError('Explicit static native preflight summary required')
 preflight_path=Path(native_preflight).resolve();preflight=inputs.read(preflight_path)
 if preflight.get('complete')is not True or preflight.get('new_physics_steps')!=0 or not isinstance(preflight.get('results'),list):raise ValueError('Complete zero-step static native preflight required')
 native={}
 for row in preflight['results']:
  oid=row.get('object_id')
  if oid in native:raise ValueError('Duplicate static native preflight object')
  native[oid]=row
 if not OBJECT_IDS<=set(native):raise ValueError('Static native preflight does not cover all internal objects')
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

def prepare_object(item,output,backend_config,prepare_one,verify_selected):
 attempts=[];selected=None
 for candidate in item['candidates'][:BUDGET]:
  job=candidate['job'];result=prepare_one(copy.deepcopy(job),output,backend_config)
  if result.get('actual_physics_steps')!=0 or result.get('training_label')is not None or result.get('inputs_unchanged')is not True:raise ValueError('Preparation returned physical/changed/unclosed result')
  if result.get('trial_id')!=job['trial_id']or result.get('object_id')!=item['object_id']:raise ValueError('Preparation returned foreign identity')
  status=result.get('status')
  if status not in('ready','no_full_arm_plan_in_declared_budget'):raise ValueError('Unknown preparation outcome: '+str(status))
  attempts.append(dict(trial_id=job['trial_id'],global_score_rank=candidate['row']['global_score_rank'],status=status,result=result))
  if status=='ready':
   verify_selected(candidate,result);selected=result;break
 return dict(object_id=item['object_id'],status='ready'if selected else'no_full_arm_plan_in_declared_budget'if item['candidates']else'no_geometry_admitted_candidate',
   attempts=attempts,selected=selected,geometry_admitted_total=item['geometry_admitted_total'],arm_budget=BUDGET,arm_preparations=len(attempts),physical_steps=0)

def verify_selected(candidate,result):
 if str(BOYA)not in sys.path:sys.path.insert(0,str(BOYA))
 from trial_record import check_binding
 b=read(result['binding_json']);check_binding(b);r=candidate['row'];job=candidate['job']
 if sha(result['binding_json'])!=result['binding_sha256']or any([
  b['source_row_index']!=r['source_candidate_index'],b['if_flip']!=r['if_flip'],
  b['canonical_CGR480_identity']!=r['canonical_CGR480_identity'],b['multifinger_pose_finger_type']!=r['gesture_id'],
  b['multifinger_pose_depth_type']!=r['extra_depth_index'],b['two_fingers_pose_depth_type']*4+b['multifinger_pose_depth_type']!=r['target_index'],
  Path(b['source_npz']).resolve()!=Path(job['source_npz']).resolve(),b['source_npz_sha256']!=sha(job['source_npz'])]):raise ValueError('Selected arm binding differs from frozen scored query')
 if result['trial_binding_sha256']!=b['trial_binding_sha256']:raise ValueError('Selected binding echo mismatch')
 plan=read(Path(result['plan_dir'])/'plan.json')
 if plan.get('complete')is not True or plan.get('admitted')is not True or plan['commands_sha256']!=result['commands_sha256']or sha(Path(result['plan_dir'])/'commands.npz')!=result['commands_sha256']:raise ValueError('Selected plan/commands changed')

def run(args,prepare_fn=None,selected_verifier=verify_selected):
 diagnostic=Path(args.diagnostic).resolve();output=Path(args.output_dir).resolve();backend=Path(args.backend_config).resolve();calibration=Path(args.calibration).resolve()
 if type(args.workers)is not int or not 1<=args.workers<=4:raise ValueError('Cross-object workers must be1..4')
 if output.exists():raise ValueError('Use a new output directory; existing results never reused')
 inputs=Inputs();report,catalog=diagnostic_catalog(diagnostic,calibration,inputs,args.native_preflight);inputs.freeze(backend)
 if output==diagnostic or diagnostic in output.parents:raise ValueError('Do not write into frozen diagnostic')
 if prepare_fn is None:
  if str(BOYA)not in sys.path:sys.path.insert(0,str(BOYA))
  from prepare_collection_queue import prepare_one
  prepare_fn=prepare_one
 sources={str(p):sha(p)for p in [Path(__file__).resolve(),BOYA/'prepare_collection_queue.py',BOYA/'trial_record.py',BOYA/'asset_adapter.py',BOYA/'native_arm.py']}
 output.mkdir(parents=True,exist_ok=False);plans=output/'plans';plans.mkdir()
 jobs_doc=dict(schema='anydex_s40_port_top1_jobs_v1',diagnostic=str(diagnostic),diagnostic_results_sha256=sha(diagnostic/'RESULTS.json'),backend_config=str(backend),calibration=str(calibration),objects=catalog,jobs=[x['job']for obj in catalog for x in obj['candidates']],workers=args.workers,arm_budget_per_object=BUDGET,benchmark_only=True,source_sha256=sources,input_sha256=inputs.files.copy())
 jobs_path=output/'jobs.json';save(jobs_path,jobs_doc)
 manifest=dict(schema='anydex_boya_collection_queue_v1',complete=False,trials=[],nulls=[],errors=[],objects=[],jobs_json=str(jobs_path),jobs_sha256=sha(jobs_path),source_sha256=sources,input_sha256=inputs.files.copy(),physical_steps=0,learned_Boya_head_loaded=False,learned_Boya_ranking_used=True,benchmark_only=True,independent_training_eligible=False,inputs_unchanged=False,scope='One final scored and arm-admitted top1 per object; no physical selection or old outcome ranking; finite4 arm candidates/object',source_diagnostic=str(diagnostic),source_diagnostic_sha256=sha(diagnostic/'RESULTS.json'),head_recipe=report['head_recipe'],native_preflight=str(Path(args.native_preflight).resolve()),native_preflight_sha256=inputs.files[str(Path(args.native_preflight).resolve())])
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
 for name in ('diagnostic','backend-config','calibration','native-preflight','output-dir'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--workers',type=int,default=4);a=p.parse_args();result=run(a)
 print(json.dumps(dict(complete=result['complete'],objects=len(result['objects']),top1=len(result['trials']),nulls=len(result['nulls']),physical_steps=0)))
if __name__=='__main__':main()
