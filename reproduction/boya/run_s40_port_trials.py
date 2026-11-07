"""Execute each frozen internal Boya port top1 once, then replay its actual trace.

No outcome-dependent retry or training write-back. No-plan objects retain their
place in the original internal-eight denominator. Reuses the original temporal
v3 executor and original RTX trace renderer, with independent CPU workers.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,html,json,os,subprocess,sys,time,traceback
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
from terminal_collection_outcome import certify_known_task_stop
from execute_collection_queue_v3 import install_temporal_contract
install_temporal_contract()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def process(argv,log,timeout=1800):
 began=time.monotonic()
 with log.open('xb')as f:rc=subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
 return dict(returncode=rc,wall_s=time.monotonic()-began,log=str(log),argv=list(map(str,argv)))
def run(a):
 queue=read(a.queue);diagnostic=read(a.diagnostic/'RESULTS.json')
 if diagnostic['schema']!='anydex_s40_boya_port_v2_internal_diagnostic' or diagnostic['status']!='complete' or diagnostic['head_recipe']!='code100':raise ValueError('Frozen complete code100 Boya port diagnostic required')
 if queue.get('complete')is not True or queue.get('errors') or queue.get('benchmark_only')is not True:raise ValueError('Complete benchmark-only top1 queue required')
 ids=[v['object_id']for v in diagnostic['objects']];trials=queue['trials']
 if len(ids)!=8 or len(set(ids))!=8 or len({v['object_id']for v in trials})!=len(trials)or not all(v['object_id']in ids and v['status']=='ready'for v in trials):raise ValueError('At most one frozen ready top1 per internal object required')
 if a.output.exists():raise ValueError('Fresh output required')
 a.output.mkdir(parents=True);(a.output/'manifests').mkdir();(a.output/'execution').mkdir()
 inputs={str(a.queue.resolve()):sha(a.queue),str((a.diagnostic/'RESULTS.json').resolve()):sha(a.diagnostic/'RESULTS.json')}
 sources={str(p):sha(p)for p in [Path(__file__).resolve(),BOYA/'execute_collection_queue_v3.py',BOYA/'execute_collection_queue_v2.py',BOYA/'terminal_collection_outcome.py',BOYA/'render_native.py']}
 plan=dict(schema='anydex_s40_port_top1_execution_plan_v1',objects=ids,selected_trial_ids=[v['trial_id']for v in trials],
  selection='Frozen learned score order, hand geometry and first arm-ready in shared four-plan budget',actual_attempts_per_object=1,
  outcome_dependent_retry=False,benchmark_only=True,training_writeback=False,workers=a.workers,physics_device='cpu',
  cross_object_parallelism='Independent native CPU processes; not one GPU physics scene',input_sha256=inputs,source_sha256=sources)
 save(a.output/'PLAN.json',plan);plan_sha=sha(a.output/'PLAN.json');began=time.monotonic();results=[];errors=[]
 def one(row):
  oid=row['object_id'];m=a.output/'manifests'/f'{oid}.json';doc=dict(queue,trials=[row],nulls=[],benchmark_only=True,
    parent_top1_manifest=dict(path=str(a.queue.resolve()),sha256=inputs[str(a.queue.resolve())]))
  save(m,doc);target=a.output/'execution'/oid
  receipt=process([sys.executable,'-B',BOYA/'execute_collection_queue_v3.py','--manifest',m,'--output-dir',target,
    '--device','cpu','--mode','serial','--batch-size','1','--benchmark-only','--paired-benchmark-id',plan_sha],a.output/f'{oid}.execution.log')
  d=read(target/'RESULTS.json')
  if receipt['returncode'] or d.get('status')not in ['complete','complete_with_null'] or len(d.get('trials',[]))!=1:raise RuntimeError('Incomplete execution: '+oid)
  v=d['trials'][0]
  if v['trial_id']!=row['trial_id']or v['trial_binding_sha256']!=row['trial_binding_sha256']or not v['benchmark_only']:raise ValueError('Execution binding changed')
  certificate=None
  if not v.get('label_eligible'):certificate=certify_known_task_stop(v)
  return dict(object_id=oid,trial=v,terminal_certificate=certificate,process=receipt,actual_grasps=d['actual_grasps'],physical_steps=d['physical_steps'])
 with ThreadPoolExecutor(max_workers=a.workers)as pool:
  active={pool.submit(one,row):row for row in trials}
  for f in as_completed(active):
   try:results.append(f.result())
   except Exception:errors.append(dict(object_id=active[f]['object_id'],error=traceback.format_exc()))
   save(a.output/'PROGRESS.json',dict(status='executing',finished=len(results),errors=errors,planned=len(trials)))
 rows=[v['trial']for v in results];successes=sum(v.get('label')==1 for v in rows)
 report=dict(schema='anydex_s40_port_top1_execution_results_v1',status='complete'if not errors else'infrastructure_error',objects=8,
  selected_top1=len(trials),no_plan_objects=[oid for oid in ids if oid not in {v['object_id']for v in trials}],
  actual_attempts=sum(v['actual_grasps']for v in results),successes=successes,eligible_failures=sum(v.get('label')==0 for v in rows),
  terminal_nulls=sum(v.get('label')is None for v in rows),physical_steps=sum(v['physical_steps']for v in results),
  success_fraction_over_internal_objects=successes/8,results=results,errors=errors,benchmark_only=True,training_writeback=False,
  external_validation=False,outcome_dependent_retry=False,inputs_and_sources_unchanged=all(sha(p)==h for p,h in {**inputs,**sources}.items()),
  new_training_updates=0,new_CGR_forwards=0,wall_s=time.monotonic()-began,plan_sha256=plan_sha,videos=[])
 save(a.output/'RESULTS.json',report)
 if errors:raise RuntimeError('Physical execution incomplete; original partial results retained')
 if a.render:
  video_root=a.output/'videos';video_root.mkdir();index=['<!doctype html><meta charset="utf-8"><title>S40 Boya internal top1</title>'];media_errors=[]
  for v in results:
   row=v['trial'];oid=v['object_id'];target=video_root/oid
   rec=process([sys.executable,'-B',BOYA/'render_native.py','--execution',row['execution_dir'],'--plan-dir',row['source_plan_dir'],
    '--output-dir',target,'--object-label',oid+' label='+str(row['label']),'--render-gpu','0','--include-failed-trace'],a.output/f'{oid}.render.log')
   video=target/'isaac_full_arm_pickup.mp4'
   if rec['returncode']or not video.is_file():media_errors.append(dict(object_id=oid,process=rec));continue
   item=dict(object_id=oid,label=row['label'],path=str(video),sha256=sha(video),process=rec)
   report['videos'].append(item);index.append('<h2>'+html.escape(oid+' label='+str(row['label']))+'</h2><video controls width="960" src="'+video.relative_to(video_root).as_posix()+'"></video>')
  (video_root/'index.html').write_text('\n'.join(index));report['media_errors']=media_errors
  if media_errors:report['status']='physical_complete_media_partial'
  report['video_index']=str(video_root/'index.html');report['wall_s']=time.monotonic()-began;save(a.output/'RESULTS.json',report)
 print(json.dumps({k:v for k,v in report.items()if k not in ['results','videos']},indent=2),flush=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--queue',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--workers',type=int,choices=range(1,5),default=3);p.add_argument('--render',action='store_true')
 run(p.parse_args())
