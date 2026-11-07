"""Continue a frozen S40 supplement into merged training; never launch physics."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection-run',type=Path,required=True);p.add_argument('--previous-dataset-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--collector-pid',type=int,required=True);p.add_argument('--timeout-hours',type=float,default=8);a=p.parse_args();out=a.output_dir.resolve()
 if out.exists():raise ValueError('Fresh automatic-training delivery required')
 out.mkdir(parents=True);run=a.collection_run.resolve();protocol=run/'protocol.json';old=a.previous_dataset_manifest.resolve()
 names=['watch_s40_supplement_training_v2.py','prepare_s40_training_data_v2.py','merge_s40_training_datasets.py','train_s40_decision.py','train_boya_decision.py','execute_collection_queue_v3.py','execute_collection_queue_v2.py','terminal_collection_outcome.py','prepare_s40_training_data.py','score_s40_raw_candidates.py','deploy_score_first.py','evaluate_boya_decision.py','trial_record.py']
 sources={str(BOYA/n):sha(BOYA/n)for n in names};inputs={str(protocol):sha(protocol),str(old):sha(old)}
 report=dict(schema='anydex_s40_supplement_training_delivery_v1',status='waiting_for_collection',collection_run=str(run),collector_pid=a.collector_pid,previous_dataset_manifest=str(old),source_sha256=sources,input_sha256=inputs,stages=[],EGAD_objects_accessed=0,new_physics_steps=0,started_unix=time.time())
 save(out/'STATUS.json',report);deadline=time.monotonic()+a.timeout_hours*3600
 def verify():
  for q,h in dict(sources,**inputs).items():
   if sha(q)!=h:raise ValueError('Frozen delivery input/source changed: '+q)
 def command(name,argv):
  verify();report['status']=name;save(out/'STATUS.json',report);t=time.monotonic()
  with(out/(name+'.log')).open('w')as log:proc=subprocess.run([sys.executable,'-B',str(BOYA/argv[0]),*map(str,argv[1:])],cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT)
  report['stages'].append(dict(name=name,returncode=proc.returncode,wall_s=time.monotonic()-t));save(out/'STATUS.json',report)
  if proc.returncode:raise RuntimeError('Delivery stage failed: '+name)
  verify()
 try:
  while True:
   verify();final=run/'execution/RESULTS.json';progress=read(run/'pipeline_progress.json')if(run/'pipeline_progress.json').exists()else{}
   if progress.get('status')=='collection_complete':
    if not final.exists()or read(final).get('complete')is not True:raise ValueError('Collection claims complete without eligible final ledger')
    break
   if progress.get('status')in('infrastructure_stopped','blocked','error'):raise ValueError('Collection stopped; no replay or labels manufactured')
   try:os.kill(a.collector_pid,0)
   except ProcessLookupError:raise ValueError('Collector exited before durable completion')
   if time.monotonic()>deadline:raise TimeoutError('Finite delivery wait expired')
   time.sleep(15)
  command('export_supplement',['prepare_s40_training_data_v2.py','--collection-run',run,'--output-dir',out/'supplement_dataset'])
  command('merge_training_data',['merge_s40_training_datasets.py','--dataset-manifests',old,out/'supplement_dataset/manifest.json','--output-dir',out/'merged_dataset'])
  command('train',['train_s40_decision.py','--dataset-manifest',out/'merged_dataset/manifest.json','--output-dir',out/'training','--device','cuda:0'])
  report.update(status='complete',completed_unix=time.time(),dataset_manifest=str(out/'merged_dataset/manifest.json'),training_results=str(out/'training/TRAINING_RESULTS.json'),deployment_evaluation_pending=True,pending_reason='Explicit Boya pre-head domain config and masked diagnostic must be frozen separately; original raw-parser control omits rig-specific pre-head masks. No deployment or external success claim.')
 except BaseException:
  report.update(status='blocked',error=traceback.format_exc());raise
 finally:save(out/'STATUS.json',report)
if __name__=='__main__':main()
