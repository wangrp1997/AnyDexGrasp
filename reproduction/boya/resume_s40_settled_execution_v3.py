"""Resume only proven zero-execution S40 slots with a new executor contract.

The stopped source run is immutable. Previous actual attempts are consumed once,
never replayed; original observations/CGR/candidates/plans/commands are reused.
A separate v1-compatible collection root permits the existing data exporter and
merger to consume the eventual complete source-bound label ledger.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya';sys.path.insert(0,str(BOYA))
from run_s40_settled_collection import Pipeline as SettledPipeline, GESTURES, DEPTHS
from run_collection_pipeline_v2 import bounded_jobs,sha,read,save,hashes,verify
from resume_collection_execution_v4 import inspect_rows
from trial_record import collect_record


def require(value,message):
 if not value:raise ValueError(message)
def zero_prepare_proof(run,oid,expected_ids):
 """An explicit dispatcher launch plus source-bound prepare traceback, no worker."""
 base=run/'execution';directory=base/oid;marker=base/'object_queues'/(oid+'.SUBMITTING.json');subpath=base/'object_queues'/(oid+'.json')
 require(marker.exists()and subpath.exists(),'Missing explicit failed-prepare dispatcher evidence')
 launched=read(marker);sub=read(subpath);require(set(launched['trial_ids'])==set(expected_ids)=={r['trial_id']for r in sub['trials']},'Failed-prepare launched identity set differs')
 require(Path(launched['manifest']).resolve()==subpath.resolve()and launched['manifest_sha256']==sha(subpath),'Failed-prepare subqueue SHA changed')
 require(directory.exists()and {str(p.relative_to(directory))for p in directory.rglob('*')}=={'.queue.lock'},'Failed prepare has possible execution artifacts; replay forbidden')
 log=run/'logs'/('execution_'+oid+'.log');text=log.read_text()
 require('prepared=prepare(args)'in text and 'contract = command_contract(plan,z)'in text and text.rstrip().endswith('ValueError: Only original contiguous approach may extend; all other phases fixed/full'),'Only exact zero-Kit prepare-contract failure can be reclassified')
 executor=BOYA/'execute_collection_queue_v2.py';source=executor.read_text()
 require(source.index("prepared=prepare(args);save(root/'PREPARED.json',prepared)")<source.index('execute(args,prepared)',source.index('def main():')),'Original executor does not preserve prepare-before-execute ordering')
 return dict(object_id=oid,trial_ids=list(expected_ids),reason='source_bound_prepare_command_contract_rejection_before_PREPARED_or_worker',new_physics_steps=0,
  evidence_sha256=hashes([marker,subpath,log,executor]),output_directory=str(directory),only_output_entries=['.queue.lock'])


def inspect_original(run):
 p=read(run/'protocol.json');verify(p['input_sha256']);verify(p['source_sha256']);verify(p['manifest_sha256'])
 require(p.get('schema')=='anydex_boya_s40_collection_v1'and p.get('collection_variant')=='settled_pose_fresh_v2','Original frozen settled S40 collection required')
 q=read(run/'planning/queue_manifest.json');require(q.get('complete')is True and not q.get('errors')and q.get('inputs_unchanged')is True,'Complete original ready/null queue required')
 ready={r['trial_id']:r for r in q['trials']};null={r['trial_id']:r for r in q['nulls']};slots={r['trial_id']:r for r in p['slots']}
 require(len(slots)==640 and len(ready)==len(q['trials'])and len(null)==len(q['nulls'])and set(ready).isdisjoint(null)and set(ready)|set(null)==set(slots),'Original640 slot partition differs')
 old=read(run/'execution/RESULTS.json');progress=read(run/'execution/PROGRESS.json')
 require(old.get('complete')is False and old.get('status')=='infrastructure_stopped'and old.get('input_sha256_unchanged')is True and old.get('source_sha256_unchanged')is True,'Stopped intact original execution required')
 actual=old['trials'];seen={r['trial_id']for r in actual};require(len(seen)==len(actual)and seen<=set(ready),'Repeated/foreign inherited actual identity')
 require(all(type(r.get('label'))is int and r['label']in(0,1)and r.get('label_eligible')is True for r in actual),'This bounded resume inherits complete actual labels only; other actual states require separate diagnosis')
 inspect_rows(actual,list(seen))
 require(old['actual_grasps']==len(actual)and old['physical_steps']==sum(r['physics_steps']for r in actual)and old['eligible_labels']==len(actual),'Inherited actual/step/label costs differ')
 exports={}
 for name in('all_records.jsonl','train_records.jsonl'):
  values={}
  for line in(run/'execution'/name).read_text().splitlines():
   if line.strip():
    value=json.loads(line);tid=value['collection_context']['trial_id'];require(tid not in values,'Repeated inherited exported label');values[tid]=value
  require(set(values)==seen,'Inherited exports differ from consumed actual IDs')
  if exports:require(values==exports,'Inherited train/all exports differ')
  else:exports=values
 for row in actual:
  value=exports[row['trial_id']];artifact=value['execution_artifacts'];expected=collect_record(value['binding'],artifact['evaluation_json'],artifact['commands_npz'])
  require(all(value.get(k)==v for k,v in expected.items())and value['result']==row['label']and value['binding']['trial_binding_sha256']==row['trial_binding_sha256'],'Inherited actual source record replay failed')
 remaining=[r for r in q['trials']if r['trial_id']not in seen];remaining_ids={r['trial_id']for r in remaining}
 unknown=set(old['unknown_execution_trial_ids']);never=set(old['unexecuted_ready_trial_ids'])
 require(unknown.isdisjoint(never)and unknown|never==remaining_ids,'Original stopped unknown/unexecuted partition differs from remaining ready IDs')
 proofs={};refs=[run/'protocol.json',run/'objects.json',run/'planning/queue_manifest.json',run/'stages/planning.json',run/'execution/RESULTS.json',run/'execution/PROGRESS.json',run/'execution/all_records.jsonl',run/'execution/train_records.jsonl']
 for oid in sorted({r['object_id']for r in remaining}):
  ids={r['trial_id']for r in remaining if r['object_id']==oid};state=progress['objects'].get(oid,{})
  require(not ids&seen,'Already consumed actual cannot be requeued')
  if ids<=unknown:
   require(state.get('infrastructure_error')is True and not state.get('result',{}).get('trials'),'Failed prepare has actual or ambiguous state')
   proof=zero_prepare_proof(run,oid,sorted(ids));proofs[oid]=proof;refs.extend(map(Path,proof['evidence_sha256']))
  else:
   require(ids<=never and state.get('status')=='not_dispatched','Remaining trial lacks explicit never-dispatched state')
   require(not(run/'execution'/oid).exists()and not(run/'execution/object_queues'/(oid+'.SUBMITTING.json')).exists(),'Never-dispatched object has launch/output ambiguity')
   proofs[oid]=dict(object_id=oid,trial_ids=sorted(ids),reason='immutable_final_ledger_not_dispatched_and_no_submission',new_physics_steps=0)
 for row in actual:
  record=exports[row['trial_id']];a=record['execution_artifacts'];refs.extend([Path(a['evaluation_json']),Path(a['commands_npz']),Path(row['execution_dir'])/'evaluation.json',Path(row['execution_dir'])/'trace.npz'])
 return p,q,old,actual,[exports[r['trial_id']]for r in actual],remaining,proofs,hashes(refs)


def freeze(args):
 run=args.original_run.resolve();out=args.output_dir.resolve();require(out!=run,'Resume requires a separate collection output directory')
 if(out/'protocol.json').exists():
  saved=read(out/'protocol.json');require(saved.get('collection_variant')=='settled_execution_resume_v3'and saved.get('original_run')==str(run)and saved['config']['executor']==str(args.executor.resolve())and saved['config']['execution_workers']==args.workers,'Frozen resume options changed')
  require(not(out/'execution').exists(),'Resume already dispatched; do not replay a partial/completed segment')
  verify(saved['input_sha256']);verify(saved['source_sha256']);verify(saved['manifest_sha256']);return saved
 require(not out.exists(),'Resume requires a fresh collection output directory')
 p,q,old,actual,records,remaining,proofs,refs=inspect_original(run)
 executor=args.executor.resolve();require(executor.is_file(),'Explicit completed v3 executor source required')
 out.mkdir(parents=True,exist_ok=False);(out/'planning').mkdir()
 shutil.copyfile(run/'objects.json',out/'objects.json');shutil.copyfile(run/'planning/queue_manifest.json',out/'planning/queue_manifest.json')
 inherited=dict(rows=actual,records=records,proofs=proofs,remaining_trial_ids=[r['trial_id']for r in remaining],original_protocol_sha256=sha(run/'protocol.json'),original_run=str(run))
 save(out/'INHERITED_EXECUTION.json',inherited)
 new=copy.deepcopy(p);new.update(collection_variant='settled_execution_resume_v3',original_run=str(run),original_protocol_sha256=sha(run/'protocol.json'),
  inherited_slot_count=len(actual)+len(q['nulls']),new_slot_count=len(remaining),inherited_actual_trials=len(actual),inherited_physics_steps=old['physical_steps'],
  original_unknown_preserved=copy.deepcopy(old['unknown_execution_trial_ids']),zero_execution_reclassification=proofs,
  amendment='Only source-proven pre-Kit failed preparation and never-dispatched ready slots resume.157 original actuals consumed once; original640 source commands/physics/labels unchanged. Earlier partial/unknown ledger remains immutable.',created_unix_s=time.time())
 new['config']=dict(p['config'],executor=str(executor),execution_workers=args.workers)
 new['input_sha256']=dict(p['input_sha256'],**refs)
 new['source_sha256']=dict(p['source_sha256'],**hashes([Path(__file__).resolve(),executor,*BOYA.glob('*.py'),*getattr(args,'extra_source',[])]))
 new['manifest_sha256']=dict(p['manifest_sha256'],**hashes([out/'objects.json',out/'planning/queue_manifest.json',out/'INHERITED_EXECUTION.json']))
 verify(new['input_sha256']);verify(new['source_sha256']);verify(new['manifest_sha256']);save(out/'protocol.json',new)
 return new

class Pipeline(SettledPipeline):
 def __init__(self,run,protocol):
  super().__init__(run,protocol);i=read(run/'INHERITED_EXECUTION.json');self.inherited_rows=i['rows'];self.inherited_records=i['records'];self.consumed={r['trial_id']for r in self.inherited_rows};self.inherited_steps=sum(r['physics_steps']for r in self.inherited_rows);self.original_protocol_sha=i['original_protocol_sha256']
 def execution(self):
     from s40_execution_outcomes import inspect_execution
     queue=read(self.run/'planning/queue_manifest.json');output=self.run/'execution'
     output.mkdir(exist_ok=False);(output/'object_queues').mkdir();(output/'terminal_certificates').mkdir()
     started=time.monotonic();jobs=[];certificates={};assets=[]
     progress=dict(schema='anydex_boya_s40_execution_v1',status='running',objects={},dispatch_events=[],peak_inflight_object_jobs=0,
         inherited_actual_grasps=len(self.inherited_rows),inherited_physics_steps=self.inherited_steps)
     for obj in self.objects:
         oid=obj['object_id'];trials=[r for r in queue['trials']if r['object_id']==oid and r['trial_id']not in self.consumed]
         nulls=[r for r in queue['nulls']if r['object_id']==oid]
         sub=dict(queue,trials=trials,nulls=nulls,parent_queue_manifest=str(self.run/'planning/queue_manifest.json'),
             parent_queue_manifest_sha256=sha(self.run/'planning/queue_manifest.json'))
         path=output/'object_queues'/(oid+'.json');save(path,sub);assets.append(path)
         jobs.append(dict(object_id=oid,manifest=str(path),ready=len(trials)))
     save(output/'PROGRESS.json',progress)
     def one(job):
         oid=job['object_id'];sub=read(job['manifest']);target=output/oid
         if not job['ready']:return dict(object_id=oid,status='no_ready_plans',result=dict(trials=[]),terminal_certificates={})
         save(output/'object_queues'/(oid+'.SUBMITTING.json'),dict(object_id=oid,trial_ids=[r['trial_id']for r in sub['trials']],manifest=job['manifest'],manifest_sha256=sha(job['manifest']),time_unix=time.time(),state='submission_intent_not_proof_of_unexecuted'))
         proc=self.command('execution_'+oid,[sys.executable,'-B',self.c['executor'],
             '--manifest',job['manifest'],'--output-dir',str(target),'--device','cpu','--mode','serial','--batch-size','16'],allowed_failure=True)
         doc=read(target/'RESULTS.json')if(target/'RESULTS.json').exists()else{}
         checked=inspect_execution(doc,[r['trial_id']for r in sub['trials']],proc['returncode'])
         return dict(object_id=oid,status='complete',output_dir=str(target),result=doc,terminal_certificates=checked['terminal_certificates'])
     def completed(job,value,bad):
         progress['objects'][job['object_id']]=dict(value,infrastructure_error=bad)
         for tid,cert in value.get('terminal_certificates',{}).items():
             certificates[tid]=cert;save(output/'terminal_certificates'/(tid+'.json'),cert)
         progress['wall_s']=time.monotonic()-started
         save(output/'PROGRESS.json',progress)
         vals=[r for v in progress['objects'].values()for r in v.get('result',{}).get('trials',[])]
         self.progress['detail']=dict(objects_finished=len(progress['objects']),objects=40,inherited_actual=len(self.inherited_rows),
             new_actual=sum(r.get('physics_steps',0)>0 for r in vals),new_physics_steps=sum(r.get('physics_steps',0)for r in vals),
             eligible=len(self.inherited_rows)+sum(r.get('label_eligible')is True for r in vals),terminal_nulls=len(certificates),last=job['object_id'],error=bad)
         self.publish('execution_running')
     def dispatched(job,active):
         progress['peak_inflight_object_jobs']=max(progress['peak_inflight_object_jobs'],sum(bool(j['ready'])for j in active))
         progress['dispatch_events'].append(dict(object_id=job['object_id'],ready=job['ready'],elapsed_s=time.monotonic()-started));save(output/'PROGRESS.json',progress)
     work=bounded_jobs(jobs,self.c['execution_workers'],one,completed,dispatched)
     failed=any(r['failed']for r in work)
     # Read every dispatched executor's durable result even if validation failed.
     for job in jobs:
         oid=job['object_id'];item=progress['objects'].setdefault(oid,dict(status='not_dispatched',result={}))
         rp=output/oid/'RESULTS.json'
         if rp.exists():item['result']=read(rp);item['output_dir']=str(output/oid)
     rows=list(self.inherited_rows)+[r for value in progress['objects'].values()for r in value.get('result',{}).get('trials',[])]
     order={j['trial_id']:i for i,j in enumerate(self.p['slots'])};byid={j['trial_id']:j for j in self.p['slots']}
     if len({r['trial_id']for r in rows})!=len(rows):raise ValueError('Duplicate physical trial identity')
     rows.sort(key=lambda r:order[r['trial_id']]);records=[]
     def add_record(value,tid,inherited=False):
         if value.get('label_eligible')is True and value.get('result')in (0,1):
             row=next(r for r in rows if r['trial_id']==tid)
             if (type(value['result'])is not int or value['result']!=row['label']
                 or value['binding']['trial_binding_sha256']!=row['trial_binding_sha256']
                 or value['total_physics_steps']!=row['physics_steps']):
                 raise ValueError('Training record differs from actual trial binding/outcome/count')
             artifact=value['execution_artifacts']
             verify({artifact['evaluation_json']:artifact['evaluation_sha256'],artifact['commands_npz']:artifact['commands_sha256']})
             if artifact['commands_sha256']!=row['commands_sha256']:
                 raise ValueError('Training record command bytes differ from actual trace')
             from trial_record import collect_record
             expected=collect_record(value['binding'],artifact['evaluation_json'],artifact['commands_npz'])
             if any(value.get(k)!=v for k,v in expected.items()):
                 raise ValueError('Training export differs from original artifact-derived record')
             slot=byid[tid];value=copy.deepcopy(value)
             if (value['binding']['multifinger_pose_finger_type']!=slot['gesture_id']
                 or value['binding']['multifinger_pose_depth_type']!=slot['extra_depth_index']):
                 raise ValueError('Training record finger type/depth differs from declared slot')
             value['collection_context']=dict(trial_id=tid,object_id=slot['object_id'],split='train',gesture_id=slot['gesture_id'],
                 extra_depth_index=slot['extra_depth_index'],benchmark_only=False,protocol_sha256=sha(self.run/'protocol.json'),
                 inherited_from_pilot=False,inherited_previous_protocol=inherited,independent_reset_from_settled_pose=True,original_protocol_sha256=self.original_protocol_sha if inherited else None)
             records.append(value)
     for inherited_record in self.inherited_records:
         core=copy.deepcopy(inherited_record);context=core.pop('collection_context');add_record(core,context['trial_id'],True)
     for item in progress['objects'].values():
         if not item.get('output_dir'):continue
         dest=Path(item['output_dir'])
         for row in item.get('result',{}).get('trials',[]):
             rp=dest/'trials'/row.get('output_identity',row['trial_id'])/'training_record.json'
             if rp.exists():add_record(read(rp),row['trial_id']);assets.append(rp)
         for name in ['RESULTS.json','PREPARED.json','training_records.jsonl']:
             if(dest/name).exists():assets.append(dest/name)
     records.sort(key=lambda r:order[r['collection_context']['trial_id']])
     eligible={r['trial_id']for r in rows if r.get('label_eligible')is True and r.get('label')in (0,1)}
     if {r['collection_context']['trial_id']for r in records}!=eligible or len(records)!=len(eligible):raise ValueError('Training exports differ from actual eligible labels')
     actual={r['trial_id']for r in rows if r.get('physics_steps',0)>0}
     ready={r['trial_id']for r in queue['trials']}
     submitted=set()
     for marker in (output/'object_queues').glob('*.SUBMITTING.json'):
         submitted.update(read(marker)['trial_ids'])
     unknown=sorted((actual-eligible-set(certificates)) | ((submitted & ready)-actual))
     unexecuted=sorted(ready-actual-submitted)
     if unknown or unexecuted:failed=True
     counts={str(g):dict(target=100,labels=sum(r['gesture_id']==g and r.get('label_eligible')is True for r in rows),
         positive=sum(r['gesture_id']==g and r.get('label')==1 for r in rows),negative=sum(r['gesture_id']==g and r.get('label')==0 for r in rows))for g in GESTURES}
     for gesture,value in counts.items():
         value.update(shortfall=max(0,100-value['labels']),extra=max(0,value['labels']-100),depths={str(d):dict(target=25,labels=sum(r['gesture_id']==int(gesture) and r['extra_depth_index']==d and r.get('label_eligible')is True for r in rows))for d in DEPTHS})
         for depth in value['depths'].values():depth['shortfall']=max(0,25-depth['labels'])
     physical_steps=sum(r.get('physics_steps',0)for r in rows)
     result=dict(schema='anydex_boya_s40_collection_results_v1',status='infrastructure_stopped'if failed else'complete_with_null',complete=not failed,
         slot_count=640,objects=40,all_objects_training=True,collection_variant='settled_execution_resume_v3',actual_grasps=len(actual),physical_steps=physical_steps,
         inherited_actual_grasps=len(self.inherited_rows),inherited_physics_steps=self.inherited_steps,new_actual_grasps=len(actual)-len(self.inherited_rows),new_physical_steps=physical_steps-self.inherited_steps,
         eligible_labels=len(records),positives=sum(r.get('label')==1 for r in rows),negatives=sum(r.get('label')==0 for r in rows),
         preparation_nulls=queue['nulls'],terminal_certificates=certificates,terminal_nulls=len(certificates),
         unexecuted_ready_trial_ids=unexecuted,unknown_execution_trial_ids=unknown,physical_cost_complete=not unknown,counts=counts,trials=rows,
         new_CGR_forwards=0,inherited_CGR_forwards=40,new_training_updates=0,no_decision_head_loaded=True,
         wall_s=time.monotonic()-started,source_sha256_unchanged=True,input_sha256_unchanged=True)
     self.check()
     for fn in ['train_records.jsonl','all_records.jsonl']:
         with(output/fn).open('x')as f:
             for value in records:f.write(json.dumps(value,allow_nan=False)+'\n')
         assets.append(output/fn)
     save(output/'RESULTS.json',result);save(self.run/'collection_counts.json',{k:v for k,v in result.items()if k not in ['trials','terminal_certificates','preparation_nulls']})
     progress.update(status=result['status'],wall_s=result['wall_s']);save(output/'PROGRESS.json',progress)
     assets += [output/'RESULTS.json',output/'PROGRESS.json',self.run/'collection_counts.json']
     if failed:raise RuntimeError('Unknown execution/infrastructure evidence retained; no automatic replay')
     return dict(eligible_labels=len(records),physical_steps=physical_steps,new_physical_steps=physical_steps-self.inherited_steps,counts=counts),assets

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original-run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--executor',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--extra-source',type=Path,action='append',default=[]);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 require(1<=a.workers<=4,'One to four bounded object workers required');run=a.original_run.resolve();out=a.output_dir.resolve();out.parent.mkdir(parents=True,exist_ok=True)
 with(run.parent/('.'+run.name+'.s40_settled.lock')).open('a')as original_lock,(out.parent/('.'+out.name+'.resume.lock')).open('a')as lock:
  fcntl.flock(original_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  spec=importlib.util.spec_from_file_location('_s40_resume_executor_v3',a.executor.resolve());executor=importlib.util.module_from_spec(spec);spec.loader.exec_module(executor)
  require(callable(getattr(executor,'install_temporal_contract',None)),'V3 executor must expose isolated temporal-contract installation');executor.install_temporal_contract()
  protocol=freeze(a)
  if a.dry_run:print(json.dumps(dict(status='frozen_resume_only',run=str(out),inherited=protocol['inherited_actual_trials'],remaining=protocol['new_slot_count'],new_physics_steps=0),indent=2));return
  pipeline=Pipeline(out,protocol)
  try:
   result=pipeline.stage('execution',pipeline.execution);pipeline.progress.update(physical_steps=result['physical_steps'],new_physical_steps=result['new_physical_steps'],eligible_labels=result['eligible_labels']);pipeline.publish('collection_complete')
  except BaseException:
   pipeline.progress['error']=traceback.format_exc();pipeline.publish('infrastructure_stopped');raise
if __name__=='__main__':main()
