"""Wait for the fixed collection, then train and compare frozen top-one policies.

All outputs are versioned. This runner never repairs/replays an ambiguous grasp
or changes the active collection, and it publishes failures instead of waiting
silently on a dead producer.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from terminal_collection_outcome import certify_known_task_stop

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
BOYA=ROOT/'reproduction/boya'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify(refs):
    for p,h in refs.items():
        if not Path(p).is_file() or sha(p)!=h:raise ValueError('Frozen delivery source changed: '+p)
def alive(pid):
    try:
        os.kill(pid,0)
        stat=Path('/proc')/str(pid)/'stat'
        return not stat.exists() or stat.read_text().split(') ',1)[1].split()[0]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False


class Delivery:
    def __init__(self,output,plan):
        self.output=output;self.plan=plan;self.start=time.monotonic();self.plan_sha256=sha(output/'DELIVERY_PLAN.json')
        self.status=dict(schema='anydex_boya_training_delivery_v2',status='starting',stages={},started_at=now())
        self.logs=output/'logs';self.logs.mkdir(exist_ok=True)
    def publish(self,status,**kwargs):
        self.status.update(status=status,updated_at=now(),wall_s=time.monotonic()-self.start,**kwargs)
        save(self.output/'DELIVERY_STATUS.json',self.status)
    def check(self):
        verify(self.plan['source_sha256'])
        if sha(self.output/'DELIVERY_PLAN.json')!=self.plan_sha256:raise ValueError('Frozen delivery plan changed')
    def command(self,name,argv):
        self.check()
        started=time.monotonic()
        with (self.logs/(name+'.log')).open('xb') as log:
            result=subprocess.run(argv,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        self.check()
        receipt=dict(argv=argv,returncode=result.returncode,wall_s=time.monotonic()-started,log=str(self.logs/(name+'.log')))
        if result.returncode:
            raise RuntimeError(name+' failed; inspect '+receipt['log'])
        return receipt
    def wait_collection(self):
        collection=Path(self.plan['collection_run']);deadline=time.monotonic()+self.plan['max_wait_hours']*3600
        while True:
            self.check()
            result=collection/'RESULTS.json'
            if result.exists():
                doc=read(result)
                if doc.get('complete')is True and not doc.get('unknown_execution_trial_ids'):
                    self.status['stages']['collection']=dict(result=str(result),sha256=sha(result),actual_grasps=doc['actual_grasps'],positives=doc['positives'],negatives=doc['negatives'])
                    self.publish('collection_complete');return
                raise RuntimeError('Collection ended incomplete; no automatic training or ambiguous replay')
            progress=read(collection/'PROGRESS.json');launch=read(collection/'LAUNCH.json')
            if progress['status']=='infrastructure_stopped':
                raise RuntimeError('Collection infrastructure stop: '+str(progress.get('error','inspect producer outputs')))
            if not alive(launch['pid']):raise RuntimeError('Collection producer exited without a complete result; inspect preserved evidence')
            if time.monotonic()>deadline:raise RuntimeError('Finite collection-wait budget exceeded; producer left untouched')
            objects=progress.get('objects',{})
            known=[r for obj in objects.values()for r in obj.get('result',{}).get('trials',[])]
            self.publish('waiting_for_collection',collection=dict(pid=launch['pid'],object_states={s:sum(o.get('status')==s for o in objects.values())for s in set(o.get('status')for o in objects.values())},known_actual_labels=sum(r.get('label_eligible')is True and r.get('label')in(0,1)for r in known)))
            time.sleep(self.plan['poll_seconds'])
    def ordered_stages(self):
        for stage in self.plan['stages']:
            self.publish(stage['name']+'_running')
            receipt=self.command(stage['name'],stage['argv'])
            for p in stage['required_outputs']:
                if not Path(p).is_file():raise RuntimeError('Stage did not create required output: '+p)
            receipt['outputs_sha256']={p:sha(p)for p in stage['required_outputs']}
            self.status['stages'][stage['name']]=receipt
            self.publish(stage['name']+'_complete')
    def physical_comparison(self):
        self.publish('physical_comparison_running')
        ranking=self.status['stages'].get('ranking',{})
        verify(ranking.get('outputs_sha256',{}))
        policies=self.plan['policies'];output=self.output/'physical_comparison';output.mkdir()
        def one(policy):
            name=policy['name'];manifest=Path(policy['manifest']);doc=read(manifest)
            if doc.get('benchmark_only')is not True:raise ValueError('Policy evaluation must be excluded from training labels')
            if not doc['trials']:
                return dict(name=name,status='no_proposal',trials=[],manifest_sha256=sha(manifest))
            target=output/name
            argv=[sys.executable,'-B',str(BOYA/'execute_collection_queue_v2.py'),'--manifest',str(manifest),'--output-dir',str(target),'--device','cpu','--mode','serial','--batch-size','16','--benchmark-only','--paired-benchmark-id',sha(self.output/'DELIVERY_PLAN.json')]
            receipt=self.command('physical_'+name,argv)
            result=read(target/'RESULTS.json')
            if result.get('error')or result.get('infrastructure_error')or result.get('status')not in('complete','complete_with_null'):
                raise RuntimeError('Policy execution incomplete: '+name)
            rows=result['trials']
            if len(rows)!=len(doc['trials']) or {r['trial_id']for r in rows}!={r['trial_id']for r in doc['trials']}:
                raise RuntimeError('Policy execution ledger incomplete: '+name)
            terminal_nulls=[]
            for row in rows:
                if row.get('label')in(0,1) and row.get('label_eligible')is True:continue
                terminal_nulls.append(certify_known_task_stop(row))
            return dict(name=name,status='complete',trials=rows,terminal_nulls=terminal_nulls,process=receipt,manifest_sha256=sha(manifest),physical_steps=result['physical_steps'],actual_grasps=result['actual_grasps'])
        results=[];errors=[]
        with ThreadPoolExecutor(max_workers=2)as executor:
            futures=[executor.submit(one,p)for p in policies]
            for policy,future in zip(policies,futures):
                try:results.append(future.result())
                except Exception:errors.append(dict(policy=policy['name'],error=traceback.format_exc()))
        bypolicy={r['name']:r for r in results}
        object_ids=self.plan['validation_object_ids']
        metrics={}
        for name,result in bypolicy.items():
            rows=result['trials'];successes=sum(r.get('label')==1 for r in rows)
            metrics[name]=dict(objects=len(object_ids),actual_attempts=len(rows),successes=successes,failures=sum(r.get('label')==0 for r in rows),known_task_terminated_nulls=sum(r.get('label')is None for r in rows),no_proposal=len(object_ids)-len(rows),success_fraction_over_objects=successes/len(object_ids),success_fraction_over_executed=successes/len(rows)if rows else None)
        paired=[]
        if len(bypolicy)==2:
            indexed={name:{r['object_id']:r for r in result['trials']}for name,result in bypolicy.items()}
            for oid in object_ids:
                paired.append(dict(object_id=oid,baseline_label=indexed['baseline'].get(oid,{}).get('label'),learned_label=indexed['learned'].get(oid,{}).get('label')))
        summary=dict(schema='anydex_boya_top1_physical_comparison_v1',complete=not errors,metrics=metrics,paired_objects=paired,policy_results=results,errors=errors,selection_without_outcome_labels=True,known_task_stops_remain_unlabelled=True,independent_reset_per_attempt=True,benchmark_only=True,new_training_steps=0,scope='One frozen top1 per policy per held-out object from the same previously geometry/arm-admitted pool. No retries or oracle selection; ten objects provide a limited initial validation.')
        save(output/'RESULTS.json',summary)
        self.status['stages']['physical_comparison']=dict(result=str(output/'RESULTS.json'),sha256=sha(output/'RESULTS.json'),metrics=metrics)
        if errors:raise RuntimeError('Physical comparison infrastructure error; partial outputs preserved')
        self.publish('physical_comparison_complete')
        return summary
    def videos(self,comparison):
        self.publish('rendering_typical_videos')
        bypolicy={p['name']:{r['object_id']:r for r in p['trials']}for p in comparison['policy_results']}
        pairs=comparison['paired_objects']
        ranked=sorted(pairs,key=lambda p:(p['baseline_label']==p['learned_label'],p['object_id']))
        selected=[]
        for pair in ranked:
            for name in ('baseline','learned'):
                row=bypolicy.get(name,{}).get(pair['object_id'])
                if row is not None:selected.append((name,row))
            if len(selected)>=self.plan['max_videos']:break
        selected=selected[:self.plan['max_videos']]
        video_root=self.output/'videos';video_root.mkdir();videos=[];errors=[]
        for i,(name,row)in enumerate(selected):
            target=video_root/('%02d_%s'%(i,name));argv=[sys.executable,'-B',str(BOYA/'render_native.py'),'--execution',row['execution_dir'],'--plan-dir',row['source_plan_dir'],'--output-dir',str(target),'--object-label',name+' '+row['object_id'],'--render-gpu','0','--include-failed-trace']
            try:
                self.command('video_%02d'%i,argv)
                check=read(target/'render_check.json')
                if check.get('physics_steps_during_replay')!=0:raise ValueError('Replay unexpectedly stepped physics')
                video=target/'isaac_full_arm_pickup.mp4'
                videos.append(dict(policy=name,object_id=row['object_id'],label=row['label'],video=str(video),sha256=sha(video)))
            except Exception:errors.append(dict(policy=name,object_id=row['object_id'],error=traceback.format_exc()))
        index=['<!doctype html><meta charset="utf-8"><title>Boya policy comparison</title><h1>Held-out top-one grasp comparison</h1>']
        for video in videos:
            label=html.escape(video['policy']+' '+video['object_id']+' result='+str(video['label']))
            relative=Path(video['video']).relative_to(video_root).as_posix()
            index.append('<h2>'+label+'</h2><video controls preload="metadata" width="960" src="'+html.escape(relative)+'"></video>')
        (video_root/'index.html').write_text('\n'.join(index))
        save(video_root/'VIDEOS.json',dict(videos=videos,errors=errors,new_physics_steps=0))
        return dict(index=str(video_root/'index.html'),videos=videos,errors=errors)
    def run(self):
        try:
            self.wait_collection();self.ordered_stages();comparison=self.physical_comparison();videos=self.videos(comparison)
            result=dict(schema='anydex_boya_training_delivery_results_v2',complete=True,comparison= str(self.output/'physical_comparison/RESULTS.json'),metrics=comparison['metrics'],training_results=self.plan['training_results'],videos=videos,stages=self.status['stages'],completed_at=now())
            save(self.output/'DELIVERY_RESULTS.json',result)
            self.publish('complete'if not videos['errors']else'core_complete_media_partial')
        except BaseException:
            self.publish('blocked',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args();plan=read(args.plan);output=args.plan.resolve().parent
    with (output/'.delivery.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'DELIVERY_STATUS.json').exists():raise ValueError('Delivery already dispatched; preserve outputs and never auto-replay')
        verify(plan['source_sha256']);Delivery(output,plan).run()


if __name__=='__main__':main()
