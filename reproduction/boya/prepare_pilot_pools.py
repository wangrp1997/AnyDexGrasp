"""Prepare only frozen pilot hand geometry; no networks or physical trials."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,d):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
 a=p.parse_args();a.protocol=a.protocol.resolve();a.run_dir=a.run_dir.resolve();root=Path(__file__).resolve().parents[2]
 if not 1<=a.workers<=2:raise ValueError('Geometry pilot concurrency limited to 1..2')
 protocol=read(a.protocol);protocol_sha=sha(a.protocol)
 for path,digest in protocol['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('Frozen pilot input changed: '+path)
 lock=threading.RLock();start=time.monotonic()
 report=dict(status='preparing',protocol_sha256=protocol_sha,geometry_workers=a.workers,jobs={},new_network_forwards=0,new_physics_steps=0)
 output=a.run_dir/'POOL_PREPARATION.json'
 def publish():
  with lock:
   report['elapsed_wall_s']=time.monotonic()-start
   atomic(output,report)
 def one(obj,g):
  pool=Path(obj['pools'][str(g)]);pool.parent.mkdir(parents=True,exist_ok=True);key=obj['object_id']+'__g'+str(g)
  with lock:report['jobs'][key]=dict(status='running',pool=str(pool));publish()
  if pool.exists():
   if (pool/'generation.json').exists():
    gen=read(pool/'generation.json')
    if not gen['complete'] or gen['candidates_sha256']!=sha(pool/'candidates.npz'):raise ValueError('Existing pool is incomplete/corrupt')
    for path,digest in gen['input_sha256'].items():
     if sha(path)!=digest:raise ValueError('Existing pool source changed: '+path)
    result=dict(status='ready',admitted=len(gen['records']),rejected=len(gen['rejected']),reused=True,pool=str(pool))
   elif (pool/'no_candidate.json').exists():
    d=read(pool/'no_candidate.json');result=dict(status='no_candidate',admitted=0,rejected=len(d['records']),reused=True,pool=str(pool))
   else:raise ValueError('Partial old pool retained; do not overwrite: '+str(pool))
  else:
   cfg=protocol['geometry'];argv=[sys.executable,'-B',str(Path(__file__).with_name('build_pilot_pool.py')),
    '--observation',obj['observation_json'],'--frame-dir',obj['representation_dir'],'--calibration',protocol['calibration'],
    '--gesture-id',str(g),'--opponent',cfg['opposing_group'],'--candidate-limit',str(cfg['candidate_limit']),
    '--keep',str(cfg['keep']),'--width-expansion',str(cfg['width_expansion']),'--output-dir',str(pool)]
   env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   with pool.with_suffix('.log').open('xb') as log:r=subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
   if (pool/'generation.json').exists():
    d=read(pool/'generation.json');result=dict(status='ready',admitted=len(d['records']),rejected=len(d['rejected']),reused=False,pool=str(pool),elapsed_s=d['elapsed_wall_s'])
   elif (pool/'no_candidate.json').exists():
    d=read(pool/'no_candidate.json');result=dict(status='no_candidate',admitted=0,rejected=len(d['records']),reused=False,pool=str(pool),elapsed_s=d['elapsed_wall_s'])
   else:raise RuntimeError('Pool failed without a normal geometry result: '+str(pool)+' returncode='+str(r.returncode))
  with lock:report['jobs'][key]=result;publish()
  print(json.dumps(dict(job=key,**result)),flush=True)
  return result
 publish();errors=[]
 with ThreadPoolExecutor(max_workers=a.workers)as exe:
  fs={exe.submit(one,o,g):(o['object_id'],g) for o in protocol['objects']for g in protocol['gesture_ids']}
  for f in as_completed(fs):
   try:f.result()
   except Exception as e:
    key=str(fs[f]);errors.append(dict(job=key,error=repr(e)))
    with lock:report['jobs'][key]=dict(status='infrastructure_error',error=repr(e));publish()
 if sha(a.protocol)!=protocol_sha:raise ValueError('Protocol changed while preparing pools')
 report.update(status='complete' if not errors else 'complete_with_infrastructure_errors',errors=errors,
  ready_jobs=sum(j['status']=='ready'for j in report['jobs'].values()),
  no_candidate_jobs=sum(j['status']=='no_candidate'for j in report['jobs'].values()),
  admitted_hand_candidates=sum(j.get('admitted',0)for j in report['jobs'].values()))
 publish()
 if errors:raise RuntimeError('Geometry preparation has infrastructure errors; no physical dispatch')

if __name__=='__main__':main()
