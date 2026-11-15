"""At most two read-only numeric audit children; one attempt per CPU8 worker."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import author_cpu_full_control as F

B,R,C=F.B,F.R,F.C
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--full-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args();p=F.verify(args.full_root)
B.verify({'full_results':B.binding(args.full_root/F.RESULTS)})
config=p['groups'][1];stop=threading.Event();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
script=Path(__file__).with_name('audit_one.py').resolve();started=time.time();begin=time.monotonic()
marker=out/'PARALLEL_AUDIT_STARTED.json'
B.save(marker,dict(protocol=B.binding(args.full_root/F.PROTOCOL),workers=config['count'],max_parallel_audits=2,
    audit_entry=B.binding(script),driver=B.binding(__file__),started_unix_s=started),exclusive=True)


def one(worker):
    identity=None;process=None;value=dict(status='not_started',worker_id=worker['worker_id'])
    if stop.is_set():return value
    target=out/(config['name']+'_'+worker['worker_id']+'.json')
    stage=args.full_root/config['name']/worker['worker_id']/'STAGE.json'
    original=B.read(stage)
    if not original['complete'] or original['returncode']!=0:
        stop.set();return dict(value,error='Execution stage was not complete')
    command=[sys.executable,'-B',str(script),'--receipt',str(Path(worker['output_dir'])/C.WORKER_RECEIPT),
        '--output',str(target)]
    value.update(started_unix_s=time.time(),command=command)
    try:
        with target.with_suffix('.log').open('xb') as stream:
            process=subprocess.Popen(command,cwd=B.ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            identity=R.process_identity(process.pid);value['process']=identity
            B.save(target.with_name(target.stem+'_STAGE.json'),value,exclusive=True)
            try:process.wait(timeout=900.)
            except subprocess.TimeoutExpired:value['signals']=C.finish_owned(process,identity,p)
        report=B.read(target)
        value.update(returncode=process.returncode,status=report['status'],report=B.binding(target))
        if process.returncode!=0 or report['status']!='full_integrity_pass':stop.set()
    except Exception as error:
        stop.set();value.update(status='failed',error=repr(error))
        if process is not None and process.poll() is None:value['signals']=C.finish_owned(process,identity,p)
    value.update(finished_unix_s=time.time())
    B.save(target.with_name(target.stem+'_STAGE.json'),value)
    return value


with R.lease(out/'AUDIT_DRIVER.lock'):
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows=[f.result() for f in as_completed([executor.submit(one,w) for w in config['workers']])]
    rows.sort(key=lambda r:r['worker_id'])
    value=dict(status='pass' if all(r['status']=='full_integrity_pass' for r in rows) else 'failed',
        workers=rows,max_parallel_audits=2,started_unix_s=started,finished_unix_s=time.time(),
        wall_s=time.monotonic()-begin,physics_steps=0,Kit_launches=0,label=None,exclude_training=True)
    B.save(out/'PARALLEL_AUDIT_RESULTS.json',value,exclusive=True)
print(json.dumps(dict(status=value['status'],workers=len(rows),wall_s=value['wall_s']),indent=2))
if value['status']!='pass':raise SystemExit(1)
