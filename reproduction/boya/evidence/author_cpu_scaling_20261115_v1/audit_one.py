"""Once-only full numeric audit of one completed, excluded CPU worker."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import author_cpu_full_audit as F
import author_collection_wave_resources as R

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
F.A.need(not args.output.exists(),'Original numeric audit is never overwritten')
started=args.output.with_name(args.output.stem+'_STARTED.json')
F.A.need(not started.exists(),'Numeric audit attempt is not retried')
began=time.monotonic();begin_unix=time.time()
start=dict(worker_receipt=dict(path=str(args.receipt.resolve()),sha256=F.A.sha(args.receipt)),
    process=R.process_identity(os.getpid()),started_unix_s=begin_unix,
    script=dict(path=str(Path(__file__).resolve()),sha256=F.A.sha(__file__)),
    new_Kit_launches=0,new_physics_steps=0)
args.output.parent.mkdir(parents=True,exist_ok=True)
with started.open('x') as stream:json.dump(start,stream,indent=2)
bindings=F.S.Bindings()
try:
    worker=F.audit_worker(args.receipt,bindings,F.native_state_auditor());bindings.verify()
    value=dict(status='full_integrity_pass',worker=worker,input_sha256=bindings.files)
except Exception:value=dict(status='failed',error=traceback.format_exc())
value.update(start=start,finished_unix_s=time.time(),wall_s=time.monotonic()-began,
    label=None,exclude_training=True,formal_training_eligible=False,
    new_Kit_launches=0,new_physics_steps=0)
with args.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in value.items() if k in ('status','wall_s','error')},indent=2))
if value['status']=='failed':raise SystemExit(1)
