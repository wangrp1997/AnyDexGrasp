"""Measure a frozen pipeline run including Python imports and final receipts."""
import time
STARTED=time.time()
BEGIN=time.monotonic()

import argparse
from pathlib import Path
import traceback
import author_pipeline_collection as P

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
args=parser.parse_args();result=None;error=None
try:
    result=P.run(args.run_dir)
except BaseException:
    error=traceback.format_exc();raise
finally:
    value=dict(started_unix_s=STARTED,finished_unix_s=time.time(),outer_wall_s=time.monotonic()-BEGIN,
        result=P.B.binding(args.run_dir/'PIPELINE_COLLECTION_RESULTS.json') if result is not None else None,
        status='complete' if result is not None and result.get('pipeline_validity')=='pass' else 'failed',
        error=error,parent_Kit_launches=0,script=P.B.binding(__file__))
    P.B.save(args.run_dir/'OUTER_WALL.json',value,exclusive=True)
    print(value)
if value['status']!='complete':raise SystemExit(1)
