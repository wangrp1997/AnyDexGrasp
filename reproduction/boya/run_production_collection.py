"""Run one frozen finite production pilot and export certified labels, no training."""
import time
STARTED=time.time()
BEGIN=time.monotonic()

import argparse
from pathlib import Path
import traceback
import author_production_collection as D

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
args=parser.parse_args();result=None;error=None;exported=None
try:
    result=D.run(args.run_dir)
    if result['status']=='complete':
        exported=D.export(args.run_dir,args.run_dir/'export')
except BaseException:
    error=traceback.format_exc();raise
finally:
    value=dict(started_unix_s=STARTED,finished_unix_s=time.time(),outer_wall_s=time.monotonic()-BEGIN,
        status='complete'if result is not None and result.get('status')=='complete'and exported is not None else'stopped',
        result=D.B.binding(args.run_dir/'PRODUCTION_RESULTS.json')if result is not None else None,
        exported=D.B.binding(args.run_dir/'export/MANIFEST.json')if exported is not None else None,
        error=error,script=D.B.binding(__file__),training_updates=0)
    D.B.save(args.run_dir/'OUTER_WALL.json',value,exclusive=True)
    print(value)
if value['status']!='complete':raise SystemExit(1)
