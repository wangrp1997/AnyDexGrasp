"""Run the original remaining15 once, export certified labels; no training."""
import time
STARTED=time.time();BEGIN=time.monotonic()
import argparse
import traceback
from pathlib import Path
import author_production_tail as D

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
a=parser.parse_args();result=exported=None;error=None
try:
    result=D.run(a.run_dir)
    if result['status']=='complete':exported=D.export(a.run_dir,a.run_dir/'export')
except BaseException:
    error=traceback.format_exc();raise
finally:
    value=dict(started_unix_s=STARTED,finished_unix_s=time.time(),outer_wall_s=time.monotonic()-BEGIN,
        status='complete'if result is not None and result.get('status')=='complete'and exported is not None else'stopped',
        result=D.B.binding(a.run_dir/'PRODUCTION_RESULTS.json')if result is not None else None,
        exported=D.B.binding(a.run_dir/'export/MANIFEST.json')if exported is not None else None,
        error=error,script=D.B.binding(__file__),training_updates=0)
    D.B.save(a.run_dir/'OUTER_WALL.json',value,exclusive=True);print(value,flush=True)
if value['status']!='complete':raise SystemExit(1)
