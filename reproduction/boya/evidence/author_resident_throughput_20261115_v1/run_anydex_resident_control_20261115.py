import time
start=time.monotonic(); started=time.time()
import sys, json, traceback
from pathlib import Path
import author_resident_collection as C
import author_collection_batch as B
root=Path(sys.argv[1]).resolve()
B.save(root/'OUTER_DRIVER_STARTED.json',dict(started_unix_s=started,wrapper=B.binding(__file__)),exclusive=True)
error=None; result=None
try:
 result=C.run(root)
except BaseException:
 error=traceback.format_exc()
 raise
finally:
 B.save(root/'OUTER_DRIVER_RESULTS.json',dict(started_unix_s=started,finished_unix_s=time.time(),
  wall_s=time.monotonic()-start,error=error,wrapper=B.binding(__file__),
  status=result.get('status') if result else None,
  wall_scope='Python wrapper including imports, protocol preflight, complete loop and final lease release; shell/PTY startup excluded'),exclusive=True)
