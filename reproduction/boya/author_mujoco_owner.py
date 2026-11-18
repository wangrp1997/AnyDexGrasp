"""Lease a serial in-process controller without claiming a spawned child slot."""
from contextlib import contextmanager
from pathlib import Path
import os
import time

import author_collection_batch as B
import author_collection_wave_resources as R


@contextmanager
def scope(root,*,lock,registry):
    root=Path(root).resolve()
    with R.lease(root/'DRIVER.lock'),R.lease(lock),R.active_run(root,registry,B.binding(root/'PROTOCOL.json')):
        identity=R.process_identity(os.getpid())
        B.save(root/'RUN_STARTED.json',dict(process=identity,started_unix_s=time.time()),exclusive=True)
        try:yield
        finally:
            B.save(root/'OWNER_SCOPE_FINISHED.json',dict(controller_scope_exited=True,process=identity,
                finished_unix_s=time.time(),child_stage_reservations=0,
                scope='Serial owner itself holds both leases; no child-stage API is used for in-process work'),exclusive=True)
