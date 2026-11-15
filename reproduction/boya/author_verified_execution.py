"""Opt-in verified storage around the unchanged original execution/runtime.

Adds durable step-cost snapshots and replaces only the contact journal. Native
commands, contacts, physical clock, finish AST and the original auditor remain
unchanged. Import starts no Kit and never reopens a prepared trial.
"""
from contextlib import contextmanager
from pathlib import Path
import traceback

import author_collection_execute as E
import author_contact_compressor as W
import author_verified_contact_journal as J

EXECUTION_SHA='80603c076d458bfb856356368c6601fa8b638996bb61019123d5e2cc561be153'


def source_bindings():
    return {str(Path(p).resolve()):W.sha(p)for p in(__file__,W.__file__,J.__file__)}


@contextmanager
def install():
    if W.sha(E.__file__)!=EXECUTION_SHA:raise ValueError('Reviewed original execution entry changed')
    journal,factory=E.ContactJournal,E.runtime_class;extra=source_bindings()
    def runtime_class(module,bindings,journals):
        bindings.update(extra);base,finish_ast=factory(module,bindings,journals)
        class VerifiedStorageRuntime(base):
            def finish(self,*args,**kwargs):
                cost=dict(schema='anydex_boya_contact_storage_cost_v1',status='before_original_finish',
                    original_runtime_steps=self.steps,observed_trace_rows=len(self.rows),
                    contact_records=self.contact_rows.records,source_sha256=extra,
                    labels_certified=False,additional_physics_steps=0,
                    scope='Original returned-step and observation counters; no label or gate certification')
                path=self.output_dir/'CONTACT_STORAGE_COST.json';W.save(path,cost)
                try:
                    result=super().finish(*args,**kwargs);cost['status']='original_finish_returned'
                    return result
                except BaseException:
                    cost.update(status='original_finish_failed',error=traceback.format_exc());raise
                finally:
                    state=getattr(self.contact_rows,'state',None)
                    if state and Path(state).exists():cost['journal_state']=dict(path=str(state),sha256=W.sha(state))
                    W.save(path,cost)
        return VerifiedStorageRuntime,finish_ast
    E.ContactJournal=J.ContactJournal;E.runtime_class=runtime_class
    try:yield
    finally:E.ContactJournal=journal;E.runtime_class=factory


def execute_many(prepared_dirs,output_dir,*,source_bindings=None):
    import author_resident_execution as R
    from author_collection_batch import normalized_bindings
    sources=normalized_bindings(source_bindings or {})
    sources.update({p:dict(path=p,sha256=h)for p,h in globals()['source_bindings']().items()})
    with install():return R.execute_many(prepared_dirs,output_dir,source_bindings=sources)


if __name__=='__main__':
    import argparse
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['execute-many'])
    p.add_argument('--prepared-dirs',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-bindings',type=Path,required=True);a=p.parse_args()
    execute_many(a.prepared_dirs,a.output_dir,source_bindings=json.loads(a.source_bindings.read_text()))
