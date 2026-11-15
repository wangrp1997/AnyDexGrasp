"""Original full all-point CPU audit plus the new producer/readback seal."""
import argparse
from pathlib import Path
import time
import traceback

import author_cpu_full_audit as ORIGINAL
import author_verified_contact_journal as J

A=ORIGINAL.A


def audit(worker):
    began=time.monotonic();bindings=ORIGINAL.S.Bindings()
    numeric=ORIGINAL.audit_worker(worker,bindings,ORIGINAL.native_state_auditor())
    record=A.read(worker);group=A.read(record['group_receipt']['path'])
    execution=Path(group['jobs'][0]['evaluation']['path']).parent
    pointer_path=execution/'MULTI_OBJECT_CONTACTS.json';pointer=A.read(pointer_path)
    seal=J.verify_pointer(pointer)
    A.need(seal['isolated']is True and seal['no_site']is True and seal['loaded_native_framework_modules']==[],
        'Compressor was not isolated from the native runtime')
    bindings.tree(pointer['integrity_seal'])
    cost=execution/'CONTACT_STORAGE_COST.json';bindings.add(cost);value=A.read(cost)
    A.need(value['status']=='original_finish_returned'and value['original_runtime_steps']==3024
        and value['observed_trace_rows']==value['contact_records']==3024 and value['additional_physics_steps']==0,
        'Original cost/observation counters changed')
    bindings.verify()
    return dict(status='full_integrity_pass',numeric=numeric,seal=pointer['integrity_seal'],
        logical=pointer['logical'],worker=A.sha(worker),inputs_sha256=bindings.files,
        audit_wall_s=time.monotonic()-began,Kit_starts=0,physics_steps=0,label=None,exclude_training=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Original first audit must not be repeated')
    try:result=audit(a.worker)
    except BaseException:result=dict(status='failed',error=traceback.format_exc(),label=None,exclude_training=True)
    from author_collection_batch import save
    save(a.output,result,exclusive=True);print(result['status'])
    if result['status']!='full_integrity_pass':raise SystemExit(1)
