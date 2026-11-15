"""Read-only final CPU diagnostic/audit ownership and immutable-input checks."""
from pathlib import Path
import json
import time

import author_cpu_prefix_control as P
import author_cpu_full_control as F

B,R,G=P.B,P.R,P.G
HERE=Path(__file__).resolve().parent
roots=[G.BASE/'cpu_prefix_control_20261115_v1',G.BASE/'cpu_full_control_20261115_v1']
protocols=[P.verify(roots[0]),F.verify(roots[1])]
processes=[];slots=[];locks=[]
for root,protocol,name in zip(roots,protocols,(P.RESULTS,F.RESULTS)):
    result=B.read(root/name)
    assert result['status']=='complete'
    for group in result['groups']:
        for worker in group['workers']:
            stage=B.read(worker['stage']['path']);identity=stage['process']
            assert R.process_identity(identity['pid']) is None and not R.group_alive(identity['pgid'])
            processes.append(dict(process=identity,stage=worker['stage'],current_identity=None,group_alive=False))
    R.assert_quiescent(root)
    for path in root.glob('resource_slots/*/*.json'):
        value=B.read(path);assert value['state']=='released'
        slots.append(dict(binding=B.binding(path),state=value['state']))
    for path in [root/'DRIVER.lock',*root.glob('resource_slots/*/*.lock')]:
        with R.lease(path):locks.append(str(path))
with R.lease(G.LOCK):locks.append(str(G.LOCK))
audits=B.ROOT/'outputs/boya_cpu_full_20261115_v1/audits'
audit_processes=[]
for path in sorted(audits.glob('cpu_*_worker_??_STARTED.json')):
    value=B.read(path);identity=value['process']
    assert R.process_identity(identity['pid']) is None and not R.group_alive(identity['pgid'])
    target=path.with_name(path.name.replace('_STARTED',''))
    assert B.read(target)['status']=='full_integrity_pass'
    audit_processes.append(dict(process=identity,report=B.binding(target),alive=False))
assert len(audit_processes)==9
with R.lease(audits/'AUDIT_DRIVER.lock'):locks.append(str(audits/'AUDIT_DRIVER.lock'))
assert B.read(G.REGISTRY)['state']=='released'
models=B.read(B.HERE/'evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')
B.verify(B.normalized_bindings(models['verified_files']))
labels=B.read(B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json')
B.verify(B.normalized_bindings(labels['verified_bindings']))
assert (labels['fit_labels'],labels['internal_labels'])==(14,1)
value=dict(status='pass',unix_s=time.time(),Kit_workers=processes,audit_workers=audit_processes,
    released_slots=slots,available_locks=locks,registry=B.binding(G.REGISTRY),
    frozen_runtime_sources_and_inputs_unchanged=True,old_data_models_count=len(models['verified_files']),
    old_data_models=models['verified_files'],old_label_reference=B.binding(
        B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json'),
    fit_labels=14,internal_labels=1,new_labels=0,training_updates=0,verification_physics_steps=0,
    verifier=B.binding(__file__))
B.save(HERE/'FINAL_STATE.json',value,exclusive=True)
print(json.dumps(dict(status='pass',Kit_workers=len(processes),audit_workers=len(audit_processes),
    released_slots=len(slots),available_locks=len(locks),old_data_models=len(models['verified_files']),new_labels=0)))
