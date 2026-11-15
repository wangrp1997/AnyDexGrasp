"""Audit the stopped64-slot prefix and six certified labels, without new physics."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import author_production_continuation as D

B,R=D.B,D.R
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--observations',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();root=a.run_dir.resolve();began=time.monotonic();p=D.verify(root)
rows=D.ledger(root,p);result=B.read(root/'PRODUCTION_RESULTS.json');outer=B.read(root/'OUTER_WALL.json')
assert result['status']==outer['status']=='stopped'and len(rows)==64
assert {(r['queue_index'],r['gesture_id'])for r in rows}=={(i,g)for i in range(16)for g in B.GESTURES}
assert not any(root.glob('windows/0[23]/RUN_STARTED.json'))
assert result['execution_starts_with_unknown_steps']==result['capture_starts_with_unknown_steps']==0
checked={}
def verify(entries):
    for value in B.normalized_bindings(entries).values():
        path,sha=value['path'],value['sha256']
        if path in checked:assert checked[path]==sha
        else:assert B.sha(path)==sha;checked[path]=sha
certified=[];failed=[];counts={s:{str(g):0 for g in B.GESTURES}for s in('fit','internal')}
for row in rows:
    spec=p['queue']['scenes'][row['queue_index']]
    if row['label']is not None:
        trial=B.validate_trial(Path(row['trial']['record']['path']).parent,spec);assert trial==row['trial']
        audit=B.read(row['artifact_bindings']['runtime_audit']['path']);B.certify_audit(audit,trial,spec)
        verify(audit['audit_inputs_sha256']);counts[row['split']][str(row['gesture_id'])]+=1
        certified.append(dict(queue_index=row['queue_index'],gesture_id=row['gesture_id'],label=row['label'],
            physics_steps=trial['actual_physics_steps'],contact_points=audit['runtime']['contact_points'],
            audit=row['artifact_bindings']['runtime_audit']))
    elif row['status']=='infrastructure_failure':
        audit=B.read(row['artifact_bindings']['runtime_audit']['path'])
        assert audit['status']=='failed'and audit['label']is None and not audit['formal_training_eligible']
        verify(audit['audit_inputs_sha256'])
        failed.append(dict(queue_index=row['queue_index'],gesture_id=row['gesture_id'],physics_steps=row['actual_physics_steps'],
            audit=row['artifact_bindings']['runtime_audit'],error=audit['error']))
assert len(certified)==6 and len(failed)==2 and counts==result['eligible_labels']
sessions=[];reported_steps=0
for path in sorted(root.glob('windows/*/scenes/*/g*/resident_session/SESSION_RECEIPT.json')):
    value=B.read(path);assert value['status']=='closed'and value['Kit_starts']==value['asset_setup_count']==len(value['jobs'])==1
    assert value['state_reset_count']==0
    target=path.parent.parent;receipt=B.read(target/'LIFT_PIPELINE_EXECUTION.json')
    evaluation=B.read(target/'execution/LIFT_EVALUATION.json');trace=B.read(target/'execution/MULTI_OBJECT_TRACE.json')
    n=evaluation['total_physics_steps'];assert receipt['physics_steps']==n==len(trace['rows'])
    assert [r['step']for r in trace['rows']]==list(range(1,n+1))
    reported_steps+=n;sessions.append(dict(receipt=B.binding(path),steps=n))
assert len(sessions)==8 and reported_steps==result['known_final_execution_physics_steps']==25462
captures=[]
for path in sorted(root.glob('windows/*/scenes/*/capture/INDEPENDENT_CAPTURE_AUDIT.json')):
    value=B.read(path);assert value['status']=='pass'and value['physics_ticks']==value['all_contact_frames']==750
    verify(value['inputs_sha256']);captures.append(dict(audit=B.binding(path),contact_points=value['all_contacts'],native_hulls=value['native_hulls']))
assert len(captures)==8 and result['captured_initialization_ticks']==6000
manifest=B.read(root/'export/MANIFEST.json');exported=[]
assert not manifest['old435_merged']and manifest['training_updates']==0 and not manifest['size_ready']
for g in B.GESTURES:
    entry=manifest['datasets'][str(g)];verify({'data':entry['binding']})
    with np.load(entry['binding']['path'],allow_pickle=False)as arrays:
        n=len(arrays['labels']);assert arrays['features'].shape==(n,480)
        assert arrays['scene_object_ids'].shape==arrays['scene_mesh_sha256'].shape==(n,3)
        assert not any(v.dtype.hasobject for v in arrays.values())
        for s,name in [('fit','train'),('internal','internal_check')]:
            assert np.count_nonzero(arrays['split']==name)==entry['counts'][s]==counts[s][str(g)]
        for i,tid in enumerate(arrays['trial_id'].tolist()):
            row=next(r for r in rows if r['trial_id']==tid);trial=row['trial']
            assert row['label']==int(arrays['labels'][i])and row['gesture_id']==g
            assert np.array_equal(arrays['features'][i],np.asarray(trial['features'],np.float32))
            assert arrays['scene_object_ids'][i].tolist()==trial['scene_object_ids']
            assert arrays['scene_mesh_sha256'][i].tolist()==trial['scene_mesh_sha256']
            exported.append(tid)
assert len(set(exported))==len(exported)==len(manifest['audits'])==6
stages=[]
for path in sorted(root.rglob('STAGE.json')):
    value=B.read(path);pid=value['child_pid']
    assert value['status']=='complete'and value['returncode']==value['retry_count']==0
    assert R.process_identity(pid)is None and not R.group_alive(pid)
    stages.append(dict(stage=B.binding(path),pid=pid,group_alive=False))
R.assert_quiescent(root);registry=B.read(D.P.OLD.REGISTRY)
assert registry['state']=='released'and registry['run_dir']==str(root)
assert R.process_identity(registry['owner']['pid'])is None
assert a.observations.with_suffix('.FINISHED.json').exists()
assert R.process_identity(B.read(a.observations.with_suffix('.STARTED.json'))['process']['pid'])is None
locks=[];slots=[]
for path in root.glob('resource_slots/*/*.json'):
    assert B.read(path)['state']=='released';slots.append(B.binding(path))
for path in[root/'DRIVER.lock',D.P.OLD.LEGACY_LOCK,*root.glob('resource_slots/*/*.lock')]:
    with R.lease(path):locks.append(str(path))
models=B.read(B.HERE/'evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')
verify(models['verified_files']);prior=B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json'
verify(B.read(prior)['verified_bindings'])
value=dict(status='partial_prefix_verified_stopped',protocol=B.binding(root/D.PROTOCOL),result=B.binding(root/'PRODUCTION_RESULTS.json'),
    outer=B.binding(root/'OUTER_WALL.json'),terminal_slots=64,not_started_scenes=15,not_started_gesture_slots=60,
    status_counts=dict(Counter(r['status']for r in rows)),counts=counts,certified_labels=certified,failed_audits=failed,
    exported_labels=6,export=B.binding(root/'export/MANIFEST.json'),reported_execution_steps=reported_steps,
    captures=captures,initialization_ticks=6000,initialization_object_steps=18000,execution_object_steps=3*reported_steps,
    total_reported_env_steps=6000+reported_steps,total_reported_object_steps=18000+3*reported_steps,
    continuation_Kit_starts=16,execution_unknown=0,capture_unknown=0,
    first_numeric_audits=8,passed_numeric_audits=6,failed_numeric_audits=2,
    stages=stages,released_slots=slots,available_locks=locks,checked_files=checked,checked_file_count=len(checked),
    old_data_models_unchanged=models['verified_files'],prior15_labels_unchanged=B.binding(prior),
    model_loads=2,CGR_forwards=8,hand_head_loads=0,training_updates=0,new_model_count=0,
    new_audit_Kit_starts=0,new_audit_physics_steps=0,repeated_full_numeric_point_audits=0,
    scope='Six saved numeric certificates and export independently checked; reported complete trace tick cost for all8 executions does not certify the two malformed contact streams',
    source=B.binding(__file__),wall_s=time.monotonic()-began)
B.save(a.output,value,exclusive=True)
print({k:value[k]for k in('status','exported_labels','counts','reported_execution_steps','checked_file_count')})
