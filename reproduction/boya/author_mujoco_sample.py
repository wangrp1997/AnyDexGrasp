"""Versioned functional collection labels and bound, non-pickled head inputs.

This is a separate MuJoCo dataset domain. Minor contact residuals remain in
the diagnostic gates; the original 5mm stop and all other gates still apply.
Nothing here promotes or changes historical records or trains a model.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np
import author_collection_batch as B
import author_collect as COLLECT
from author_lift import POLICY as LIFT_POLICY, ORIGINAL_GATES

POLICY=dict(schema='anydex_boya_mujoco_functional_sample_v1',
    penetration='Diagnostic below original 5mm hard stop; retain original 1mm gate in raw results',
    early_stop_penetration_m=LIFT_POLICY['early_stop_penetration_m'],
    other_numeric_gates='All original six required',lift_policy=copy.deepcopy(LIFT_POLICY),
    positive='Complete original lift with source-bottom clearance, 5cm rise and thumb/opposing load',
    negative='Complete qualified execution without task success',
    null='Unexecuted, initialization/infrastructure error, incomplete/invalid execution or hard stop',
    authorization='User permits minor penetration, asks to accept functional success/failure collection',
    dataset_domain='Independent MuJoCo source-0.5ms functional acceptance; never silently merge with PhysX')


def classify(execution):
    if execution is None:return dict(status='not_executed',label=None,eligible=False)
    gates=execution['gates']
    # Source names are checked explicitly below, rather than defaulting missing
    # diagnostics to a successful negative label.
    non_penetration={k:v for k,v in gates.items() if k!='penetration'}
    if set(gates)!=set(ORIGINAL_GATES):
        raise ValueError('All original seven numeric gate results required')
    complete=(execution['status']=='complete' and execution['steps']==execution['expected_steps']
        and execution['stop'] is None)
    valid=(complete and execution.get('native_warning_count')==0
        and all(non_penetration.values()) and np.isfinite(list(execution['maxima'].values())).all()
        and execution['maxima']['penetration_m']<=POLICY['early_stop_penetration_m'])
    label=int(execution['evaluation']['task_success']) if valid else None
    return dict(status=('grasp_success' if label else 'grasp_failure') if valid else 'invalid_execution',
        label=label,eligible=bool(valid),original_gates=copy.deepcopy(gates),
        raw_task_success=bool(execution['evaluation']['task_success']))


def write_record(directory, spec, *, execution=None, terminal=None, error=None):
    directory=Path(directory);decision=classify(execution)
    if terminal is not None:
        if execution is not None:raise ValueError('An execution cannot be replaced by a terminal null')
        decision=dict(status=terminal,label=None,eligible=False)
    record=dict(schema='anydex_boya_mujoco_sample_record_v1',trial_id=spec['scene_id']+':g1:functional-v1',
        scene_id=spec['scene_id'],split=spec['split'],gesture_id=1,policy=copy.deepcopy(POLICY),
        **decision,error=error,bindings={},training_updates=0,
        separate_acceptance_cohort=True,formal_production_quota_frozen=False)
    if execution is not None:
        ready=B.read(directory/'live/READY.json')
        record['bindings']={k:ready[k] for k in ('scene','candidate','commands','command_binding','plan')}
        record['bindings'].update(execution=B.binding(directory/'execution/RESULTS.json'),trace=execution['trace'],
            observation=B.binding(directory/'capture/observation.json'),representation=B.binding(directory/'representation/RESULTS.json'))
        B.verify(record['bindings'])
    B.save(directory/'RECORD.json',record,exclusive=True)
    return record


def validate_record(path):
    record=B.read(path)
    if record['policy']!=POLICY:raise ValueError('Frozen functional sample policy mismatch')
    if record['label'] is None:
        if record['eligible']:raise ValueError('A null cannot be eligible')
        return record,None
    B.verify(record['bindings'])
    execution=B.read(record['bindings']['execution']['path'])
    decision=classify(execution)
    if any(record[k]!=v for k,v in decision.items()):raise ValueError('Sample label disagrees with saved execution')
    candidate=B.read(record['bindings']['candidate']['path'])
    feature=np.asarray(candidate['canonical_cgr480'],dtype=np.float32)
    if feature.shape!=(480,) or not np.isfinite(feature).all():raise ValueError('Finite float32 CGR480 required')
    if hashlib.sha256(feature.tobytes()).hexdigest()!=candidate['canonical_cgr480_identity']['c_order_bytes_sha256']:
        raise ValueError('Canonical feature binding changed')
    # Recompute from the actual raw source feature and saved flip, once.
    canonical=COLLECT.canonical_features(np.asarray([candidate['source_feature2261']]),[candidate['source_if_flip']])
    np.testing.assert_array_equal(feature,canonical['cgr480'][0].astype(np.float32))
    if int(canonical['raw_depth_index'][0])!=candidate['raw_depth_index']:
        raise ValueError('Original depth target changed')
    target=candidate['raw_depth_index']*4+candidate['extra_depth_index']
    if type(target) is not int or not 0<=target<20 or candidate['gesture_id']!=1:
        raise ValueError('Original g1 raw-depth/extra output required')
    scene=B.read(record['bindings']['scene']['path']);ids=sorted(o['object_id'] for o in scene['objects'])
    digests={o['object_id']:o['mesh_group_sha256'] for o in scene['objects']}
    return record,dict(features=feature,target_index=target,scene_object_ids=ids,
        scene_mesh_sha256=[digests[i] for i in ids],mesh_sha256=digests[scene['target_object_id']],
        candidate_sha256=candidate['candidate_sha256'])


def export(root, output):
    root=Path(root);output=Path(output);p=B.read(root/'PROTOCOL.json')
    # Export requires the independent trace audit, not just an eligible flag.
    audit=B.read(root/'READ_ONLY_AUDIT.json')
    if audit['status']!='read_only_audit_pass' or audit['original_result']!=B.binding(root/'RESULTS.json'):
        raise ValueError('Bound independent raw trace audit required')
    audited={r['index']:r for r in audit['rounds']};items=[]
    for spec in p['rows']:
        path=root/'scenes'/f"{spec['index']:02d}"/'RECORD.json'
        record,item=validate_record(path)
        if record['label'] is None:continue
        raw=audited[spec['index']]
        if not raw.get('original_controls_exact') or not raw.get('CGR_input_and_receipt_exact') or not raw['capture_stable']:
            raise ValueError('Label lacks raw camera/control/stability audit')
        if raw['trace']!=record['bindings']['trace']:raise ValueError('Audited execution trace changed')
        items.append((path,record,item))
    output.mkdir(parents=True,exist_ok=False)
    data=dict(features=np.asarray([t['features'] for _,_,t in items],np.float32).reshape(-1,480),
        labels=np.asarray([r['label'] for _,r,_ in items],np.int64),
        target_index=np.asarray([t['target_index'] for _,_,t in items],np.int64),
        trial_id=np.asarray([r['trial_id'] for _,r,_ in items],str),
        split=np.asarray(['train' if r['split']=='fit' else 'internal_check' for _,r,_ in items],str),
        mesh_sha256=np.asarray([t['mesh_sha256'] for _,_,t in items],str),
        scene_object_ids=np.asarray([t['scene_object_ids'] for _,_,t in items],str).reshape(-1,3),
        scene_mesh_sha256=np.asarray([t['scene_mesh_sha256'] for _,_,t in items],str).reshape(-1,3),
        candidate_sha256=np.asarray([t['candidate_sha256'] for _,_,t in items],str),
        record_json=np.asarray([str(path.resolve()) for path,_,_ in items],str),
        record_sha256=np.asarray([B.sha(path) for path,_,_ in items],str),
        commands_sha256=np.asarray([r['bindings']['commands']['sha256'] for _,r,_ in items],str),
        evaluation_sha256=np.asarray([r['bindings']['execution']['sha256'] for _,r,_ in items],str))
    if any(a.dtype.hasobject for a in data.values()):raise ValueError('Pickled arrays forbidden')
    from author_training import validate_arrays
    try:fit,check=validate_arrays(data);consumer_ready=True;error=None
    except ValueError as exc:consumer_ready=False;error=str(exc)
    # Even below the training minimum, enforce whole-scene split isolation.
    a={x for _,r,t in items if r['split']=='fit' for x in t['scene_mesh_sha256']}
    b={x for _,r,t in items if r['split']=='internal' for x in t['scene_mesh_sha256']}
    if a&b:raise ValueError('Whole-scene source identity leakage')
    path=output/'gesture_1.npz';np.savez_compressed(path,**data)
    with np.load(path,allow_pickle=False) as z:
        for k,v in data.items():np.testing.assert_array_equal(z[k],v)
    result=dict(status='exported',dataset=B.binding(path),records=len(items),
        positive=int(data['labels'].sum()),negative=int(len(items)-data['labels'].sum()),
        fit=int(np.count_nonzero(data['split']=='train')),internal=int(np.count_nonzero(data['split']=='internal_check')),
        training_consumer_ready=consumer_ready,consumer_error=error,policy=POLICY,
        source_protocol=B.binding(root/'PROTOCOL.json'),audit=B.binding(root/'READ_ONLY_AUDIT.json'),
        old_datasets_merged=False,training_updates=0)
    B.save(output/'MANIFEST.json',result,exclusive=True);return result
