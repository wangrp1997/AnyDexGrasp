"""Export audited diagnostic 0/1 labels and verify the original head input."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import author_collection_batch as B


def finish(root):
    import author_collect as COLLECT
    import author_mujoco_sample as SAMPLE
    root=Path(root);p=B.read(root/'PROTOCOL.json');audit=B.read(root/'READ_ONLY_AUDIT.json')
    if audit['status']!='read_only_audit_pass'or audit['original_result']!=B.binding(root/'RESULTS.json'):raise ValueError('Bound raw GPU audit required')
    checked={r['index']:r for r in audit['rounds']};items=[]
    for spec in p['rows']:
        path=root/'scenes'/f"{spec['index']:02d}"/'RECORD.json';record=B.read(path)
        if record['label']is None:
            if record['eligible']:raise ValueError('Null cannot be eligible')
            continue
        B.verify(record['bindings']);execution=B.read(record['bindings']['execution']['path'])
        for k,v in SAMPLE.classify(execution).items():
            if record[k]!=v:raise ValueError('Label differs from original functional classification')
        row=checked[spec['index']]
        if not row.get('original_controls_exact')or not row.get('CGR_input_and_receipt_exact')or not row['capture_stable']:raise ValueError('Eligible label lacks full raw provenance')
        candidate=B.read(record['bindings']['candidate']['path']);feature=np.asarray(candidate['canonical_cgr480'],dtype=np.float32)
        if feature.shape!=(480,)or not np.isfinite(feature).all():raise ValueError('Finite canonical480 required')
        if hashlib.sha256(feature.tobytes()).hexdigest()!=candidate['canonical_cgr480_identity']['c_order_bytes_sha256']:raise ValueError('Feature hash changed')
        canonical=COLLECT.canonical_features(np.asarray([candidate['source_feature2261']]),[candidate['source_if_flip']])
        np.testing.assert_array_equal(feature,canonical['cgr480'][0].astype(np.float32))
        target=candidate['raw_depth_index']*4+candidate['extra_depth_index']
        if not 0<=target<20:raise ValueError('Original depth index required')
        scene=B.read(record['bindings']['scene']['path']);objects=sorted(scene['objects'],key=lambda o:o['object_id'])
        items.append(dict(path=path,record=record,feature=feature,target=target,objects=objects,scene=scene,candidate=candidate))
    data=dict(features=np.asarray([i['feature']for i in items],np.float32).reshape(-1,480),
        labels=np.asarray([i['record']['label']for i in items],np.int64),target_index=np.asarray([i['target']for i in items],np.int64),
        trial_id=np.asarray([i['record']['scene_id']+':g1:mjx:'+B.sha(root/'PROTOCOL.json')[:12]for i in items],str),
        split=np.asarray(['train'if i['record']['split']=='fit'else'internal_check'for i in items],str),
        mesh_sha256=np.asarray([next(o['mesh_group_sha256']for o in i['objects']if o['object_id']==i['scene']['target_object_id'])for i in items],str),
        scene_object_ids=np.asarray([[o['object_id']for o in i['objects']]for i in items],str).reshape(-1,3),
        scene_mesh_sha256=np.asarray([[o['mesh_group_sha256']for o in i['objects']]for i in items],str).reshape(-1,3),
        candidate_sha256=np.asarray([i['candidate']['candidate_sha256']for i in items],str),
        record_json=np.asarray([str(i['path'].resolve())for i in items],str),record_sha256=np.asarray([B.sha(i['path'])for i in items],str))
    fit={h for i in items if i['record']['split']=='fit'for h in [o['mesh_group_sha256']for o in i['objects']]}
    check={h for i in items if i['record']['split']=='internal'for h in [o['mesh_group_sha256']for o in i['objects']]}
    if fit&check:raise ValueError('Whole-scene source split leakage')
    output=root/'export';output.mkdir(exist_ok=False);dataset=output/'gesture_1.npz';np.savez_compressed(dataset,**data)
    with np.load(dataset,allow_pickle=False)as z:
        for k,v in data.items():np.testing.assert_array_equal(z[k],v)
    from author_training import validate_arrays
    try:
        a,b=validate_arrays(data)
        from author_mujoco_acceptance import dry_consumer
        consumer=dry_consumer(dataset,root/'HEAD_INPUT_CHECK.json');error=None
    except ValueError as exc:consumer=None;error=str(exc)
    result=dict(status='diagnostic_export_complete',dataset=B.binding(dataset),records=len(items),positive=int(data['labels'].sum()),
        negative=int(len(items)-data['labels'].sum()),fit=int(np.count_nonzero(data['split']=='train')),
        internal=int(np.count_nonzero(data['split']=='internal_check')),original_head_input_check=consumer,consumer_error=error,
        exclude_training=True,formal_training_eligible=False,training_updates=0,old_data_merged=False,
        source_protocol=B.binding(root/'PROTOCOL.json'),source_audit=B.binding(root/'READ_ONLY_AUDIT.json'))
    B.save(output/'MANIFEST.json',result,exclusive=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    p=B.read(a.root/'PROTOCOL.json');marker=B.sha(a.root/'PROTOCOL.json')
    if os.environ.get('ANYDEX_MJX_FINISH_ENV')!=marker:
        env=B._environment(p);env['ANYDEX_MJX_FINISH_ENV']=marker
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(a.root.resolve())],env)
    print(finish(a.root))
