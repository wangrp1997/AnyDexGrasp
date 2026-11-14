"""Export only SHA-verified new lift labels, without collecting or training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

import author_collection_batch as batch


def _audit_trial(prepared_dir,protocol):
    source=protocol['sources']['audit_entry']
    batch.verify({'audit_entry':source})
    spec=importlib.util.spec_from_file_location('author_batch_bound_runtime_audit',source['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    split=protocol['queue'].get('split_manifest')
    if not split:raise ValueError('Frozen explicit object split manifest is required for independent export audit')
    batch.verify({'split':split})
    return module.audit_trial(prepared_dir,split_manifest=split['path'])


def export(batch_dir, output_dir):
    source=Path(batch_dir).resolve();out=Path(output_dir).resolve()
    protocol=batch.read(source/'BATCH_PROTOCOL.json')
    if protocol.get('schema')!=batch.SCHEMA or protocol.get('task')!=batch.TASK:raise ValueError('New lift batch required; old435 is never merged')
    batch.verify(protocol['inputs']);batch.verify(protocol['sources'])
    rows=batch.ledger(source,protocol)
    queue=protocol['queue']['scenes']
    ids={'fit':set(),'internal':set()};meshes={'fit':set(),'internal':set()};trials=set();data={g:[]for g in batch.GESTURES}
    audits={}
    for row in rows:
        if row.get('label')is None:continue
        spec=queue[row['queue_index']]
        if row['trial_id']in trials:raise ValueError('Duplicate actually executed trial')
        trials.add(row['trial_id'])
        path=Path(row['trial']['record']['path']).parent
        trial=batch.validate_trial(path,spec)
        if trial!=row['trial']:raise ValueError('Ledger trial changed after collection')
        audit=_audit_trial(path,protocol)
        batch.certify_audit(audit,trial,spec)
        audits[row['trial_id']]=audit
        split=row['split'];g=row['gesture_id']
        if split!=spec['split']:raise ValueError('Whole scene split changed')
        ids[split].update(trial['scene_object_ids']);meshes[split].update(trial['scene_mesh_sha256'])
        if len(data[g])and (row['queue_index']<=data[g][-1][0]):raise ValueError('Fixed scene prefix ordering required')
        data[g].append((row['queue_index'],row,trial))
    if ids['fit']&ids['internal'] or meshes['fit']&meshes['internal']:
        raise ValueError('Nonprimary source object leakage between fit and internal data')
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema='anydex_boya_author_collection_export_v1',task=batch.TASK,
        source_protocol=batch.binding(source/'BATCH_PROTOCOL.json'),old435_merged=False,training_updates=0,
        mesh_sha256_semantics='Compatibility primary only; scene_mesh_sha256 is the required all-body isolation column',
        fit_source_object_ids=sorted(ids['fit']),internal_source_object_ids=sorted(ids['internal']),
        fit_source_mesh_sha256=sorted(meshes['fit']),internal_source_mesh_sha256=sorted(meshes['internal']),datasets={},audits={})
    for trial_id,audit in audits.items():
        path=out/'audits'/f'{hashlib.sha256(trial_id.encode()).hexdigest()}.json'
        batch.save(path,audit,exclusive=True);manifest['audits'][trial_id]=batch.binding(path)
    for g,entries in data.items():
        arrays=dict(features=np.asarray([t['features']for _,_,t in entries],dtype=np.float32).reshape(-1,480),
            labels=np.asarray([t['label']for _,_,t in entries],dtype=np.int64),
            target_index=np.asarray([t['target_index']for _,_,t in entries],dtype=np.int64),
            gesture_ids=np.full(len(entries),g,dtype=np.int64),
            trial_id=np.asarray([r['trial_id']for _,r,_ in entries],dtype=str),
            split=np.asarray(['train'if r['split']=='fit'else'internal_check'for _,r,_ in entries],dtype=str),
            mesh_sha256=np.asarray([t['mesh_sha256']for _,_,t in entries],dtype=str),
            scene_object_ids=np.asarray([t['scene_object_ids']for _,_,t in entries],dtype=str).reshape(-1,3),
            scene_mesh_sha256=np.asarray([t['scene_mesh_sha256']for _,_,t in entries],dtype=str).reshape(-1,3),
            scene_id=np.asarray([r['scene_id']for _,r,_ in entries],dtype=str),
            candidate_sha256=np.asarray([t['candidate_sha256']for _,_,t in entries],dtype=str),
            commands_sha256=np.asarray([t['commands']['sha256']for _,_,t in entries],dtype=str),
            evaluation_sha256=np.asarray([t['evaluation']['sha256']for _,_,t in entries],dtype=str),
            record_json=np.asarray([t['record']['path']for _,_,t in entries],dtype=str),
            record_sha256=np.asarray([t['record']['sha256']for _,_,t in entries],dtype=str))
        if any(v.dtype.hasobject for v in arrays.values()):raise ValueError('Pickled training arrays forbidden')
        path=out/f'gesture_{g}.npz';np.savez_compressed(path,**arrays)
        count={s:int(np.count_nonzero(arrays['split']==v))for s,v in [('fit','train'),('internal','internal_check')]}
        if any(count[s]>protocol['quotas'][s]for s in count):raise ValueError('Export exceeds frozen prefix quota')
        manifest['datasets'][str(g)]=dict(binding=batch.binding(path),counts=count,positive=int(arrays['labels'].sum()),negative=int(len(entries)-arrays['labels'].sum()))
    manifest['size_ready']=all(r['counts'][s]==protocol['quotas'][s]for r in manifest['datasets'].values()for s in ('fit','internal'))
    batch.verify(protocol['inputs']);batch.verify(protocol['sources'])
    batch.save(out/'MANIFEST.json',manifest,exclusive=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();print(export(a.batch_dir,a.output_dir))


if __name__=='__main__':main()
