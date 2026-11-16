"""Export already certified preclosure labels separately; no numerical rerun."""
import argparse
from pathlib import Path
import numpy as np
import author_contact_pilot as P
import author_contact_pipeline as M

B=P.B


def export(root,output):
    root=Path(root).resolve();p=P.verify(root);results=B.read(root/'RESULTS.json');out=Path(output).resolve()
    if results['status']!='complete':raise ValueError('Only a completed pilot can export')
    entries=[];bindings={'protocol':B.binding(root/'PROTOCOL.json'),'results':B.binding(root/'RESULTS.json')};ids=set()
    for row in results['rows']:
        if row['status']!='labelled':continue
        actual=p['rows'][row['index']];prepared=Path(row['trial']['record']['path']).parent
        B.verify({k:row[k]for k in ('runtime_audit','mapping_audit')})
        with M.installed():trial=B.validate_trial(prepared,actual)
        if trial!=row['trial']:raise ValueError('Certified trial changed')
        B.certify_audit(B.read(row['runtime_audit']['path']),trial,actual)
        audit=B.read(row['runtime_audit']['path'])
        B.verify({k:dict(path=k,sha256=v)for k,v in audit['audit_inputs_sha256'].items()})
        mapping=M.audit_mapping(prepared)
        if mapping!=B.read(row['mapping_audit']['path']):raise ValueError('Actual mapping/commands changed')
        identity=row['scene_id']+':g1:preclosure_v1'
        if identity in ids:raise ValueError('Repeated trial')
        ids.add(identity);entries.append((identity,row,actual,trial))
        bindings[identity+':runtime_audit']=row['runtime_audit'];bindings[identity+':mapping_audit']=row['mapping_audit']
    arrays=dict(features=np.asarray([t['features']for _,_,_,t in entries],dtype=np.float32).reshape(-1,480),
        labels=np.asarray([t['label']for _,_,_,t in entries],dtype=np.int64),
        target_index=np.asarray([t['target_index']for _,_,_,t in entries],dtype=np.int64),
        gesture_ids=np.ones(len(entries),dtype=np.int64),trial_id=np.asarray([i for i,_,_,_ in entries],dtype=str),
        split=np.asarray(['train'if a['split']=='fit'else'internal_check'for _,_,a,_ in entries],dtype=str),
        mesh_sha256=np.asarray([t['mesh_sha256']for _,_,_,t in entries],dtype=str),
        scene_object_ids=np.asarray([t['scene_object_ids']for _,_,_,t in entries],dtype=str).reshape(-1,3),
        scene_mesh_sha256=np.asarray([t['scene_mesh_sha256']for _,_,_,t in entries],dtype=str).reshape(-1,3),
        scene_id=np.asarray([r['scene_id']for _,r,_,_ in entries],dtype=str),
        candidate_sha256=np.asarray([t['candidate_sha256']for _,_,_,t in entries],dtype=str),
        commands_sha256=np.asarray([t['commands']['sha256']for _,_,_,t in entries],dtype=str),
        evaluation_sha256=np.asarray([t['evaluation']['sha256']for _,_,_,t in entries],dtype=str),
        record_json=np.asarray([t['record']['path']for _,_,_,t in entries],dtype=str),
        record_sha256=np.asarray([t['record']['sha256']for _,_,_,t in entries],dtype=str))
    if any(v.dtype.hasobject for v in arrays.values()):raise ValueError('Pickled data forbidden')
    out.mkdir(parents=True,exist_ok=False);path=out/'gesture_1.npz';np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False)as z:
        for k,value in arrays.items():np.testing.assert_array_equal(z[k],value)
    counts={s:int(np.count_nonzero(arrays['split']==v))for s,v in [('fit','train'),('internal','internal_check')]}
    manifest=dict(schema='anydex_boya_preclosure_pilot_export_v1',mapping_variant=M.SCHEMA,
        source_bindings=bindings,exporter=B.binding(__file__),datasets={'1':dict(binding=B.binding(path),counts=counts,
        positive=int(arrays['labels'].sum()),negative=int(len(entries)-arrays['labels'].sum()))},
        size_ready=False,old_mapping_labels_merged=False,old435_merged=False,training_updates=0,
        audit_reused='Full numerical certificates already completed; SHA, record, feature and new mapping/command bindings rechecked',
        new_physics_steps=0,new_Kit_starts=0)
    B.verify(bindings);P.verify(root);B.save(out/'MANIFEST.json',manifest,exclusive=True);return manifest


if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--root',type=Path,required=True)
    q.add_argument('--output-dir',type=Path,required=True);a=q.parse_args();print(export(a.root,a.output_dir)['datasets'])
