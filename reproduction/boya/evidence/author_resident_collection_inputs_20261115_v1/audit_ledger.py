"""Independent read-only parent admission/cost/identity checks after completion."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(HERE))
import author_collection_batch as B

def audit(root):
    began=time.monotonic();root=Path(root).resolve();checked={}
    def binding(entry):
        path=str(Path(entry['path']).resolve());expected=entry['sha256']
        if path in checked:assert checked[path]==expected,path
        else:assert B.sha(path)==expected,path;checked[path]=expected
    def bindings(entries):
        for entry in entries.values():binding(entry)
    protocol=B.read(root/'RESIDENT_COLLECTION_PROTOCOL.json');protocol_sha=B.sha(root/'RESIDENT_COLLECTION_PROTOCOL.json')
    assert protocol['diagnostic_only']is True and protocol['exclude_training']is True and protocol['automatic_training']is False
    bindings(protocol['inputs']);bindings(protocol['sources']);binding(protocol['source_batch_protocol'])
    assert protocol['gestures']==[1,6,7,8]
    scene_rows=dict(zip(protocol['source_scene_indices'],protocol['queue']['scenes']))
    expected={(i,g)for i in scene_rows for g in(1,6,7,8)};seen=set();labels=[];rows=[];stages=set()
    for path in sorted((root/'ledger').glob('*.json')):
        row=B.read(path);key=(row['queue_index'],row['gesture_id']);assert key in expected and key not in seen;seen.add(key)
        assert row['protocol_sha256']==protocol_sha and row['label']is None and row['formal_training_eligible']is False and row['exclude_training']is True
        frozen=scene_rows[row['queue_index']]
        assert row['scene_id']==frozen['scene_id'] and row['split']==frozen['split']
        assert row['source_object_ids']==frozen['source_object_ids'] and row['source_object_digests']==frozen['source_object_digests']
        bindings(row['artifact_bindings'])
        for entry in row['artifact_bindings'].values():
            if Path(entry['path']).name=='STAGE.json'and entry['path']not in stages:
                stages.add(entry['path']);bindings(B.read(entry['path']).get('outputs',{}))
        if row['diagnostic_label']is not None:
            assert row['status']=='diagnostic_labelled' and type(row['diagnostic_label'])is int and row['diagnostic_label']in(0,1)
            assert row['raw_label']==row['diagnostic_label'] and row['execution_steps_unknown']is False
            trial=row['trial'];record=B.read(trial['record']['path']);candidate=B.read(trial['candidate']['path'])
            evaluation=B.read(trial['evaluation']['path']);scene=B.read(trial['scene']['path']);audit=B.read(row['artifact_bindings']['runtime_audit']['path'])
            assert record['label']==trial['label']==audit['label']==row['diagnostic_label']
            assert record['formal_training_eligible']is True and audit['formal_training_eligible']is True and audit['status']=='pass'
            assert audit['split']==row['split'] and dict(zip(audit['original_object_ids'],audit['all_scene_source_mesh_sha256']))==row['source_object_digests']
            assert scene['bin']is None and scene['task']==evaluation['task']==B.TASK
            actual={o['object_id']:o['source_visual_mesh_sha256']for o in scene['objects']};assert actual==row['source_object_digests']
            assert candidate['gesture_id']==row['gesture_id'] and candidate['candidate_sha256']==trial['candidate_sha256']
            features=np.asarray(candidate['canonical_cgr480'],dtype=np.float32)
            assert features.shape==(480,)and np.isfinite(features).all()
            assert hashlib.sha256(features).hexdigest()==candidate['canonical_cgr480_identity']['c_order_bytes_sha256']
            assert np.array_equal(features,np.asarray(trial['features'],np.float32))
            assert trial['target_index']==candidate['raw_depth_index']*4+candidate['extra_depth_index']
            outcomes=evaluation['object_outcomes'];ids=[o['object_id']for o in outcomes]
            assert len(ids)==3 and len(set(ids))==3 and set(ids)==set(actual)
            assert evaluation['task_success']is any(o['task_success']for o in outcomes)
            assert int(evaluation['task_success'])==row['diagnostic_label']
            assert evaluation['total_physics_steps']==row['actual_physics_steps']
            labels.append(dict(queue_index=key[0],gesture_id=key[1],label=row['diagnostic_label'],physics_steps=row['actual_physics_steps']))
        else:assert row['raw_label']is None
        rows.append(row)
    assert seen==expected,('Missing terminal slots',expected-seen)
    progress=B.read(root/'PROGRESS.json');results=B.read(root/'RESIDENT_COLLECTION_RESULTS.json')
    assert progress==results and progress['status']=='complete'and progress['ledger_rows']==len(expected)
    assert progress['known_final_execution_physics_steps']==sum(r['actual_physics_steps']for r in rows)
    assert progress['execution_starts_with_unknown_steps']==sum(r['execution_steps_unknown']for r in rows)
    assert B.sha(root/'RESIDENT_COLLECTION_PROTOCOL.json')==protocol_sha
    return dict(status='pass',protocol=B.binding(root/'RESIDENT_COLLECTION_PROTOCOL.json'),terminal_slots=len(rows),diagnostic_labels=labels,
        formal_training_labels=0,all_diagnostic_rows_excluded=True,checked_file_count=len(checked),checked_files=checked,
        complete_clock_wall_s=progress['timing_seconds']['total_wall'],scope='Read saved numeric audit PASS and independently verify admission/identity/features/cost and all bound artifacts; no repeated physics or source-bottom audit',
        audit_physics_steps=0,audit_model_forwards=0,wall_s=time.monotonic()-began)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    result=audit(a.run_dir);result['audit_source']=B.binding(__file__);B.save(a.output,result,exclusive=True)
    print(json.dumps({k:result[k]for k in('status','terminal_slots','formal_training_labels','checked_file_count','wall_s')}))
if __name__=='__main__':main()
