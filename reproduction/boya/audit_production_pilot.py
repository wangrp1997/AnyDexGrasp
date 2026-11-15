"""Independent saved-certificate, export, cost and owned-host audit; zero physics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import author_production_collection as D

B,R=D.B,D.R


def audit(root,observations,output):
    began=time.monotonic();p=D.verify(root);rows=D.ledger(root,p)
    result=B.read(root/'PRODUCTION_RESULTS.json');outer=B.read(root/'OUTER_WALL.json')
    assert result['status']==outer['status']=='complete' and len(rows)==128
    assert {(r['queue_index'],r['gesture_id'])for r in rows}=={(i,g)for i in range(32)for g in B.GESTURES}
    assert all(r['status']in D.GOOD for r in rows)
    assert result['execution_starts_with_unknown_steps']==result['capture_starts_with_unknown_steps']==0
    counts={s:{str(g):0 for g in B.GESTURES}for s in('fit','internal')};positive={s:{str(g):0 for g in B.GESTURES}for s in counts}
    contacts=0;steps=0;certificates=[];ids={s:set()for s in counts};meshes={s:set()for s in counts}
    for row in rows:
        spec=p['queue']['scenes'][row['queue_index']]
        ids[row['split']].update(spec['source_object_ids']);meshes[row['split']].update(spec['source_object_digests'].values())
        if row['label']is None:continue
        assert row['formal_training_eligible']and not row['exclude_training']
        trial=B.validate_trial(Path(row['trial']['record']['path']).parent,spec);assert trial==row['trial']
        cert=B.read(row['artifact_bindings']['runtime_audit']['path']);B.certify_audit(cert,trial,spec)
        counts[row['split']][str(row['gesture_id'])]+=1;positive[row['split']][str(row['gesture_id'])]+=row['label']
        contacts+=cert['runtime']['contact_points'];steps+=cert['runtime']['physics_steps']
        certificates.append(dict(trial_id=row['trial_id'],split=row['split'],gesture_id=row['gesture_id'],label=row['label'],
            physics_steps=cert['runtime']['physics_steps'],contact_points=cert['runtime']['contact_points'],
            certificate=row['artifact_bindings']['runtime_audit']))
    assert not ids['fit']&ids['internal'] and not meshes['fit']&meshes['internal']
    assert result['eligible_labels']==counts and result['positive_labels']==positive
    assert result['known_final_execution_physics_steps']==steps
    exported=B.read(root/'export/MANIFEST.json');B.verify({'protocol':exported['source_protocol']})
    assert exported['source_protocol']==B.binding(root/D.PROTOCOL)and not exported['old435_merged']and exported['training_updates']==0
    exported_trials=[]
    for g in B.GESTURES:
        entry=exported['datasets'][str(g)];B.verify({'dataset':entry['binding']})
        with np.load(entry['binding']['path'],allow_pickle=False)as arrays:
            n=len(arrays['labels']);assert arrays['features'].shape==(n,480)
            assert arrays['scene_object_ids'].shape==arrays['scene_mesh_sha256'].shape==(n,3)
            assert not any(a.dtype.hasobject for a in arrays.values())
            for s,name in [('fit','train'),('internal','internal_check')]:
                assert int(np.count_nonzero(arrays['split']==name))==counts[s][str(g)]==entry['counts'][s]
            for i,tid in enumerate(arrays['trial_id'].tolist()):
                row=next(r for r in rows if r['trial_id']==tid);trial=row['trial']
                assert row['gesture_id']==g and int(arrays['labels'][i])==row['label']
                assert np.array_equal(arrays['features'][i],np.asarray(trial['features'],np.float32))
                assert arrays['scene_object_ids'][i].tolist()==trial['scene_object_ids']
                assert arrays['scene_mesh_sha256'][i].tolist()==trial['scene_mesh_sha256']
                exported_trials.append(tid)
    assert len(set(exported_trials))==len(exported_trials)==len(certificates)
    assert len(exported['audits'])==len(certificates)
    captures=[];sessions=[];first_audits=0;loads=0;forwards=0
    for path in sorted(root.glob('windows/*/scenes/*/capture/INDEPENDENT_CAPTURE_AUDIT.json')):
        value=B.read(path);assert value['status']=='pass'and value['physics_ticks']==value['all_contact_frames']==750
        B.verify(B.normalized_bindings(value['inputs_sha256']))
        captures.append(dict(binding=B.binding(path),contact_points=value['all_contacts'],native_hulls=value['native_hulls'],
            physics_ticks=750,independent_full_source_rays=value['independent_full_source_rays']))
    for path in sorted(root.glob('windows/*/scenes/*/g*/resident_session/SESSION_RECEIPT.json')):
        value=B.read(path)
        assert value['status']=='closed'and value['Kit_starts']==value['asset_setup_count']==len(value['jobs'])==1
        assert value['state_reset_count']==0
        sessions.append(B.binding(path))
    for path in root.glob('windows/*/AUDIT_PIPELINE_RESULTS.json'):
        value=B.read(path);assert value['all_futures_terminal']and value['peak_active_audits']<=2
        first_audits+=value['queued_audits']
    assert len(sessions)==first_audits==len(certificates)
    for path in root.glob('windows/*/representation/BATCH_REPRESENTATION.json'):
        value=B.read(path);assert value['hand_head_load_count']==0 and value['model_load_count']==1
        loads+=1;forwards+=value['network_forward_count']
    assert forwards==len(captures)
    assert result['captured_initialization_ticks']==750*len(captures)
    stages=[]
    for path in sorted(root.rglob('STAGE.json')):
        value=B.read(path);pid=value['child_pid'];assert value['status']=='complete'and value['returncode']==value['retry_count']==0
        assert R.process_identity(pid)is None and not R.group_alive(pid)
        stages.append(dict(stage=B.binding(path),pid=pid,group_alive=False,wall_s=value['wall_s']))
    R.assert_quiescent(root);registry=B.read(D.P.OLD.REGISTRY)
    assert registry['state']=='released'and registry['run_dir']==str(root)
    assert R.process_identity(registry['owner']['pid'])is None
    assert observations.with_suffix('.FINISHED.json').exists()
    assert R.process_identity(B.read(observations.with_suffix('.STARTED.json'))['process']['pid'])is None
    samples=[json.loads(line)for line in observations.read_text().splitlines()]
    assert max(s['active_kits']for s in samples)<=8 and max(s['active_audits']for s in samples)<=2
    slots=[];locks=[]
    for path in root.glob('resource_slots/*/*.json'):
        assert B.read(path)['state']=='released';slots.append(B.binding(path))
    for path in[root/'DRIVER.lock',D.P.OLD.LEGACY_LOCK,*root.glob('resource_slots/*/*.lock')]:
        with R.lease(path):locks.append(str(path))
    old=B.read(B.HERE/'evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')
    B.verify(B.normalized_bindings(old['verified_files']))
    prior=B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json'
    B.verify(B.normalized_bindings(B.read(prior)['verified_bindings']))
    summary=dict(status='pass',protocol=B.binding(root/D.PROTOCOL),result=B.binding(root/'PRODUCTION_RESULTS.json'),outer=B.binding(root/'OUTER_WALL.json'),
        terminal_slots=128,status_counts=dict(Counter(r['status']for r in rows)),counts=counts,positive=positive,
        new_labels=len(certificates),certificates=certificates,export=B.binding(root/'export/MANIFEST.json'),
        physics_steps=steps,execution_object_steps=3*steps,initialization_ticks=750*len(captures),
        initialization_object_steps=2250*len(captures),total_Kit_starts=len(captures)+len(sessions),
        total_env_steps=steps+750*len(captures),total_object_steps=3*steps+2250*len(captures),
        unknown_execution_costs=0,unknown_capture_costs=0,execution_contact_points=contacts,captures=captures,
        model_loads=loads,model_forwards=forwards,hand_head_loads=0,training_updates=0,old435_merged=False,
        outer_wall_s=outer['outer_wall_s'],host_samples=len(samples),peak_kits=max(s['active_kits']for s in samples),
        peak_audits=max(s['active_audits']for s in samples),stages=stages,released_slots=slots,available_locks=locks,
        old_data_models_unchanged=old['verified_files'],prior15_labels_unchanged=B.binding(prior),
        new_audit_physics_steps=0,new_audit_Kit_starts=0,repeated_execution_point_decodes=0,
        auditor=B.binding(__file__),wall_s=time.monotonic()-began)
    B.save(output,summary,exclusive=True)
    print({k:summary[k]for k in('status','new_labels','counts','outer_wall_s','total_Kit_starts','total_object_steps')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.run_dir,a.observations,a.output)
