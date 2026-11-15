"""Summarize completed full collection clocks and independent host observations."""
import argparse
import json
from pathlib import Path

import author_pipeline_collection as P

B=P.B
HERE=Path(__file__).resolve().parent


def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return merged


def duration(intervals):return sum(b-a for a,b in union(intervals))


def summarize(root,observations):
    protocol=P.verify(root);outer=B.read(root/'OUTER_WALL.json')
    result=B.read(root/'PIPELINE_COLLECTION_RESULTS.json');pipeline=B.read(root/'AUDIT_PIPELINE_RESULTS.json')
    assert outer['status']==result['status']=='complete' and result['pipeline_validity']=='pass'
    assert pipeline['all_futures_terminal'] and pipeline['queued_audits']==8
    assert pipeline['peak_active_audits']==2
    stages=[];intervals={}
    for path in sorted(root.rglob('STAGE.json')):
        stage=B.read(path);assert stage['status']=='complete' and stage['retry_count']==0
        command=stage['command'];scripts=[Path(x).name for x in command if str(x).endswith('.py')]
        if 'author_resident_execution.py' in scripts:kind='execute'
        elif Path(protocol['sources']['audit_entry']['path']).name in scripts:kind='audit'
        elif Path(protocol['sources']['capture_entry']['path']).name in scripts:kind='capture'
        elif 'run_collection_representation.py' in scripts:kind='representation'
        elif 'author_lift_pipeline.py' in scripts and 'prepare' in command:kind='prepare'
        elif Path(protocol['sources']['scene_entry']['path']).name in scripts:kind='build'
        else:raise AssertionError(scripts)
        a=stage['started_unix_s'];b=a+stage['wall_s']
        intervals.setdefault(kind,[]).append([a,b])
        stages.append(dict(kind=kind,stage=B.binding(path),child_pid=stage['child_pid'],
            started_unix_s=a,estimated_finished_unix_s=b,wall_s=stage['wall_s']))
    assert {k:len(v) for k,v in intervals.items()}==dict(build=2,capture=2,representation=1,prepare=8,execute=8,audit=8)
    timing={k:dict(count=len(v),sum_worker_wall_s=sum(b-a for a,b in v),
        active_union_s=duration(v),envelope_s=max(b for a,b in v)-min(a for a,b in v)) for k,v in intervals.items()}
    overlap=duration([[max(a,c),min(b,d)] for a,b in union(intervals['execute'])
        for c,d in union(intervals['audit']) if max(a,c)<min(b,d)])
    samples=[json.loads(line) for line in observations.read_text().splitlines()]
    assert samples and observations.with_suffix('.FINISHED.json').exists()
    peak_kits=max(x['active_kits'] for x in samples);peak_audits=max(x['active_audits'] for x in samples)
    assert peak_kits==8 and peak_audits==2
    max_growing=max(sum(v>0 for v in x['journal_growth_bytes'].values()) for x in samples)
    assert max_growing==8
    host=dict(sample_count=len(samples),peak_observed_kits=peak_kits,peak_observed_audits=peak_audits,
        max_journals_growing_in_one_sample=max_growing,
        samples_with_execute_and_audit=sum(x['active_kits']>0 and x['active_audits']>0 for x in samples),
        peak_owned_workers_rss_bytes=max(sum(w['rss_bytes'] for w in x['workers']) for x in samples),
        gpu_uuids=sorted({x['gpu0_nvml_csv'].split(',')[0].strip() for x in samples if 'gpu0_nvml_csv' in x}),
        sampling_error_count=sum('gpu_sampling_error' in x for x in samples),
        samples=B.binding(observations),observer_end=B.binding(observations.with_suffix('.FINISHED.json')))
    assert host['gpu_uuids']==[P.GPU_UUID]
    rows=P.build_namespace()['ledger'](root);assert len(rows)==8
    labels=[];points=0
    for row in rows:
        assert row['status']=='diagnostic_labelled' and row['label'] is None and row['exclude_training']
        audit=B.read(row['artifact_bindings']['runtime_audit']['path']);assert audit['status']=='pass'
        points+=audit['runtime']['contact_points']
        labels.append(dict(queue_index=row['queue_index'],gesture_id=row['gesture_id'],
            diagnostic_label=row['diagnostic_label'],steps=row['actual_physics_steps'],
            numeric_audit=row['artifact_bindings']['runtime_audit'],audit_wall_s=audit['wall_s'],
            contact_points=audit['runtime']['contact_points']))
    assert sum(r['steps'] for r in labels)==result['known_final_execution_physics_steps']==25656
    assert result['captured_initialization_ticks']==1500 and result['captured_initialization_object_steps']==4500
    reference=B.HERE/'evidence/author_resident_throughput_20261115_v1/CONTROL_RECEIPTS.json'
    old=B.read(reference)['fit']['outer']['wall_s'];wall=outer['outer_wall_s']
    capture_audits=[]
    for i in (1,4):
        path=observations.parent/f'SCENE{i}_CAPTURE_AUDIT.json';value=B.read(path)
        assert value['status']=='pass'
        B.verify(B.normalized_bindings(value['inputs_sha256']))
        capture_audits.append(dict(binding=B.binding(path),**{k:v for k,v in value.items() if k not in
            ('inputs_sha256','object_ids','object_sources','source_meshes','audit_source')}))
    summary=dict(status='pass',protocol=B.binding(root/P.PROTOCOL),outer=B.binding(root/'OUTER_WALL.json'),
        result=B.binding(root/'PIPELINE_COLLECTION_RESULTS.json'),outer_wall_s=wall,outer_minutes=wall/60,
        stage_timings=timing,stages=stages,execute_audit_stage_overlap_s=overlap,
        timing_scope='Stage intervals use saved start plus monotonic wall; overlapping worker sums are not additive to full outer wall',
        host=host,labels=labels,positive=sum(r['diagnostic_label'] for r in labels),
        negative=sum(r['diagnostic_label']==0 for r in labels),execution_steps=25656,execution_object_steps=76968,
        capture_ticks=1500,capture_object_steps=4500,total_Kit_starts=10,total_env_steps=27156,total_object_steps=81468,
        execution_contact_points=points,capture_audits=capture_audits,
        historical_reference=B.binding(reference),historical_resident_outer_s=old,historical_wall_ratio=old/wall,
        historical_comparison_scope='Same two known fit scenes and eight first-candidate trials; historical clocks, not randomized or paired timing. Old uses two hot sessions, new eight cold sessions.',
        diagnostic_labels_per_hour=8*3600/wall,formal_new_labels=0,training_updates=0,
        runtime_audits_per_trial=1,retries=0,verification_Kit_launches=0,verification_physics_steps=0,
        summarizer=B.binding(__file__))
    B.save(HERE/'SUMMARY.json',summary,exclusive=True)
    print({k:summary[k] for k in ('status','outer_wall_s','historical_wall_ratio','positive','negative','execution_steps')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True);args=parser.parse_args()
    summarize(args.run_dir,args.observations)
