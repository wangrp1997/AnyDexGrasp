"""Independently recount exact-window events and close this test's resources."""
import json
from pathlib import Path
import argparse

import author_collection_batch as B
import author_collection_wave_resources as R


def review(root):
    root=Path(root).resolve();directory=root/'collection'
    measured=B.read(root/'ACTUAL_HOUR_RESULTS.json');p=B.read(directory/'PROTOCOL.json')
    window=B.read(directory/'HOUR_WINDOW.json');start=window['started_monotonic_ns'];end=window['deadline_monotonic_ns']
    if end-start!=3600_000_000_000:raise ValueError('Not an exact hour')
    records={};returns={};audited={};admitted={};started={};events=[];previous=start
    for line in(directory/'HOUR_EVENTS.jsonl').open():
        row=json.loads(line);now=row['monotonic_ns']
        if now<previous or row['within_window']!=(start<=now<end)or row['elapsed_s']!=(now-start)/1e9:
            raise ValueError('Incorrect event window/clock')
        previous=now;events.append(row);kind=row['kind']
        if kind=='record_committed':
            if row['index']in records:raise ValueError('Duplicate record event')
            records[row['index']]=row
        elif kind=='physical_return_complete':
            if row['index']in returns:raise ValueError('Duplicate return event')
            returns[row['index']]=row
        elif kind=='grasp_started':started[row['index']]=row
        elif kind=='wave_started':
            for i in row['indices']:admitted[i]=row
        elif kind=='wave_audit_pass':
            finish=B.read(directory/'wave_audits'/f"{row['wave']:03d}"/'FINISHED.json')
            if finish['event']!=row:raise ValueError('Audit event not bound to completed audit')
            B.verify({'audit':finish['audit']})
            for i in row['eligible_indices']:
                if i in audited:raise ValueError('Duplicate audited label')
                audited[i]=row
    for i,row in records.items():
        actual=B.read(directory/'scenes'/f'{i:02d}'/'RECORD.json')
        if actual['eligible']!=row['eligible']or actual['label']!=row['label']or actual['status']!=row['status']:
            raise ValueError('Event label differs from immutable record')
    eligible={i:r for i,r in records.items()if r['eligible']and r['within_window']}
    returned={i:r for i,r in eligible.items()if i in returns and returns[i]['passed']and returns[i]['within_window']}
    verified={i:r for i,r in returned.items()if i in audited and audited[i]['within_window']}
    expected=dict(eligible_labels_within_window=len(eligible),grasp_and_return_within_window=len(returned),
        audited_labels_within_window=len(verified),eligible_positive_within_window=sum(r['label']==1 for r in eligible.values()),
        eligible_negative_within_window=sum(r['label']==0 for r in eligible.values()),
        audited_positive_within_window=sum(r['label']==1 for r in verified.values()),
        audited_negative_within_window=sum(r['label']==0 for r in verified.values()))
    for key,value in expected.items():
        if measured['counts'][key]!=value:raise ValueError('Reported count differs from exact events '+key)
    latencies=[]
    for i,row in records.items():
        if not row['eligible']:continue
        item=dict(index=i,label=row['label'],record_elapsed_s=row['elapsed_s'],within_window=row['within_window'])
        if i in admitted:item['wave_admission_to_label_s']=row['elapsed_s']-admitted[i]['elapsed_s']
        if i in started:item['grasp_start_to_label_s']=row['elapsed_s']-started[i]['elapsed_s']
        if i in audited:item['audit_elapsed_s']=audited[i]['elapsed_s']
        latencies.append(item)
    B.save(root/'SAMPLE_TIMESTAMPS.json',dict(records=latencies,
        note='Wave admission is shared by the 16 input slots; these are elapsed latencies, not normalized throughput.'),exclusive=True)
    last=None;calls=0
    for line in(directory/'NATIVE_CALLS.jsonl').open():
        value=json.loads(line)
        if calls==0:
            if value!={'completed_native_environment_steps':0,'pending_native_environment_steps_upper':0}:raise ValueError('Missing initial native cost')
        elif calls%2:
            if value['completed_native_environment_steps']!=last['completed_native_environment_steps']or value['pending_native_environment_steps_upper']!=128:
                raise ValueError('Missing native block begin')
        elif value['pending_native_environment_steps_upper']or value['completed_native_environment_steps']!=last['completed_native_environment_steps']+128:
            raise ValueError('Missing native completion')
        last=value;calls+=1
    world=measured['world']
    if calls!=1+2*world['global_control_ticks']or last['pending_native_environment_steps_upper']or last['completed_native_environment_steps']!=world['native_environment_integrations']:
        raise ValueError('Native clock/cost discontinuity')
    B.verify(p['inputs']);B.verify(p['sources'])
    for name,source in p['sources'].items():
        if B.sha(directory/'source_archive'/name)!=source['sha256']:raise ValueError('Archived source changed')
    R.assert_quiescent(directory);owners={}
    paths=[root/'PIPELINE_STARTED.json',directory/'RUN_STARTED.json',directory/'OWNER_SCOPE_FINISHED.json']
    paths+=list(root.glob('timing/*/PROCESS.json'))
    paths+=list(directory.glob('planning/*/worker_*/PROCESS.json'))
    paths+=list(directory.glob('wave_audits/*/PROCESS.json'))
    for path in paths:
        value=B.read(path);identity=value.get('process',value.get('identity'))
        if identity is None:raise ValueError('Missing process identity '+str(path))
        if R.process_identity(identity['pid'])==identity or R.group_alive(identity['pgid']):raise ValueError('Registered group still alive '+str(path))
        owners[(identity['pid'],identity['start_ticks'])]=identity
    locks=list(directory.rglob('*.lock'))+[B.ROOT/'outputs/author_collection_gpu0.lock']
    for path in locks:
        with R.lease(path):pass
    reservations=[]
    for path in(directory/'resource_slots').glob('*/*.json'):
        if B.read(path).get('state')!='released':raise ValueError('Unreleased slot')
        reservations.append(str(path))
    registry=B.read(B.ROOT/'outputs/author_collection_gpu0.wave_active.json')
    if registry['state']!='released'or Path(registry['run_dir'])!=directory:raise ValueError('Hour registry not released')
    result=dict(status='actual_hour_counts_and_resources_verified',exact_window_counts=expected,event_count=len(events),
        native_environment_steps=world['native_environment_integrations'],unknown_native_steps=0,
        registered_owners=list(owners.values()),all_registered_groups_exited=True,
        available_locks=list(map(str,locks)),released_reservations=reservations,registry_released=True,
        additional_physics_steps=0,training_updates=0,implementation=B.binding(__file__),
        measured_results=B.binding(root/'ACTUAL_HOUR_RESULTS.json'))
    B.save(root/'FINAL_REVIEW.json',result,exclusive=True);print({'status':result['status'],'counts':expected},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    review(parser.parse_args().root)
