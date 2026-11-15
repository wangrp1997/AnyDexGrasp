"""Verify and combine the once-only prefix/full and pipelined numeric audits."""
from pathlib import Path
import json

import author_cpu_full_control as F
import author_scaling_audit as S

B,A=F.B,S.A
HERE=Path(__file__).resolve().parent
ROOT=F.G.BASE
full_root=ROOT/'cpu_full_control_20261115_v1'
prefix_path=B.ROOT/'outputs/boya_cpu_prefix_20261115_v1/INDEPENDENT_AUDIT.json'
audit_dir=B.ROOT/'outputs/boya_cpu_full_20261115_v1/audits'
bindings=S.Bindings()
bindings.add(prefix_path);prefix=A.read(prefix_path);bindings.mapping(prefix['inputs_sha256'])
bindings.add(full_root/F.RESULTS);result=A.read(full_root/F.RESULTS);bindings.tree(result)
p=F.verify(full_root);bindings.tree(p)
parallel_path=audit_dir/'PARALLEL_AUDIT_RESULTS.json';bindings.add(parallel_path)
parallel=A.read(parallel_path);bindings.tree(parallel)
A.need(prefix['status']=='prefix_integrity_pass' and result['status']=='complete' and parallel['status']=='pass',
       'All terminal controls and once-only audits must pass')
samples=[json.loads(line) for line in Path(result['resource_samples']['path']).read_text().splitlines()]
groups=[];audits=[]
for config,group in zip(p['groups'],result['groups']):
    workers=[];wrappers=[]
    for expected,observed in zip(config['workers'],group['workers']):
        path=audit_dir/(group['name']+'_'+expected['worker_id']+'.json')
        bindings.add(path);report=A.read(path);bindings.mapping(report['input_sha256']);bindings.tree(report['start'])
        A.need(report['status']=='full_integrity_pass' and report['start']['worker_receipt']==observed['receipt'],
               'Audit belongs to another original worker or failed')
        workers.append(report['worker']);wrappers.append(report);audits.append(report)
    intervals=S.interval_stats([(w['prefix_started_unix_s'],w['prefix_finished_unix_s']) for w in workers])
    n=group['count'];wall=group['wall_s']
    audit_wall=wrappers[0]['wall_s'] if n==1 else parallel['wall_s']
    groups.append(dict(name=group['name'],count=n,full_execution_intervals=intervals,
        full_execution_span_s=intervals['span_s'],outer_group_wall_s=wall,
        execution_diagnostic_trials_per_hour=3600*n/intervals['span_s'],
        startup_inclusive_diagnostic_trials_per_hour=3600*n/wall,
        first_audit_phase_wall_s=audit_wall,first_audit_worker_wall_sum_s=sum(r['wall_s'] for r in wrappers),
        accounted_execution_outer_plus_audit_phase_s=wall+audit_wall,
        diagnostic_trials_per_accounted_hour=3600*n/(wall+audit_wall),
        actual_group_start_to_last_certificate_s=max(r['finished_unix_s'] for r in wrappers)-group['started_unix_s'],
        diagnostic_labels=[w['diagnostic_label'] for w in workers],
        contact_points=sum(w['original_all_body_numeric']['contact_points'] for w in workers),
        resources=S.resource_summary(samples,group,workers,p['gpu_uuid']),
        first_audit_certificates=[dict(path=str((audit_dir/(group['name']+'_'+w['worker_id']+'.json')).resolve()),
            sha256=A.sha(audit_dir/(group['name']+'_'+w['worker_id']+'.json'))) for w in config['workers']]))
serial,batch=groups
base=audits[0];windows=[]
for report in audits[1:]:
    w=report['worker']
    lo=max(base['start']['started_unix_s'],w['prefix_started_unix_s'])
    hi=min(base['finished_unix_s'],w['prefix_finished_unix_s'])
    if hi>lo:windows.append((lo,hi))
overlap=0 if not windows else S.interval_stats(windows)['union_s']
audit_windows=[(w['started_unix_s'],w['finished_unix_s']) for w in parallel['workers']]
audit_intervals=S.interval_stats(audit_windows)
A.need(audit_intervals['peak_concurrency']<=2,'Independent audit exceeded two-child budget')
comparisons=dict(full_execution_throughput_gain=batch['execution_diagnostic_trials_per_hour']/serial['execution_diagnostic_trials_per_hour'],
    startup_inclusive_gain=batch['startup_inclusive_diagnostic_trials_per_hour']/serial['startup_inclusive_diagnostic_trials_per_hour'],
    accounted_outer_plus_audit_gain=batch['diagnostic_trials_per_accounted_hour']/serial['diagnostic_trials_per_accounted_hour'])
bindings.verify()
summary=dict(status='all_prefix_and_full_integrity_pass',prefix_groups=[
    dict(name=g['name'],count=g['count'],metrics=g['metrics'],resources=g['resources']) for g in prefix['groups']],
    prefix_selection=prefix['selection'],full_groups=groups,full_comparisons=comparisons,
    pipeline=dict(baseline_audit_overlaps_parallel_execution_s=overlap,
        audit_child_intervals=audit_intervals,parallel_audit_phase_wall_s=parallel['wall_s'],
        parallel_audit_worker_wall_sum_s=sum(r['wall_s'] for r in audits[1:]),
        source='Actual once-only audit subprocesses; original numeric audit retained'),
    actual_Kit_launches=22,prefix_physical_steps=1300,full_physical_steps=27216,
    total_environment_steps=28516,total_dynamic_object_steps=85548,
    full_diagnostic_trials=9,full_diagnostic_positive=9,
    total_contact_points=prefix['contact_points']+sum(g['contact_points'] for g in groups),
    label=None,exclude_training=True,formal_training_eligible=False,new_training_labels=0,training_updates=0,
    audit_Kit_launches=0,audit_physics_steps=0,input_sha256=bindings.files,
    limitations=['Known g6 template repeated, not new training examples or model success rates',
        'Accounted phases sum measured execution/first-audit costs; development and handoff gaps are not erased from the separately reported certificate latency',
        'Only baseline audit overlapped physics; remaining eight first audits used at most two children',
        'No new capture/preparation in this benchmark; no 4000+800 collection ETA',
        'Existing resident collector is still pinned to at most three workers and serial audit; this is an independent performance/control branch'])
B.save(B.ROOT/'outputs/boya_cpu_full_20261115_v1/SUMMARY_WITH_INPUTS.json',summary,exclusive=True)
print(json.dumps({k:v for k,v in summary.items() if k in ('status','full_comparisons','pipeline','total_contact_points')},indent=2))
