"""Read-only CPU 1/4/8 prefix audit, throughput comparison and fixed selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import author_scaling_audit as S

A=S.A
SCHEMA='anydex_boya_author_cpu_prefix_audit_v1'


def select(groups,policy):
    baseline=groups[0]['metrics']
    comparisons=[]
    for group in groups:
        m=group['metrics'];prefix=m['prefix_envsteps_per_s']/baseline['prefix_envsteps_per_s']
        outer=m['startup_inclusive_envsteps_per_s']/baseline['startup_inclusive_envsteps_per_s']
        eligible=group['count']>1 and min(prefix,outer)>=policy['minimum_both_rate_gain']
        eligible=eligible and m['prefix_intervals']['peak_concurrency']>=2
        comparisons.append(dict(count=group['count'],prefix_gain=prefix,startup_inclusive_gain=outer,
            score=min(prefix,outer),eligible=eligible))
    candidates=[r for r in comparisons if r['eligible']]
    best=None
    if candidates:
        peak=max(r['score'] for r in candidates)
        best=min(r['count'] for r in candidates if r['score']>=peak*(1-policy['tie_relative_band']))
    return dict(selected_parallel_count=best,comparisons=comparisons,
        full_followup_allowed=best is not None,
        reason='Frozen throughput rule; no labels/success outcomes used',
        full_followup_is_formal_collection=False)


def audit(path):
    began=time.monotonic();bindings=S.Bindings();cache={}
    path=Path(path).resolve();bindings.add(path);result=A.read(path);bindings.tree(result)
    p=A.read(result['protocol']['path']);bindings.tree(p)
    A.need(p['schema']==result['schema']=='anydex_boya_author_cpu_prefix_control_v1'
        and result['status']=='complete' and p['prefix_ticks']==100 and p['retry_budget']==0
        and p['total_worker_attempt_budget']==13 and p['total_integrated_environment_steps']==1300
        and result['label'] is None and result['raw_label'] is None
        and result['exclude_training'] is True and result['formal_training_eligible'] is False
        and result['new_training_labels']==result['training_updates']==result['retries']==0,
        'Complete frozen CPU diagnostic required')
    A.need(len(p['groups'])==len(result['groups'])==3,'All three CPU sizes required')
    archive=A.read(p['source_archive']['path']);bindings.tree(archive)
    with Path(result['resource_samples']['path']).open() as stream:samples=[json.loads(line) for line in stream]
    groups=[];template=None;previous=None
    for config,group,count in zip(p['groups'],result['groups'],(1,4,8)):
        A.need(config['name']==group['name']==f'cpu_{count}' and config['count']==group['count']==count
            and config['mode']==group['mode']=='cpu' and group['status']=='partial_scaling_complete'
            and len(config['workers'])==len(group['workers'])==count,'Frozen CPU group differs')
        if previous is not None:A.need(previous<=group['started_unix_s'],'CPU groups overlap')
        previous=group['finished_unix_s'];workers=[]
        for expected,actual in zip(config['workers'],group['workers']):
            stage=A.read(actual['stage']['path'])
            A.need(expected['worker_id']==actual['worker_id'] and stage['returncode']==0
                and stage['complete'] is True and stage['physical_steps_unknown'] is False,
                'Missing, failed or unknown CPU worker')
            worker=S.audit_worker(actual['receipt']['path'],bindings,cache)
            A.need(worker['mode']=='cpu' and worker['lanes']==1 and worker['worker_pid']==stage['process']['pid'],
                'Actual CPU tensor/backend/PID differs')
            if template is None:template=worker['template']
            A.need(template==worker['template'],'A source template differs')
            workers.append(worker)
        metrics=S.comparison_metrics('cpu',count,workers,group['wall_s'])
        A.need(abs(group['finished_unix_s']-group['started_unix_s']-group['wall_s'])<.05,
            'Group wall excludes startup or actual process exit')
        groups.append(dict(name=group['name'],count=count,metrics=metrics,workers=workers,
            resources=S.resource_summary(samples,group,workers,p['gpu_uuid'])))
    decision=select(groups,p['full_followup_selection']);bindings.verify()
    return dict(schema=SCHEMA,status='prefix_integrity_pass',groups=groups,selection=decision,
        physical_clock_steps=1300,environment_steps=1300,dynamic_object_steps=3900,
        trace_rows=1300,contact_rows=1300,
        contact_points=sum(w['lanes_audit'][0]['contact_points'] for g in groups for w in g['workers']),
        label=None,raw_label=None,exclude_training=True,formal_training_eligible=False,
        full_gates_certified=False,full_lift_certified=False,full_collection_ETA=None,
        inputs_sha256=bindings.files,audit_wall_s=time.monotonic()-began,
        audit_physics_steps=0,audit_Kit_launches=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();A.need(not args.output.exists(),'First audit must be preserved')
    try:result=audit(args.control_results)
    except Exception:result=dict(schema=SCHEMA,status='failed',error=traceback.format_exc(),audit_physics_steps=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','selection','audit_wall_s','error')},indent=2))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
