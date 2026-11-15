"""Independent full-trajectory CPU audit using original all-point numeric audit."""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import time
import traceback

import author_scaling_audit as S

A,G=S.A,S.G
SCHEMA='anydex_boya_author_cpu_full_audit_v1'
GPU_AUDIT_SHA='7c508445f7905845b952b622d15a7767ad61eabcae038ed1f4b372796ae5b7ec'


def native_state_auditor():
    """Change only the original GPU-only admission to explicit CPU device.

    All native material, 28-DOF velocity, palm reconstruction and stability
    checks remain the original function body, in a private namespace.
    """
    A.need(A.sha(G.__file__)==GPU_AUDIT_SHA,'Original native-state auditor changed')
    tree=ast.parse(Path(G.__file__).read_text())
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_verify_lane_state'))
    edits=0
    for node in ast.walk(fn):
        if isinstance(node,ast.Compare) and ast.unparse(node)=="native.get('device') in ('cuda:0', 'cuda')":
            node.ops=[ast.Eq()];node.comparators=[ast.Constant(value='cpu')];edits+=1
    A.need(edits==1,'Exactly one device admission adaptation required')
    namespace=dict(vars(G))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(G.__file__),'exec'),namespace)
    return namespace['_verify_lane_state']


def audit_worker(path,bindings,state_audit):
    began=time.monotonic();path=Path(path).resolve();bindings.add(path);r=A.read(path);bindings.tree(r)
    A.need(r['schema']==S.WORKER_SCHEMA and r['status']=='diagnostic_full' and r['complete'] is True
        and r['full_trial'] is True and r['prefix_ticks']==0 and r['mode']==r['device']=='cpu'
        and r['lanes']==1 and r['label'] is None and r['exclude_training'] is True
        and r['formal_training_eligible'] is False,'Excluded complete CPU trial required')
    bindings.mapping(r['source_sha256']);group_path=Path(r['group_receipt']['path'])
    group=A.read(group_path);profile=A.read(r['profile']['path'])
    bindings.mapping(group['source_sha256']);bindings.mapping(profile['source_sha256'])
    A.need(group['status']=='complete' and group['complete'] is True and group['device']=='cpu'
        and group['label'] is None and group['exclude_training'] is True
        and profile['status']=='diagnostic_full' and profile['complete'] is True
        and profile['label'] is None and profile['raw_label'] is None
        and len(group['jobs'])==1,'Complete native CPU group/profile required')
    job=group['jobs'][0]
    ready,scene,plan,candidate,command,arrays=S._source_commands(job,bindings)
    n=len(arrays['phase'])
    A.need(n==3024 and r['expected_command_steps']==n and group['clock']['lane_steps']==[n]
        and group['clock']['commands_added']==0,'Complete frozen g6 commands required')
    for key in ('physical_clock_steps','returned_physical_steps','actual_simulation_step_counter',
                'physical_steps_lower_bound','physical_steps_upper_bound','actual_integrated_environment_steps',
                'recorded_command_environment_steps','integrated_environment_steps_lower_bound',
                'integrated_environment_steps_upper_bound'):
        A.need(r[key]==n,'Full counter differs: '+key)
    A.need(r['observations_completed_steps_per_lane']==[n] and r['physical_steps_unknown'] is False
        and r['inflight_step_unconfirmed'] is False and group['physical_clock_steps']==n
        and group['actual_integrated_environment_steps']==n,'Unknown/incomplete full integration')
    native=group['native_scene'];attrs=native['native_scene']
    A.need(native['actual_cpu_validated'] is True and native['actual_gpu_validated'] is False
        and attrs['physxScene:enableGPUDynamics'] is False and attrs['physxScene:broadphaseType']=='MBP'
        and attrs['physxScene:timeStepsPerSecond']==250 and attrs['physxScene:enableCCD'] is False,
        'Actual CPU physical backend differs')
    probes=S._snapshot_probes(group,bindings,'cpu',['/World/envs/env_0'])
    evaluation=A.read(job['evaluation']['path']);bindings.tree(job['evaluation']);bindings.tree(job['diagnostic_record'])
    execution=Path(job['evaluation']['path']).parent
    A.need(execution==group_path.parent/'env_0/execution' and evaluation['complete'] is True
        and evaluation['label_eligible'] is True and all(evaluation['gates'].get(k) is True for k in A.GATES)
        and evaluation['completed_phases']==A.PHASES and evaluation['total_physics_steps']==n
        and evaluation['original_object_ids']==[o['object_id'] for o in scene['objects']]
        and evaluation['commands_sha256']==ready['commands']['sha256']
        and evaluation['scene_sha256']==ready['scene']['sha256']
        and evaluation['collection_candidate_sha256']==candidate['candidate_sha256']
        and evaluation['state_writes_during_execution']==0 and not evaluation['error']
        and not evaluation['physical_stop'],'Full original gate/source/phase binding differs')
    record=A.read(job['diagnostic_record']['path'])
    A.need(record['label'] is None and record['formal_training_eligible'] is False
        and record['exclude_training'] is True and record['candidate']==candidate
        and record['command_binding']==command and record['evaluation_binding']==job['evaluation'],
        'Diagnostic record changed inputs or training exclusion')
    numeric=G._projected_runtime(scene,plan,arrays,execution,evaluation,bindings,
        '/World/envs/env_0',S.np.zeros(3))
    state=state_audit(execution,plan,arrays,S.np.zeros(3),bindings)
    raw=int(any(numeric['object_success'].values()))
    A.need(type(r['raw_label']) is int and r['raw_label']==job['raw_label']==record['diagnostic_raw_label']==raw
        and r['original_gates']==job['gates']==evaluation['gates'],'Diagnostic result differs from independent recomputation')
    duration=r['execution_finished_monotonic_s']-r['execution_started_monotonic_s']
    A.need(duration>0 and abs(duration-r['execution_wall_s'])<1e-9,'Full execution wall differs')
    timing=r['timing']['execution_full']
    for key in ('physx.simulate','physx.fetch_results','sim.step'):
        A.need(timing[key]['calls']==n,'Exactly one simulate/fetch per command required')
    return dict(path=str(path),worker_pid=r['worker_pid'],template=r['prepared_template'],
        physics_clock_steps=n,environment_steps=n,prefix_wall_s=duration,
        prefix_started_unix_s=r['execution_started_unix_s'],prefix_finished_unix_s=r['execution_finished_unix_s'],
        full_execution_wall_s=duration,diagnostic_label=raw,label=None,exclude_training=True,
        full_gates_certified=True,original_all_body_numeric=numeric,original_native_state=state,
        snapshots=probes,timing=r['timing'],independent_audit_wall_s=time.monotonic()-began)


def audit(path):
    began=time.monotonic();bindings=S.Bindings();state_audit=native_state_auditor()
    path=Path(path).resolve();bindings.add(path);result=A.read(path);bindings.tree(result)
    p=A.read(result['protocol']['path']);bindings.tree(p)
    A.need(p['schema']==result['schema']=='anydex_boya_author_cpu_full_control_v1'
        and result['status']=='complete' and p['retry_budget']==result['retries']==0
        and result['new_training_labels']==result['training_updates']==0
        and result['label'] is None and result['exclude_training'] is True,
        'Complete frozen full diagnostic comparison required')
    n=p['selected_count'];A.need(n in (4,8) and len(result['groups'])==2,'Frozen selected full size required')
    samples=[json.loads(line) for line in Path(result['resource_samples']['path']).read_text().splitlines()]
    groups=[];template=None
    for config,group,count in zip(p['groups'],result['groups'],(1,n)):
        A.need(group['name']==config['name']==f'cpu_{count}' and group['status']=='full_complete'
            and len(group['workers'])==count,'Full group count/status differs')
        workers=[]
        for expected,observed in zip(config['workers'],group['workers']):
            stage=A.read(observed['stage']['path'])
            A.need(stage['complete'] is True and stage['returncode']==0 and stage['physical_steps_unknown'] is False
                and expected['worker_id']==observed['worker_id'],'Failed/unknown full worker')
            worker=audit_worker(observed['receipt']['path'],bindings,state_audit)
            A.need(worker['worker_pid']==stage['process']['pid'],'Actual full PID differs')
            if template is None:template=worker['template']
            A.need(template==worker['template'],'One original full template required')
            workers.append(worker)
        intervals=S.interval_stats([(w['prefix_started_unix_s'],w['prefix_finished_unix_s']) for w in workers])
        execution_span=intervals['span_s'];audit_sum=sum(w['independent_audit_wall_s'] for w in workers)
        groups.append(dict(name=group['name'],count=count,workers=workers,intervals=intervals,
            full_execution_span_s=execution_span,outer_group_wall_s=group['wall_s'],
            complete_trials_per_execution_hour=3600*count/execution_span,
            complete_trials_per_outer_hour=3600*count/group['wall_s'],
            first_independent_audit_sum_s=audit_sum,
            accounted_outer_plus_serial_audit_s=group['wall_s']+audit_sum,
            diagnostic_trials_per_accounted_hour=3600*count/(group['wall_s']+audit_sum),
            resources=S.resource_summary(samples,group,workers,p['gpu_uuid'])))
    serial,parallel=groups
    comparisons=dict(execution_throughput_gain=parallel['complete_trials_per_execution_hour']/serial['complete_trials_per_execution_hour'],
        startup_inclusive_gain=parallel['complete_trials_per_outer_hour']/serial['complete_trials_per_outer_hour'],
        outer_plus_serial_audit_accounted_gain=parallel['diagnostic_trials_per_accounted_hour']/serial['diagnostic_trials_per_accounted_hour'])
    bindings.verify()
    return dict(schema=SCHEMA,status='full_integrity_pass',groups=groups,comparisons=comparisons,
        actual_trials=1+n,environment_steps=3024*(1+n),dynamic_object_steps=9072*(1+n),
        contact_points=sum(w['original_all_body_numeric']['contact_points'] for g in groups for w in g['workers']),
        diagnostic_positive=sum(w['diagnostic_label'] for g in groups for w in g['workers']),
        label=None,exclude_training=True,formal_training_eligible=False,new_training_labels=0,training_updates=0,
        inputs_sha256=bindings.files,audit_wall_s=time.monotonic()-began,audit_physics_steps=0,audit_Kit_launches=0,
        limitations=['Known same-template diagnostic trials, not random-scene/new-model success rates',
            'Accounted outer plus serial audit is a cost sum; audits ran after the execution groups, not inside their measured wall',
            'No new capture/preparation, no production yield or full collection ETA',
            'One fixed-order full serial/parallel pair; no independent replication'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--control-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    A.need(not args.output.exists(),'Preserve first full audit result')
    try:result=audit(args.control_results)
    except Exception:result=dict(schema=SCHEMA,status='failed',error=traceback.format_exc(),audit_physics_steps=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','comparisons','audit_wall_s','error')},indent=2))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
