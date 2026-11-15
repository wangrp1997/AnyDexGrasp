"""Independent read-only audit and paired comparison of Fabric prefix outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import author_scaling_audit as S

A = S.A
SCHEMA = 'anydex_boya_author_fabric_audit_v1'
KEYS = ('/physics/fabricUpdateTransformations', '/physics/fabricUpdateVelocities')
OTHER = ('/physics/fabricUseGPUInterop', '/physics/suppressReadback', '/physics/disableContactProcessing')


def check_sidecar(value, arm, worker_path, bindings):
    A.need(value['schema'] == 'anydex_boya_author_fabric_execution_v1'
        and value['status'] == 'partial_fabric' and value['mode'] == arm
        and value['label'] is None and value['raw_label'] is None
        and value['formal_training_eligible'] is False and value['exclude_training'] is True
        and value['full_lift_certified'] is False and value['prefix_ticks'] == 100,
        'Only unlabeled complete prefixes may be compared')
    bindings.tree(value)
    A.need(Path(value['scaling_receipt']['path']).resolve() == Path(worker_path).resolve(),
           'Fabric output bound to another worker')
    app = value['after_app_before_scene']
    A.need(all(app['before'][k] == app['after'][k] for k in OTHER), 'Unrelated App settings changed')
    A.need(value['only_changed_settings'] == (list(KEYS) if arm == 'both_disabled' else []),
           'Declared intervention differs')
    if arm == 'baseline':
        A.need(app['before'] == app['after'], 'Baseline changed default settings')
    else:
        A.need(all(app['after'][k] is False for k in KEYS), 'Both switches must be off after App creation')
    before, after = value['before_prefix'], value['after_prefix']
    A.need(before['physics_clock'] == 0 and after['physics_clock'] == 100
        and before['cameras'] == [] and before['visualizer_cfg_count'] == 0
        and before['use_fabric'] is True and before['physics_device'] == 'cuda:0',
        'Original GPU tensor scene without rendering consumers required')
    rows = value['settings_after_ticks']
    A.need([r['tick'] for r in rows] == list(range(1,101)), 'Missing per-tick settings')
    for settings in [before['settings'], *(r['values'] for r in rows), after['settings']]:
        A.need(all(type(settings[k]) is bool for k in KEYS)
            and settings['/physics/disableContactProcessing'] is False,
            'Unknown publication state or disabled full contacts')
        A.need(all(settings[k] == before['settings'][k] for k in OTHER), 'Unrelated setting drifted')
        A.need(all(settings[k] == (False if arm == 'both_disabled' else before['settings'][k]) for k in KEYS),
               'Actual Fabric publication arm drifted')
    compatibility = value['convex_compatibility']
    if compatibility['status'] == 'queried':
        rows = compatibility['rows']
        A.need(rows and len({r['path'] for r in rows}) == len(rows)
            and len(rows) == compatibility['checked_convex_prims']
            and compatibility['gpu_compatible'] == sum(r['gpu_compatible'] for r in rows)
            and compatibility['gpu_incompatible'] == sum(not r['gpu_compatible'] for r in rows)
            and compatibility['physics_clock_before'] == compatibility['physics_clock_after'] == 100
            and compatibility['physics_steps_added'] == 0
            and compatibility['root_layer_sha256_before'] == compatibility['root_layer_sha256_after']
            and compatibility['measured_prefix_includes_this_query'] is False,
            'Compatibility query changed stage/clock or its accounting')
    return dict(arm=arm, loaded_settings=before['settings'],
        checked_ticks=100, compatibility=compatibility, intervention_valid=True)


def compare_arrays(first, second):
    A.need(set(first) == set(second), 'Paired trace fields differ')
    result = {}
    for key in first:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        A.need(a.shape == b.shape and a.dtype == b.dtype, 'Paired trace shape/dtype differs: '+key)
        same = np.array_equal(a,b)
        value = dict(exact=same, shape=list(a.shape), dtype=str(a.dtype))
        if a.dtype.kind in 'fiu':
            A.need(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite paired trace')
            value['max_abs_difference'] = float(np.max(abs(a.astype(float)-b.astype(float)))) if a.size else 0.
        result[key] = value
    return dict(all_arrays_exact=all(v['exact'] for v in result.values()), fields=result)


def compare_artifacts(directories, bindings):
    executions = [Path(p)/'env_0/execution' for p in directories]
    arrays = []
    for directory in executions:
        path = directory/'trace.npz'; bindings.add(path)
        with np.load(path,allow_pickle=False) as z:
            arrays.append({k:z[k].copy() for k in z.files})
    native = compare_arrays(*arrays)
    traces = [A.read(p/'MULTI_OBJECT_TRACE.json')['rows'] for p in executions]
    A.need(len(traces[0]) == len(traces[1]) == 100, 'Paired all-body length differs')
    first_difference = next((i+1 for i,(a,b) in enumerate(zip(*traces)) if a != b),None)
    streams = [A.iter_contact_rows(p/'MULTI_OBJECT_CONTACTS.json',bindings) for p in executions]
    contact_rows = 0; first_contact_difference = None
    while True:
        rows = [next(s,None) for s in streams]
        if rows == [None,None]: break
        A.need(all(r is not None for r in rows), 'Paired contact lengths differ')
        contact_rows += 1
        if rows[0] != rows[1] and first_contact_difference is None:
            first_contact_difference = contact_rows
    A.need(contact_rows == 100, 'Exactly 100 complete contact rows per arm required')
    setup = [A.read(p/'MULTI_OBJECT_SETUP.json') for p in executions]
    return dict(native_trace=native, all_body_trace_exact=first_difference is None,
        all_body_first_differing_tick=first_difference,
        full_contact_rows_exact=first_contact_difference is None,
        full_contacts_first_differing_tick=first_contact_difference, contact_rows_compared=contact_rows,
        all_three_body_entry_exact=setup[0]['entry'] == setup[1]['entry'],
        scope='Observed prefix equality only; no full trajectory equivalence or population claim')


def audit(results_path):
    started=time.monotonic(); bindings=S.Bindings(); mesh_cache={}
    results_path=Path(results_path).resolve(); bindings.add(results_path)
    result=A.read(results_path); bindings.tree(result)
    p=A.read(result['protocol']['path']); bindings.tree(p)
    A.need(result['schema'] == p['schema'] == 'anydex_boya_author_fabric_control_v1'
        and result['status']=='complete' and result['retries']==0
        and result['new_training_labels']==result['training_updates']==0
        and result['label'] is None and result['raw_label'] is None
        and result['exclude_training'] is True and result['formal_training_eligible'] is False,
        'Complete two-arm diagnostic results required')
    A.need(p['prefix_ticks']==100 and p['retry_budget']==0 and p['total_worker_attempt_budget']==2
        and p['total_integrated_environment_steps']==200 and len(result['groups'])==2
        and [g['name'] for g in p['groups']]==['baseline','both_disabled'], 'Frozen pair budget changed')
    archive=A.read(p['source_archive']['path']); bindings.tree(archive)
    A.need({(r['source']['path'],r['source']['sha256']) for r in archive['sources']} ==
        {(r['path'],r['sha256']) for r in p['sources'].values()}, 'Source archive incomplete')
    groups=[]; directories=[]; template=None
    for config,group in zip(p['groups'],result['groups']):
        A.need(group['name']==config['name'] and group['status']=='partial_scaling_complete'
            and len(group['workers'])==1 and group['mode']=='gpu' and group['count']==1,
            'One completed GPU worker per frozen arm required')
        observed=group['workers'][0]; stage=A.read(observed['stage']['path'])
        A.need(stage['returncode']==0 and stage['complete'] is True and stage['retry_count']==0
            and stage['physical_steps_unknown'] is False, 'Failed or unknown-cost stage')
        worker=S.audit_worker(observed['receipt']['path'],bindings,mesh_cache)
        A.need(worker['mode']=='gpu' and worker['lanes']==1 and stage['process']['pid']==worker['worker_pid'],
               'Worker physical device/count/identity differs')
        if template is None: template=worker['template']
        A.need(worker['template']==template, 'Pair uses different source plans')
        directory=Path(observed['receipt']['path']).parent; directories.append(directory)
        sidecar_path=directory/'FABRIC_EXECUTION.json'; bindings.add(sidecar_path)
        sidecar=check_sidecar(A.read(sidecar_path),group['name'],observed['receipt']['path'],bindings)
        A.need(abs(group['finished_unix_s']-group['started_unix_s']-group['wall_s'])<.05,
               'Group wall does not include actual startup/exit')
        timing=worker['timing']['execution_prefix']
        metrics=dict(prefix_wall_s=worker['prefix_wall_s'],outer_group_wall_s=group['wall_s'],
            fetch_results_wall_s=timing['physx.fetch_results']['inclusive_wall_s'],
            sim_step_wall_s=timing['sim.step']['inclusive_wall_s'],
            full_filtered_contact_readback_wall_s=timing['contact.filtered6_readback']['inclusive_wall_s'],
            prefix_envsteps_per_s=100/worker['prefix_wall_s'])
        groups.append(dict(name=group['name'],worker=worker,fabric=sidecar,metrics=metrics))
    A.need(result['groups'][0]['finished_unix_s'] <= result['groups'][1]['started_unix_s'],
           'A/B arms unexpectedly overlapped')
    before,after=[g['metrics'] for g in groups]
    rates={key:before[key]/after[key] for key in ('prefix_wall_s','outer_group_wall_s','fetch_results_wall_s')}
    paired=compare_artifacts(directories,bindings)
    bindings.verify()
    return dict(schema=SCHEMA,status='prefix_integrity_pass',groups=groups,baseline_over_disabled_wall= rates,
        paired_outputs=paired,physical_clock_steps=200,environment_steps=200,dynamic_object_steps=600,
        trace_rows=200,contact_rows=200,contact_points=sum(g['worker']['lanes_audit'][0]['contact_points'] for g in groups),
        label=None,raw_label=None,exclude_training=True,formal_training_eligible=False,
        full_gates_certified=False,full_lift_certified=False,full_collection_ETA=None,
        audit_Kit_launches=0,audit_physics_steps=0,audit_wall_s=time.monotonic()-started,
        inputs_sha256=bindings.files,
        limitations=['One fixed-order pair without independent replication',
            'Inclusive timers overlap; no added CUDA synchronization or kernel attribution',
            'Compatibility status does not identify the actual kernel path',
            '100-tick prefix cannot certify complete grasp or new collection ETA'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    A.need(not args.output.exists(),'Preserve the first audit result')
    try: result=audit(args.control_results)
    except Exception: result=dict(schema=SCHEMA,status='failed',error=traceback.format_exc(),audit_physics_steps=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','baseline_over_disabled_wall','audit_wall_s','error')},indent=2))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
