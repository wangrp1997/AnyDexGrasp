"""Once-only baseline/Fabric-off GPU prefix pair using existing lifecycle controls."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import author_scaling_control as C
import author_fabric_execution as E

B, R, G = C.B, C.R, C.G
SCHEMA = 'anydex_boya_author_fabric_control_v1'
PROTOCOL = 'FABRIC_CONTROL_PROTOCOL.json'
RESULTS = 'FABRIC_CONTROL_RESULTS.json'
ARMS = ('baseline', 'both_disabled')


def freeze(output_dir):
    old = B.read(C.REFERENCE)
    prepared = Path(old['jobs'][0]['prepared_dirs'][0]).resolve()
    checked = E.V.load_jobs([prepared])[0]
    C.need(checked['command_steps'] == 3024, 'Original fixed g6 template required')
    sources = copy.deepcopy(old['sources'])
    sources.update({path: dict(path=path, sha256=sha) for path, sha in E.V.source_bindings().items()})
    for name in ('author_fabric_execution.py', 'author_fabric_control.py', 'author_fabric_audit.py',
                 'author_scaling_execution.py', 'author_scaling_control.py', 'author_scaling_audit.py',
                 'author_gpu_vector_profile.py'):
        sources[name] = B.binding(B.HERE/name)
    # Bind concrete tensor reads and the installed compatibility predicate wrapper.
    for path in (E.V.CANON/'canondex/scene/physx_compat.py',):
        sources[str(path)] = B.binding(path)
    for pattern in ('isaacsim/extscache/omni.physx.asset_validator-*/omni/physxassetvalidator/scripts/cookingApproximationChecker.py',):
        paths = sorted(E.V.SITE.glob(pattern))
        C.need(len(paths) == 1, 'Exactly one installed compatibility implementation required')
        sources[str(paths[0])] = B.binding(paths[0])
    inputs = copy.deepcopy(old['inputs'])
    inputs['reference_protocol'] = B.binding(C.REFERENCE)
    for key, item in checked['bindings'].items():
        inputs['g6_'+key] = item
    smi = shutil.which('nvidia-smi')
    C.need(smi is not None, 'Declared GPU0 query executable required')
    inputs['nvidia_smi'] = B.binding(smi)
    identity = C.gpu_identity(smi, old['gpu_uuid'])
    out = Path(output_dir).resolve()
    groups = [dict(name=name, mode='gpu', count=1, physical_ticks_per_worker=100,
        integrated_environment_steps=100, workers=[dict(worker_id='worker_00',
            prepared_dir=str(prepared), lanes=1,
            stage_dir=str(out/name/'worker_00'), output_dir=str(out/name/'worker_00/result'))])
        for name in ARMS]
    p = dict(schema=SCHEMA, task=B.TASK, diagnostic_only=True, exclude_training=True,
        formal_training_eligible=False, label=None, raw_label=None, automatic_training=False,
        groups=groups, prefix_ticks=100, retry_budget=0, stage_timeout_s=600.,
        cancel_sigint_grace_s=30., cancel_sigterm_grace_s=5., cancel_sigkill_grace_s=5.,
        sampling_interval_s=2., gpu_uuid=old['gpu_uuid'], gpu_identity=identity,
        env=dict(old['env']), inherited_env=old['inherited_env'], inputs=inputs, sources=sources,
        worker_source=B.binding(B.HERE/'author_fabric_execution.py'),
        prepared_template=dict(path=str(prepared), ready=B.binding(prepared/'READY.json'),
            commands=checked['bindings']['commands']),
        total_worker_attempt_budget=2, total_shared_ticks=200,
        total_integrated_environment_steps=200, total_dynamic_object_steps=600,
        groups_are_sequential=True, created_unix_s=time.time(),
        settings_policy=dict(baseline='Read original loaded values without changing settings',
            both_disabled={key:False for key in E.KEYS}, changes_only_after_app_before_scene=True,
            preserve_fabric_interop=True, preserve_physics_contacts_commands_and_dt=True),
        compatibility_scope='Per-prim official predicate after timing; never full CheckStage/reset/cache release',
        reference_urls=[
            'https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html',
            'https://github.com/isaac-sim/IsaacSim/issues/740'],
        limitations=['One A/B pair, fixed order and known template; no independent replication',
            'Settle/HOME prefixes cannot certify full lift, labels or collection ETA',
            'Publication output changes are allowed only without task cameras/rendering/Fabric consumers',
            'Compatibility predicates do not prove where each contact kernel executed'])
    B.verify(inputs); B.verify(sources)
    out.mkdir(parents=True, exist_ok=False)
    p['source_archive'] = C.archive_sources(out, sources)
    B.save(out/PROTOCOL, p, exclusive=True)
    return verify(out)


def verify(root):
    root = Path(root).resolve()
    p = B.read(root/PROTOCOL)
    C.need(p['schema'] == SCHEMA and p['prefix_ticks'] == 100 and p['retry_budget'] == 0
        and p['stage_timeout_s'] == 600. and p['total_worker_attempt_budget'] == 2
        and p['total_integrated_environment_steps'] == 200 and p['exclude_training'] is True,
        'Frozen two-arm 100-tick diagnostic only')
    C.need([g['name'] for g in p['groups']] == list(ARMS), 'Frozen baseline then disabled order required')
    for group in p['groups']:
        C.need(group['mode'] == 'gpu' and group['count'] == 1 and len(group['workers']) == 1,
            'Only one GPU lane per arm')
        worker = group['workers'][0]
        directory = root/group['name']/'worker_00'
        C.need(worker == dict(worker_id='worker_00', prepared_dir=p['prepared_template']['path'],
            lanes=1, stage_dir=str(directory), output_dir=str(directory/'result')),
            'Output or source escaped frozen pair')
    B.verify(p['inputs']); B.verify(p['sources']); B.verify({'archive':p['source_archive']})
    archive = B.read(p['source_archive']['path'])
    C.need({(r['source']['path'],r['source']['sha256']) for r in archive['sources']} ==
        {(r['path'],r['sha256']) for r in p['sources'].values()}, 'Full source archive required')
    for row in archive['sources']:
        B.verify({'source':row['source'], 'archive':row['archive']})
    C.need(p['env']['CUDA_VISIBLE_DEVICES'] == p['gpu_uuid'] and p['env']['NVIDIA_VISIBLE_DEVICES'] == '0',
           'Declared physical GPU0 only')
    return p


def admit_fabric(receipt, worker, group, protocol, *, original):
    result = original(receipt, worker, group, protocol)
    path = Path(worker['output_dir'])/'FABRIC_EXECUTION.json'
    state = B.read(path)
    C.need(state['schema'] == E.SCHEMA and state['status'] == 'partial_fabric'
        and state['mode'] == group['name'] and state['label'] is None and state['raw_label'] is None
        and state['exclude_training'] is True and state['formal_training_eligible'] is False,
        'Original-prefix Fabric receipt required')
    B.verify({'scaling':state['scaling_receipt'], 'protocol':state['protocol']})
    C.need(state['scaling_receipt'] == B.binding(Path(worker['output_dir'])/C.WORKER_RECEIPT),
           'Fabric sidecar must bind this exact measured worker')
    rows = state['settings_after_ticks']
    C.need([r['tick'] for r in rows] == list(range(1,101)), 'Every measured tick needs setting evidence')
    for row in rows:
        E.check_settings(row['values'], state['mode'], state['before_prefix']['settings'])
    C.need(state['before_prefix']['physics_clock'] == 0 and state['after_prefix']['physics_clock'] == 100
        and state['before_prefix']['cameras'] == [] and state['before_prefix']['visualizer_cfg_count'] == 0,
        'No additional integration or task rendering consumers permitted')
    return result


def run(root):
    root = Path(root).resolve(); p = verify(root)
    C.need(not (root/'RUN_STARTED.json').exists(), 'Started diagnostic cannot resume or retry')
    C.need(C.gpu_identity(p['inputs']['nvidia_smi']['path'],p['gpu_uuid']) == p['gpu_identity'],
           'Frozen GPU0 identity changed')
    began = time.monotonic(); groups = []; error = None
    with R.lease(root/'DRIVER.lock'), R.lease(G.LOCK) as fd:
        with R.active_run(root,G.REGISTRY,B.binding(root/PROTOCOL)):
            B.save(root/'RUN_STARTED.json',dict(protocol=B.binding(root/PROTOCOL),
                process=R.process_identity(os.getpid()),unix_s=time.time()),exclusive=True)
            pool = R.ResourcePool(root/'resource_slots',fd,B.binding(root/PROTOCOL))
            sampler = C.ResourceSampler(root,p); sampler.start()
            original = C.admit
            C.admit = lambda *a,**kw:admit_fabric(*a,**kw,original=original)
            try:
                runtime = dict(p,_protocol_path=str(root/PROTOCOL))
                for group in p['groups']:
                    B.verify(p['inputs']); B.verify(p['sources'])
                    result = C.run_group(root,group,runtime,pool,sampler)
                    groups.append(result)
                    if result['status'] != 'partial_scaling_complete':
                        break
            except BaseException:
                error = traceback.format_exc()
            finally:
                C.admit = original
                sampler.close()
            result = dict(schema=SCHEMA,status='complete' if len(groups)==2 and error is None
                and all(g['status']=='partial_scaling_complete' for g in groups) else 'stopped_after_failure',
                protocol=B.binding(root/PROTOCOL),groups=groups,error=error,wall_s=time.monotonic()-began,
                resource_samples=B.binding(sampler.path),label=None,raw_label=None,exclude_training=True,
                formal_training_eligible=False,new_training_labels=0,training_updates=0,retries=0,
                full_lift_certified=False,full_collection_ETA=None)
            B.verify(p['inputs']); B.verify(p['sources'])
            B.save(root/RESULTS,result,exclusive=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=freeze(args.output_dir) if args.action=='freeze' else run(args.output_dir)
    print(json.dumps(dict(status=result.get('status','frozen'),
        protocol=str(args.output_dir/PROTOCOL),groups=len(result['groups'])),indent=2))
    if args.action=='run' and result['status']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
