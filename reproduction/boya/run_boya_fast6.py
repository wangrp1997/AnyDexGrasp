"""Freeze and run six new Boya validation objects with bounded arm recovery.

Stage completion is persisted. A failed stage cannot be silently retried;
initialization, forward passes and physical attempts are never replayed here.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'reproduction/boya'
OLD = ROOT / 'outputs/egad_external12_20261107_v1'
AUDIT = ROOT / 'outputs/boya_fast_audit_20261111_v1'
IDS = tuple('egad_' + r + str(c) for r in 'ACE' for c in (1, 4))
BACKEND = ROOT / 'outputs/boya_pilot8_20261105_v1/backend_pilot.json'
CALIBRATION = ROOT / 'outputs/boya_adapter_preparation_20261105_v2/calibration.json'
DOMAIN = B / 'evidence/s40_hand_domain_20261107/POLICY.json'
NEW_SOURCES = ('prepare_boya_fast6_assets.py', 'prepare_boya_fast6_runtime.py',
               'render_boya_fast6_observations.py', 'evaluate_boya_fast6.py',
               'prepare_boya_fast6_top1.py', 'run_boya_fast6_trials.py',
               'run_boya_fast6.py', 'planning_recovery.py')
STAGES = ('initialize', 'observe', 'represent', 'score', 'plan', 'execute')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, doc):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def verify(doc):
    for key in ('source_sha256', 'input_sha256'):
        for p, digest in doc[key].items():
            if sha(p) != digest:
                raise ValueError('Frozen dependency changed: ' + p)


def validate_development(planning_results, candidate_results):
    from planning_recovery import POLICY
    planning_results, candidate_results = Path(planning_results), Path(candidate_results)
    plan, cases = read(planning_results), read(candidate_results)
    if (plan.get('complete') is not True or plan.get('errors') or plan.get('changed_inputs')
            or plan.get('physics_steps') != 0 or len(plan.get('results', [])) != 29):
        raise ValueError('Complete zero-physics 29-candidate planning diagnosis required')
    pp = planning_results.parent / 'PROTOCOL.json'
    preregistered = read(pp)
    if plan.get('protocol_sha256') != sha(pp) or preregistered.get('policy') != POLICY:
        raise ValueError('Planning results do not bind the deployed policy')
    cp = candidate_results.parent / 'protocol.json'
    frozen = read(cp)
    if (cases.get('status') != 'complete' or cases.get('errors')
            or cases.get('actual_grasps') != 9 or len(cases.get('results', [])) != 9
            or cases.get('training_writeback') is not False
            or cases.get('protocol_sha256') != sha(cp)
            or frozen.get('physics_attempt_budget') != 9
            or frozen.get('planned_physics_attempts') != 9
            or not 0 < cases.get('physical_steps', 0) <= 9*2100):
        raise ValueError('Completed bounded nine-candidate diagnosis required; infrastructure errors cannot qualify')
    for doc in (preregistered, frozen):
        verify(doc)


def freeze(root, planning_results, candidate_results):
    from planning_recovery import POLICY
    from prepare_boya_fast6_runtime import validate_fast6_manifest
    pp = root / 'protocol.json'
    if pp.exists():
        raise FileExistsError('Protocol already frozen')
    old = read(OLD / 'protocol.json')
    manifest = read(root / 'scenes/objects.json')
    selection = read(AUDIT / 'EXTERNAL6_SELECTION_AUDIT.json')
    if [r['object_id'] for r in selection['objects']] != list(IDS):
        raise ValueError('Selection differs from independent pre-score audit')
    validate_development(planning_results, candidate_results)
    # Explicit existing dependency closure, not a glob of concurrently developed files.
    sources = dict(read(OLD / 'EVALUATION_FREEZE.json')['source_sha256'])
    sources.update(read(OLD / 'settling/PLAN.json')['source_sha256'])
    for p in NEW_SOURCES:
        path = B / p
        sources[str(path)] = sha(path)
    inputs = {}
    baseline = read(AUDIT / 'IMMUTABLE_BASELINE.json')
    for p, digest in baseline['input_sha256'].items():
        if sha(p) != digest:
            raise ValueError('Original model, data or helper changed: ' + p)
        inputs[p] = digest
    # Old inputs are retained as provenance; none supplies new held-out labels.
    for path, digest in old['input_sha256'].items():
        if str(OLD) not in path and '/egad_eval_set/' not in path:
            inputs[path] = digest
    for p in (AUDIT / 'IMMUTABLE_BASELINE.json', AUDIT / 'EXTERNAL6_SELECTION_AUDIT.json',
              root / 'scenes/SELECTION.json', root / 'scenes/objects.json',
              planning_results, candidate_results):
        inputs[str(Path(p).resolve())] = sha(p)
    for r in selection['objects']:
        inputs[r['source_mesh']] = r['source_mesh_sha256']
    protocol = copy.deepcopy(old)
    for key in ('asset_preparation_amendment_ref', 'convex_face_normalization_ref',
                'coacd_policy', 'coacd_policy_note'):
        protocol.pop(key, None)
    protocol.update(schema='anydex_boya_fast6_protocol_v1', object_count=6,
                    objects=[dict(object_id=r['object_id'], source_mesh=r['source_mesh'],
                                  source_mesh_sha256=r['source_mesh_sha256']) for r in selection['objects']],
                    selection_rule=selection['rule'], source_sha256=sources, input_sha256=inputs,
                    created_unix_s=time.time(), planning_recovery_policy=POLICY,
                    sole_deployment_change='Bounded additional arm IK branches and collision-checked home transit waypoints',
                    head_mapping_closure_and_physics_unchanged=True,
                    development_evidence=dict(planning=str(Path(planning_results).resolve()),
                                              candidate_diagnostic=str(Path(candidate_results).resolve())),
                    validation_boundary='Six new-to-Boya shapes, one fixed placement policy and at most one physical attempt each; small-scale validation, not paper reproduction or broad generalization proof',
                    failure_policy='Keep all six in denominator; no replacement or outcome-guided retry; infrastructure errors prevent completed-success-rate certification',
                    plan_workers=3, execution_workers=3)
    protocol['active_asset_policy'] = dict(method='Uniform original V-HACD plus source-vertex convex envelope normalization',
          parameters=read(root / 'scenes/SELECTION.json')['vhacd_parameters'],
          source_scene_manifest=str(root / 'scenes/objects.json'),
          selection_ref=dict(path=str(root / 'scenes/SELECTION.json'), sha256=sha(root / 'scenes/SELECTION.json')),
          geometry_approximation_limit='Approximate convex proxies; source visual triangles unchanged. Native20um verifies import, not source-shape fidelity.')
    validate_fast6_manifest(manifest, protocol)
    verify(protocol)
    save(pp, protocol)
    save(root / 'EVALUATION_FREEZE.json', dict(protocol_sha256=sha(pp),
         source_sha256=sources, input_sha256=inputs, ranking_and_outcomes_not_yet_observed=True,
         new_training_updates=0, physical_attempt_limit=6))
    print('FAST6_FROZEN', sha(pp), flush=True)


def arguments(root, stage):
    protocol = read(root / 'protocol.json')
    common = ['--run-dir', root]
    if stage == 'initialize':
        return ['prepare_boya_fast6_runtime.py', '--scene-manifest', root/'scenes/objects.json', *common, '--workers', '3']
    if stage == 'observe':
        return ['render_boya_fast6_observations.py', '--objects-json', root/'objects.json', '--output-dir', root/'observations']
    if stage == 'represent':
        return ['run_collection_representation.py', '--manifest', root/'objects.json', '--run-dir', root/'representation_stage',
                '--device', 'cuda:0', '--checkpoint', protocol['representation']['checkpoint']['path'], '--seed', str(protocol['representation']['seed'])]
    if stage == 'score':
        return ['evaluate_boya_fast6.py', '--objects-manifest', root/'objects.json', '--protocol', root/'protocol.json',
                '--training-results', protocol['training_results']['path'], '--calibration', CALIBRATION,
                '--domain-policy', DOMAIN, '--output-dir', root/'diagnostic', '--geometry-workers', '3']
    if stage == 'plan':
        return ['prepare_boya_fast6_top1.py', '--protocol', root/'protocol.json', '--diagnostic', root/'diagnostic',
                '--backend-config', BACKEND, '--calibration', CALIBRATION, '--native-preflight', root/'native_preprocessing/SUMMARY.json',
                '--output-dir', root/'planning', '--workers', '3']
    if stage == 'execute':
        return ['run_boya_fast6_trials.py', '--queue', root/'planning/queue_manifest.json', '--diagnostic', root/'diagnostic',
                '--output', root/'execution', '--workers', '3', '--render']
    raise ValueError(stage)


def stage_run(root, stage):
    protocol = read(root / 'protocol.json')
    freeze_doc = read(root / 'EVALUATION_FREEZE.json')
    if sha(root / 'protocol.json') != freeze_doc['protocol_sha256']:
        raise ValueError('Protocol changed after freeze')
    verify(protocol)
    status_path = root / 'EVALUATION_PROGRESS.json'
    status = read(status_path) if status_path.exists() else dict(object_count=6, stages=[], status='prepared')
    completed = [x['stage'] for x in status['stages'] if x.get('returncode') == 0]
    if stage in completed:
        print('FAST6_ALREADY_COMPLETE', stage, flush=True)
        return
    if any(x['stage'] == stage for x in status['stages']) or (root / (stage+'.log')).exists():
        raise ValueError('Prior stage attempt exists; inspect evidence and create explicit recovery protocol, never replay automatically')
    if tuple(completed) != STAGES[:STAGES.index(stage)]:
        raise ValueError('Stage order or previous completion differs')
    command = arguments(root, stage)
    argv = [sys.executable, '-B', str(B / command[0]), *map(str, command[1:])]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1',
               CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    status.update(status='running', stage=stage)
    save(status_path, status)
    start = time.monotonic()
    with (root / (stage+'.log')).open('xb') as out:
        proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    rec = dict(stage=stage, argv=argv, returncode=proc.returncode, wall_s=time.monotonic()-start)
    status['stages'].append(rec)
    status['status'] = 'infrastructure_error' if proc.returncode else ('complete' if stage=='execute' else 'stage_complete')
    if stage == 'execute' and proc.returncode == 0:
        report = read(root / 'execution/RESULTS.json')
        if report.get('status') != 'complete' or report.get('media_errors'):
            status['status'] = 'physical_complete_media_incomplete'
        status['physical_results'] = dict(successes=report['successes'], objects=report['objects'],
            actual_attempts=report['actual_attempts'], physical_steps=report['physical_steps'],
            no_plan_objects=report['no_plan_objects'], results_sha256=sha(root/'execution/RESULTS.json'))
    save(status_path, status)
    verify(protocol)
    print(json.dumps(rec), flush=True)
    if proc.returncode:
        raise RuntimeError('Stage failed; inspect ' + str(root/(stage+'.log')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--stage', choices=('freeze', *STAGES), required=True)
    p.add_argument('--planning-results', type=Path)
    p.add_argument('--candidate-results', type=Path)
    a = p.parse_args()
    root = a.run_dir.resolve()
    if a.stage == 'freeze':
        if not a.planning_results or not a.candidate_results:
            p.error('freeze requires both development evidence paths')
        freeze(root, a.planning_results, a.candidate_results)
    else:
        stage_run(root, a.stage)
