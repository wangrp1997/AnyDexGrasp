"""Bounded original-augmentation comparison on three known failures plus A0.

Stages are explicit and never resume/retry physical attempts automatically.
Freeze all sixteen first-geometry choices before consulting historical outcomes.
Reuse only exact physical arrays under the original controlled-wrist settings.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np

from run_boya_fixed_pool import read, save, sha, check_hashes
from audit_boya_fixed_pool import POSE_KEYS, CONFIG_KEYS, ROW_KEYS, EXECUTOR, EXECUTOR_SHA

ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'reproduction/boya'
BASE = ROOT/'outputs/egad_external12_20261107_v1'
OBJECTS = ('egad_A3', 'egad_A6', 'egad_E0', 'egad_A0')
RECIPES = ('original_mse', 'preactivation')
INPUTS = ('identity', 'augmented11')


def baseline(oid):
    if oid == 'egad_A0':
        return ROOT/'outputs/boya_contact_proxy_alignment_20261108_v2/jobs/egad_A0_coarse.json'
    return ROOT/'outputs/egad_floating_three_20261108_v1/jobs'/(oid+'.json')


def same_pose(left, right):
    with np.load(Path(left)/'candidates.npz', allow_pickle=False) as a, np.load(Path(right)/'candidates.npz', allow_pickle=False) as b:
        return all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k])
                   for k in (*POSE_KEYS, 'independent_motor_names'))


def eligible(job, reference):
    return (all(job[k] == reference[k] for k in CONFIG_KEYS)
            and all(job['object_row'][k] == reference['object_row'][k] for k in ROW_KEYS)
            and job.get('candidate_index', 0) == 0
            and not job.get('closure_control') and not job.get('collision_refinement')
            and job.get('variant') != 'refined'
            and read(job['protocol_path'])['source_sha256'].get(str(EXECUTOR)) == EXECUTOR_SHA
            and sha(job['protocol_path']) == job['protocol_sha256'])


def prepare(out):
    if out.exists():
        raise ValueError('Fresh cohort directory required')
    out.mkdir(); frozen = {}
    # Preserve the old dataset/model/source locks as well as the new entry points.
    parents = [ROOT/'outputs/boya_shared_score_20261111_v2/PROTOCOL.json',
               ROOT/'outputs/boya_augmentation_deployment_20261114_v1/PROTOCOL.json',
               ROOT/'outputs/boya_reference_alignment_20261114_v2/PROTOCOL.json']
    for path in parents:
        doc = read(path); frozen[str(path)] = sha(path)
        for section in ('input_sha256', 'source_sha256', 'frozen_files'):
            frozen.update(doc.get(section, {}))
    old = read(ROOT/'outputs/egad_floating_three_20261108_v1/protocol.json')
    frozen.update(old['source_sha256']); frozen.update(old['input_sha256'])
    for name in ('run_augmented_cohort.py', 'compare_augmented_cohort.py', 'audit_augmented_cohort.py',
                 'run_augmented_representation.py', 'augmented_candidates.py', 'publish_boya_fixed_pool.py',
                 'audit_boya_fixed_pool.py', 'audit_boya_closure_refinement.py',
                 'audit_augmented_cohort_physics.py', 'test_augmented_cohort.py'):
        path = B/name; frozen[str(path)] = sha(path)
    for oid in OBJECTS:
        job_path = baseline(oid); job = read(job_path)
        frozen[str(job_path)] = sha(job_path)
        for section in ('source_sha256', 'input_sha256'):
            frozen.update(read(job['protocol_path']).get(section, {}))
    protocol = dict(schema='boya_augmented_cohort_predeclared_v1', object_order=list(OBJECTS),
        primary_objects=list(OBJECTS[:3]), separate_positive_control='egad_A0',
        recipe_order=list(RECIPES), input_order=list(INPUTS),
        primary_comparison='original_mse identity versus augmented11',
        secondary_comparison='Frozen preactivation identity versus augmented11',
        candidate_rule='First full-geometry admitted candidate in frozen score order; no arm or fallback',
        physical_reuse='Earliest exact physical-array and unchanged controlled-wrist configuration match; never by outcome',
        unique_attempts_maximum=16, max_steps_each=2100, max_new_physics_steps=33600,
        closure_mapping_threshold_and_budgets_unchanged=True, new_training_updates=0,
        new_RGBD=0, new_CGR_forwards_maximum=33, new_CGR_loads_maximum=3,
        no_full_arm=True, independent_generalization=False, training_writeback=False,
        outcome_dependent_retry=False, default_promotion=False, sources_and_inputs=frozen,
        stop='Complete the frozen queue regardless of success. Infrastructure errors halt new waves. No parameter search.')
    check_hashes((frozen,)); save(out/'STUDY_PROTOCOL.json', protocol)
    for oid in OBJECTS[:3]:
        rep = copy.deepcopy(read(parents[2])); rep.pop('recovery_from', None)
        rep.update(observation_json=str(BASE/'observations'/oid/'observation.json'),
            known_object=oid+' already used diagnostic; no unseen claim',
            representation_dir=str(out/oid/'representation'), identity_reference_dir=str(BASE/'representations'/oid),
            geometry_output=None, allowed=['Exactly eleven original CGR branches for this frozen observation'],
            not_authorized=['training', 'physics by this representation subprotocol', 'mapping/threshold changes'],
            parent_study=dict(path=str(out/'STUDY_PROTOCOL.json'), sha256=sha(out/'STUDY_PROTOCOL.json')))
        rep['frozen_files'] = dict(frozen)
        obs_path = Path(rep['observation_json']); obs = read(obs_path)
        rep['frozen_files'][str(obs_path)] = sha(obs_path)
        for key in ('rgb_path', 'depth_path'):
            if key in obs:
                path = Path(obs[key]); rep['frozen_files'][str(path)] = sha(path)
        for path in (BASE/'representations'/oid).glob('*.npz'):
            rep['frozen_files'][str(path)] = sha(path)
        save(out/oid/'REPRESENTATION_PROTOCOL.json', rep)
    save(out/'PROGRESS.json', dict(status='prepared', new_physics_steps=0))


def compare_protocol(out, oid):
    protocol = copy.deepcopy(read(ROOT/'outputs/boya_augmentation_deployment_20261114_v1/PROTOCOL.json'))
    protocol.update(schema='boya_augmented_cohort_offline_v1', object_id=oid,
        scope='Previously used object, label-blind fixed comparison',
        observation_json=str(BASE/'observations'/oid/'observation.json'),
        representation_dirs=dict(identity=str(BASE/'representations'/oid), augmented11=str(out/oid/'representation')),
        output_dir=str(out/oid/'comparison'), parent_protocol=dict(path=str(out/'STUDY_PROTOCOL.json'), sha256=sha(out/'STUDY_PROTOCOL.json')))
    protocol['frozen_files'] = dict(read(out/'STUDY_PROTOCOL.json')['sources_and_inputs'])
    for path in (out/oid/'representation').glob('*'):
        if path.is_file(): protocol['frozen_files'][str(path)] = sha(path)
    protocol['frozen_files'][protocol['observation_json']] = sha(protocol['observation_json'])
    save(out/oid/'COMPARISON_PROTOCOL.json', protocol)
    return out/oid/'COMPARISON_PROTOCOL.json'


def freeze_physical(out):
    study = read(out/'STUDY_PROTOCOL.json'); check_hashes((study['sources_and_inputs'],))
    if (out/'protocol.json').exists(): raise ValueError('Choices already frozen')
    choices = []; frozen = dict(study['sources_and_inputs'])
    for oid in OBJECTS:
        comparison = (ROOT/'outputs/boya_augmentation_deployment_20261114_v1/comparison'
                      if oid == 'egad_A0' else out/oid/'comparison')
        report = read(comparison/'RESULTS.json'); assert report['status'] == 'complete'
        frozen[str(comparison/'RESULTS.json')] = sha(comparison/'RESULTS.json')
        frozen.update(report['input_sha256'])
        for input_kind in INPUTS:
            for recipe in RECIPES:
                directory = comparison/f'{input_kind}__{recipe}'
                geometry = read(directory/'geometry.json')
                frozen[str(directory/'geometry.json')] = sha(directory/'geometry.json')
                first = geometry['admitted'][0] if geometry['admitted'] else None
                cid = f'{oid}_{input_kind}_{recipe}'
                choice = dict(condition_id=cid, source_object_id=oid, input_kind=input_kind, recipe=recipe,
                    candidate_dir=first['candidate_dir'] if first else None,
                    candidate=first['candidate'] if first else None,
                    rank=first['global_score_rank'] if first else None)
                choices.append(choice)
                if first:
                    for name in ('generation.json', 'candidates.npz', 'world_object_pose.json'):
                        p = Path(first['candidate_dir'])/name; frozen[str(p)] = sha(p)
    # This file is immutable before ANY historical execution/outcome is opened.
    save(out/'ALL_CHOICES_FREEZE.json', dict(choices=choices, input_sha256=frozen,
        study_sha256=sha(out/'STUDY_PROTOCOL.json'), outcomes_consulted_for_selection=False))
    history_paths = set()
    old_results = read(ROOT/'outputs/boya_fixed_pool_20261112_v1/RESULTS.json')
    history_paths.update(r['job_path'] for r in old_results['results'])
    catalog = read(ROOT/'outputs/boya_fixed_pool_20261112_v1/CATALOG.json')
    for c in catalog['candidates']:
        history_paths.update(h['job_path'] for h in c['eligible_history'])
    history_paths.add(str(baseline('egad_A0')))
    histories = []
    for path in sorted(history_paths, key=lambda p:(Path(p).stat().st_mtime_ns, p)):
        job = read(path); histories.append((path, job))
    cases = []; pending = {}; aliases = []; reused = []; nulls = []
    for choice in choices:
        cid, oid = choice['condition_id'], choice['source_object_id']
        if not choice['candidate_dir']:
            nulls.append(cid); aliases.append(dict(**choice, candidate_id=None, kind='no_geometry')); continue
        duplicate = next((c for c in cases if c['source_object_id'] == oid and same_pose(c['candidate_dir'], choice['candidate_dir'])), None)
        if duplicate:
            aliases.append(dict(**choice, candidate_id=duplicate['object_id'], kind='same_physical_pose')); continue
        reference = read(baseline(oid))
        matches = [(p,j) for p,j in histories if eligible(j, reference) and same_pose(j['candidate_dir'], choice['candidate_dir'])]
        case = dict(object_id=cid, source_object_id=oid, candidate_dir=choice['candidate_dir'], old_execution=None,
                    kind='reused' if matches else 'new', matching_history=[p for p,j in matches])
        if matches:
            path, job = matches[0]; case['job_path'] = path; reused.append(cid)
        else:
            job = copy.deepcopy(reference)
            for key in ('variant', 'collision_refinement', 'closure_control', 'source_job', 'source_pool', 'delta_world_m', 'static_report'):
                job.pop(key, None)
            generation = read(Path(choice['candidate_dir'])/'generation.json')
            job.update(object_id=cid, source_object_id=oid, candidate_dir=choice['candidate_dir'], candidate_index=0,
                candidate_record=generation['records'][0], candidate_rank=choice['rank'], old_execution=None,
                output_dir=str(out/'execution'/cid), selection_reason='Frozen first geometry admitted; one attempt')
            case['job_path'] = str(out/'jobs'/(cid+'.json')); pending[cid] = job
        cases.append(case); aliases.append(dict(**choice, candidate_id=cid, kind=case['kind']))
    sources = {p:h for p,h in frozen.items() if p.endswith('.py')}
    for case in cases:
        if case['kind'] == 'reused':
            job = read(case['job_path']); frozen[case['job_path']] = sha(case['job_path'])
            for name in ('evaluation.json', 'commands.npz', 'trace.npz', 'initialization_audit.json', 'native_audit.json'):
                p = Path(job['output_dir'])/name; frozen[str(p)] = sha(p)
    protocol = dict(schema='boya_augmented_cohort_physical_v1', cases=cases, conditions=aliases, nulls=nulls,
        new_queue=list(pending), source_sha256=sources, input_sha256=frozen,
        choices_freeze_sha256=sha(out/'ALL_CHOICES_FREEZE.json'), study_sha256=sha(out/'STUDY_PROTOCOL.json'),
        new_attempts=len(pending), reused=len(reused), unique=len(cases), concurrency=4,
        maximum_new_physics_steps=len(pending)*2100, physical_attempts_per_unique_pose=1,
        new_training_updates=0, new_CGR_forwards=0, outcome_dependent_retry=False,
        stop_on_first_success=False, default_promotion=False, training_writeback=False)
    check_hashes((frozen,)); save(out/'protocol.json', protocol)
    jobs = {}
    for cid, job in pending.items():
        job.update(protocol_path=str(out/'protocol.json'), protocol_sha256=sha(out/'protocol.json'))
        p = out/'jobs'/(cid+'.json'); save(p, job); jobs[str(p)] = sha(p)
    save(out/'PREPARED.json', dict(jobs_sha256=jobs, protocol_sha256=sha(out/'protocol.json'),
        new_attempts=len(pending), unique=len(cases), reused=len(reused), no_geometry=len(nulls)))
    print('COHORT_PREPARED', len(cases), 'unique', len(pending), 'new', len(reused), 'reused', flush=True)


def execute(out):
    p = read(out/'protocol.json'); ready = read(out/'PREPARED.json')
    check_hashes((p['input_sha256'], ready['jobs_sha256']))
    assert ready['protocol_sha256'] == sha(out/'protocol.json')
    if (out/'DISPATCH.json').exists(): raise ValueError('Already dispatched; no implicit physical repeat')
    save(out/'DISPATCH.json', dict(queue=p['new_queue'], protocol_sha256=sha(out/'protocol.json'), pid=os.getpid()))
    began = time.monotonic(); rows = {}; errors = []
    by_id = {c['object_id']:c for c in p['cases']}
    def result(cid):
        c = by_id[cid]; j = read(c['job_path']); ep = Path(j['output_dir'])/'evaluation.json'; e = read(ep)
        if not e['complete']: raise ValueError('Incomplete '+cid)
        return dict(candidate_id=cid, object_id=cid, source_object_id=c['source_object_id'], kind=c['kind'],
            job_path=c['job_path'], evaluation=str(ep), evaluation_sha256=sha(ep), task_success=e['task_success'],
            physical_steps=e['total_physics_steps'], gates=e['gates'], failure_reason=e['failure_reason'])
    for c in p['cases']:
        if c['kind'] == 'reused': rows[c['object_id']] = result(c['object_id'])
    def one(slot, cid):
        argv = ['taskset', '-c', f'{32+8*slot}-{39+8*slot}', sys.executable, '-B', str(EXECUTOR), '--job', by_id[cid]['job_path']]
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
        log = out/'logs'/(cid+'.log'); log.parent.mkdir(exist_ok=True)
        start = time.monotonic()
        with log.open('xb') as stream:
            rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=600).returncode
        save(out/'attempts'/(cid+'.json'), dict(argv=argv, returncode=rc, wall_s=time.monotonic()-start, log=str(log)))
        if rc: raise RuntimeError(f'{cid}: infrastructure exit {rc}')
        return result(cid)
    for first in range(0, len(p['new_queue']), 4):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(one,i,cid):cid for i,cid in enumerate(p['new_queue'][first:first+4])}
            for future in as_completed(futures):
                cid = futures[future]
                try: rows[cid] = future.result()
                except Exception as exc: errors.append(dict(candidate_id=cid, error=repr(exc)))
                save(out/'PROGRESS.json', dict(status='executing', results=list(rows.values()), errors=errors))
                print('COHORT_PHYSICS', cid, rows.get(cid), flush=True)
        if errors: break
    results = [rows[c['object_id']] for c in p['cases'] if c['object_id'] in rows]
    report = dict(schema='boya_augmented_cohort_results_v1', status='complete' if not errors and len(rows)==len(p['cases']) else 'infrastructure_error',
        protocol_sha256=sha(out/'protocol.json'), results=results, errors=errors,
        new_physics_steps=sum(r['physical_steps'] for r in results if r['kind']=='new'),
        reused_physics_steps=sum(r['physical_steps'] for r in results if r['kind']=='reused'),
        new_attempts=sum(r['kind']=='new' for r in results), reused=sum(r['kind']=='reused' for r in results),
        conditions=p['conditions'], wall_s=time.monotonic()-began, new_training_updates=0, training_writeback=False)
    check_hashes((p['input_sha256'], ready['jobs_sha256']))
    save(out/'RESULTS.json', report); save(out/'PROGRESS.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare','compare_protocol','freeze_physical','execute'), required=True)
    parser.add_argument('--object-id', choices=OBJECTS[:3])
    args = parser.parse_args(); out = args.output.resolve()
    if args.stage == 'compare_protocol': compare_protocol(out, args.object_id)
    else: globals()[args.stage](out)


if __name__ == '__main__': main()
