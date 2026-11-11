"""Opt-in bounded arm-branch and transit-path recovery for frozen Boya grasps.

The original candidate, hand commands, object pose, collision gates and local
IK tolerances are retained. This is a finite arm planning control, not cuRobo,
new grasp synthesis or a proof of reachability. No physical scene is launched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANON = Path('/home/wangrenpeng/CanonDex')
PYTHON = Path('/home/wangrenpeng/boya-inhand-wm/.venv/bin/python')
POLICY = dict(
    schema='anydex_boya_bounded_arm_recovery_policy_v1',
    baseline_first=True, extra_halton_seeds=16, halton_scramble=False,
    halton_skip=1, seed_range_fraction=[.05, .95], per_seed_max_nfev=300,
    max_new_ik_branches=4,
    branch_order='original maximum unwrapped joint displacement, L2, seed index',
    alternate_home_routes=['cable_home', 'qsfp_home', 'xr_test_home'],
    alternate_routes_only_for_home_collisions=True,
    grasp_pose_change=False, object_pose_change=False,
    hand_command_change=False, collision_tolerance_change=False,
    approach_steps=500, close_steps=250, hold_steps=50, lift_steps=500,
    dt_s=.004, curobo_used=False, physics_steps=0,
    scope='Finite local IK branches and named joint-space transit waypoints; sampled source-hull gates retained',
)
KNOWN_REJECTIONS = ('Six-arm IK did not converge', 'Six-arm source bridge exhausted',
                    'Bounded trajectory time scaling', 'Interpolated arm path fails',
                    'Direct path rejected')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extra_seeds(names, lower, upper):
    """Fixed low-discrepancy seeds, independent of object, pose and outcome."""
    from scipy.stats import qmc
    names = list(names)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if len(names) != 6 or lower.shape != (6,) or upper.shape != (6,):
        raise ValueError('Exactly six arm coordinates required')
    if not np.isfinite([lower, upper]).all() or np.any(upper <= lower):
        raise ValueError('Finite ordered arm bounds required')
    unit = qmc.Halton(d=6, scramble=False).random(POLICY['extra_halton_seeds'] + 1)[1:]
    values = lower + (upper-lower) * (.05 + .9*unit)
    return [dict(zip(names, map(float, row))) for row in values]


def different_branches(solutions, baseline_state, names):
    result = []
    for row in solutions:
        q = np.array([row['state'][n] for n in names])
        if baseline_state is not None:
            original = np.array([baseline_state[n] for n in names])
            if np.max(abs(q-original)) < 1e-5:
                continue
        if any(np.max(abs(q-np.array([old['state'][n] for n in names]))) < 1e-5 for old in result):
            continue
        result.append(row)
        if len(result) == POLICY['max_new_ik_branches']:
            break
    return result


def only_home_collisions(plan_dir):
    """A transit change is useful only if the rest of the path passed."""
    plan_dir = Path(plan_dir)
    if not (plan_dir/'plan.json').exists():
        return False
    plan = read(plan_dir/'plan.json')
    path = read(plan_dir/'path_geometry.json')
    lift = read(plan_dir/'lift_arm_geometry.json')
    home = next(p for p in plan['phases'] if p['name'] == 'home_to_pregrasp')
    violations = path['violations']
    return (bool(violations) and lift['ok'] and all(
        50 <= v['sample'] < home['end_step'] for v in violations))


def backend_modules():
    os.environ.setdefault('CANONDEX_COLLISION_PARTITION_CACHE', str(ROOT/'outputs/boya_collision_partition_cache'))
    sys.path[:0] = [str(CANON/'scripts/research'), str(CANON)]
    import run_template_arm_positive as backend
    from canondex.grasp import template_arm_admission as admission
    from canondex.grasp import template_arm_bridge as bridge
    return backend, admission, bridge


def native_prepare(argv):
    """Drop-in native prepare command; original executor can consume its plan."""
    backend, admission, bridge = backend_modules()
    args = backend.parse_args(argv)
    if args.stage != 'prepare':
        raise ValueError('Recovery entry does not launch physics')
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    args.process_start_monotonic_s = time.monotonic()
    backend.bind_hand(args)
    save(root/'RECOVERY_PROTOCOL.json', dict(policy=POLICY,
        source_sha256=sha(__file__), candidate_generation_sha256=sha(args.candidates/'generation.json'),
        candidate_npz_sha256=sha(args.candidates/'candidates.npz'),
        world_object_pose_sha256=sha(args.world_object_pose), physics_steps=0))
    original_rank, original_segments = admission.rank_arm_ik_solutions, bridge.arm_segments
    state = dict(selected=None, route=None, expanded=None, context=None, original=None)
    began = time.monotonic()
    attempts = []

    def ranked(model, reference, initial, target, seeds, **kwargs):
        if state['selected'] is None:
            result = original_rank(model, reference, initial, target, seeds, **kwargs)
            state['context'] = (model, reference, initial, target.copy(), copy.deepcopy(seeds))
            state['original'] = copy.deepcopy(result)
            return result
        result = copy.deepcopy(state['expanded'])
        result['solutions'] = [state['selected']]
        return result

    def segments(model, reference, initial, pregrasp, palm, direction, **kwargs):
        result, audit = original_segments(model, reference, initial, pregrasp, palm, direction, **kwargs)
        for phase in ('approach', 'lift'):
            if phase in result and len(result[phase][0])-1 != POLICY[phase+'_steps']:
                raise RuntimeError('Bounded trajectory time scaling exceeds frozen two-second '+phase)
        if state['route']:
            from canondex.robots.reference import load_reference
            names = bridge.ARM_JOINTS
            indexes = [model.active_names.index(n) for n in names]
            middle = load_reference(args.reference_root, pose_name=state['route']).arm_home
            vector = lambda s: [s[n] for n in names]
            result['home_to_pregrasp'] = bridge.timed_segment(
                [vector(initial), vector(middle), vector(pregrasp)], dt=.004,
                velocity_limits=model.velocity[indexes], minimum_s=3.,
                position_lower=model.lower[indexes], position_upper=model.upper[indexes])
        return result, audit

    admission.rank_arm_ik_solutions, bridge.arm_segments = ranked, segments

    def attempt(label, branch=None, route=None):
        state.update(selected=branch, route=route)
        local = copy.copy(args)
        local.output_dir = root/label
        local.output_dir.mkdir()
        start = time.monotonic()
        try:
            backend.prepare(local)
        except RuntimeError as error:
            if not str(error).startswith(KNOWN_REJECTIONS):
                raise
            failure = dict(complete=True, kind='arm_plan_rejected', reason=str(error), physical_executed=False)
            save(local.output_dir/'admission_failure.json', failure)
            success = False
        else:
            success = True
        row = dict(label=label, admitted=success, plan_dir=str(local.output_dir),
                   route=route, seed_index=None if branch is None else branch['seed_index'],
                   wall_s=time.monotonic()-start)
        if not success:
            row['reason'] = read(local.output_dir/'admission_failure.json')['reason']
        attempts.append(row)
        save(root/'RECOVERY_PROGRESS.json', dict(policy=POLICY, attempts=attempts, physics_steps=0))
        return row

    selected = None
    try:
        baseline = attempt('baseline')
        if baseline['admitted']:
            selected = baseline
        else:
            model, reference, initial, target, seeds = state['context']
            names = bridge.ARM_JOINTS
            original = state['original']
            save(root/'ORIGINAL_IK_SEARCH.json', original)
            indexes = [model.active_names.index(n) for n in names]
            added = extra_seeds(names, model.lower[indexes], model.upper[indexes])
            expanded = original_rank(model, reference, initial, target, seeds+added,
                                     max_nfev=POLICY['per_seed_max_nfev'])
            state['expanded'] = expanded
            save(root/'EXPANDED_IK_SEARCH.json', expanded)
            original_best = original['solutions'][0] if original['solutions'] else None
            branches = different_branches(expanded['solutions'],
                None if original_best is None else original_best['state'], names)
            # Original branch may be viable with a different transit path.
            if original_best is not None and only_home_collisions(root/'baseline'):
                for route in POLICY['alternate_home_routes']:
                    row = attempt('baseline_via_'+route, original_best, route)
                    if row['admitted']:
                        selected = row
                        break
            if selected is None:
                for number, branch in enumerate(branches):
                    direct = attempt(f'branch_{number:02d}_direct', branch)
                    if direct['admitted']:
                        selected = direct
                        break
                    if only_home_collisions(direct['plan_dir']):
                        for route in POLICY['alternate_home_routes']:
                            row = attempt(f'branch_{number:02d}_via_'+route, branch, route)
                            if row['admitted']:
                                selected = row
                                break
                    if selected:
                        break
        summary = dict(complete=True, policy=POLICY, attempts=attempts, selected=selected,
                       baseline_admitted=baseline['admitted'], recovered=bool(selected and not baseline['admitted']),
                       physics_steps=0, wall_s=time.monotonic()-began)
        save(root/'RECOVERY_RESULTS.json', summary)
        if selected:
            for name in ('frozen_input.json', 'commands.npz', 'plan.json', 'path_geometry.json', 'lift_arm_geometry.json'):
                shutil.copyfile(Path(selected['plan_dir'])/name, root/name)
            plan = read(root/'plan.json')
            plan['planning_recovery'] = dict(policy=POLICY, results_json=str(root/'RECOVERY_RESULTS.json'),
                results_sha256=sha(root/'RECOVERY_RESULTS.json'), source_sha256=sha(__file__),
                selected_attempt=selected['label'], baseline_admitted=baseline['admitted'])
            save(root/'plan.json', plan)
        else:
            save(root/'admission_failure.json', dict(complete=True, kind='arm_plan_rejected',
                reason='Bounded arm recovery exhausted declared branches and transit paths', physical_executed=False))
            return 2
        return 0
    finally:
        admission.rank_arm_ik_solutions, bridge.arm_segments = original_rank, original_segments


def bind_worker(job_path, output, config):
    """Reuse original source/label binding in an isolated preparation process."""
    import prepare_collection_queue as original
    real_subprocess = original.subprocess

    def run(command, **kwargs):
        command = list(command)
        expected = str(Path(original.__file__).with_name('native_arm.py'))
        if len(command) < 3 or command[2] != expected:
            raise ValueError('Unexpected child command in recovery binding')
        command[2:3] = [str(Path(__file__).resolve()), 'native', '--']
        kwargs['timeout'] = 1200
        return real_subprocess.run(command, **kwargs)

    original.subprocess = SimpleNamespace(run=run, STDOUT=real_subprocess.STDOUT)
    try:
        result = original.prepare_one(read(job_path), Path(output), Path(config))
        print(json.dumps(dict(status=result['status'], trial_id=result['trial_id'])), flush=True)
    finally:
        original.subprocess = real_subprocess


def prepare_one(job, output, config_path):
    """Same signature/result as prepare_collection_queue.prepare_one.

    Safe for concurrent caller threads: all backend overrides live in a child.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    job_path = output/(job['trial_id']+'.recovery_job.json')
    if job_path.exists():
        raise ValueError('Existing job is never overwritten or re-executed')
    save(job_path, job)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
               CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
    with (output/(job['trial_id']+'.recovery.log')).open('xb') as log:
        proc = subprocess.run([str(PYTHON), '-B', str(Path(__file__).resolve()), 'bind',
            '--job', str(job_path), '--output', str(output), '--config', str(config_path)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1300)
    receipt = output/job['trial_id']/'preparation.json'
    if proc.returncode:
        raise RuntimeError(f'Recovery binding failed, see {log.name}')
    return read(receipt)


def compare(prior, output, workers=2):
    """Evaluate every original declared candidate for the eight arm-null objects."""
    prior, output = Path(prior).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('New comparison output required')
    queue, jobs = read(prior/'queue_manifest.json'), read(prior/'jobs.json')
    objects = {r['object_id'] for r in queue['objects'] if not r['selected'] and r['arm_preparations']}
    if len(objects) != 8:
        raise ValueError('Expected the original eight arm-null objects')
    selected_jobs = [r['job'] for obj in jobs['objects'] if obj['object_id'] in objects for r in obj['candidates']]
    output.mkdir(parents=True)
    sources = [Path(__file__).resolve(), ROOT/'reproduction/boya/prepare_collection_queue.py',
               CANON/'scripts/research/run_template_arm_positive.py', CANON/'canondex/grasp/template_arm_admission.py',
               CANON/'canondex/grasp/template_arm_bridge.py', CANON/'canondex/grasp/admm/arm_only_ik.py']
    freeze = dict(schema='anydex_boya_arm_recovery_comparison_v1', policy=POLICY,
        original_planning=str(prior), object_ids=sorted(objects), candidate_count=len(selected_jobs),
        jobs=selected_jobs, workers=workers, physics_steps=0, training_writeback=False,
        input_sha256={str(p): sha(p) for p in (prior/'queue_manifest.json', prior/'jobs.json', Path(jobs['backend_config']))},
        source_sha256={str(p):sha(p) for p in sources})
    save(output/'PROTOCOL.json', freeze)
    plans = output/'plans'
    plans.mkdir()
    began = time.monotonic()
    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(prepare_one, job, plans, jobs['backend_config']):job for job in selected_jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                recovery = read(plans/job['trial_id']/'plan_000/RECOVERY_RESULTS.json')
                results[job['trial_id']] = dict(object_id=job['object_id'], trial_id=job['trial_id'],
                    status=result['status'], plan_dir=result.get('plan_dir'), recovery=recovery,
                    preparation_json=str(plans/job['trial_id']/'preparation.json'))
            except Exception as exc:
                errors.append(dict(trial_id=job['trial_id'], error=repr(exc)))
            save(output/'PROGRESS.json', dict(completed=len(results), total=len(selected_jobs), errors=errors))
    changed = [p for section in ('input_sha256','source_sha256') for p,h in freeze[section].items() if sha(p)!=h]
    report = dict(complete=not errors and not changed, protocol_sha256=sha(output/'PROTOCOL.json'),
        results=[results[j['trial_id']] for j in selected_jobs if j['trial_id'] in results], errors=errors,
        changed_inputs=changed, physics_steps=0, wall_s=time.monotonic()-began)
    report['ready_candidates'] = sum(r['status']=='ready' for r in report['results'])
    report['ready_objects'] = sorted({r['object_id'] for r in report['results'] if r['status']=='ready'})
    save(output/'RESULTS.json', report)
    print(json.dumps({k:report[k] for k in ('complete','ready_candidates','ready_objects','physics_steps','wall_s')}))
    return report


def main():
    if len(sys.argv)>1 and sys.argv[1]=='native':
        argv=sys.argv[2:]
        if argv and argv[0]=='--':argv=argv[1:]
        raise SystemExit(native_prepare(argv))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    bind = sub.add_parser('bind')
    for name in ('job','output','config'):bind.add_argument('--'+name, type=Path, required=True)
    comparison = sub.add_parser('compare')
    comparison.add_argument('--prior',type=Path,required=True)
    comparison.add_argument('--output',type=Path,required=True)
    comparison.add_argument('--workers',type=int,choices=(1,2,3,4),default=2)
    args = parser.parse_args()
    if args.mode=='bind':bind_worker(args.job,args.output,args.config)
    else:compare(args.prior,args.output,args.workers)


if __name__=='__main__':main()
