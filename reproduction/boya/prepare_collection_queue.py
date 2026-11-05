"""Prepare and bind AnyDex-Boya trials without building a physical scene.

A job declares one frozen gesture/depth pool and an ordered finite plan budget.
The first arm-admitted plan is queued once, even if its later outcome is a
failure. Candidate ranking never consults executed outcomes. Inputs, rejected
plans and zero-step nulls are retained. Execution is a separate queue stage.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from asset_adapter import sha
from trial_record import prepare_binding, write_bound_commands

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare_one(job, output, config_path):
    started = time.monotonic()
    if (not isinstance(job['trial_id'], str) or not job['trial_id']
            or Path(job['trial_id']).name != job['trial_id'] or job['trial_id'] in ('.', '..')):
        raise ValueError('Trial ID must be one safe path segment')
    run = output / job['trial_id']
    run.mkdir(exist_ok=False)
    cfg = read(config_path)
    pool = Path(job['candidate_dir']).resolve()
    if not (pool / 'generation.json').exists() and (pool / 'no_candidate.json').exists():
        null = read(pool / 'no_candidate.json')
        observed = read(job['observation_json'])
        for key, path in [('source_candidates_sha256', job['source_npz']),
                          ('source_observation_sha256', job['observation_json']),
                          ('calibration_sha256', job['calibration'])]:
            if sha(path) != null[key]:
                raise ValueError('Null-pool source changed: ' + str(path))
        if observed.get('object_id') != job['object_id']:
            raise ValueError('Null-pool object identity mismatch')
        if job.get('gesture_id', null['gesture_id']) != null['gesture_id']:
            raise ValueError('Null-pool gesture mismatch')
        depths = {r['additional_depth_index'] for r in null.get('records', [])
                  if 'additional_depth_index' in r}
        if 'extra_depth_index' in job and depths and depths != {job['extra_depth_index']}:
            raise ValueError('Null-pool depth mismatch')
        result = dict(trial_id=job['trial_id'], object_id=job['object_id'],
                      status='no_hand_geometry_in_declared_budget', training_label=None,
                      actual_physics_steps=0, no_candidate_json=str(pool / 'no_candidate.json'),
                      no_candidate_sha256=sha(pool / 'no_candidate.json'))
        save(run / 'preparation.json', result)
        return result
    gen = read(pool / 'generation.json')
    observation = Path(job['observation_json']).resolve()
    obs = read(observation)
    source = Path(job['source_npz']).resolve()
    calibration = Path(job['calibration']).resolve()
    indices = job.get('candidate_indices', gen['selected_indices'])
    budget = job.get('max_arm_plans', 4)
    if (type(budget) is not int or budget < 1 or len(indices) != len(set(indices))
            or any(i not in gen['selected_indices'] for i in indices)):
        raise ValueError('Finite positive plan budget and unique frozen indices required')
    if observation != Path(gen['source_observation']).resolve():
        raise ValueError('Job observation is not the frozen pool observation')
    if source != Path(gen['source_frame']).resolve() / 'raw_camera_candidates.npz':
        raise ValueError('Job CGR features are not the frozen pool source')
    if job['object_id'] != obs.get('object_id') or Path(obs['object_root']).name != job['object_id']:
        raise ValueError('Job identity differs from the observed object')
    if job.get('gesture_id', gen['gesture_id']) != gen['gesture_id']:
        raise ValueError('Job gesture differs from its pool')
    if not gen.get('complete') or sha(pool / 'candidates.npz') != gen['candidates_sha256']:
        raise ValueError('Candidate pool incomplete or changed')
    if str(calibration) not in gen['input_sha256']:
        raise ValueError('Calibration is not the frozen pool calibration')
    depths = {r['additional_depth_index'] for r in gen['records']}
    if 'extra_depth_index' in job and depths != {job['extra_depth_index']}:
        raise ValueError('Declared extra-depth differs from pool metadata')
    inputs = dict(gen['input_sha256'])
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError('Frozen input changed: ' + path)
    inputs.update({str(p): sha(p) for p in (config_path, observation, source, calibration,
                                          pool / 'generation.json', pool / 'candidates.npz')})
    repair = Path(job['collision_refinement']).resolve() if job.get('collision_refinement') else None
    if repair:
        inputs[str(repair)] = sha(repair)
        if job.get('collision_refinement_sha256', sha(repair)) != sha(repair):
            raise ValueError('Declared object repair changed')
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError('Frozen input changed: ' + path)
    shutil.copyfile(cfg['arm_collision_refinement'], run / 'arm_collision_refinements.json')
    inputs[cfg['arm_collision_refinement']] = sha(cfg['arm_collision_refinement'])
    result = dict(trial_id=job['trial_id'], object_id=job['object_id'],
                  gesture_id=gen['gesture_id'], split=job.get('split', 'development'),
                  status='preparing', plans=[], actual_physics_steps=0, training_label=None,
                  input_sha256=inputs, candidate_dir=str(pool))
    save(run / 'preparation.json', result)
    env = os.environ.copy()
    env.update(CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT / 'outputs/boya_collision_partition_cache'),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1')
    for index in indices[:budget]:
        plan_dir = run / f'plan_{index:03d}'
        common = ['--robot-urdf', cfg['robot_urdf'], '--reference-root', cfg['reference_root'],
                  '--candidates', str(pool), '--index', str(index), '--object-root', obs['object_root'],
                  '--world-object-pose', str(pool / 'world_object_pose.json'),
                  '--pregrasp-distance-m', '.1', '--device', 'cpu']
        for key in ('hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config'):
            common.extend(['--' + key.replace('_', '-'), cfg[key]])
        if repair:
            common.extend(['--collision-refinement', str(repair)])
        if cfg.get('restore_object_initial_state', False):
            common.append('--restore-object-initial-state')
        command = [sys.executable, '-B', str(Path(__file__).with_name('native_arm.py')),
                   '--stage', 'prepare', '--output-dir', str(plan_dir),
                   '--arm-select-min-motion', '--vectorized-arm-broadphase', *common]
        with (run / f'prepare_{index:03d}.log').open('xb') as log:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=360)
        plan = read(plan_dir / 'plan.json') if (plan_dir / 'plan.json').exists() else {}
        result['plans'].append(dict(index=index, plan_dir=str(plan_dir), returncode=proc.returncode,
                                    admitted=plan.get('admitted', False)))
        save(run / 'preparation.json', result)
        if proc.returncode or not plan.get('admitted'):
            if proc.returncode and not (plan_dir / 'plan.json').exists():
                failure_path = plan_dir / 'admission_failure.json'
                failure = read(failure_path) if failure_path.exists() else {}
                if not (failure.get('complete') is True and failure.get('kind') == 'arm_plan_rejected'
                        and failure.get('physical_executed') is False):
                    raise RuntimeError('Arm preparation failed without a rejection receipt: ' + str(plan_dir))
                result['plans'][-1]['admission_failure_json'] = str(failure_path)
                result['plans'][-1]['admission_failure_sha256'] = sha(failure_path)
            continue
        meta = gen['records'][index]
        with np.load(pool / 'candidates.npz', allow_pickle=False) as z:
            spec = dict(world_T_palm=z['proposal_palm'][index].tolist(),
                        open_motors=z['proposal_q'][index].tolist(),
                        closed_motors=z['proposal_q_close'][index].tolist(),
                        motor_names=z['independent_motor_names'].tolist(),
                        world_T_object=z['world_T_object'].tolist(),
                        backend_config_sha256=sha(config_path), candidate_pool_sha256=sha(pool / 'candidates.npz'),
                        source_observation_sha256=sha(observation))
        binding = prepare_binding(source, int(meta['source_candidate_index']), ROOT / 'robot_dh3.py',
                    gesture_id=gen['gesture_id'], aperture_id=meta['source_slot'],
                    aperture_sha256=sha(calibration), calibrated_width_m=meta['actual_reference_width_m'],
                    extra_depth_index=meta['additional_depth_index'], execution_spec=spec,
                    if_flip=meta['if_flip'])
        binding_path = run / 'trial_binding.json'
        save(binding_path, binding)
        commands = plan_dir / 'commands.npz'
        with np.load(commands, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        command_sha = write_bound_commands(plan_dir / 'bound_commands.npz', arrays, binding)
        with np.load(plan_dir / 'bound_commands.npz', allow_pickle=False) as z:
            if not all(np.array_equal(arrays[k], z[k]) for k in arrays):
                raise ValueError('Binding changed a motion command')
        (plan_dir / 'bound_commands.npz').replace(commands)
        plan.update(commands_sha256=command_sha, trial_binding_sha256=binding['trial_binding_sha256'],
                    added_metadata_only_original_control_arrays_unchanged=True)
        save(plan_dir / 'plan.json', plan)
        result.update(status='ready', selected_candidate_index=index, selected_candidate=meta,
                      extra_depth_index=meta['additional_depth_index'], plan_dir=str(plan_dir),
                      binding_json=str(binding_path), binding_sha256=sha(binding_path),
                      commands_sha256=command_sha, trial_binding_sha256=binding['trial_binding_sha256'])
        break
    else:
        result['status'] = 'no_full_arm_plan_in_declared_budget'
    result['elapsed_wall_s'] = time.monotonic() - started
    result['inputs_unchanged'] = all(sha(path) == h for path, h in inputs.items())
    if not result['inputs_unchanged']:
        raise ValueError('Source changed during preparation')
    save(run / 'preparation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('1..8 CPU preparation workers required')
    source = args.jobs_json.resolve()
    jobs_doc = read(source)
    jobs = jobs_doc['jobs']
    if not jobs or len({j['trial_id'] for j in jobs}) != len(jobs):
        raise ValueError('Nonempty jobs with unique trial IDs required')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / 'jobs.json')
    manifest = dict(schema='anydex_boya_collection_queue_v1', complete=False, trials=[], nulls=[],
                    errors=[], jobs_json=str(source), jobs_sha256=sha(source),
                    source_sha256={str(p): sha(p) for p in [Path(__file__), *[Path(__file__).with_name(n) for n in ('trial_record.py', 'asset_adapter.py', 'native_arm.py')]]},
                    physical_steps=0, learned_Boya_head_loaded=False)
    config = Path(jobs_doc['backend_config']).resolve()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(prepare_one, job, output, config): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                manifest['trials' if result['status'] == 'ready' else 'nulls'].append(result)
                print(json.dumps(dict(trial_id=job['trial_id'], status=result['status'])), flush=True)
            except Exception as error:
                manifest['errors'].append(dict(trial_id=job['trial_id'], error=repr(error)))
            manifest['elapsed_wall_s'] = time.monotonic() - started
            save(output / 'queue_manifest.json', manifest)
    order = {job['trial_id']: index for index, job in enumerate(jobs)}
    manifest['trials'].sort(key=lambda job: order[job['trial_id']])
    manifest['complete'] = not manifest['errors']
    manifest['inputs_unchanged'] = sha(source) == manifest['jobs_sha256']
    save(output / 'queue_manifest.json', manifest)
    if manifest['errors'] or not manifest['inputs_unchanged']:
        raise RuntimeError('Preparation infrastructure error; queue retained without dispatch')


if __name__ == '__main__':
    main()
