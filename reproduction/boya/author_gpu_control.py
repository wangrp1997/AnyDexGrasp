"""Freeze/run two bounded GPU execution controls through existing stage leases.

No candidates, collection export or training. The old prepared directories stay
read-only. Each new group owns its once-only marker and fresh output directory.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

import author_collection_batch as B
import author_collection_wave_resources as R

SCHEMA = 'anydex_boya_author_gpu_control_v1'
PROTOCOL = 'GPU_CONTROL_PROTOCOL.json'
LOCK = B.ROOT/'outputs/author_collection_gpu0.lock'
REGISTRY = LOCK.with_suffix('.wave_active.json')
SCENE = '0000004_s40_lift_fit_00004_s2026111401'
BASE = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')


def freeze(output_dir):
    old = BASE/'resident_collection_control_20261115_v1/fit'
    inherited = B.read(old/'RESIDENT_COLLECTION_PROTOCOL.json')
    # Keep all original environment/native bindings, not just the new adapter.
    inputs = dict(reference_protocol=B.binding(old/'RESIDENT_COLLECTION_PROTOCOL.json'))
    sources = copy.deepcopy(inherited['sources'])
    for name in ('author_gpu_control.py', 'author_gpu_vector_execution.py',
                 'author_gpu_capability_probe.py'):
        sources[name] = B.binding(B.HERE/name)
    jobs = []
    for name, gestures in (('single_g6', [6]), ('batch_g1_g6', [1, 6])):
        prepared = [old/'scenes'/SCENE/f'g{g}' for g in gestures]
        steps = []
        for p in prepared:
            ready = B.read(p/'READY.json')
            for key in ('scene', 'candidate', 'plan', 'commands', 'command_binding', 'protocol'):
                inputs[f'{name}_{p.name}_{key}'] = ready[key]
            inputs[f'{name}_{p.name}_ready'] = B.binding(p/'READY.json')
            plan = B.read(ready['plan']['path'])
            steps.append(plan['total_command_steps'])
        expected = [3024] if len(gestures) == 1 else [3374, 3024]
        if steps != expected:
            raise ValueError('Reviewed positive/negative control lengths changed')
        roots = [f'/World/envs/env_{i}' for i in range(len(gestures))]
        origins = [[4*i, 0, 0] for i in range(len(gestures))]
        jobs.append(dict(name=name, prepared_dirs=list(map(str, prepared)), gestures=gestures,
            env_roots=roots, origins=origins, command_steps=steps,
            shared_ticks=max(steps), recorded_command_envsteps=sum(steps),
            actual_integrated_envsteps=len(steps)*max(steps),
            inactive_tail_envsteps=len(steps)*max(steps)-sum(steps)))
    protocol = dict(schema=SCHEMA, task=B.TASK, diagnostic_only=True,
        exclude_training=True, formal_training_eligible=False, automatic_training=False,
        physics_device='cuda:0', gpu_uuid='GPU-4d3badb8-5fb0-b880-f08e-6407f0093e86',
        selection_basis='known_same_geometry_positive_and_negative_controls',
        inputs=inputs, sources=sources, jobs=jobs, execution_attempt_budget=3,
        retry_budget=0, extra_probe_physics_steps=0,
        total_recorded_command_envsteps=9422, total_actual_integrated_envsteps=9772,
        total_shared_ticks=6398, inactive_tail_envsteps=350,
        env=dict(inherited['env'], CUDA_VISIBLE_DEVICES='GPU-4d3badb8-5fb0-b880-f08e-6407f0093e86',
            NVIDIA_VISIBLE_DEVICES='0'),
        inherited_env=inherited['inherited_env'], stage_timeout_s=1800,
        created_unix_s=time.time(),
        limitations=['Execution-only controls cannot establish full collection ETA',
            'Known controls cannot estimate random-scene yield',
            'GPU physics need not reproduce CPU contact trajectories bit for bit',
            'No diagnostic outcome is a formal training label'])
    B.verify(inputs); B.verify(sources)
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=False)
    B.save(output_dir/PROTOCOL, protocol, exclusive=True)
    return protocol


def run(root):
    root = Path(root).resolve(); protocol = B.read(root/PROTOCOL)
    if (protocol.get('schema') != SCHEMA or protocol.get('exclude_training') is not True
            or protocol.get('total_actual_integrated_envsteps') != 9772
            or protocol.get('retry_budget') != 0):
        raise ValueError('Frozen bounded diagnostic required')
    B.verify(protocol['inputs']); B.verify(protocol['sources'])
    started = time.monotonic(); stages = []
    with R.lease(root/'DRIVER.lock'), R.lease(LOCK) as lease_fd:
        if (root/'RUN_STARTED.json').exists():
            raise FileExistsError('Started GPU control is never relaunched')
        with R.active_run(root, REGISTRY, B.binding(root/PROTOCOL)):
            pool = R.ResourcePool(root/'resource_slots', lease_fd, B.binding(root/PROTOCOL))
            B.save(root/'RUN_STARTED.json', dict(protocol=B.binding(root/PROTOCOL),
                owner=R.process_identity(__import__('os').getpid()), unix_s=time.time()), exclusive=True)
            for job in protocol['jobs']:
                stage = root/job['name']; output = stage/'result'
                stage.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, '-B', str(B.HERE/'author_gpu_vector_execution.py'),
                    'run', '--prepared-dirs', *job['prepared_dirs'], '--output-dir', str(output),
                    '--device', 'cuda:0', '--env-roots', *job['env_roots'],
                    '--origins-json', __import__('json').dumps(job['origins']),
                    '--protocol', str(root/PROTOCOL)]
                previous = B.subprocess
                with pool.stage(('kit',), stage):
                    B.subprocess = R.SubprocessProxy(pool)
                    try:
                        result = B._stage(stage, command, [output/'GPU_VECTOR_EXECUTION.json'], protocol)
                    finally:
                        B.subprocess = previous
                stages.append(dict(name=job['name'], stage=B.binding(stage/'STAGE.json')))
                if not result['complete']:
                    break
                native = B.read(output/'GPU_VECTOR_EXECUTION.json')
                if (native.get('status') != 'complete' or native.get('complete') is not True
                        or native.get('physical_clock_steps') != job['shared_ticks']
                        or native.get('actual_integrated_environment_steps') != job['actual_integrated_envsteps']
                        or not native.get('jobs') or any(not all(j['gates'].values()) for j in native['jobs'])):
                    stages[-1]['admission'] = 'stopped_after_incomplete_or_invalid_gpu_control'
                    break
                stages[-1]['admission'] = 'native_control_complete_all_gates_passed'
            B.verify(protocol['inputs']); B.verify(protocol['sources'])
            result = dict(schema=SCHEMA, status='complete' if len(stages) == 2 and all(
                s.get('admission') == 'native_control_complete_all_gates_passed'
                for s in stages) else 'stopped_after_failure',
                protocol=B.binding(root/PROTOCOL), stages=stages, wall_s=time.monotonic()-started,
                exclude_training=True, formal_training_eligible=False, training_updates=0,
                label=None, retry_count=0)
            B.save(root/'GPU_CONTROL_RESULTS.json', result, exclusive=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'run'))
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    result = freeze(a.output_dir) if a.action == 'freeze' else run(a.output_dir)
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__':
    main()
