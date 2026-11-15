"""Freeze and run two 20-tick diagnostic prefixes through existing leases.

No sampling, new full trial, training export, retry, or background work.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import author_collection_batch as B
import author_collection_wave_resources as R
import author_gpu_control as G

SCHEMA = 'anydex_boya_author_gpu_profile_control_v1'
PROTOCOL = 'GPU_PROFILE_CONTROL_PROTOCOL.json'


def freeze(root):
    reference = G.BASE/'gpu_vector_control_20261115_v2'/G.PROTOCOL
    old = B.read(reference)
    protocol = copy.deepcopy(old)
    protocol.update(schema=SCHEMA, purpose='bounded_partition_timing_only',
        prefix_ticks=20, stage_timeout_s=600, total_shared_ticks=40,
        total_recorded_command_envsteps=60, total_actual_integrated_envsteps=60,
        inactive_tail_envsteps=0, created_unix_s=time.time(),
        next_unintegrated_command_may_have_been_prepared=True,
        limitations=old['limitations']+[
            'Incomplete prefixes cannot certify full physical validity or a binary label',
            'Early settle timing does not establish a full trajectory or collection ETA'])
    protocol['inputs']['reference_gpu_protocol'] = B.binding(reference)
    for name in ('author_gpu_vector_profile.py', 'author_gpu_profile_control.py'):
        protocol['sources'][name] = B.binding(B.HERE/name)
    for job in protocol['jobs']:
        job.update(shared_ticks=20, recorded_command_envsteps=20*len(job['gestures']),
            actual_integrated_envsteps=20*len(job['gestures']), inactive_tail_envsteps=0)
    B.verify(protocol['inputs']); B.verify(protocol['sources'])
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    B.save(root/PROTOCOL, protocol, exclusive=True)
    return protocol


def admitted(native, profile, lanes):
    return (native.get('status') == 'partial_profile'
        and native.get('complete') is False
        and native.get('physical_clock_steps') == 20
        and native.get('actual_integrated_environment_steps') == 20*lanes
        and native.get('recorded_command_environment_steps') == 20*lanes
        and native.get('clock', {}).get('lane_steps') == [20]*lanes
        and len(native.get('jobs', [])) == lanes
        and all(j.get('label') is None and j.get('raw_label') is None
            and j.get('completion_tick') == 20 for j in native['jobs'])
        and all(native.get(k, {}).get('status') == 'live_api_snapshot_pass'
            for k in ('initial_gpu_probe', 'first_tick_gpu_probe'))
        and profile.get('status') == 'partial_profile'
        and profile.get('actual_physics_steps') == 20
        and profile.get('label') is None and profile.get('raw_label') is None
        and profile.get('exclude_training') is True)


def run(root):
    root = Path(root).resolve(); protocol = B.read(root/PROTOCOL)
    if (protocol.get('schema') != SCHEMA or protocol.get('prefix_ticks') != 20
            or protocol.get('total_actual_integrated_envsteps') != 60
            or protocol.get('retry_budget') != 0):
        raise ValueError('Frozen 20-tick diagnostic protocol required')
    B.verify(protocol['inputs']); B.verify(protocol['sources'])
    began = time.monotonic(); stages = []
    with R.lease(root/'DRIVER.lock'), R.lease(G.LOCK) as lease_fd:
        if (root/'RUN_STARTED.json').exists():
            raise FileExistsError('A started prefix diagnostic is never relaunched')
        with R.active_run(root, G.REGISTRY, B.binding(root/PROTOCOL)):
            pool = R.ResourcePool(root/'resource_slots', lease_fd, B.binding(root/PROTOCOL))
            B.save(root/'RUN_STARTED.json', dict(protocol=B.binding(root/PROTOCOL),
                owner=R.process_identity(__import__('os').getpid()), unix_s=time.time()), exclusive=True)
            for job in protocol['jobs']:
                stage = root/job['name']; output = stage/'result'
                stage.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, '-B', str(B.HERE/'author_gpu_vector_profile.py'),
                    '--prepared-dirs', *job['prepared_dirs'], '--output-dir', str(output),
                    '--env-roots', *job['env_roots'], '--origins-json', json.dumps(job['origins']),
                    '--prefix-ticks', '20', '--protocol', str(root/PROTOCOL)]
                expected = [output/'GPU_VECTOR_EXECUTION.json', output/'GPU_VECTOR_PROFILE.json']
                previous = B.subprocess
                with pool.stage(('kit',), stage):
                    B.subprocess = R.SubprocessProxy(pool)
                    try:
                        result = B._stage(stage, command, expected, protocol)
                    finally:
                        B.subprocess = previous
                stages.append(dict(name=job['name'], stage=B.binding(stage/'STAGE.json')))
                if not result['complete']:
                    break
                native, profile = map(B.read, expected)
                if not admitted(native, profile, len(job['gestures'])):
                    stages[-1]['admission'] = 'stopped_after_invalid_prefix'
                    break
                stages[-1]['admission'] = 'bounded_prefix_complete_excluded_training'
            B.verify(protocol['inputs']); B.verify(protocol['sources'])
            result = dict(schema=SCHEMA, status='bounded_prefixes_complete' if len(stages) == 2
                and all(s.get('admission') == 'bounded_prefix_complete_excluded_training'
                    for s in stages) else 'stopped_after_failure',
                protocol=B.binding(root/PROTOCOL), stages=stages, wall_s=time.monotonic()-began,
                full_physical_validation=False, exclude_training=True, formal_training_eligible=False,
                training_updates=0, label=None, raw_label=None, retry_count=0)
            B.save(root/'GPU_PROFILE_CONTROL_RESULTS.json', result, exclusive=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.output_dir) if args.action == 'freeze' else run(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
