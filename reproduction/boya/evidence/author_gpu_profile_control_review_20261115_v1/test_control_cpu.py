"""Independent CPU review: real leases/registry, mocked children and slot pool.

No Kit, model, child process, physics step, signal, or runtime source edit.
"""
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import author_gpu_profile_control as M


class MockPool:
    def __init__(self, *args, **kwargs):
        pass

    @contextlib.contextmanager
    def stage(self, *args, **kwargs):
        yield


class MockProxy:
    def __init__(self, *args):
        pass


def fixtures(lanes):
    native = dict(status='partial_profile', complete=False,
        physical_clock_steps=20, actual_integrated_environment_steps=20*lanes,
        recorded_command_environment_steps=20*lanes,
        clock=dict(lane_steps=[20]*lanes),
        jobs=[dict(label=None, raw_label=None, completion_tick=20) for _ in range(lanes)],
        initial_gpu_probe=dict(status='live_api_snapshot_pass'),
        first_tick_gpu_probe=dict(status='live_api_snapshot_pass'))
    profile = dict(status='partial_profile', actual_physics_steps=20,
        label=None, raw_label=None, exclude_training=True)
    return native, profile


def check():
    began = time.monotonic()
    results = []
    cases = ('first_stage_failure', 'clock_count_mismatch', 'integrated_count_mismatch',
        'recorded_count_mismatch', 'lane_count_mismatch', 'completion_tick_mismatch',
        'missing_initial_probe', 'missing_first_tick_probe', 'native_binary_label',
        'profile_binary_label', 'profile_count_mismatch', 'profile_training_eligible',
        'second_stage_count_mismatch', 'both_20_tick_prefixes_complete')
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            root = temp/'run'
            root.mkdir()
            calls = []
            jobs = [
                dict(name='single_g6', gestures=[6], prepared_dirs=['/notexecuted/g6'],
                    env_roots=['/World/envs/env_0'], origins=[[0, 0, 0]]),
                dict(name='batch_g1_g6', gestures=[1, 6],
                    prepared_dirs=['/notexecuted/g1', '/notexecuted/g6'],
                    env_roots=['/World/envs/env_0', '/World/envs/env_1'],
                    origins=[[0, 0, 0], [4, 0, 0]]),
            ]
            source = temp/'synthetic_source.txt'
            source.write_text('Frozen synthetic dependency, not a physical asset.\n')
            M.B.save(root/M.PROTOCOL, dict(schema=M.SCHEMA, prefix_ticks=20,
                total_actual_integrated_envsteps=60, retry_budget=0,
                inputs={}, sources={'synthetic_source': M.B.binding(source)}, jobs=jobs))

            def fake_stage(stage, command, expected, protocol):
                calls.append(list(command))
                assert command[command.index('--prefix-ticks')+1] == '20'
                assert len(expected) == 2
                M.B.save(stage/'STAGE.json', dict(status='mock_complete'))
                if case == 'first_stage_failure':
                    return dict(complete=False, status='infrastructure_failure')
                lanes = len(jobs[len(calls)-1]['gestures'])
                native, profile = fixtures(lanes)
                if case == 'clock_count_mismatch': native['physical_clock_steps'] = 19
                if case == 'integrated_count_mismatch': native['actual_integrated_environment_steps'] -= 1
                if case == 'recorded_count_mismatch': native['recorded_command_environment_steps'] -= 1
                if case == 'lane_count_mismatch': native['clock']['lane_steps'][0] = 19
                if case == 'completion_tick_mismatch': native['jobs'][0]['completion_tick'] = 19
                if case == 'missing_initial_probe': native.pop('initial_gpu_probe')
                if case == 'missing_first_tick_probe': native.pop('first_tick_gpu_probe')
                if case == 'native_binary_label': native['jobs'][0]['raw_label'] = 0
                if case == 'profile_binary_label': profile['label'] = 0
                if case == 'profile_count_mismatch': profile['actual_physics_steps'] = 19
                if case == 'profile_training_eligible': profile['exclude_training'] = False
                if case == 'second_stage_count_mismatch' and len(calls) == 2:
                    native['recorded_command_environment_steps'] = 39
                M.B.save(expected[0], native)
                M.B.save(expected[1], profile)
                return dict(complete=True, status='complete')

            with patch.object(M.G, 'LOCK', temp/'gpu.lock'), \
                    patch.object(M.G, 'REGISTRY', temp/'registry.json'), \
                    patch.object(M.R, 'ResourcePool', MockPool), \
                    patch.object(M.R, 'SubprocessProxy', MockProxy), \
                    patch.object(M.B, '_stage', fake_stage):
                value = M.run(root)
                calls_before = len(calls)
                try:
                    M.run(root)
                except FileExistsError:
                    rerun_rejected = True
                else:
                    rerun_rejected = False
                assert rerun_rejected and len(calls) == calls_before
                expected_calls = 2 if case in ('second_stage_count_mismatch',
                    'both_20_tick_prefixes_complete') else 1
                assert len(calls) == expected_calls
                complete = case == 'both_20_tick_prefixes_complete'
                assert value['status'] == ('bounded_prefixes_complete' if complete else 'stopped_after_failure')
                assert value['label'] is None and value['raw_label'] is None
                assert value['exclude_training'] is True and value['formal_training_eligible'] is False
                assert value['full_physical_validation'] is False and value['retry_count'] == 0
                assert (root/'RUN_STARTED.json').is_file()
                assert M.B.read(temp/'registry.json')['state'] == 'released'
                assert M.B.read(root/'GPU_PROFILE_CONTROL_RESULTS.json') == value
                # Prove the actual driver/global flock contexts returned their leases.
                with M.R.lease(root/'DRIVER.lock'), M.R.lease(temp/'gpu.lock'):
                    pass
                results.append(dict(case=case, dispatched_stages=len(calls),
                    status=value['status'], admissions=[x.get('admission') for x in value['stages']],
                    rerun_rejected=True, registry_released=True, leases_reacquired=True,
                    synthetic_only=True))

    # A changed declared dependency must be rejected before a start marker or child.
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp); root = temp/'run'; root.mkdir()
        source = temp/'source'; source.write_text('before')
        M.B.save(root/M.PROTOCOL, dict(schema=M.SCHEMA, prefix_ticks=20,
            total_actual_integrated_envsteps=60, retry_budget=0, jobs=[], inputs={},
            sources={'synthetic_source': M.B.binding(source)}))
        source.write_text('after')
        with patch.object(M.B, '_stage', side_effect=AssertionError('No child may be dispatched')):
            try:
                M.run(root)
            except ValueError:
                pass
            else:
                raise AssertionError('Changed declared source was admitted')
        assert not (root/'RUN_STARTED.json').exists()
        results.append(dict(case='changed_source_before_start', status='rejected',
            dispatched_stages=0, start_marker=False, synthetic_only=True))

    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return dict(schema='anydex_boya_gpu_profile_control_independent_cpu_review_v1',
        status='pass', cases=results,
        source_sha256={str(Path(p).resolve()): sha(p) for p in
            (__file__, M.__file__, M.G.__file__, M.B.__file__, M.R.__file__,
             HERE/'author_gpu_vector_profile.py')},
        actual_existing_paths_exercised=['driver/global flock leases and release',
            'active_run registry release', 'RUN_STARTED guard', 'admitted predicate',
            'stop on first failed or inconsistent prefix', 'JSON result persistence',
            'synthetic declared source binding verification'],
        mocked=['ResourcePool stage context', 'SubprocessProxy', 'B._stage'],
        limitations=['No real slot FD inheritance, Kit/GPU startup, physical clock or live artifact validity tested',
            'Synthetic fixture counts are test inputs, not measured physical steps',
            'No real freeze command or public run artifact was modified'],
        Kit_launches=0, physics_steps=0, child_processes=0, model_forwards=0,
        wall_s=time.monotonic()-began)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
