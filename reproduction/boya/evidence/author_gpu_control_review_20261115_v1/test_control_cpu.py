"""Independent CPU control-flow check; child stages/slot pool are mocked.

Actual existing driver/global leases, run registry, durable start guard and
receipt writes are exercised in temporary directories. No child is spawned.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import author_gpu_control as M


class MockPool:
    def __init__(self, *args, **kwargs):
        pass

    @contextlib.contextmanager
    def stage(self, *args, **kwargs):
        yield


class MockProxy:
    def __init__(self, *args):
        pass


def check():
    started = time.monotonic()
    results = []
    for case in ('first_stage_failure', 'first_native_gate_failure', 'both_complete'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            root.mkdir()
            calls = []
            jobs = [
                dict(name='single_g6', prepared_dirs=['/notexecuted/g6'],
                     env_roots=['/World/envs/env_0'], origins=[[0, 0, 0]],
                     shared_ticks=3024, actual_integrated_envsteps=3024),
                dict(name='batch_g1_g6', prepared_dirs=['/notexecuted/g1', '/notexecuted/g6'],
                     env_roots=['/World/envs/env_0', '/World/envs/env_1'],
                     origins=[[0, 0, 0], [4, 0, 0]], shared_ticks=3374,
                     actual_integrated_envsteps=6748),
            ]
            M.B.save(root/M.PROTOCOL, dict(schema=M.SCHEMA, exclude_training=True,
                total_actual_integrated_envsteps=9772, retry_budget=0,
                inputs={}, sources={}, jobs=jobs))

            def fake_stage(stage, command, expected, protocol):
                calls.append(command)
                M.B.save(stage/'STAGE.json', dict(status='complete'))
                if case == 'first_stage_failure':
                    return dict(complete=False, status='infrastructure_failure')
                job = jobs[len(calls)-1]
                M.B.save(expected[0], dict(status='complete', complete=True,
                    physical_clock_steps=job['shared_ticks'],
                    actual_integrated_environment_steps=job['actual_integrated_envsteps'],
                    jobs=[dict(gates=dict(original_gate=case != 'first_native_gate_failure'))]))
                return dict(complete=True, status='complete')

            with patch.object(M, 'LOCK', Path(tmp)/'gpu.lock'), \
                    patch.object(M, 'REGISTRY', Path(tmp)/'registry.json'), \
                    patch.object(M.R, 'ResourcePool', MockPool), \
                    patch.object(M.R, 'SubprocessProxy', MockProxy), \
                    patch.object(M.B, '_stage', fake_stage):
                value = M.run(root)
                before = len(calls)
                try:
                    M.run(root)
                except FileExistsError:
                    once = True
                else:
                    once = False
                assert once and len(calls) == before
                assert len(calls) == (2 if case == 'both_complete' else 1)
                assert value['status'] == ('complete' if case == 'both_complete' else 'stopped_after_failure')
                registry = M.B.read(Path(tmp)/'registry.json')
                assert registry['state'] == 'released'
                results.append(dict(case=case, stage_count=len(calls), status=value['status'],
                    second_run_rejected=once, registry_released=True))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return dict(schema='anydex_boya_gpu_control_independent_cpu_review_v1', status='pass',
        source_sha256={str(Path(p).resolve()): sha(p) for p in
            (__file__, M.__file__, M.B.__file__, M.R.__file__)},
        actual_existing_paths_exercised=['driver/global flock leases', 'active-run registry',
            'durable RUN_STARTED guard', 'stop-after-first-failure', 'receipt writes'],
        mocked=['ResourcePool stage context', 'SubprocessProxy', 'B._stage child execution'],
        input_source_verification_scope='Synthetic empty bindings only; not a live artifact verification test',
        limitations=['Does not prove actual slot FD inheritance, GPU initialization or GPU physics'],
        Kit_launches=0, physics_steps=0, child_processes=0, model_forwards=0,
        cases=results, wall_s=time.monotonic()-started)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
