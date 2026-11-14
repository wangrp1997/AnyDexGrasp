"""Frozen, once-only pure transport continuation queue; no candidate search."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def verify(bindings):
    for key, binding in bindings.items():
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Frozen queue dependency changed: '+key)


def main():
    protocol_path = OUT/'COMPONENT_INTEGRATION_PROTOCOL.json'
    protocol = read(protocol_path)
    verify(protocol['sources']); verify(protocol['inputs'])
    if protocol['candidate_order'] != [1, 6, 7] or protocol['max_transport_dispatches'] != 3:
        raise ValueError('Fixed three-job queue required')
    save(OUT/'DISPATCH_STARTED.json', dict(protocol_sha256=sha(protocol_path),
        started_unix_s=time.time(), pid=os.getpid(), physics_steps=0,
        policy='One prepare per frozen already-ready HOME; never execute, retry or fallback'))
    began = time.monotonic()

    def worker(job):
        command = [sys.executable, '-B', str(OUT/'run_one_integration.py'), str(job['gesture_id'])]
        started = time.monotonic()
        record = dict(gesture_id=job['gesture_id'], command=command, physics_steps=0)
        try:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                LD_LIBRARY_PATH=str(ROOT/'outputs/native_deps/os_libs'))
            with (OUT/f"integration_g{job['gesture_id']}.log").open('xb') as log:
                run = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=protocol['per_job_timeout_s'])
            record['returncode'] = run.returncode
            path = Path(job['output'])/'COMPONENT_INTEGRATION_RESULTS.json'
            result = read(path)
            record.update(status=result['status'], result_path=str(path), result_sha256=sha(path),
                          component_wall_s=result.get('wall_s'))
            if run.returncode and result['status'] != 'infrastructure_failure':
                raise ValueError('Nonzero process exit without infrastructure receipt')
        except BaseException as error:
            record.update(status='infrastructure_failure', error=repr(error), traceback=traceback.format_exc())
        record['dispatch_wall_s'] = time.monotonic()-started
        save(OUT/f"DISPATCH_g{job['gesture_id']}.json", record)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(worker, protocol['jobs']))
    verify(protocol['sources']); verify(protocol['inputs'])
    save(OUT/'COMPONENT_INTEGRATION_RESULTS.json', dict(
        schema='anydex_boya_author_component_queue_v1', complete=True,
        protocol_sha256=sha(protocol_path), jobs=records,
        ready=sum(row['status'] == 'ready' for row in records),
        null=sum(row['status'].startswith('null') or row['status'] == 'no_plan' for row in records),
        infrastructure_failures=sum(row['status'] == 'infrastructure_failure' for row in records),
        wall_s=time.monotonic()-began, physics_steps=0, training_updates=0, new_labels=0,
        new_CGR_forwards=0, new_collection_collision_queries=0,
        new_native_approach_IK=0, new_HOME_search_calls=0,
        prior_g8_null_preserved=True, physical_execution_dispatched=False))
    print(json.dumps(records, indent=2))
    return int(any(row['status'] == 'infrastructure_failure' for row in records))


if __name__ == '__main__':
    raise SystemExit(main())
