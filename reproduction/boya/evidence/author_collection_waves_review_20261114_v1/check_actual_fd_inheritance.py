"""Independent real-subprocess FD audit; no simulator or planner imports.

Run with the existing venv and reproduction/boya on PYTHONPATH. Temporary lock
files are removed after both child processes have exited. No collection runs.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import author_collection_wave_resources as resources


def main():
    source = Path(resources.__file__)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    child_code = """import json, os, pathlib
paths = []
for entry in pathlib.Path('/proc/self/fd').iterdir():
    try:
        target = os.readlink(entry)
        if target.endswith('.lock'):
            paths.append(target)
    except FileNotFoundError:
        pass
print(json.dumps(sorted(paths)))
"""
    with tempfile.TemporaryDirectory(prefix='boya_wave_fd_review_') as raw:
        root = Path(raw)
        lease_path = root / 'global.lock'
        with resources.lease(lease_path) as lease_fd:
            pool = resources.ResourcePool(root / 'slots', lease_fd,
                {'path': 'independent-review-only', 'sha256': '0' * 64})
            barrier = threading.Barrier(2)
            results, errors = {}, []

            def work(index):
                try:
                    with pool.stage(('prepare',), root / f'prepare_{index}') as context:
                        own = [os.readlink(f'/proc/self/fd/{slot["fd"]}')
                               for slot in context['slots']]
                        barrier.wait(timeout=5)
                        process = pool.popen([sys.executable, '-B', '-c', child_code],
                            start_new_session=True, stdout=subprocess.PIPE, text=True)
                        output, _ = process.communicate(timeout=5)
                        actual = json.loads(output)
                        expected = sorted([str(lease_path), *own])
                        assert actual == expected, (actual, expected)
                        assert process.returncode == 0
                        results[index] = dict(
                            only_own_slot_and_parent_lease=True,
                            visible_lock_files=[str(Path(path).relative_to(root))
                                                for path in actual],
                            returncode=process.returncode)
                except BaseException as error:
                    errors.append(repr(error))

            threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            assert not any(thread.is_alive() for thread in threads)
            assert not errors, errors
            assert len(results) == 2
            assert not any((root / 'slots' / 'quarantine').glob('*.json'))
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert before == after, 'Reviewed resource source changed during the audit'
    print(json.dumps(dict(status='pass', test='two_actual_prepare_subprocesses',
        source=dict(path=str(source.resolve()), sha256=before),
        results=results, sources_unchanged=True,
        simulator_processes=0, physics_steps=0, planning_calls=0),
        indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
