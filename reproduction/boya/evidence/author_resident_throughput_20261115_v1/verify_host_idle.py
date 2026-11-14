"""Read host identities of this fixed control and probe released flock leases."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import time


def read(p):
    return json.loads(p.read_text())


def main():
    repo = Path(__file__).resolve().parents[4]
    root = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/resident_collection_control_20261115_v1')
    registry_path = repo/'outputs/author_collection_gpu0.wave_active.json'
    registry = read(registry_path)
    assert registry['state'] == 'released'
    pids = {registry['owner']['pid']}
    locks = [repo/'outputs/author_collection_gpu0.lock']
    reservations = []
    for part in ('fit', 'internal'):
        run = root/part
        assert read(run/'OUTER_DRIVER_RESULTS.json')['status'] == 'complete'
        for path in run.glob('scenes/**/SESSION_STARTED.json'):
            pids.add(read(path)['pid'])
        for path in run.glob('**/STAGE.json'):
            j = read(path)
            if j.get('child_pid'):
                pids.add(j['child_pid'])
        for path in sorted((run/'resource_slots').glob('*/*.json')):
            state = read(path)
            assert state['state'] == 'released', str(path)
            if state.get('worker', {}).get('pid'):
                pids.add(state['worker']['pid'])
            reservations.append(dict(path=str(path), state=state['state']))
        locks.extend(sorted((run/'resource_slots').glob('*/*.lock')))
    observed = {str(pid): (Path('/proc')/str(pid)).exists() for pid in sorted(pids)}
    assert not any(observed.values()), observed
    probes = []
    for path in locks:
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            probes.append(dict(path=str(path), exclusive_lock_available=True))
        finally:
            os.close(fd)
    result = dict(status='pass', unix_s=time.time(), host_proc_namespace=True,
        registry=registry, owned_stage_pids_present=observed,
        released_reservations=reservations, exclusive_lock_probes=probes,
        training_started=False, bulk_collection_started=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='No broad process enumeration; only this completed control PIDs and released resource slots')
    with (Path(__file__).parent/'HOST_FINAL_IDLE.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='pass', exited_stage_pids=len(pids), released_slots=len(reservations), lock_probes=len(probes))))


if __name__ == '__main__':
    main()
