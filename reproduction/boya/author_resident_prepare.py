"""Finite CPU planning requests in one serial process; never starts simulation."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


def parser():
    p = argparse.ArgumentParser()
    for name in ('scene', 'representation-dir', 'observation', 'calibration',
                 'cloud-manifest', 'backend', 'output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--domain')
    p.add_argument('--archive-sources')
    return p


def request_from_command(command):
    import author_collection_batch as B
    entry = str(B.HERE/'author_contact_pipeline.py')
    values = list(map(str, command))
    index = values.index(entry)
    if values[index+1] != 'prepare':
        raise ValueError('Only the existing contact prepare command is supported')
    return vars(parser().parse_args(values[index+2:]))


def bind_backend(request):
    import author_transport as T
    T._backend_path()
    from run_template_arm_positive import bind_hand
    cfg = T.read(request['backend'])
    # Before PlanningModel imports HAND_SCENE or any geometry constructor.
    bind_hand(SimpleNamespace(hand_urdf=Path(cfg['hand_urdf']), hand_scene=Path(cfg['hand_scene'])))
    return cfg


def prepare(request, session=None):
    import author_collection_batch as B
    import author_contact_pipeline as M
    output = Path(request['output_dir']).resolve()
    before = dict(session.stats) if session else {}
    began = time.monotonic()
    if session:
        session.verify()
    result = M.prepare(Path(request['scene']), Path(request['representation_dir']),
        Path(request['observation']), request['seed'], Path(request['calibration']),
        Path(request['cloud_manifest']), Path(request['backend']), output,
        domain=B.read(request['domain']) if request.get('domain') else None,
        archive_sources=B.read(request['archive_sources']) if request.get('archive_sources') else None)
    mapping = M.audit_mapping(output/'pipeline') if result['status'] == 'ready' else None
    if session:
        session.verify()
    receipt = dict(status=result['status'], wall_s=time.monotonic()-began,
        planning_result=B.binding(output/'pipeline/LIFT_PIPELINE_RESULTS.json'),
        mapping_audit=mapping, physics_steps=0, labels=0, Kit_starts=0,
        simulation_started=False, new_CGR_forwards=0, training_updates=0,
        implementation=B.binding(__file__),
        cache=session.receipt() if session else None,
        job_cache_delta={k:session.stats[k]-before[k] for k in before} if session else None)
    if session:
        import author_planning_reuse as reuse
        receipt['cache_implementation'] = B.binding(reuse.__file__)
    B.save(output/'PLANNING_REUSE.json', receipt, exclusive=True)
    return receipt


def batch(requests, *, reuse=True):
    if not requests or len(requests) > 8:
        raise ValueError('One to eight explicit finite planning requests required')
    bind_backend(requests[0])
    if reuse:
        from author_planning_reuse import PlanningSession
        session = PlanningSession()
    else:
        session = None
    results = []
    with session.installed() if session else nullcontext():
        for request in requests:
            cfg = bind_backend(request)
            if session:
                session.bind(Path(cfg[n]) for n in ('robot_urdf', 'hand_urdf', 'hand_scene', 'package'))
            result = prepare(request, session)
            results.append(result)
            if result['status'] == 'infrastructure_failure':
                raise RuntimeError('Planning infrastructure failure; finite queue stopped')
    return results


def serve(root, protocol, lease_fd):
    """A resource-owned CPU worker using the same finite mailbox contract."""
    import author_collection_batch as B
    import author_collection_wave_resources as R
    from author_planning_reuse import PlanningSession
    root, protocol = Path(root), Path(protocol)
    p = B.read(protocol)
    B.verify(p['sources']); B.verify(p['inputs'])
    if not p.get('resident_planning') or p['exclude_training'] is not True:
        raise ValueError('Explicit finite development planning protocol required')
    lock = B.ROOT/'outputs/author_collection_gpu0.lock'
    R.check_inherited_lease(lease_fd, lock, [lock.stat().st_dev, lock.stat().st_ino])
    mailbox = root/'mailbox'; mailbox.mkdir(parents=True, exist_ok=False)
    binding = B.binding(protocol)
    bootstrap = dict(backend=p['inputs']['backend']['path'])
    bind_backend(bootstrap)
    session = PlanningSession(); error = None; index = 0
    try:
        with session.installed():
            B.save(root/'SERVICE_READY.json', dict(pid=os.getpid(), protocol=binding), exclusive=True)
            while True:
                request_path = mailbox/f'{index:03d}.json'
                deadline = time.monotonic()+p['service_idle_timeout_s']
                while not request_path.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Resident CPU planning idle timeout')
                    time.sleep(.02)
                value = B.read(request_path)
                if value['sequence'] != index or value['protocol'] != binding:
                    raise ValueError('Unbound planning request')
                if value['role'] == 'close':
                    break
                if value['role'] == 'abort':
                    raise RuntimeError('Parent aborted planning')
                if value['role'] != 'prepare' or index >= len(p['rows']):
                    raise ValueError('Finite planning request count changed')
                request = value['request']
                expected_output = protocol.parent/'scenes'/f'{index:02d}'/'contact'
                if Path(request['output_dir']).resolve() != expected_output.resolve():
                    raise ValueError('Planning output escaped the frozen round')
                expected = p['rows'][index]['scene_id']
                scene = B.read(request['scene'])
                if scene['scene_id'] != expected:
                    raise ValueError('Planning scene differs from frozen round')
                B.verify(p['sources']); session.verify()
                from author_planning_reuse import deadline
                with deadline(p['stage_timeout_s'], ['resident_prepare', str(index)]):
                    result = prepare(request, session)
                if result['status'] == 'infrastructure_failure':
                    raise RuntimeError('Planning infrastructure failure')
                B.verify(p['sources'])
                B.save(mailbox/f'{index:03d}.RESULT.json', result, exclusive=True)
                index += 1
    except BaseException:
        error = traceback.format_exc()
        B.save(root/'SERVICE_ERROR.json', dict(error=error, sequence=index), exclusive=True)
    finally:
        B.save(root/'SERVICE_FINISHED.json', dict(status='complete' if error is None else 'failed',
            error=error, jobs_completed=index, cache=session.receipt(), Kit_starts=0, physics_steps=0), exclusive=True)
    if error is not None:
        raise RuntimeError(error)


def main():
    # Match the existing native CPU subprocess numerical thread settings.
    os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='4',
                      PYTHONDONTWRITEBYTECODE='1')
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('batch'); q.add_argument('--requests', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True); q.add_argument('--original', action='store_true')
    q = sub.add_parser('serve'); q.add_argument('--root', type=Path, required=True)
    q.add_argument('--protocol', type=Path, required=True); q.add_argument('--lease-fd', type=int, required=True)
    a = p.parse_args()
    if a.action == 'serve':
        serve(a.root, a.protocol, a.lease_fd)
        return
    import author_collection_batch as B
    began = time.monotonic()
    results = batch(B.read(a.requests), reuse=not a.original)
    B.save(a.output, dict(status='complete', jobs=results, wall_s=time.monotonic()-began,
        Kit_starts=0, physics_steps=0, labels=0), exclusive=True)
    print(dict(status='complete', jobs=len(results), wall_s=time.monotonic()-began), flush=True)


if __name__ == '__main__':
    main()
