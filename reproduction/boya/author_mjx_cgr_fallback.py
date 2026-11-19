"""Bounded CGR order planning in the resident voxel and native arm pipeline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import author_collection_batch as B
import author_collection_wave_resources as R


MAX_CANDIDATES = 5


def run_planner(requests, receipt):
    import author_resident_prepare as resident

    resident.bind_backend(requests[0])
    import author_collect as collector
    import author_contact_pipeline as contact
    import author_mjx_measured_planning as measured

    original_prepare = resident.prepare
    original_mapping = contact.create_mapping_bundle
    original_collect = collector.prepare_collection

    def prepare(request, session):
        limit = request['cgr_planning_limit']
        if type(limit) is not int or not 1 <= limit <= MAX_CANDIDATES:
            raise ValueError('A bounded number of original CGR ranks is required')
        destination = Path(request['output_dir'])
        attempts_root = destination.with_name('cgr_attempts')
        attempts_root.mkdir()
        mapping = None
        original_pool = original_arrays = None
        events = []

        def create_mapping(*args, **kwargs):
            nonlocal mapping
            if mapping is None:
                mapping = original_mapping(*args, **kwargs)
            return mapping

        def prepare_pool(*args, **kwargs):
            nonlocal original_pool, original_arrays
            if original_pool is None:
                original_pool, original_arrays = original_collect(*args, **kwargs)
                B.save(attempts_root/'ORIGINAL_POOL.json', original_pool, exclusive=True)
            prepared = copy.deepcopy(original_pool)
            candidates = prepared['candidates']
            if candidates:
                selected = candidates[rank]
                prepared.update(candidates=[selected], selected_candidate_sha256=selected['candidate_sha256'],
                                selection='Fixed rank from original post-collision CGR order')
                prepared['counts']['top10'] = 1
            return prepared, original_arrays

        for rank in range(limit):
            attempt = attempts_root/f'rank_{rank:02d}'
            with patch.object(collector, 'prepare_collection', prepare_pool), patch.object(contact, 'create_mapping_bundle', create_mapping):
                result = original_prepare(dict(request, output_dir=str(attempt)), session)
            if original_pool is None:
                raise RuntimeError('Missing original post-collision pool')
            candidates = original_pool['candidates']
            digest = candidates[rank]['candidate_sha256'] if candidates else None
            status = result['status']
            if status not in ('ready', 'null_geometry', 'no_candidate') or (status == 'no_candidate') != (digest is None):
                raise RuntimeError(f'Unexpected planning result {status} for rank {rank}')
            if digest:
                selected = B.read(attempt/'pipeline/pool/selected_author_candidate.json')
                if selected['candidate_sha256'] != digest:
                    raise ValueError('Original candidate changed during preparation')
            events.append(dict(rank=rank if digest else None, status=status, candidate_sha256=digest,
                               plan=result['planning_result'], wall_s=result['wall_s']))
            if status != 'null_geometry' or rank+1 == len(candidates):
                break
        destination.symlink_to(attempt, target_is_directory=True)
        selection = dict(status=status, candidate_limit=limit, candidate_count=len(candidates),
                         selected_rank=rank if status == 'ready' else None, attempts=events,
                         shared_pool=B.binding(attempts_root/'ORIGINAL_POOL.json'),
                         observation=B.binding(request['observation']), physics_steps=0)
        B.save(attempts_root/'SELECTION.json', selection, exclusive=True)
        print(dict(scene=destination.parent.name, candidates=len(candidates), plans=len(events) if candidates else 0,
                   selected_rank=selection['selected_rank'], status=status), flush=True)
        return dict(result, cgr_fallback=B.binding(attempts_root/'SELECTION.json'))

    with patch.object(resident, 'prepare', prepare):
        measured.run(requests, receipt)


def planning_batch(root, requests, protocol, pool):
    def run_group(group_number, entries):
        directory = root/f'worker_{group_number}'
        directory.mkdir(parents=True)
        request_file = directory/'REQUESTS.json'
        limit = protocol['cgr_planning_limit']
        B.save(request_file, [dict(req, cgr_planning_limit=limit) for _, req in entries], exclusive=True)
        output = directory/'RESULTS.json'
        command = [sys.executable, '-B', str(Path(__file__).resolve()), 'plan', str(request_file), str(output)]
        B.save(directory/'DISPATCH.json', dict(command=command, requests=B.binding(request_file)), exclusive=True)
        with pool.stage(('prepare',), directory), (directory/'process.log').open('x') as log:
            child = pool.popen(command, cwd=B.ROOT, env=B._environment(protocol), stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
            identity = R.process_identity(child.pid)
            B.save(directory/'PROCESS.json', dict(identity=identity), exclusive=True)
            try:
                code = child.wait(timeout=protocol['stage_timeout_s'])
            except BaseException:
                if R.process_identity(child.pid) == identity:
                    os.killpg(identity['pgid'], signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(identity['pgid'], signal.SIGKILL)
                        child.wait(timeout=10)
                raise
            if code:
                raise RuntimeError(f'Bounded CGR planning failed in {directory}')
        jobs = B.read(output)['jobs']
        return dict(zip((index for index, _ in entries), jobs, strict=True))

    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(run_group, i, requests[i::3]) for i in range(min(3, len(requests)))]
        return {key: row for future in futures for key, row in future.result().items()}


def collect(root):
    import author_mjx_batch_collection_v9 as frozen

    protocol_file = root/'PROTOCOL.json'
    protocol = B.read(protocol_file)
    marker = B.sha(protocol_file)
    if os.environ.get('ANYDEX_MJX_CGR_FALLBACK_ENV') != marker:
        env = B._environment(protocol)
        env.update(PYTHONPATH=os.pathsep.join((str(frozen.WARP), str(frozen.DEPS), env['PYTHONPATH'])),
                   CUDA_VISIBLE_DEVICES=frozen.GPU, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   WARP_CACHE_PATH=str(root/'warp_cache'), ANYDEX_MJX_CGR_FALLBACK_ENV=marker)
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), 'collect', str(root)], env)
    from author_mjx_policy_compare_collection import run

    with patch.object(frozen, 'planning_batch', planning_batch):
        if run(root)['status'] != 'complete':
            raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    worker = commands.add_parser('plan')
    worker.add_argument('requests', type=Path)
    worker.add_argument('output', type=Path)
    controller = commands.add_parser('collect')
    controller.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        run_planner(B.read(args.requests), args.output)
    else:
        collect(args.root.resolve())
