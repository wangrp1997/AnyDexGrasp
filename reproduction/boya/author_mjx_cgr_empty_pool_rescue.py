"""Plan a zero-extra-depth roll only when the original CGR pool is empty."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

import numpy as np

import author_collection_batch as B
import author_collection_wave_resources as R
import author_mjx_cgr_fallback as primary
from audit_cgr_approach_depth_recovery import candidate_rows


RESCUE_LIMIT = 2
VARIANT = 'preserve_approach_zero_extra'


def selection(job):
    reference = job['cgr_fallback']
    B.verify({'selection': reference})
    result = B.read(reference['path'])
    if result['status'] != job['status']:
        raise ValueError('CGR selection and planner result disagree')
    return result


def original_pool_empty(result):
    empty = result['candidate_count'] == 0
    if empty != (result['status'] == 'no_candidate'):
        raise ValueError('Inconsistent CGR pool count and status')
    return empty


def run_planner(requests, output):
    primary_output = output.with_name('PRIMARY_RESULTS.json')
    primary.run_planner(requests, primary_output)
    jobs = B.read(primary_output)['jobs']
    rescue_requests = []
    rescue_positions = []
    for position, (request, job) in enumerate(zip(requests, jobs, strict=True)):
        if not original_pool_empty(selection(job)):
            continue
        destination = Path(request['output_dir']).parent/'empty_rescue'
        destination.mkdir()
        rescue_requests.append(dict(request, output_dir=str(destination/'contact'), cgr_planning_limit=RESCUE_LIMIT))
        rescue_positions.append(position)

    rescue_output = output.with_name('RESCUE_RESULTS.json')
    if rescue_requests:
        import author_collect as collector
        import author_protocol as author

        original_source = author.collection_candidates
        original_collect = collector.prepare_collection

        def orient(rows, features, **kwargs):
            source = original_source(rows, features, **kwargs)
            original = rows[source['source_indices']]
            raw = np.c_[original[:, :2], original[:, 3:16]]
            source['rows'] = candidate_rows(source['rows'], raw, source['if_flip'])['preserve_approach']
            source['extra_depth_m'] = np.zeros_like(source['extra_depth_m'])
            return source

        def annotate(*args, **kwargs):
            prepared, trace = original_collect(*args, **kwargs)
            candidates = prepared['candidates']
            for candidate in candidates:
                raw = np.asarray(candidate['source_row15_camera'])
                actual = np.asarray(candidate['actual_row15_camera'])
                np.testing.assert_array_equal(actual[[2, 3, 6, 9]], raw[[2, 3, 6, 9]])
                if candidate['extra_depth_m'] != 0:
                    raise ValueError('Extra depth in the zero-extra-depth recovery pool')
                candidate['canonicalization']['source_collection_flip_columns'] = [1, 2]
                candidate['recovery_variant'] = VARIANT
                candidate['candidate_sha256'] = collector._candidate_id(candidate)
            prepared['selected_candidate_sha256'] = candidates[0]['candidate_sha256'] if candidates else None
            return prepared, trace

        with patch.object(author, 'collection_candidates', orient), patch.object(collector, 'prepare_collection', annotate):
            primary.run_planner(rescue_requests, rescue_output)

        for position, replacement in zip(rescue_positions, B.read(rescue_output)['jobs'], strict=True):
            request = requests[position]
            original = jobs[position]
            before, after = selection(original), selection(replacement)
            if before['observation'] != after['observation'] or original['live_state'] != replacement['live_state']:
                raise ValueError('Recovery must use the same photo and measured start')
            contact = Path(request['output_dir'])
            receipt = contact.parent/'EMPTY_RESCUE.json'
            B.save(receipt, dict(variant=VARIANT, original_selection=original['cgr_fallback'],
                recovery_selection=replacement['cgr_fallback'], observation=after['observation'],
                live_state=replacement['live_state'], recovery_status=after['status'], physics_steps=0), exclusive=True)
            if after['status'] == 'ready':
                temporary = contact.with_name('contact_recovered')
                temporary.symlink_to(contact.parent/'empty_rescue/contact', target_is_directory=True)
                temporary.replace(contact)
            jobs[position] = dict(replacement, empty_pool_rescue=B.binding(receipt))

    B.save(output, dict(status='complete', jobs=jobs, primary=B.binding(primary_output),
        recovery=B.binding(rescue_output) if rescue_requests else None, physics_steps=0), exclusive=True)


def planning_batch(root, requests, protocol, pool):
    if protocol['rescue_planning_limit'] != RESCUE_LIMIT:
        raise ValueError('Recovery rank limit differs from the frozen worker')

    def run_group(number, entries):
        directory = root/f'worker_{number}'
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
                raise RuntimeError(f'Empty-pool recovery planning failed in {directory}')
        return dict(zip((index for index, _ in entries), B.read(output)['jobs'], strict=True))

    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(run_group, i, requests[i::3]) for i in range(min(3, len(requests)))]
        return {key: row for future in futures for key, row in future.result().items()}


def collect(root):
    import author_mjx_batch_collection_v9 as frozen

    protocol_file = root/'PROTOCOL.json'
    protocol = B.read(protocol_file)
    marker = B.sha(protocol_file)
    if os.environ.get('ANYDEX_MJX_EMPTY_RESCUE_ENV') != marker:
        env = B._environment(protocol)
        env.update(PYTHONPATH=os.pathsep.join((str(frozen.WARP), str(frozen.DEPS), env['PYTHONPATH'])),
                   CUDA_VISIBLE_DEVICES=frozen.GPU, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   WARP_CACHE_PATH=str(root/'warp_cache'), ANYDEX_MJX_EMPTY_RESCUE_ENV=marker)
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
