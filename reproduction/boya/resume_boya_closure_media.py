"""Recover unfinished closure-diagnostic videos in isolated replay processes.

This is a media-only view adapter for the immutable original RTX renderer.
It never invokes a physical executor or changes the original run receipts.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / 'reproduction/boya/render_egad_floating_three.py'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_media(directory, original_row):
    receipt = read(directory / 'render_check.json')
    evaluation = read(original_row['evaluation'])
    assert receipt['complete'] and receipt['physics_steps_during_replay'] == 0
    assert receipt['object_id'] == original_row['object_id']
    assert receipt['task_success'] == original_row['task_success']
    assert receipt['source_evaluation_sha256'] == original_row['evaluation_sha256'] == sha(original_row['evaluation'])
    assert receipt['source_trace_sha256'] == evaluation['trace_sha256']
    assert receipt['video_sha256'] == sha(directory / 'hand_only.mp4')
    assert max(receipt['max_render_matrix_errors']) <= 1e-6
    assert receipt['source_visual_meshes'] and receipt['root_and_fingers_from_actual_recorded_native_body_states']
    import imageio_ffmpeg
    frames = imageio_ffmpeg.read_frames(str(directory / 'hand_only.mp4'), pix_fmt='rgb24')
    metadata = next(frames); fps = float(metadata['fps']); count = 0
    assert tuple(metadata['size']) == (1280, 560)
    for frame in frames:
        assert len(frame) == 560 * 1280 * 3
        count += 1
    assert count == 235 and abs(fps - 25.) < 1e-9
    stream = dict(width=1280, height=560, decoded_frames=count, fps=fps, duration_s=count / fps,
                  decoder=imageio_ffmpeg.get_ffmpeg_exe())
    return dict(**receipt, media_directory=str(directory), render_check=ref(directory / 'render_check.json'),
                decoded_video_stream=stream)


def recover(run, stack_path, previous_signal):
    started = time.monotonic()
    original_protocol = read(run / 'protocol.json'); original_results = read(run / 'RESULTS.json')
    assert original_results['status'] == 'complete'
    assert original_results['protocol_sha256'] == sha(run / 'protocol.json')
    assert len(original_results['results']) == 8
    frozen = {str(run / name): sha(run / name) for name in ('protocol.json', 'RESULTS.json', 'PREPARED.json')}
    frozen[str(RENDERER)] = sha(RENDERER)
    recovery = run / 'media_recovery_v3'; delivery = run / 'delivery_videos'
    recovery.mkdir(exist_ok=False); delivery.mkdir(exist_ok=False)
    source_sha = {str(Path(__file__).resolve()): sha(__file__), str(RENDERER): sha(RENDERER)}
    failure = dict(previous_process_pid=2894330, previous_process_signal=previous_signal, previous_process_exit_code=139,
                   signal_provenance='Parent observed original renderer process exit 139; no kill was issued',
                   original_log=ref(run / 'render.log'), original_partial_videos_preserved=True,
                   last_progress='egad_A3_baseline frame 50 of 210',
                   crash_is_replay_only=True, new_physics_steps=0)
    failure['recovery_preparation_zero_process_failures'] = [ref(path) for path in
        (run / 'media_recovery_v1/PREPARATION_FAILURE.json', run / 'media_recovery_v2/PREPARATION_FAILURE.json')
        if path.exists()]
    if stack_path is not None:
        stack = stack_path.read_text()
        assert '_post_process_graph_tick' in stack and 'render_egad_floating_three.py' in stack
        (recovery / 'ORIGINAL_CRASH_STACK.txt').write_bytes(stack_path.read_bytes())
        failure.update(stack_source=ref(stack_path), stack_copy=ref(recovery / 'ORIGINAL_CRASH_STACK.txt'),
                       crash_stack_function='SyntheticData._post_process_graph_tick')
    save(recovery / 'ORIGINAL_FAILURE.json', failure)
    retained, pending = {}, []
    for row in original_results['results']:
        oid = row['object_id']; existing = run / 'videos' / oid
        if (existing / 'render_check.json').exists():
            retained[oid] = verify_media(existing, row)
            continue
        view = recovery / oid; view.mkdir(); (view / 'jobs').mkdir()
        case = next(c for c in original_protocol['cases'] if c['object_id'] == oid)
        protocol = dict(schema='boya_closure_single_case_media_view_v1', media_only=True,
            physics_execution_authorized_by_view=False, cases=[case],
            source_protocol=ref(run / 'protocol.json'), source_results=ref(run / 'RESULTS.json'),
            source_sha256=source_sha,
            boundary='Replay original saved native trace only; this is not an additional grasp or modified physical protocol.')
        save(view / 'protocol.json', protocol)
        save(view / 'RESULTS.json', dict(status='complete', media_only=True,
            protocol_sha256=sha(view / 'protocol.json'), results=[row],
            source_results=ref(run / 'RESULTS.json'), new_physics_steps=0))
        (view / 'jobs' / (oid + '.json')).write_bytes((run / 'jobs' / (oid + '.json')).read_bytes())
        pending.append((row, view))
    assert set(retained) == {'egad_A0_baseline', 'egad_A0_refined'} and len(pending) == 6
    save(recovery / 'DISPATCH.json', dict(media_only=True, concurrency=2, GPU=0,
        source_sha256=source_sha, frozen_original_sha256=frozen,
        retained_cases=list(retained), pending_cases=[row['object_id'] for row, _ in pending],
        attempts_per_pending_case=1, physics_steps=0))
    completed, errors = {}, []

    def one(row, view):
        begin = time.monotonic()
        argv = [sys.executable, '-B', str(RENDERER), '--run', str(view)]
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (view / 'render.log').open('xb') as log:
            process = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=600, check=False)
        attempt = dict(object_id=row['object_id'], argv=argv, returncode=process.returncode,
                       wall_s=time.monotonic() - begin, new_physics_steps=0,
                       renderer_source_sha256=sha(RENDERER), log=ref(view / 'render.log'))
        save(view / 'ATTEMPT.json', attempt)
        if process.returncode:
            raise RuntimeError(f"{row['object_id']}: replay process {process.returncode}; no automatic retry")
        media = verify_media(view / 'videos' / row['object_id'], row)
        media['recovery_attempt'] = attempt
        return media

    for first in range(0, len(pending), 2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(one, row, view): row['object_id'] for row, view in pending[first:first + 2]}
            for future in as_completed(futures):
                oid = futures[future]
                try:
                    completed[oid] = future.result()
                except Exception as error:
                    errors.append(dict(object_id=oid, error=repr(error)))
                save(recovery / 'PROGRESS.json', dict(complete=False, completed=list(completed), errors=errors,
                                                       physics_steps_during_replay=0))
                print('MEDIA_RECOVERY', oid, 'complete' if oid in completed else errors[-1], flush=True)
    assert all(sha(path) == digest for path, digest in frozen.items()), 'Original receipt/renderer changed'
    all_media = retained | completed
    ordered = []
    for row in original_results['results']:
        oid = row['object_id']
        if oid not in all_media:
            continue
        media = all_media[oid]; target = Path(media['media_directory'])
        (delivery / oid).symlink_to(os.path.relpath(target, delivery), target_is_directory=True)
        ordered.append(media)
    report = dict(schema='boya_closure_recovered_media_v1', complete=not errors and len(ordered) == 8,
        cases=ordered, source_protocol=ref(run / 'protocol.json'), source_results=ref(run / 'RESULTS.json'),
        source_sha256=source_sha, original_failure=failure,
        preserved_original_complete_videos=len(retained), recovered_videos=len(completed), errors=errors,
        physics_steps_during_replay=0, new_grasp_executions=0, replay_gpu=0, concurrency=2,
        recovery_wall_s=time.monotonic() - started,
        sum_isolated_process_wall_s=sum(m['recovery_attempt']['wall_s'] for m in completed.values()),
        initial_failed_replay_cost_included_in_recovery_wall_s=False,
        boundary='Two original complete videos plus six isolated trace replays; original partial media and failure logs retained.')
    save(delivery / 'RESULTS.json', report); save(recovery / 'RESULTS.json', report)
    if not report['complete']:
        raise RuntimeError('Some isolated media attempts failed; preserve receipts and do not repeat implicitly')
    print('MEDIA_RECOVERY_COMPLETE', json.dumps(dict(videos=8, physics_steps=0, wall_s=report['recovery_wall_s'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--original-stack', type=Path)
    parser.add_argument('--previous-signal', default='SIGSEGV')
    args = parser.parse_args()
    recover(args.run.resolve(), args.original_stack, args.previous_signal)


if __name__ == '__main__':
    main()
