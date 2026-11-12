"""Publish bound saved-trace videos for the finite Boya candidate-pool diagnostic.

Historical videos are reused, retaining their original execution IDs. Each new
execution gets one isolated invocation of the immutable RTX replay renderer.
This module contains no physics executor and never retries failed replays.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / 'reproduction/boya/render_egad_floating_three.py'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_id(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]+', value), 'Unsafe case ID')
    return value


def inputs(run):
    results = read(run / 'RESULTS.json')
    require(results['status'] == 'complete', 'Complete physical results required')
    require(results['protocol_sha256'] == sha(run / 'protocol.json'), 'Protocol binding changed')
    rows = results['results']
    require(len({r['candidate_id'] for r in rows}) == len(rows), 'Duplicate candidate IDs')
    for row in rows:
        cid = safe_id(row['candidate_id'])
        require(row['object_id'] == cid, 'New view ID must be candidate ID')
        safe_id(row['source_object_id'])
        require(row['kind'] in ('reused', 'new'), 'Unknown execution origin')
        require(sha(row['evaluation']) == row['evaluation_sha256'], 'Evaluation binding changed')
        evaluation = read(row['evaluation'])
        require(evaluation['complete'], 'Incomplete evaluation')
        require(evaluation['task_success'] == row['task_success'], 'Outcome binding changed')
        require(evaluation['total_physics_steps'] == row['physical_steps'], 'Step count binding changed')
        trace = Path(row['evaluation']).parent / 'trace.npz'
        require(sha(trace) == evaluation['trace_sha256'], 'Trace binding changed')
        job = read(row['job_path'])
        require(job['object_id'] == evaluation['object_id'], 'Job/evaluation execution IDs differ')
        if row['kind'] == 'new':
            require(job['object_id'] == cid, 'New execution ID differs from candidate ID')
    return rows


def encoded_frame_expectation(directory, receipt):
    """Mirror the immutable encoder's GIF-duration rounding, including early stops."""
    from PIL import Image, ImageSequence
    fps = float(receipt['fps'])
    nominal = int(receipt['frames']) + round(float(receipt['final_pause_s']) * fps)
    require(nominal > 0 and fps == 25., 'Unexpected replay frame clock')
    gif = directory / 'hand_only.gif'
    require(gif.is_file(), 'Original encoded GIF missing')
    with Image.open(gif) as image:
        durations = [int(frame.info.get('duration', 80)) for frame in ImageSequence.Iterator(image)]
    actual = sum(max(1, round(duration * fps / 1000)) for duration in durations)
    require(abs(sum(durations) / 1000 - nominal / fps) <= .010001, 'GIF duration differs from renderer clock')
    return dict(renderer_motion_frames=int(receipt['frames']), nominal_total_frames=nominal,
                gif_frames=len(durations), gif_duration_ms=sum(durations), encoder_expected_frames=actual,
                rule='sum(max(1, round(GIF frame duration_ms * fps / 1000))); original encoder rule')


def verify_media(directory, row, original_id):
    directory = Path(directory).resolve()
    receipt = read(directory / 'render_check.json')
    evaluation = read(row['evaluation'])
    require(receipt['complete'] and receipt['physics_steps_during_replay'] == 0, 'Incomplete/nonzero-step replay')
    require(receipt['object_id'] == original_id == evaluation['object_id'], 'Original media execution ID changed')
    require(receipt['task_success'] == row['task_success'], 'Video outcome differs')
    require(receipt['source_evaluation_sha256'] == row['evaluation_sha256'] == sha(row['evaluation']),
            'Video/evaluation binding changed')
    trace = Path(row['evaluation']).parent / 'trace.npz'
    require(receipt['source_trace_sha256'] == evaluation['trace_sha256'] == sha(trace), 'Video/trace binding changed')
    require(receipt['video_sha256'] == sha(directory / 'hand_only.mp4'), 'Video SHA changed')
    require(max(receipt['max_render_matrix_errors']) <= 1e-6, 'Replay matrix mismatch')
    require(receipt['source_visual_meshes'] and receipt['root_and_fingers_from_actual_recorded_native_body_states'],
            'Source visual/state binding missing')
    require(min(receipt['pixel_motion'].values()) > .1, 'No visible recorded motion')
    for path, digest in receipt['visual_mesh_sha256'].items():
        require(sha(path) == digest, 'Source visual asset changed: ' + path)
    expectation = encoded_frame_expectation(directory, receipt)
    import imageio_ffmpeg
    frames = imageio_ffmpeg.read_frames(str(directory / 'hand_only.mp4'), pix_fmt='rgb24')
    metadata = next(frames)
    count = 0
    require(tuple(metadata['size']) == (1280, 560), 'Wrong decoded dimensions')
    require(float(metadata['fps']) == 25., 'Wrong decoded FPS')
    for frame in frames:
        require(len(frame) == 1280 * 560 * 3, 'Truncated decoded frame')
        count += 1
    require(count == expectation['encoder_expected_frames'], 'Wrong fully decoded frame count')
    return dict(candidate_id=row['candidate_id'], source_object_id=row['source_object_id'],
                original_execution_id=original_id, kind=row['kind'], task_success=row['task_success'],
                physical_steps=row['physical_steps'], media_directory=str(directory),
                evaluation=ref(row['evaluation']), trace=ref(trace), source_job=ref(row['job_path']),
                video=ref(directory / 'hand_only.mp4'), render_check=ref(directory / 'render_check.json'),
                original_render_receipt=receipt, frame_expectation=expectation,
                decoded_video_stream=dict(width=1280, height=560, decoded_frames=count, fps=25.,
                    duration_s=count / 25., decoder=imageio_ffmpeg.get_ffmpeg_exe()),
                physics_steps_during_replay=0,
                alias_boundary='candidate_id labels this finite pool; original_execution_id and saved media remain unchanged')


def frozen_sources(run):
    return {str(path.resolve()): sha(path) for path in
            (run / 'protocol.json', run / 'RESULTS.json', Path(__file__), RENDERER)}


def check_frozen(frozen):
    for path, digest in frozen.items():
        require(sha(path) == digest, 'Frozen media input/source changed: ' + path)


def prepare(run, media_name, workers):
    rows = inputs(run)
    media = run / safe_id(media_name)
    require(not media.exists(), 'Media preparation already exists; use render/publish without overwriting')
    media.mkdir()
    frozen = frozen_sources(run)
    reused, pending = [], []
    for row in rows:
        cid = row['candidate_id']
        job = Path(row['job_path'])
        original_id = read(job)['object_id']
        if row['kind'] == 'reused':
            directory = job.parent.parent / 'videos' / original_id
            reused.append(verify_media(directory, row, original_id))
            continue
        view = media / cid
        view.mkdir()
        (view / 'jobs').mkdir()
        save(view / 'protocol.json', dict(schema='boya_fixed_pool_single_case_media_view_v1',
            media_only=True, physics_execution_authorized_by_view=False,
            cases=[dict(object_id=cid, source_object_id=row['source_object_id'], old_execution=None)],
            source_protocol=ref(run / 'protocol.json'), source_results=ref(run / 'RESULTS.json'),
            source_job=ref(job), source_sha256={str(RENDERER): sha(RENDERER), str(Path(__file__).resolve()): sha(__file__)},
            boundary='Render this saved native trace only. No additional grasp or integration is authorized by this view.'))
        save(view / 'RESULTS.json', dict(status='complete', media_only=True,
            protocol_sha256=sha(view / 'protocol.json'), results=[row],
            source_results=ref(run / 'RESULTS.json'), new_physics_steps=0))
        (view / 'jobs' / (cid + '.json')).write_bytes(job.read_bytes())
        pending.append(dict(candidate_id=cid, view=str(view), protocol=ref(view / 'protocol.json'),
                            results=ref(view / 'RESULTS.json'), copied_job=ref(view / 'jobs' / (cid + '.json'))))
    check_frozen(frozen)
    dispatch = dict(schema='boya_fixed_pool_media_dispatch_v1', media_only=True,
                    source_run=str(run), frozen_input_sha256=frozen,
                    retained_cases=reused, pending_cases=pending, workers=workers, replay_gpu=0,
                    attempts_per_pending_case=1, new_grasp_executions=0, physics_steps_during_replay=0)
    save(media / 'DISPATCH.json', dispatch)
    return dispatch


def render_one(row, item, timeout_s):
    view = Path(item['view'])
    for key in ('protocol', 'results', 'copied_job'):
        require(sha(item[key]['path']) == item[key]['sha256'], 'Media view changed')
    require(not (view / 'ATTEMPT.json').exists() and not (view / 'render.log').exists(),
            'Replay was already attempted; preserve it and use a separately authorized recovery view')
    argv = [sys.executable, '-B', str(RENDERER), '--run', str(view)]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    begin = time.monotonic()
    attempt = dict(candidate_id=row['candidate_id'], argv=argv, new_grasp_executions=0,
                   physics_steps_during_replay=0, renderer_source_sha256=sha(RENDERER))
    try:
        with (view / 'render.log').open('xb') as log:
            process = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=timeout_s, check=False)
        attempt['returncode'] = process.returncode
    except Exception as error:
        attempt.update(returncode=None, error=repr(error))
    attempt.update(wall_s=time.monotonic() - begin, log=ref(view / 'render.log'))
    save(view / 'ATTEMPT.json', attempt)
    require(attempt['returncode'] == 0, f"{row['candidate_id']}: replay failed; no automatic retry")
    report = verify_media(view / 'videos' / row['candidate_id'], row, row['candidate_id'])
    report['attempt'] = ref(view / 'ATTEMPT.json')
    save(view / 'VERIFIED.json', report)
    return report


def render(run, media_name, timeout_s):
    rows = inputs(run)
    media = run / media_name
    dispatch = read(media / 'DISPATCH.json')
    check_frozen(dispatch['frozen_input_sha256'])
    by_id = {row['candidate_id']: row for row in rows}
    completed, errors, pending = {}, [], []
    started = time.monotonic()
    for item in dispatch['pending_cases']:
        cid = item['candidate_id']
        view = Path(item['view'])
        if (view / 'VERIFIED.json').exists():
            completed[cid] = read(view / 'VERIFIED.json')
        elif (view / 'ATTEMPT.json').exists() or (view / 'render.log').exists():
            errors.append(dict(candidate_id=cid, error='Prior replay attempt retained; no implicit retry'))
        else:
            pending.append(item)
    # A fresh process per candidate avoids the known multi-case SyntheticData crash.
    with ThreadPoolExecutor(max_workers=dispatch['workers']) as pool:
        futures = {pool.submit(render_one, by_id[item['candidate_id']], item, timeout_s): item['candidate_id']
                   for item in pending}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                completed[cid] = future.result()
            except Exception as error:
                errors.append(dict(candidate_id=cid, error=repr(error)))
            save(media / 'PROGRESS.json', dict(completed=list(completed), errors=errors,
                pending_count=len(dispatch['pending_cases']) - len(completed) - len(errors),
                physics_steps_during_replay=0, new_grasp_executions=0))
            print('FIXED_POOL_MEDIA', cid, 'complete' if cid in completed else errors[-1], flush=True)
    check_frozen(dispatch['frozen_input_sha256'])
    report = dict(complete=not errors, completed=list(completed), errors=errors,
                  wall_s=time.monotonic() - started, workers=dispatch['workers'],
                  physics_steps_during_replay=0, new_grasp_executions=0)
    # Retain every render invocation's timing, including incomplete/interrupted recovery calls.
    invocations = sorted(media.glob('RENDER_INVOCATION_*.json'))
    save(media / f'RENDER_INVOCATION_{len(invocations) + 1:03d}.json', report)
    return report


def score_map(scores):
    """Only display explicitly saved pre-physics ranks; never rank using outcomes."""
    if scores is None:
        return {}
    rows = scores.get('candidates', scores.get('scores', []))
    mapped = {row['candidate_id']: dict(row) for row in rows} if isinstance(rows, list) else rows
    for obj in scores.get('objects', []):
        for recipe, ranking in obj['rankings'].items():
            for rank, cid in enumerate(ranking, 1):
                require(cid in mapped, 'Ranked candidate lacks frozen scores')
                mapped[cid].setdefault('object_local_ranks_1based', {})[recipe] = rank
    return mapped


def write_index(run, rows, media_rows, scores_path):
    esc = html.escape
    score_rows = score_map(read(scores_path)) if scores_path and scores_path.exists() else {}
    by_id = {row['candidate_id']: row for row in media_rows}
    sections = ['<!doctype html><meta charset="utf-8"><title>Boya 固定候选池诊断</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:28px auto;padding:0 18px;background:#eef1f5;color:#172d3f}'
        'section,details{background:white;padding:16px;margin:14px 0;border-radius:8px}summary{cursor:pointer}'
        'video{display:block;width:100%;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}'
        'a{color:#075b96}.ok{color:#18713b}.fail{color:#a12d27}small{color:#536572}</style>',
        '<h1>Boya 固定候选池逐个执行诊断</h1>',
        '<p>三个已知物体的冻结候选池；每个姿态使用一次兼容历史执行或本轮一次新执行。'
        '以下展示实际保存轨迹，回放新增物理步数为 0。</p>',
        '<p>池中“存在成功姿态”只回答这批候选与固定控制器能否抓起，不能称为部署成功率、'
        '独立外部泛化或算法上限。原外部 1/12 与新六件 3/6 成绩保持不变。</p>',
        '<p>视频字幕 Object rise 是物体参考点位移，不等于底面离桌；成功标签来自原执行评价。'
        '提前停止的失败视频按实际保存时长展示。历史视频保留原执行 ID，下面另列候选 ID。</p>',
        '<p><a href="delivery_videos/RESULTS.json">媒体绑定与完整解码报告</a> · '
        '<a href="RESULTS.json">原始执行结果</a></p>']
    for oid in sorted({row['source_object_id'] for row in rows}):
        group = [row for row in rows if row['source_object_id'] == oid]
        sections.append(f'<section><h2>{esc(oid)} · {len(group)} 个冻结候选</h2></section>')
        for row in group:
            cid = row['candidate_id']; item = by_id.get(cid)
            label = '成功' if row['task_success'] else '失败'
            origin = '复用历史执行' if row['kind'] == 'reused' else '本轮新执行'
            color = 'ok' if row['task_success'] else 'fail'
            sections.append(f'<details><summary><b class="{color}">{label}</b> · {esc(cid)} · {origin}</summary>')
            sections.append(f'<p>实际物理步数 {int(row["physical_steps"])}；{origin}。</p>')
            if cid in score_rows:
                ranks = score_rows[cid].get('object_local_ranks_1based', {})
                sections.append('<p>冻结对象内名次：' + ' · '.join(
                    f'{esc(name)} #{ranks[recipe]}' for recipe, name in
                    (('original_mse', '原 MSE'), ('preactivation', 'Preactivation'), ('cgr', 'CGR'))
                    if recipe in ranks) + '</p>')
            if item:
                sections.append('<p>视频原执行 ID：<code>' + esc(item['original_execution_id']) + '</code></p>')
                sections.append(f'<video controls preload="none" src="delivery_videos/{esc(cid)}/hand_only.mp4"></video>')
            else:
                sections.append('<p>媒体尚未通过验证；未重跑物理。</p>')
            if cid in score_rows:
                sections.append('<details><summary>冻结评分与名次（不使用执行结果重新排序）</summary><pre>' +
                                esc(json.dumps(score_rows[cid], indent=2, ensure_ascii=False)) + '</pre></details>')
            evaluation_url = esc(os.path.relpath(row['evaluation'], run))
            sections.append(f'<p><a href="{evaluation_url}">原执行评价</a></p></details>')
    (run / 'index.html').write_text('\n'.join(sections) + '\n')


def publish(run, media_name, scores_path=None):
    rows = inputs(run); media = run / media_name
    dispatch = read(media / 'DISPATCH.json')
    check_frozen(dispatch['frozen_input_sha256'])
    by_id = {row['candidate_id']: row for row in rows}
    completed, errors = {}, []
    for original in dispatch['retained_cases']:
        cid = original['candidate_id']
        completed[cid] = verify_media(original['media_directory'], by_id[cid], original['original_execution_id'])
    for item in dispatch['pending_cases']:
        cid = item['candidate_id']; view = Path(item['view'])
        if not (view / 'VERIFIED.json').exists():
            errors.append(dict(candidate_id=cid, error='No verified completed replay; physical trace preserved'))
            continue
        completed[cid] = verify_media(view / 'videos' / cid, by_id[cid], cid)
        completed[cid]['attempt'] = ref(view / 'ATTEMPT.json')
    delivery = run / 'delivery_videos'; delivery.mkdir(exist_ok=True)
    for cid, item in completed.items():
        target = Path(item['media_directory']); link = delivery / cid
        if link.exists() or link.is_symlink():
            require(link.is_symlink() and link.resolve() == target, 'Delivery path already points elsewhere')
        else:
            link.symlink_to(os.path.relpath(target, delivery), target_is_directory=True)
    ordered = [completed[row['candidate_id']] for row in rows if row['candidate_id'] in completed]
    invocation_files = sorted(media.glob('RENDER_INVOCATION_*.json'))
    attempts = [read(Path(item['view']) / 'ATTEMPT.json') for item in dispatch['pending_cases']
                if (Path(item['view']) / 'ATTEMPT.json').exists()]
    report = dict(schema='boya_fixed_pool_media_delivery_v1', complete=not errors and len(ordered) == len(rows),
        cases=ordered, expected_cases=len(rows), reused_videos=sum(item['kind'] == 'reused' for item in ordered),
        new_replay_videos=sum(item['kind'] == 'new' for item in ordered), errors=errors,
        source_protocol=ref(run / 'protocol.json'), source_results=ref(run / 'RESULTS.json'),
        dispatch=ref(media / 'DISPATCH.json'), render_invocations=[ref(path) for path in invocation_files],
        render_invocation_wall_s=sum(read(path)['wall_s'] for path in invocation_files),
        sum_attempt_process_wall_s=sum(attempt['wall_s'] for attempt in attempts),
        failed_replay_attempts=[attempt for attempt in attempts if attempt['returncode'] != 0],
        physics_steps_during_replay=0, new_grasp_executions=0, replay_gpu=0, workers=dispatch['workers'],
        boundary='Finite known-object pool, fixed controller, saved native traces; oracle-any is not deployment success rate.')
    if scores_path and scores_path.exists():
        report['source_prephysics_scores'] = ref(scores_path)
    check_frozen(dispatch['frozen_input_sha256'])
    save(delivery / 'RESULTS.json', report)
    write_index(run, rows, ordered, scores_path)
    print('FIXED_POOL_MEDIA_DELIVERY', json.dumps(dict(complete=report['complete'], videos=len(ordered),
          reused=report['reused_videos'], new=report['new_replay_videos'], physics_steps=0)), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--mode', choices=('prepare', 'render', 'publish', 'all'), default='all')
    parser.add_argument('--media-name', default='media_v1')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--timeout-s', type=float, default=600.)
    parser.add_argument('--scores', type=Path)
    args = parser.parse_args(); run = args.run.resolve(); safe_id(args.media_name)
    scores = args.scores or run / 'SCORES_PREPHYSICS.json'
    if args.mode in ('prepare', 'all'):
        prepare(run, args.media_name, args.workers)
    if args.mode in ('render', 'all'):
        render(run, args.media_name, args.timeout_s)
    if args.mode in ('publish', 'all'):
        report = publish(run, args.media_name, scores)
        require(report['complete'], 'Media incomplete; failures retained, no physics or replay implicitly repeated')


if __name__ == '__main__':
    main()
