"""Shared RGB-D, frozen rankings, and copied integration state for each pair."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

import author_collection_batch as B
import author_collection_wave_resources as R
from author_expansion_progress import Progress
from author_hour_audit import AuditPool, combine, snapshot
import author_mjx_batch_collection_v9 as F


STATE = ('qpos', 'qvel', 'act', 'history', 'qacc_warmstart', 'ctrl', 'qfrc_applied',
         'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat', 'userdata', 'tree_asleep', 'time')


def integration_state(batch):
    with batch.wp.ScopedStream(batch.core.stream):
        return {name: getattr(batch.gpu, name).numpy().copy() for name in STATE}


def copy_lanes(batch, pairs, before):
    """Keep graph pointers and native time intact, copying only receiver states."""
    expected = {key: value.copy() for key, value in before.items()}
    donors = {donor for donor, _ in pairs}
    receivers = {receiver for _, receiver in pairs}
    if len(receivers) != len(pairs) or donors & receivers:
        raise ValueError('Independent lane pairs required')
    for donor, receiver in pairs:
        if batch.views[donor].active != batch.views[receiver].active:
            raise ValueError('Paired lanes must have the same active bodies')
        np.testing.assert_array_equal(before['time'][donor], before['time'][receiver])
        for key in STATE:
            expected[key][receiver] = before[key][donor]
    with batch.wp.ScopedStream(batch.core.stream):
        for key, value in expected.items():
            target = getattr(batch.gpu, key)
            if target.size and not np.array_equal(value, before[key]):
                pointer = target.ptr
                # Existing Warp array, with the same dtype and layout used by the graph.
                target.assign(np.ascontiguousarray(value))
                if target.ptr != pointer:
                    raise ValueError('State copy changed native graph addresses')
        batch.wp.synchronize_stream(batch.core.stream)
    for key, value in integration_state(batch).items():
        np.testing.assert_array_equal(value, expected[key], err_msg=key)
    for _, receiver in pairs:
        batch._sync_lane(receiver, expected['qpos'][receiver], expected['qvel'][receiver], expected['time'][receiver])
    return expected


def plan_pairs(output, groups, protocol, resources):
    if not groups:
        return {}

    def job(number, items):
        directory = output/f'worker_{number}'
        directory.mkdir(parents=True)
        requests = [request for pair in items for _, request in pair]
        request_file = directory/'REQUESTS.json'
        B.save(request_file, requests, exclusive=True)
        command = [sys.executable, '-B', str(B.HERE/'author_mjx_shared_pair_worker.py'),
                   '--requests', str(request_file), '--output', str(directory/'RESULTS.json')]
        B.save(directory/'DISPATCH.json', dict(command=command, requests=B.binding(request_file)), exclusive=True)
        with resources.stage(('prepare',), directory), (directory/'process.log').open('x') as stream:
            process = resources.popen(command, cwd=B.ROOT, env=B._environment(protocol),
                                      stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            identity = R.process_identity(process.pid)
            B.save(directory/'PROCESS.json', dict(identity=identity), exclusive=True)
            try:
                code = process.wait(timeout=protocol['stage_timeout_s'])
            except BaseException:
                if R.process_identity(process.pid) == identity:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                raise
            if code:
                raise RuntimeError(f'Paired planning failed in {directory}')
        results = B.read(directory/'RESULTS.json')['jobs']
        return dict(zip((index for pair in items for index, _ in pair), results, strict=True))

    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = [executor.submit(job, i, groups[i::3]) for i in range(min(3, len(groups)))]
        return {key: value for job in jobs for key, value in job.result().items()}


def wave(root, p, batch, engine, resources, progress, home, specs, number, cost):
    from author_contact_pipeline import audit_mapping
    from author_live_world_contract import bind_live_plan
    from author_mjx_native_batch import photograph

    start = batch.ticks
    assigned = {lane: (spec, B.read(spec['scene']['path'])) for lane, spec in enumerate(specs)}
    segment = root/'journals'/f'wave_{number:03d}.jsonl'
    segment.parent.mkdir(exist_ok=True)
    pair_info = []
    with segment.open('x', buffering=1) as journal:
        batch.journal = journal
        progress.stage('wave_started', wave=number, indices=[s['index'] for s in specs])
        B.save(root/'PROGRESS.json', dict(phase='settle', wave=number, global_tick=batch.ticks, cost=cost))
        stable, last, ticks = F.settle(batch, assigned, root, p, home)
        lanes = []
        for lane in range(0, len(specs), 2):
            first, second = specs[lane:lane+2]
            if (first['pair_index'] != second['pair_index'] or first['scene'] != second['scene']
                    or {first['policy'], second['policy']} != {'cgr', 'head'}):
                raise ValueError('Broken pair assignment')
            if stable[lane]['pass_'] and stable[lane+1]['pass_']:
                lanes.append((lane, lane+1))
            else:
                for item in (first, second):
                    F.write_record(root, item, terminal='pair_initialization_unstable')
        captured = integration_state(batch)
        cloned = copy_lanes(batch, lanes, captured)
        requests = []
        for donor, receiver in lanes:
            spec, scene = assigned[donor]
            pair_root = root/'pairs'/f"{spec['pair_index']:03d}"
            pair_root.mkdir(parents=True)
            a, b = [root/'scenes'/f"{specs[i]['index']:02d}" for i in (donor, receiver)]
            photograph(batch, donor, scene, a/'capture', last[donor], stable[donor]['stability'], ticks)
            engine.forward(a/'capture/observation.json', a/'representation')
            cost['CGR_forwards'] += 1
            # Audit both old settling traces while retaining one set of capture bytes.
            for name in ('observed_scene.json', 'observation.json', 'LIVE_STATE.json', 'RAY_CHECK.json',
                         'native_rgbd.npz', 'color.png', 'depth.png', 'meta.mat', 'RESULTS.json'):
                (b/'capture'/name).symlink_to(a/'capture'/name)
            (b/'representation').symlink_to(a/'representation', target_is_directory=True)
            state_path = pair_root/'INTEGRATION_STATE.npz'
            np.savez_compressed(state_path, **{k: v[[donor, receiver]] for k, v in cloned.items()})
            observation = a/'capture/observation.json'
            B.save(pair_root/'SHARED_START.json', dict(donor_lane=donor, receiver_lane=receiver,
                global_tick=batch.ticks, indices=[specs[i]['index'] for i in (donor, receiver)],
                state=B.binding(state_path), observation=B.binding(observation),
                measured_scene=B.binding(a/'capture/observed_scene.json'), live_state=B.binding(a/'capture/LIVE_STATE.json'),
                representation=B.binding(a/'representation/RESULTS.json'), copied_integration_fields=list(STATE),
                native_time_preserved=True, integrations_during_copy=0), exclusive=True)
            domain = pair_root/'DOMAIN.json'
            B.save(domain, p['domain'], exclusive=True)
            requests.append([(specs[lane]['index'], dict(scene=str(a/'capture/observed_scene.json'),
                observation=str(observation), representation_dir=str(a/'representation'), seed=p['seed'],
                output_dir=str(path/'contact'), domain=str(domain), calibration=p['inputs']['calibration']['path'],
                cloud_manifest=p['inputs']['cloud_manifest']['path'], backend=p['inputs']['backend']['path'],
                pair_root=str(pair_root), comparison_head=p['inputs']['comparison_head'],
                comparison_policy=specs[lane]['policy'])) for lane, path in ((donor, a), (receiver, b))])
            pair_info.append((donor, receiver, pair_root))
        B.save(root/'PROGRESS.json', dict(phase='planning', wave=number, global_tick=batch.ticks, cost=cost))
        plans = plan_pairs(root/'planning'/f'{number:03d}', requests, p, resources)
        cost['planning_requests'] += len(plans)
        for key, value in integration_state(batch).items():
            np.testing.assert_array_equal(value, cloned[key], err_msg=f'{key} changed during ranking/planning')
        ready = {}
        for donor, receiver, pair_root in pair_info:
            ranking = B.read(pair_root/'RANKING.json')
            disagree = ranking['disagreement']
            allowed = disagree and cost['disagreement_pairs'] < p['target_pairs']
            if allowed:
                cost['disagreement_pairs'] += 1
            B.save(pair_root/'ADMISSION.json', dict(disagreement=disagree, admitted=allowed,
                selected_by='Fixed draw order with differing ranks, before reading arm-plan results',
                start=B.binding(pair_root/'SHARED_START.json'), ranking=B.binding(pair_root/'RANKING.json')), exclusive=True)
            for lane in (donor, receiver):
                spec = specs[lane]
                directory = root/'scenes'/f"{spec['index']:02d}"
                if not allowed:
                    terminal = 'pair_capacity_after_target' if disagree else 'pair_same_choice' if ranking['candidate_count'] else 'pair_empty_pool'
                    F.write_record(root, spec, terminal=terminal)
                    continue
                plan = plans[spec['index']]
                if plan['status'] != 'ready':
                    F.write_record(root, spec, terminal='planning_'+plan['status'])
                    continue
                bind_live_plan(directory/'contact/pipeline', directory/'capture/LIVE_STATE.json', directory/'live')
                audit_mapping(directory/'live')
                bound = B.read(directory/'live/READY.json')
                chosen = B.read(bound['candidate']['path'])['candidate_sha256']
                if chosen != ranking['hashes'][ranking[spec['policy']+'_rank']]:
                    raise ValueError('Actual plan candidate differs from shared ranking')
                with np.load(bound['commands']['path'], allow_pickle=False) as arrays:
                    commands = {key: arrays[key].copy() for key in arrays.files}
                if len(commands['phase']) > p['max_execution_controls']:
                    raise ValueError('Execution exceeds original command budget')
                ready[lane] = spec, commands
                progress.stage('grasp_started', index=spec['index'], wave=number)
        cost['grasp_attempts'] += len(ready)
        B.save(root/'PROGRESS.json', dict(phase='motion', wave=number, global_tick=batch.ticks, cost=cost))
        results = F.motion(batch, ready, root, p, home) if ready else {}
        for lane, (spec, _) in assigned.items():
            directory = root/'scenes'/f"{spec['index']:02d}"
            batch.views[lane].phase = 'boundary'
            if batch.views[lane].active:
                batch.exchange(lane, dict(scene_id='external_clear_between_rounds', objects=[]))
            B.save(directory/'ROUND.json', dict(index=spec['index'], lane=lane, motion=results.get(lane),
                record=B.read(directory/'RECORD.json')), exclusive=True)
    batch.journal = None
    if any(not r['physical_return']['pass_'] for r in results.values()):
        raise RuntimeError('Original physical return failed')
    return snapshot(root, p, batch, start, segment, specs, number)


def run(root):
    import author_resident_prepare as resident

    p = B.read(root/'PROTOCOL.json')
    B.verify(p['sources']); B.verify(p['inputs'])
    resident.bind_backend(dict(backend=p['inputs']['backend']['path']))
    from author_compiled_kinematics import CompiledKinematics
    from author_mjx_native_batch import Batch
    from author_mujoco_representation import Stream

    template = B.read(p['inputs']['template:0']['path'])
    plan = B.read(template['plan']['path'])
    with np.load(template['commands']['path'], allow_pickle=False) as z:
        names, home = z['joint_names'].tolist(), z['initial_position'].copy()
    scenes = [B.read(spec['scene']['path']) for spec in p['rows'][::2]]
    lock = B.ROOT/'outputs/author_collection_gpu0.lock'
    began = time.monotonic()
    cost = dict(CGR_forwards=0, planning_requests=0, grasp_attempts=0, disagreement_pairs=0)
    error = batch = engine = auditor = None
    reason = 'draw_capacity_exhausted'
    with R.lease(root/'DRIVER.lock'), R.lease(lock) as lease, R.active_run(
            root, lock.with_suffix('.wave_active.json'), B.binding(root/'PROTOCOL.json')):
        B.save(root/'RUN_STARTED.json', dict(process=R.process_identity(os.getpid()), started_unix_s=time.time()), exclusive=True)
        resources = R.ResourcePool(root/'resource_slots', lease, B.binding(root/'PROTOCOL.json'))
        progress = Progress(root, len(p['rows']))
        original_save = B.save

        def save(path, value, **kwargs):
            original_save(path, value, **kwargs)
            path = Path(path)
            if path.parent.parent != root/'scenes':
                return
            index = int(path.parent.name)
            if path.name == 'RECORD.json':
                receipt = value['bindings'].get('execution')
                progress.record(index, value, execution_complete=bool(receipt and B.read(receipt['path'])['status'] == 'complete'))
            elif path.name == 'RETURN.json':
                progress.returned(index, value['pass_'])

        with patch.object(B, 'save', save):
            try:
                with CompiledKinematics().installed():
                    batch = Batch(plan, names, home, scenes, count=p['environments'], max_ticks=p['max_global_control_ticks'])
                    batch.set_cost_journal(root/'NATIVE_CALLS.jsonl')
                    (root/'MODEL.xml').write_text(batch.base.spec.to_xml())
                    B.save(root/'MODEL_RECEIPT.json', dict(model=B.binding(root/'MODEL.xml'),
                        GPU_CCD=batch.compatibility, camera_count=1), exclusive=True)
                    engine = Stream(SimpleNamespace(checkpoint=Path(p['inputs']['checkpoint']['path']),
                        native_deps=Path(p['native_deps']), device='cuda:0', seed=p['seed'], num_seed=2048,
                        voxel_size=.005, depth_min=.2, depth_max=.65))
                    auditor = AuditPool(p, resources, progress)
                    for number, start in enumerate(range(0, len(p['rows']), p['environments'])):
                        auditor.raise_if_failed()
                        if shutil.disk_usage(root).free < p['minimum_free_disk_bytes']:
                            reason = 'storage_headroom_guard'
                            break
                        specs = p['rows'][start:start+p['environments']]
                        proof = wave(root, p, batch, engine, resources, progress, home, specs, number, cost)
                        auditor.submit(proof, number)
                        print(dict(wave=number, tick=batch.ticks, **cost), flush=True)
                        B.save(root/'WAVE_PROGRESS.json', dict(wave=number, tick=batch.ticks, **cost))
                        if cost['disagreement_pairs'] >= p['target_pairs']:
                            reason = 'paired_disagreement_target_reached'
                            break
            except BaseException:
                reason = 'controller_error'
                error = traceback.format_exc()
            finally:
                if batch is not None:
                    try:
                        batch.close()
                    except BaseException:
                        error = (error or '') + traceback.format_exc()
                if auditor is not None:
                    failures = auditor.close()
                    if failures:
                        error = (error or '') + repr(failures)
                for spec in p['rows']:
                    directory = root/'scenes'/f"{spec['index']:02d}"
                    directory.mkdir(parents=True, exist_ok=True)
                    if not (directory/'RECORD.json').exists():
                        F.write_record(root, spec, terminal='pair_not_started', error=error)
                result = dict(status='failed' if error else 'complete', error=error, stop_reason=reason, cost=cost,
                    world=batch.receipt() if batch is not None else None, records=[B.read(root/'scenes'/f"{s['index']:02d}"/'RECORD.json') for s in p['rows']],
                    representation=engine.receipt() if engine is not None else None, wall_s=time.monotonic()-began,
                    video_recording=False, exclude_training=True, training_updates=0)
                B.verify(p['sources']); B.verify(p['inputs'])
                B.save(root/'RESULTS.json', result, exclusive=True)
                if error is None:
                    combine(root, p, result['world'], result)
                progress.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    p = B.read(root/'PROTOCOL.json')
    marker = B.sha(root/'PROTOCOL.json')
    if os.environ.get('ANYDEX_MJX_SHARED_PAIR_ENV') != marker:
        env = B._environment(p)
        env.update(PYTHONPATH=os.pathsep.join((str(F.WARP), str(F.DEPS), env['PYTHONPATH'])),
                   CUDA_VISIBLE_DEVICES=F.GPU, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   WARP_CACHE_PATH=str(root/'warp_cache'), ANYDEX_MJX_SHARED_PAIR_ENV=marker)
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), '--root', str(root)], env)
    result = run(root)
    if result['status'] != 'complete':
        print(result['error'], flush=True)
        raise SystemExit(1)
