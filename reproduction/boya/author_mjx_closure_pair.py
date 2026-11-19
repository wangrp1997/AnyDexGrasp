"""Paired MJX replay of the original closure and a contact-triggered position hold."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import author_collection_batch as B
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F
from author_mjx_shared_pair import STATE, copy_lanes, integration_state
from run_author_mjx_expansion import dispatch


CASES = (
    ('boya_mjx_cgr_fallback_20261119_v2', (0,)),
    ('boya_mjx_cgr_empty_pool_rescue_20261119_v1', (3, 15)),
)
POLICY = dict(consecutive_close_controls=8, min_thumb_n=1., min_opposing_n=1.,
              retain_last_applied_finger_positions=True, reclose_after_loss=False)
WRISTS = ('FAJ1', 'FAJ3')


def opposing_objects(row):
    return {key for key, obj in row['objects'].items()
            if obj['finger_loads_n'][0] >= POLICY['min_thumb_n']
            and sum(obj['finger_loads_n'][1:]) >= POLICY['min_opposing_n']}


class ClosureHold:
    def __init__(self, finger_columns):
        self.columns = finger_columns
        self.streak = {}
        self.trigger = None
        self.target = None

    def apply(self, target):
        if self.target is not None:
            target[self.columns] = self.target

    def observe(self, row, applied):
        if row['phase'] != 'close' or self.trigger is not None:
            return
        self.streak = {k: self.streak.get(k, 0)+1 for k in opposing_objects(row)}
        sustained = sorted(k for k, length in self.streak.items() if length >= POLICY['consecutive_close_controls'])
        if sustained:
            self.trigger = dict(step=row['step'], objects=sustained)
            self.target = applied[self.columns].copy()


def arrays(path):
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key].copy() for key in npz.files}


def freeze(root):
    root.mkdir(parents=True, exist_ok=False)
    batches = []
    for source, indices in CASES:
        old = B.ROOT/'outputs'/source/'collection'
        run = B.read(old/'RESULTS.json')
        audit = B.read(old/'READ_ONLY_AUDIT.json')
        assert run['status'] == 'complete' and audit['status'] == 'read_only_audit_pass'
        assert audit['original_result'] == B.binding(old/'RESULTS.json')
        original = B.read(old/'PROTOCOL.json')
        bank = [row['scene'] for row in original['rows'] if row['scene']]
        inputs = dict(protocol=B.binding(old/'PROTOCOL.json'), audit=B.binding(old/'READ_ONLY_AUDIT.json'),
                      model=B.read(old/'MODEL_RECEIPT.json')['model'], template=original['inputs']['template:0'],
                      backend=original['inputs']['backend'])
        inputs.update({f'bank:{index}': ref for index, ref in enumerate(bank)})
        cases = []
        for index in indices:
            d = old/'scenes'/f'{index:02d}'
            record = B.read(d/'RECORD.json')
            ready = B.read(d/'live/READY.json')
            settle = B.read(d/'capture/SETTLING.json')['steps']
            assert record['eligible'] and record['bindings']['commands'] == ready['commands']
            assert record['bindings']['scene'] == ready['scene']
            saved = B.read(record['bindings']['execution']['path'])
            script = arrays(ready['commands']['path'])
            close_at = int(np.flatnonzero(script['phase'] == 'close')[0])
            assert len(script['phase']) == saved['steps'] == saved['expected_steps']
            spec = original['rows'][index]
            assert spec['scene'] in bank and spec['scene_id'] == B.read(ready['scene']['path'])['scene_id']
            cases.append(dict(index=index, original_label=record['label'], initial_scene=spec['scene'],
                              original_execution=record['bindings']['execution'], original_trace=saved['trace'],
                              commands=ready['commands'], candidate=ready['candidate'], settle_controls=settle,
                              close_at=close_at, motion_controls=saved['steps']))
            inputs.update({f'{index}:{key}': cases[-1][key]
                           for key in ('initial_scene', 'original_execution', 'original_trace', 'commands', 'candidate')})
        assert len({case['settle_controls'] for case in cases}) == 1
        ticks = cases[0]['settle_controls'] + max(case['motion_controls'] for case in cases)
        count = 2*len(cases)
        B.verify(inputs)
        B.verify(original['sources'])
        sources = dict(original['sources'])
        for name in (Path(__file__).name, 'author_mjx_shared_pair.py'):
            sources[name] = B.binding(B.HERE/name)
        batches.append(dict(source=source, inputs=inputs, sources=sources, cases=cases, object_bank=bank,
                            count=count, max_ticks=ticks, native_environment_budget=ticks*count*8))
    protocol = dict(status='frozen', policy=POLICY, batches=batches, video_recording=False, new_CGR_forwards=0,
                    no_new_planning=True, training_updates=0, exclude_training=True, max_grasp_attempts=6,
                    max_native_environment_steps=sum(b['native_environment_budget'] for b in batches))
    B.save(root/'PROTOCOL.json', protocol, exclusive=True)
    print(dict(root=str(root), max_grasps=6, native_environment_budget=protocol['max_native_environment_steps']), flush=True)


def save_pair_start(batch, lane, case_root):
    before = integration_state(batch)
    after = copy_lanes(batch, [(lane, lane+1)], before)
    pair = [lane, lane+1]
    equal_before = {name: np.array_equal(before[name][lane], before[name][lane+1]) for name in STATE}
    state_file = case_root/'PRE_CLOSE_STATE.npz'
    np.savez_compressed(state_file, **{key: value[pair] for key, value in after.items()})
    B.save(case_root/'PRE_CLOSE.json', dict(state=B.binding(state_file), lanes=pair, tick=batch.ticks,
        copied_state_fields=list(STATE), paired_exact_after_copy=True, equal_before_copy=equal_before,
        physics_steps_during_copy=0), exclusive=True)


def motion(batch, stage, cases, finger_columns):
    import author_mujoco_control as C
    from author_mujoco_sample import classify

    states = {}
    start = time.monotonic()
    with ExitStack() as stack:
        for j, case in enumerate(cases):
            arr = arrays(case['commands']['path'])
            speeds = C.source_velocity_targets(arr['joint_target'], batch.base.names, batch.base.arm_names)
            case_root = stage/f"case_{case['index']:02d}"
            case_root.mkdir()
            for lane, mode in enumerate(('original', 'contact_hold'), start=2*j):
                out = case_root/mode
                out.mkdir()
                stream = stack.enter_context((out/'TRACE.jsonl').open('x', buffering=1))
                states[lane] = dict(arr=arr, speed=speeds, rows=[], out=out, stream=stream, stop=None,
                                    history={}, policy=ClosureHold(finger_columns) if mode == 'contact_hold' else None)
        for i in range(max(case['motion_controls'] for case in cases)):
            for j, case in enumerate(cases):
                if i == case['close_at'] and all(states[lane]['stop'] is None for lane in (2*j, 2*j+1)):
                    save_pair_start(batch, 2*j, stage/f"case_{case['index']:02d}")
            active = {lane:s for lane,s in states.items() if s['stop'] is None and i < len(s['arr']['phase'])}
            if not active:
                break
            targets = batch.targets.copy()
            velocities = np.zeros_like(targets)
            for lane, state in active.items():
                targets[lane] = state['arr']['joint_target'][i]
                velocities[lane] = state['speed'][i]
                if state['policy'] is not None:
                    state['policy'].apply(targets[lane])
            batch.step(targets, velocities)
            for lane, state in active.items():
                phase = str(state['arr']['phase'][i])
                row = batch.observe(lane, phase, i+1)
                row.update(joint_target=targets[lane].tolist(), velocity_target=velocities[lane].tolist(),
                    applied_position_target_float32=targets[lane].astype(np.float32).tolist(),
                    applied_velocity_target_float32=velocities[lane].astype(np.float32).tolist())
                state['rows'].append(row)
                state['stream'].write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                if phase == 'settle':
                    state['history'] = {key:np.asarray(obj['pose_xyzw'][:2]) for key,obj in row['objects'].items()}
                if max([0.]+[-c['separation_m'] for c in row['contacts']]) > .005:
                    state['stop'] = 'original_5mm_deep_collision_stop'
                if C.T.preclosure_pushes(row['objects'], state['history'], phase):
                    state['stop'] = 'original_preclosure_push_stop'
                if state['policy'] is not None:
                    state['policy'].observe(row, targets[lane])
            if (i+1) % 250 == 0:
                B.save(stage/'PROGRESS.json', dict(phase='motion', control=i+1, ticks=batch.ticks))
        for lane, state in states.items():
            state['stream'].close()
            result = F.evaluate_execution(batch.views[lane], state['rows'], state['arr'], state['stop'], state['out'], time.monotonic()-start)
            B.save(state['out']/'OUTCOME.json', dict(result=B.binding(state['out']/'RESULTS.json'),
                decision=classify(result), trigger=state['policy'].trigger if state['policy'] is not None else None), exclusive=True)
            print(dict(lane=lane, steps=result['steps'], outcome=classify(result)['status']), flush=True)


def run_batch(root, index):
    from author_resident_prepare import bind_backend

    protocol_file = root/'PROTOCOL.json'
    p = B.read(protocol_file)
    assert p['policy'] == POLICY
    cfg = p['batches'][index]
    stage = root/f'batch_{index}'
    if not stage.exists():
        stage.mkdir()
    old = B.read(cfg['inputs']['protocol']['path'])
    marker = B.sha(protocol_file)
    if os.environ.get('ANYDEX_MJX_CLOSURE_PAIR_ENV') != marker:
        env = B._environment(old)
        env.update(PYTHONPATH=os.pathsep.join((str(F.WARP), str(F.DEPS), env['PYTHONPATH'])),
            CUDA_VISIBLE_DEVICES=F.GPU, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
            WARP_CACHE_PATH=str(root/'warp_cache'), ANYDEX_MJX_CLOSURE_PAIR_ENV=marker)
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', str(root), str(index)], env)
    B.verify(cfg['inputs']); B.verify(cfg['sources'])
    bind_backend(dict(backend=cfg['inputs']['backend']['path']))
    from author_compiled_kinematics import CompiledKinematics
    from author_mjx_native_batch import Batch

    template = B.read(cfg['inputs']['template']['path'])
    B.verify({key:template[key] for key in ('plan', 'commands')})
    plan = B.read(template['plan']['path'])
    initial = arrays(template['commands']['path'])
    names, home = initial['joint_names'].tolist(), initial['initial_position']
    bank = [B.read(scene['path']) for scene in cfg['object_bank']]
    lock = B.ROOT/'outputs/author_collection_gpu0.lock'
    started = time.monotonic()
    with R.lease(stage/'DRIVER.lock'), R.lease(lock), R.active_run(stage, lock.with_suffix('.wave_active.json'), B.binding(protocol_file)):
        B.save(stage/'RUN_STARTED.json', dict(pid=R.process_identity(os.getpid())), exclusive=True)
        with CompiledKinematics().installed():
            batch = Batch(plan, names, home, bank, count=cfg['count'], max_ticks=cfg['max_ticks'])
            try:
                fingerprint = hashlib.sha256(batch.base.spec.to_xml().encode()).hexdigest()
                if fingerprint != cfg['inputs']['model']['sha256']:
                    raise ValueError('Rebuilt model differs from the original native MJCF')
                B.save(stage/'MODEL_MATCH.json', dict(source=cfg['inputs']['model'], rebuilt_sha256=fingerprint), exclusive=True)
                batch.set_cost_journal(stage/'NATIVE_CALLS.jsonl')
                finger_columns = [i for i, name in enumerate(names) if name not in batch.base.arm_names and name not in WRISTS]
                assert len(finger_columns) == 16
                assignment = {}
                for j, case in enumerate(cfg['cases']):
                    assert arrays(case['commands']['path'])['joint_names'].tolist() == names
                    for lane in (2*j, 2*j+1):
                        assignment[lane] = dict(index=lane), B.read(case['initial_scene']['path'])
                ticks = cfg['cases'][0]['settle_controls']
                B.save(stage/'PROGRESS.json', dict(phase='settle', expected_controls=ticks))
                stable, _, _ = F.settle(batch, assignment, stage, dict(min_settle=ticks, max_settle=ticks), home)
                if not all(row['pass_'] for row in stable.values()):
                    raise ValueError('Historical layout did not pass the original stability check')
                motion(batch, stage, cfg['cases'], finger_columns)
                B.save(stage/'RESULTS.json', dict(status='complete', steps=batch.ticks, world=batch.receipt(),
                    finger_columns=finger_columns, wrist_and_arm_exact_by_design=True, returned=False,
                    wall_s=time.monotonic()-started), exclusive=True)
            finally:
                batch.close()


def audit_pair(case, stage, columns):
    """Reconstruct the changed command row from the raw consecutive contact window."""
    import author_mujoco_control as C
    from author_mujoco_sample import classify

    arr = arrays(case['commands']['path'])
    arm = arr['joint_names'][:6].tolist()
    speeds = C.source_velocity_targets(arr['joint_target'], arr['joint_names'].tolist(), arm)
    case_root = stage/f"case_{case['index']:02d}"
    preclose = case_root/'PRE_CLOSE.json'
    if preclose.exists():
        shared = B.read(preclose)
        B.verify({'state':shared['state']})
        for name, value in arrays(shared['state']['path']).items():
            np.testing.assert_array_equal(value[0], value[1], err_msg=name)
    branches = {}
    for mode in ('original', 'contact_hold'):
        out = case_root/mode
        result = B.read(out/'RESULTS.json')
        outcome = B.read(out/'OUTCOME.json')
        assert outcome['result'] == B.binding(out/'RESULTS.json')
        assert outcome['decision'] == classify(result)
        B.verify({'trace':result['trace']})
        trace = [json.loads(line) for line in (out/'TRACE.jsonl').open()]
        assert len(trace) == result['steps']
        trigger = None
        for i, row in enumerate(trace):
            assert row['step'] == i+1 and row['phase'] == arr['phase'][i]
            expected = arr['joint_target'][i].copy()
            if trigger is not None:
                expected[columns] = arr['joint_target'][trigger['step']-1, columns]
            np.testing.assert_array_equal(row['joint_target'], expected)
            np.testing.assert_array_equal(row['velocity_target'], speeds[i])
            np.testing.assert_array_equal(row['applied_position_target_float32'], expected.astype(np.float32))
            np.testing.assert_array_equal(row['applied_velocity_target_float32'], speeds[i].astype(np.float32))
            k = POLICY['consecutive_close_controls']
            if mode == 'contact_hold' and trigger is None and i >= case['close_at']+k-1 and row['phase'] == 'close':
                shared_objects = set.intersection(*(opposing_objects(x) for x in trace[i-k+1:i+1]))
                if shared_objects:
                    trigger = dict(step=i+1, objects=sorted(shared_objects))
        assert trigger == outcome['trigger']
        if result['status'] == 'complete':
            assert C.LIFT.evaluate_lift(trace, list(trace[0]['objects']), .004) == result['evaluation']
        lifts = [{'object':obj['object_id'], 'rise_m':obj['final_reference_rise_m'],
                  'loaded_lift_s':obj['trailing_observed_pickup_run_s']}
                 for obj in result['evaluation'].get('object_outcomes', []) if obj['task_success']]
        both = {phase:sum(bool(opposing_objects(r)) for r in trace if r['phase'] == phase) for phase in ('close','hold','lift')}
        branches[mode] = dict(outcome=outcome['decision']['status'], label=outcome['decision']['label'], trigger=trigger,
            strict_1mm_pass=result['gates']['penetration'], penetration_m=result['maxima']['penetration_m'],
            steps=result['steps'], stop=result['stop'], sustained_load_controls=both, lifts=lifts,
            exact_arm_wrist_velocity_and_phase=True, trace=result['trace'])
    return dict(index=case['index'], historical_label=case['original_label'],
                historical_outcome_reproduced=branches['original']['label'] == case['original_label'],
                preclose_state_equality_verified=preclose.exists(), branches=branches)


def audit(root):
    from author_resident_prepare import bind_backend

    protocol = B.read(root/'PROTOCOL.json')
    bind_backend(dict(backend=protocol['batches'][0]['inputs']['backend']['path']))
    batches = []
    for index, cfg in enumerate(protocol['batches']):
        stage = root/f'batch_{index}'
        B.verify(cfg['inputs']); B.verify(cfg['sources'])
        result = B.read(stage/'RESULTS.json')
        finishing = B.read(root/f'dispatch_{index}/FINISHED.json')
        assert finishing['process_group_exited'] and finishing['returncode'] == 0
        last = 0
        rows = 0
        per_control = cfg['count']*8
        for rows, line in enumerate((stage/'NATIVE_CALLS.jsonl').open(), 1):
            saved = json.loads(line)
            count = ((rows-1)//2)*per_control
            assert saved == dict(completed_native_environment_steps=count,
                                 pending_native_environment_steps_upper=per_control if rows % 2 == 0 else 0)
            last = count
        assert rows == 2*result['steps']+1 and last <= cfg['native_environment_budget']
        audited = [audit_pair(case, stage, result['finger_columns']) for case in cfg['cases']]
        with R.lease(stage/'DRIVER.lock'):
            pass
        batches.append(dict(source=cfg['source'], cases=audited, native_environment_steps=last,
                            stage_wall_s=finishing['wall_s'], audit_without_integration=True))
    lock = B.ROOT/'outputs/author_collection_gpu0.lock'
    with R.lease(lock):
        assert B.read(lock.with_suffix('.wave_active.json'))['state'] == 'released'
    reviewed = dict(status='complete', policy=POLICY, batches=batches,
                    native_environment_steps=sum(row['native_environment_steps'] for row in batches),
                    video_recording=False, training_updates=0, exclude_training=True)
    B.save(root/'COMPARISON_RESULTS.json', reviewed, exclusive=True)
    print(dict(status='complete', native_environment_steps=reviewed['native_environment_steps'],
               results=[(case['index'], {mode:x['label'] for mode,x in case['branches'].items()})
                        for batch in batches for case in batch['cases']]), flush=True)


def run(root):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=B.ROOT).strip():
        raise ValueError('Start the paired test from a committed clean tree')
    B.save(root/'PIPELINE_STARTED.json', dict(git=subprocess.check_output(['git','rev-parse','HEAD'], cwd=B.ROOT,text=True).strip()), exclusive=True)
    p = B.read(root/'PROTOCOL.json')
    cache = B.ROOT/'outputs'/p['batches'][0]['source']/'collection/warp_cache'
    shutil.copytree(cache, root/'warp_cache')
    for index, _ in enumerate(p['batches']):
        dispatch([sys.executable, '-B', str(Path(__file__).resolve()), 'worker', str(root), str(index)],
                 root/f'dispatch_{index}', 1500)
    audit(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'worker', 'audit'))
    parser.add_argument('root', type=Path)
    parser.add_argument('index', nargs='?', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'worker':
        run_batch(root, args.index)
    else:
        {'freeze':freeze, 'run':run, 'audit':audit}[args.action](root)
