"""Compare two closure policies on fresh, shared-observation Boya grasps."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

import author_collection_batch as B
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F
from author_mjx_cgr_empty_pool_rescue import RESCUE_LIMIT, planning_batch, selection
from author_mjx_cgr_fallback import MAX_CANDIDATES
from author_mjx_closure_pair import POLICY, WRISTS, ClosureHold, arrays, audit_pair, save_pair_start
from author_mjx_shared_pair import STATE, copy_lanes, integration_state
from run_author_mjx_expansion import CATALOG, dispatch


SEED = 2026111907
MAX_DRAWS = 32
MAX_PAIRS = 8


def freeze(root):
    import author_collection_scenes as S
    from author_scene_prepare_cache import PreparationCache

    root.mkdir(parents=True, exist_ok=False)
    catalog = B.read(CATALOG)
    S.verify(catalog['input_sha256']); S.verify(catalog['source_sha256'])
    rows = []
    draws = []
    # Freeze the side assignment before checking any layout, plan, or outcome.
    held_on_even = np.random.default_rng(SEED).permutation([True]*4+[False]*4)
    with PreparationCache().installed():
        for draw in range(MAX_DRAWS):
            spec = S.sample_scene(catalog, 'internal', draw, SEED)
            spec['catalog'] = B.binding(CATALOG)
            path = root/'draws'/f'{draw:02d}.json'
            B.save(path, spec, exclusive=True)
            draws.append(dict(index=draw, placement=spec['status'], spec=B.binding(path)))
            if spec['status'] != 'ready':
                continue
            index = len(rows)
            out = root/'layouts'/f'{index:02d}'
            S.build_scene(path, out)
            rows.append(dict(index=index, draw=draw, scene_id=spec['scene_id'], status='ready', split='internal',
                scene=B.binding(out/'scene.json'), scene_spec=B.binding(path), source_object_ids=spec['source_object_ids'],
                source_object_digests=spec['source_object_digests'], held_on_even_lane=bool(held_on_even[index])))
            if len(rows) == MAX_PAIRS:
                break
    S.verify(catalog['input_sha256']); S.verify(catalog['source_sha256'])
    if not rows:
        raise ValueError('Fixed evaluation layouts yielded no valid placements')
    queue = root/'QUEUE_SOURCE.json'
    B.save(queue, dict(rows=rows), exclusive=True)
    B.save(root/'DRAWS.json', draws, exclusive=True)
    p = F.freeze(root/'base_freeze', environments=2*len(rows), source_protocol=queue)
    out = root/'collection'
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT'] = str(out/'contact_spool')
    p.update(schema='anydex_mjx_fresh_closure_pair_v1', seed=SEED, policy=POLICY,
        max_pairs=len(rows), max_CGR_forwards=len(rows), max_candidate_plans=len(rows)*MAX_CANDIDATES,
        max_grasp_attempts=2*len(rows), cgr_planning_limit=MAX_CANDIDATES, rescue_planning_limit=RESCUE_LIMIT,
        max_global_control_ticks=p['max_settle']+p['max_execution_controls'], video_recording=False,
        candidate_selection='First feasible original CGR rank, with existing fallback for an empty original pool',
        evaluation_only=True, no_return=True)
    p['max_native_environment_steps'] = p['max_global_control_ticks']*p['environments']*8
    p['stage_native_budget'] = p['max_native_environment_steps']
    p['inputs'].update(scene_catalog=B.binding(CATALOG), draws=B.binding(root/'DRAWS.json'))
    p['inputs'].update({f'draw:{d["index"]}':d['spec'] for d in draws})
    for name in (Path(__file__).name, 'author_mjx_closure_pair.py', 'author_mjx_shared_pair.py',
                 'author_mjx_cgr_empty_pool_rescue.py', 'author_mjx_cgr_fallback.py',
                 'audit_cgr_approach_depth_recovery.py', 'run_author_mjx_expansion.py'):
        p['sources'][name] = B.binding(B.HERE/name)
    B.verify(p['inputs']); B.verify(p['sources'])
    archive = out/'source_archive'
    archive.mkdir(parents=True)
    for name, ref in p['sources'].items():
        shutil.copyfile(ref['path'], archive/name)
    B.save(out/'PROTOCOL.json', p, exclusive=True)
    cache = B.ROOT/'outputs/boya_mjx_closure_pair_20261119_v1/warp_cache'
    shutil.copytree(cache, out/'warp_cache')
    ready = dict(status='frozen', protocol=B.binding(out/'PROTOCOL.json'), seed=SEED, draws=len(draws),
                 placement_ready=len(rows), max_grasps=p['max_grasp_attempts'],
                 max_native_environment_steps=p['max_native_environment_steps'], video_recording=False)
    B.save(root/'READY.json', ready, exclusive=True)
    print(ready, flush=True)


def select_cases(root, p, batch, engine, resources, home):
    import author_contact_pipeline as M
    from author_live_world_contract import bind_live_plan
    from author_mjx_native_batch import photograph

    assigned = {}
    for spec in p['rows']:
        scene = B.read(spec['scene']['path'])
        for lane in (2*spec['index'], 2*spec['index']+1):
            assigned[lane] = dict(index=lane), scene
    B.save(root/'PROGRESS.json', dict(phase='settle', ticks=batch.ticks))
    stable, last, ticks = F.settle(batch, assigned, root, p, home)
    lanes = [(i, i+1) for i in range(0, p['environments'], 2) if stable[i]['pass_'] and stable[i+1]['pass_']]
    before = integration_state(batch)
    shared = copy_lanes(batch, lanes, before)
    requests = []
    records = {}
    for row in p['rows']:
        index = row['index']; lane = 2*index
        record = dict(index=index, draw=row['draw'], scene=row['scene'], status='initialization_unstable')
        records[index] = record
        if (lane, lane+1) not in lanes:
            continue
        directory = root/'scenes'/f'{lane:02d}'
        capture = directory/'capture'
        scene = assigned[lane][1]
        photograph(batch, lane, scene, capture, last[lane], stable[lane]['stability'], ticks)
        engine.forward(capture/'observation.json', directory/'representation')
        domain = directory/'DOMAIN.json'
        B.save(domain, p['domain'], exclusive=True)
        requests.append((index, dict(scene=str(capture/'observed_scene.json'), observation=str(capture/'observation.json'),
            representation_dir=str(directory/'representation'), seed=p['seed'], output_dir=str(directory/'contact'),
            domain=str(domain), **{key:p['inputs'][key]['path'] for key in ('calibration','cloud_manifest','backend')})))
        initial = directory/'PAIRED_CAPTURE_STATE.npz'
        np.savez_compressed(initial, **{name:value[[lane,lane+1]] for name,value in shared.items()})
        record.update(initial_shared_state=B.binding(initial), observation=B.binding(capture/'observation.json'),
            live_state=B.binding(capture/'LIVE_STATE.json'), representation=B.binding(directory/'representation/RESULTS.json'),
            physical_controls_during_capture_and_planning=0)
    B.save(root/'PROGRESS.json', dict(phase='planning', jobs=len(requests), ticks=batch.ticks))
    plans = planning_batch(root/'planning', requests, p, resources)
    for name,value in integration_state(batch).items():
        np.testing.assert_array_equal(value, shared[name], err_msg=name)
    cases = []
    for index, job in sorted(plans.items()):
        record = records[index]
        result = selection(job)
        assert result['observation'] == record['observation'] and job['live_state'] == record['live_state']
        record.update(status='planning_'+job['status'], selection=job['cgr_fallback'],
                      empty_pool_rescue=job.get('empty_pool_rescue'), collision_candidates=result['candidate_count'])
        if job['status'] != 'ready':
            continue
        lane = 2*index
        directory = root/'scenes'/f'{lane:02d}'
        bind_live_plan(directory/'contact/pipeline', directory/'capture/LIVE_STATE.json', directory/'live')
        M.audit_mapping(directory/'live')
        ready = B.read(directory/'live/READY.json')
        B.verify({name:ready[name] for name in ('commands','candidate','plan','scene')})
        candidate = B.read(ready['candidate']['path'])
        assert candidate['candidate_sha256'] == result['attempts'][-1]['candidate_sha256']
        script = arrays(ready['commands']['path'])
        phases = script['phase']
        close_at = int(np.flatnonzero(phases == 'close')[0])
        assert 0 < close_at < len(phases) <= p['max_execution_controls']
        treatment = lane if p['rows'][index]['held_on_even_lane'] else lane+1
        record.update(status='paired_execution', plan=ready['plan'], candidate=ready['candidate'], commands=ready['commands'],
            motion_controls=len(phases), close_at=close_at, contact_hold_lane=treatment, original_lane=2*lane+1-treatment)
        cases.append(record)
    B.save(root/'SELECTION.json', dict(records=list(records.values()), paired_cases=len(cases), CGR_forwards=len(requests)), exclusive=True)
    print(dict(phase='prepared', captured=len(requests), paired_cases=[case['index'] for case in cases]), flush=True)
    return cases


def execute_pairs(root, batch, cases, columns):
    import author_mujoco_control as C
    from author_mujoco_sample import classify

    B.save(root/'PROGRESS.json', dict(phase='motion', selected=len(cases), ticks=batch.ticks))
    states = {}
    started = time.monotonic()
    with ExitStack() as stack:
        for case in cases:
            arr = arrays(case['commands']['path'])
            speed = C.source_velocity_targets(arr['joint_target'], batch.base.names, batch.base.arm_names)
            for mode in ('original', 'contact_hold'):
                out = root/f"case_{case['index']:02d}"/mode
                out.mkdir(parents=True)
                states[case[mode+'_lane']] = dict(arr=arr, speed=speed, rows=[], out=out, stop=None, settled={},
                    stream=stack.enter_context((out/'TRACE.jsonl').open('x', buffering=1)),
                    hold=ClosureHold(columns) if mode == 'contact_hold' else None)
        for i in range(max((case['motion_controls'] for case in cases), default=0)):
            for case in cases:
                lane = 2*case['index']
                if i == case['close_at'] and all(states[j]['stop'] is None for j in (lane,lane+1)):
                    save_pair_start(batch, lane, root/f"case_{case['index']:02d}")
            active = {lane:s for lane,s in states.items() if s['stop'] is None and i < len(s['arr']['phase'])}
            if not active:
                break
            targets = batch.targets.copy()
            speed = np.zeros_like(targets)
            for lane,state in active.items():
                targets[lane] = state['arr']['joint_target'][i]
                speed[lane] = state['speed'][i]
                if state['hold'] is not None:
                    state['hold'].apply(targets[lane])
            batch.step(targets, speed)
            for lane,state in active.items():
                phase = str(state['arr']['phase'][i])
                row = batch.observe(lane, phase, i+1)
                row.update(joint_target=targets[lane].tolist(), velocity_target=speed[lane].tolist(),
                    applied_position_target_float32=targets[lane].astype(np.float32).tolist(),
                    applied_velocity_target_float32=speed[lane].astype(np.float32).tolist())
                state['rows'].append(row)
                state['stream'].write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                if phase == 'settle':
                    state['settled'] = {key:np.asarray(obj['pose_xyzw'][:2]) for key,obj in row['objects'].items()}
                if max([0.]+[-c['separation_m'] for c in row['contacts']]) > .005:
                    state['stop'] = 'original_5mm_deep_collision_stop'
                if C.T.preclosure_pushes(row['objects'], state['settled'], phase):
                    state['stop'] = 'original_preclosure_push_stop'
                if state['hold'] is not None:
                    state['hold'].observe(row, targets[lane])
            if (i+1) % 250 == 0:
                B.save(root/'PROGRESS.json', dict(phase='motion', controls=i+1, ticks=batch.ticks, active=len(active)))
        for lane,state in states.items():
            state['stream'].close()
            result = F.evaluate_execution(batch.views[lane], state['rows'], state['arr'], state['stop'], state['out'], time.monotonic()-started)
            decision = classify(result)
            B.save(state['out']/'OUTCOME.json', dict(result=B.binding(state['out']/'RESULTS.json'), decision=decision,
                trigger=state['hold'].trigger if state['hold'] is not None else None), exclusive=True)
            print(dict(phase='grasp_finished', lane=lane, outcome=decision['status']), flush=True)


def collect(root):
    path = root/'PROTOCOL.json'
    p = B.read(path)
    marker = B.sha(path)
    if os.environ.get('ANYDEX_MJX_FRESH_CLOSURE_ENV') != marker:
        env = B._environment(p)
        env.update(PYTHONPATH=os.pathsep.join((str(F.WARP), str(F.DEPS), env['PYTHONPATH'])), CUDA_VISIBLE_DEVICES=F.GPU,
                   JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false', WARP_CACHE_PATH=str(root/'warp_cache'),
                   ANYDEX_MJX_FRESH_CLOSURE_ENV=marker)
        os.execve(sys.executable, [sys.executable,'-B',str(Path(__file__).resolve()),'collect',str(root)], env)
    from author_resident_prepare import bind_backend

    assert p['policy'] == POLICY
    B.verify(p['inputs']); B.verify(p['sources'])
    bind_backend(dict(backend=p['inputs']['backend']['path']))
    from author_compiled_kinematics import CompiledKinematics
    from author_mjx_native_batch import Batch
    from author_mujoco_representation import Stream

    template = B.read(p['inputs']['template:0']['path'])
    B.verify({key:template[key] for key in ('commands','plan')})
    initial = arrays(template['commands']['path'])
    names, home = initial['joint_names'].tolist(), initial['initial_position']
    scenes = [B.read(row['scene']['path']) for row in p['rows']]
    lock = B.ROOT/'outputs/author_collection_gpu0.lock'
    started = time.monotonic()
    with R.lease(root/'DRIVER.lock'), R.lease(lock) as lease, R.active_run(root, lock.with_suffix('.wave_active.json'), B.binding(path)):
        B.save(root/'RUN_STARTED.json', dict(process=R.process_identity(os.getpid())), exclusive=True)
        pool = R.ResourcePool(root/'resource_slots',lease,B.binding(path))
        with CompiledKinematics().installed():
            batch = Batch(B.read(template['plan']['path']),names,home,scenes,count=p['environments'],max_ticks=p['max_global_control_ticks'])
            try:
                batch.set_cost_journal(root/'NATIVE_CALLS.jsonl')
                (root/'MODEL.xml').write_text(batch.base.spec.to_xml())
                engine = Stream(SimpleNamespace(checkpoint=Path(p['inputs']['checkpoint']['path']), native_deps=Path(p['native_deps']),
                    device='cuda:0', seed=p['seed'], num_seed=2048, voxel_size=.005, depth_min=.2, depth_max=.65))
                cases = select_cases(root,p,batch,engine,pool,home)
                columns = [i for i,n in enumerate(names) if n not in batch.base.arm_names and n not in WRISTS]
                assert len(columns) == 16
                execute_pairs(root,batch,cases,columns)
                B.save(root/'RESULTS.json', dict(status='complete', selection=B.binding(root/'SELECTION.json'),
                    world=batch.receipt(), representation=engine.receipt(), model=B.binding(root/'MODEL.xml'), finger_columns=columns,
                    physical_attempts=2*len(cases), elapsed_s=time.monotonic()-started), exclusive=True)
            finally:
                batch.close()


def audit_numerics(case, root, plan, planning, reference, clocks):
    import author_mujoco_control as C
    import author_fast_geometry_checks as geometry
    import trimesh

    hand = {body.get('name') for body in ET.parse(plan['arguments']['hand_scene']).getroot().findall('.//body')}
    legal = {planning.fk.fixed_attachment(name,np.eye(4))[0] for name in hand if name in planning.fk.links}
    robot = sorted({'/World/Robot/'+planning.fk.fixed_attachment(name,np.eye(4))[0] for name in planning.fk.links})
    vertices = {o['object_id']:np.asarray(trimesh.load(o['source_mesh'],force='mesh',process=False).vertices)*o['scale']
                for o in B.read(case['scene']['path'])['objects']}
    desired = arrays(case['commands']['path'])['desired_palm']
    start = B.read(case['observation']['path'])['world_step']
    for mode in ('original','contact_hold'):
        folder = root/f"case_{case['index']:02d}"/mode
        result = B.read(folder/'RESULTS.json')
        trace = [json.loads(line) for line in Path(result['trace']['path']).open()]
        max_object_error = 0.
        for i,row in enumerate(trace,1):
            assert row['world_step'] == start+i
            np.testing.assert_array_equal(np.float32(row['native_time_s']),clocks[start+i])
            views = C.object_views(row['contacts'],list(vertices))
            for key,obj in row['objects'].items():
                metrics = geometry.source_metrics(vertices[key],obj['pose_xyzw'])
                loads = C.T.object_robot_loads('/World/'+key,views,robot,legal)
                for name,value in (metrics|loads).items():
                    max_object_error = max(max_object_error,float(np.max(np.abs(np.asarray(obj[name])-value))))
        assert max_object_error <= 1e-12
        maxima,first = C.numeric_maxima(trace,planning,list(planning.joints),desired,reference,legal,list(vertices))
        assert maxima == result['maxima'] and first == result['first_failure']
        assert C.source_gates(maxima,len(trace),case['motion_controls']) == result['gates']


def audit(root):
    from author_resident_prepare import bind_backend

    p = B.read(root/'PROTOCOL.json')
    B.verify(p['sources']); B.verify(p['inputs'])
    B.verify({name:dict(path=str(root/'source_archive'/name),sha256=ref['sha256']) for name,ref in p['sources'].items()})
    bind_backend(dict(backend=p['inputs']['backend']['path']))
    from author_compiled_kinematics import CompiledKinematics
    from author_mjx_live_batch_v4 import clock_table
    from audit_author_mujoco_collection import model_for

    result = B.read(root/'RESULTS.json')
    assert result['status'] == 'complete'
    B.verify({name:result[name] for name in ('selection','model')})
    selected = B.read(result['selection']['path'])
    counts = Counter(valid_pairs=0, both_pass=0, both_fail=0, rescued=0, regressed=0, invalid_pairs=0)
    cases = []
    template = B.read(p['inputs']['template:0']['path'])
    plan = B.read(template['plan']['path'])
    clocks = clock_table(p['max_global_control_ticks'])
    planning,reference = model_for(template)
    with CompiledKinematics().installed():
        for record in selected['records']:
            if 'observation' in record:
                B.verify({name:record[name] for name in ('initial_shared_state','observation','live_state','representation','selection')})
                for name,value in arrays(record['initial_shared_state']['path']).items():
                    np.testing.assert_array_equal(value[0],value[1],err_msg=name)
            if record['status'] != 'paired_execution':
                continue
            assert set(arrays(record['initial_shared_state']['path'])) == set(STATE)
            checked = audit_pair(dict(record,original_label=None),root,result['finger_columns'])
            del checked['historical_label'],checked['historical_outcome_reproduced']
            if checked['preclose_state_equality_verified']:
                preclose = B.read(root/f"case_{record['index']:02d}"/'PRE_CLOSE.json')
                assert sorted(preclose['lanes']) == sorted([record['original_lane'],record['contact_hold_lane']])
                assert preclose['tick'] == B.read(record['observation']['path'])['world_step']+record['close_at']
                assert set(preclose['copied_state_fields']) == set(STATE)
            audit_numerics(record,root,plan,planning,reference,clocks)
            checked.update(draw=record['draw'], candidate=record['candidate'], original_lane=record['original_lane'],
                contact_hold_lane=record['contact_hold_lane'], source_gates_and_contact_loads_recomputed=True)
            cases.append(checked)
            a,b = [checked['branches'][mode]['label'] for mode in ('original','contact_hold')]
            if a is None or b is None or not checked['preclose_state_equality_verified']:
                counts['invalid_pairs'] += 1
            else:
                counts['valid_pairs'] += 1
                counts[{(1,1):'both_pass',(0,0):'both_fail',(0,1):'rescued',(1,0):'regressed'}[a,b]] += 1
            print(dict(phase='audited', pair=record['index'], labels=[a,b]),flush=True)
    per_control = p['environments']*8
    tick_count = result['world']['global_control_ticks']
    lines = 0
    with (root/'NATIVE_CALLS.jsonl').open() as stream:
        for lines,line in enumerate(stream,1):
            assert json.loads(line) == dict(completed_native_environment_steps=((lines-1)//2)*per_control,
                pending_native_environment_steps_upper=per_control if lines%2==0 else 0)
    native = tick_count*per_control
    assert lines == 2*tick_count+1 and native == result['world']['native_environment_integrations'] <= p['max_native_environment_steps']
    assert len(cases)*2 == result['physical_attempts'] <= p['max_grasp_attempts']
    R.assert_quiescent(root)
    report = dict(status='complete', counts=dict(counts), cases=cases, layout_statuses=dict(Counter(r['status'] for r in selected['records'])),
        CGR_forwards=selected['CGR_forwards'], native_environment_steps=native, unknown_native_environment_steps=0,
        runtime=B.binding(root/'RESULTS.json'), selection=result['selection'], video_recording=False, training_updates=0, exclude_training=True)
    B.save(root.parent/'COMPARISON_RESULTS.json',report,exclusive=True)
    print(dict(status='complete',counts=dict(counts),native_environment_steps=native),flush=True)


def run(root):
    assert B.read(root/'READY.json')['protocol'] == B.binding(root/'collection/PROTOCOL.json')
    if subprocess.check_output(['git','status','--porcelain'],cwd=B.ROOT).strip():
        raise ValueError('Start the fixed comparison from a committed clean tree')
    entry = [sys.executable,'-B',str(Path(__file__).resolve())]
    collection = str(root/'collection')
    B.save(root/'PIPELINE_STARTED.json',dict(timestamp=time.time(),git=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=B.ROOT,text=True).strip()),exclusive=True)
    try:
        live = dispatch(entry+['collect',collection],root/'timing',2400)
        checked = dispatch(entry+['audit',collection],root/'audit_timing',1200)
        B.save(root/'PIPELINE_FINISHED.json',dict(run=live,audit=checked,result=B.binding(root/'COMPARISON_RESULTS.json')),exclusive=True)
    except BaseException:
        B.save(root/'PIPELINE_FAILURE.json',dict(error=traceback.format_exc()),exclusive=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','collect','audit'))
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    {'freeze':freeze,'run':run,'collect':collect,'audit':audit}[args.action](args.root.resolve())
