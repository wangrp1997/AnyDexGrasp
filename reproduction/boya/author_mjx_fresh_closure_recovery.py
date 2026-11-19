"""Compare load-loss recovery with original closing on fixed fresh layouts."""
import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np

import author_collection_batch as B
import author_collection_scenes as S
import author_collection_wave_resources as R
import author_mjx_closure_recovery as recovery
import author_mjx_fresh_closure_pair as paired
from author_scene_prepare_cache import PreparationCache
from run_author_mjx_expansion import dispatch


SEED = 2026112001
DRAWS_PER_SPLIT = 32
LAYOUTS_PER_SPLIT = 4
PRIOR_LAYOUTS = B.ROOT/'outputs/boya_mjx_cross_object_closure_20261119_v1'
PRIOR_RECOVERY = B.ROOT/'outputs/boya_mjx_closure_recovery_20261120_v1'


def freeze(root):
    root.mkdir(parents=True,exist_ok=False)
    catalog = B.read(paired.CATALOG)
    S.verify(catalog['input_sha256']); S.verify(catalog['source_sha256'])
    old = B.read(PRIOR_LAYOUTS/'collection/PROTOCOL.json')
    prior_control = B.read(PRIOR_RECOVERY/'PROTOCOL.json')
    assert prior_control['recovery_policy']==recovery.RECOVERY
    prior_review = PRIOR_RECOVERY/'RECOVERY_REVIEW.json'
    assert B.read(prior_review)['status']=='verified_complete'
    B.verify(old['sources']); B.verify(prior_control['sources'])
    split_ids = {split:[obj for obj in ids if obj!='graspnet_021'] for split,ids in catalog['split_object_ids'].items()}
    restricted = dict(catalog,split_object_ids=split_ids)
    limited_catalog = root/'CATALOG_WITHOUT_021.json'
    B.save(limited_catalog,restricted,exclusive=True)
    ref = B.binding(limited_catalog)
    rows, draws = [], []
    rng = np.random.default_rng(SEED)
    for split in ('fit','internal'):
        even = rng.permutation([True,True,False,False])
        accepted = 0
        with PreparationCache().installed():
            for draw in range(DRAWS_PER_SPLIT):
                spec = S.sample_scene(restricted,split,draw,SEED)
                spec['catalog'] = ref
                path = root/'draws'/split/f'{draw:02d}.json'
                B.save(path,spec,exclusive=True)
                draws.append(dict(split=split,draw=draw,placement=spec['status'],spec=B.binding(path)))
                if spec['status']!='ready':
                    continue
                index = len(rows)
                out = root/'layouts'/f'{index:02d}'
                S.build_scene(path,out)
                rows.append(dict(index=index,split=split,draw=draw,scene_id=spec['scene_id'],status='ready',
                    scene=B.binding(out/'scene.json'),scene_spec=B.binding(path),source_object_ids=spec['source_object_ids'],
                    source_object_digests=spec['source_object_digests'],held_on_even_lane=bool(even[accepted])))
                accepted += 1
                if accepted==LAYOUTS_PER_SPLIT:
                    break
    if not rows:
        raise ValueError('The two prescribed draws yielded no usable layout')
    queue = root/'QUEUE_SOURCE.json'
    B.save(queue,dict(rows=rows),exclusive=True)
    B.save(root/'DRAWS.json',draws,exclusive=True)
    p = paired.F.freeze(root/'base_freeze',environments=2*len(rows),source_protocol=queue)
    out = root/'collection'
    names = ('policy','cgr_planning_limit','rescue_planning_limit','candidate_selection','evaluation_only','no_return','video_recording')
    p.update({key:old[key] for key in names})
    p.update(schema='anydex_mjx_fresh_closure_recovery_v1',seed=SEED,max_pairs=len(rows),max_grasp_attempts=2*len(rows),
        max_candidate_plans=len(rows)*p['cgr_planning_limit'],recovery_policy=recovery.RECOVERY,
        max_global_control_ticks=p['max_settle']+p['max_execution_controls'])
    p['max_native_environment_steps'] = p['max_global_control_ticks']*p['environments']*8
    p['stage_native_budget'] = p['max_native_environment_steps']
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT'] = str(out/'contact_spool')
    p['inputs'].update(source_catalog=B.binding(paired.CATALOG),evaluation_catalog=ref,draws=B.binding(root/'DRAWS.json'),
                       paired_recovery_baseline=B.binding(prior_review))
    p['inputs'].update({f'draw:{item["split"]}:{item["draw"]}':item['spec'] for item in draws})
    p['sources'].update(prior_control['sources'])
    p['sources'][Path(__file__).name] = B.binding(__file__)
    B.verify(p['inputs']); B.verify(p['sources'])
    archive = out/'source_archive'
    archive.mkdir(parents=True)
    for name,ref in p['sources'].items():
        shutil.copyfile(ref['path'],archive/name)
    B.save(out/'PROTOCOL.json',p,exclusive=True)
    shutil.copytree(PRIOR_LAYOUTS/'collection/warp_cache',out/'warp_cache')
    ready = dict(status='frozen',protocol=B.binding(out/'PROTOCOL.json'),seed=SEED,draws=len(draws),
        placement_ready=dict(Counter(r['split'] for r in rows)),max_grasps=p['max_grasp_attempts'],
        max_native_environment_steps=p['max_native_environment_steps'],video_recording=False)
    B.save(root/'READY.json',ready,exclusive=True)
    print(ready,flush=True)


def collect(root):
    path = root/'PROTOCOL.json'
    p = B.read(path)
    assert p['recovery_policy']==recovery.RECOVERY
    marker = B.sha(path)
    if os.environ.get('ANYDEX_MJX_FRESH_CLOSURE_ENV')!=marker:
        env = B._environment(p)
        F = paired.F
        env.update(PYTHONPATH=os.pathsep.join((str(F.WARP),str(F.DEPS),env['PYTHONPATH'])),CUDA_VISIBLE_DEVICES=F.GPU,
                   JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',WARP_CACHE_PATH=str(root/'warp_cache'),
                   ANYDEX_MJX_FRESH_CLOSURE_ENV=marker)
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'collect',str(root)],env)
    original = paired.execute_pairs

    def execute_recovery(root,batch,cases,columns):
        scripts = iter(paired.arrays(case['commands']['path']) for case in cases)

        def controller(columns):
            return recovery.RecoveringHold(columns,next(scripts))

        with patch.object(paired,'ClosureHold',controller):
            return original(root,batch,cases,columns)

    with patch.object(paired,'execute_pairs',execute_recovery):
        paired.collect(root)


def audit_pair(case,root,columns):
    import author_mujoco_control as C

    script = paired.arrays(case['commands']['path'])
    joints = script['joint_names'].tolist()
    speed = C.source_velocity_targets(script['joint_target'],joints,joints[:6])
    receipt = root/f"case_{case['index']:02d}"/'PRE_CLOSE.json'
    if receipt.exists():
        state = B.read(receipt)
        B.verify({'state':state['state']})
        assert state['physics_steps_during_copy']==0
        for name,value in paired.arrays(state['state']['path']).items():
            np.testing.assert_array_equal(value[0],value[1],err_msg=name)
    # The sample verdict accepts any of its three bodies. The nominal scene
    # target is supplied only for the archived replay helper's focused fields.
    focus = B.read(case['scene']['path'])['target_object_id']
    branches = {}
    for mode in ('original','contact_hold'):
        result = recovery.validate_branch(dict(case,target=focus),root,mode,columns,script,speed)
        for key in ('target_lifted','target_rise_m','loaded_lift_s'):
            del result[key]
        branches[mode] = result
    return dict(index=case['index'],historical_label=None,historical_outcome_reproduced=False,
                preclose_state_equality_verified=receipt.exists(),branches=branches)


def audit(root):
    with patch.object(paired,'audit_pair',audit_pair):
        paired.audit(root)


def review(root):
    collection = root/'collection'
    p = B.read(collection/'PROTOCOL.json')
    B.verify(p['inputs']); B.verify(p['sources'])
    stages = B.read(root/'PIPELINE_FINISHED.json')
    B.verify({'result':stages['result']})
    result = B.read(stages['result']['path'])
    assert result['status']=='complete'
    process_groups = [stages[key]['process']['pgid'] for key in ('run','audit')]
    process_groups.extend(B.read(path)['identity']['pgid'] for path in collection.glob('planning/*/PROCESS.json'))
    assert all(not R.group_alive(group) for group in process_groups)
    R.assert_quiescent(collection)
    lock = B.ROOT/'outputs/author_collection_gpu0.lock'
    locks = [lock,collection/'DRIVER.lock',*(collection/'resource_slots').glob('*/*.lock')]
    for path in locks:
        with R.lease(path):
            if path==lock:
                assert B.read(lock.with_suffix('.wave_active.json'))['state']=='released'
    summary = []
    objects = {'original':set(),'contact_hold':set()}
    for case in result['cases']:
        row = p['rows'][case['index']]
        assert 'graspnet_021' not in row['source_object_ids']
        branches = case['branches']
        lifted = {mode:[obj['object'] for obj in branch['lifts']] for mode,branch in branches.items()}
        for mode,ids in lifted.items():
            assert set(ids)<=set(row['source_object_ids'])
            if branches[mode]['label']==1:
                objects[mode].update(ids)
        summary.append(dict(index=case['index'],split=row['split'],draw=row['draw'],paired_state=case['preclose_state_equality_verified'],
            original_label=branches['original']['label'],recovery_label=branches['contact_hold']['label'],lifted_objects=lifted,
            static_hold_triggered=branches['contact_hold']['trigger'] is not None,
            resumed_source_close_controls=branches['contact_hold']['resumed_source_close_controls'],
            penetration_m={mode:branch['penetration_m'] for mode,branch in branches.items()}))
    review = dict(status='verified_complete',comparison=stages['result'],protocol=B.binding(collection/'PROTOCOL.json'),
        counts=result['counts'],layout_statuses=result['layout_statuses'],cases=summary,lifted_objects={mode:sorted(v) for mode,v in objects.items()},
        CGR_forwards=result['CGR_forwards'],native_environment_steps=result['native_environment_steps'],
        unknown_native_environment_steps=result['unknown_native_environment_steps'],runtime_wall_s=stages['run']['wall_s'],
        audit_wall_s=stages['audit']['wall_s'],exited_process_groups=len(set(process_groups)),available_locks=len(locks),
        registry_released=True,video_recording=False,training_updates=0,exclude_training=True)
    B.save(root/'FRESH_RECOVERY_REVIEW.json',review,exclusive=True)
    print(review,flush=True)


def run(root):
    assert B.read(root/'READY.json')['protocol']==B.binding(root/'collection/PROTOCOL.json')
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
        review(root)
    except BaseException:
        B.save(root/'PIPELINE_FAILURE.json',dict(error=traceback.format_exc()),exclusive=True)
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','collect','audit','run','review'))
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    {'freeze':freeze,'collect':collect,'audit':audit,'run':run,'review':review}[args.action](args.root.resolve())
