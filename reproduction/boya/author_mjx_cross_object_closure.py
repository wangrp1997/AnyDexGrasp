"""Evaluate the existing closure pair on height strata with disjoint objects."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np

import author_collection_batch as B
import author_collection_scenes as S
import author_collection_wave_resources as R
import author_mjx_fresh_closure_pair as paired
from author_scene_prepare_cache import PreparationCache


SEED = 2026111908
EXCLUDED = 'graspnet_021'
HEIGHT_BOUNDARY_M = .05
DRAWS_PER_STRATUM = 32
LAYOUTS_PER_STRATUM = 4
FRESH = B.ROOT/'outputs/boya_mjx_fresh_closure_pair_20261119_v1'


def strata(catalog):
    heights = {}
    cache = {}
    groups = {'below_50mm': [], 'at_least_50mm': []}
    for key in catalog['split_object_ids']['internal']:
        if key == EXCLUDED:
            continue
        record = catalog['objects'][key]
        vertices, _ = S._geometry(record, cache)
        z = vertices @ np.asarray(record['source_settled_R'])[2]
        height = float(np.ptp(z))
        heights[key] = height
        groups['below_50mm' if height < HEIGHT_BOUNDARY_M else 'at_least_50mm'].append(key)
    if any(len(ids) < S.POLICY['objects_per_scene'] for ids in groups.values()):
        raise ValueError('Each height stratum requires three eligible internal objects')
    return groups, heights


def freeze(root):
    root.mkdir(parents=True, exist_ok=False)
    catalog = B.read(paired.CATALOG)
    S.verify(catalog['input_sha256']); S.verify(catalog['source_sha256'])
    previous = B.read(FRESH/'collection/PROTOCOL.json')
    B.verify(previous['sources'])
    grouping, heights = strata(catalog)
    source = B.binding(paired.CATALOG)
    B.save(root/'STRATA.json', dict(height_boundary_m=HEIGHT_BOUNDARY_M, source_settled_heights_m=heights,
        excluded_object_id=EXCLUDED, internal_groups=grouping, source_catalog=source), exclusive=True)
    rows, draws, derived_refs = [], [], {}
    rng = np.random.default_rng(SEED)
    for number, (tag, ids) in enumerate(grouping.items()):
        restricted = dict(catalog, split_object_ids=dict(catalog['split_object_ids'], internal=ids))
        derived = root/'catalogs'/f'{tag}.json'
        B.save(derived, restricted, exclusive=True)
        ref = B.binding(derived)
        derived_refs[f'catalog:{tag}'] = ref
        held_on_even = rng.permutation([True, True, False, False])
        accepted = 0
        with PreparationCache().installed():
            for ordinal in range(DRAWS_PER_STRATUM):
                draw = number*DRAWS_PER_STRATUM+ordinal
                spec = S.sample_scene(restricted,'internal',draw,SEED)
                spec['catalog'] = ref
                path = root/'draws'/f'{draw:02d}.json'
                B.save(path,spec,exclusive=True)
                draws.append(dict(index=draw,stratum=tag,placement=spec['status'],spec=B.binding(path)))
                if spec['status'] != 'ready':
                    continue
                index = len(rows)
                out = root/'layouts'/f'{index:02d}'
                S.build_scene(path,out)
                rows.append(dict(index=index,draw=draw,stratum=tag,scene_id=spec['scene_id'],status='ready',split='internal',
                    scene=B.binding(out/'scene.json'),scene_spec=B.binding(path),source_object_ids=spec['source_object_ids'],
                    source_object_digests=spec['source_object_digests'],held_on_even_lane=bool(held_on_even[accepted])))
                accepted += 1
                if accepted == LAYOUTS_PER_STRATUM:
                    break
    if not rows:
        raise ValueError('No usable layout in either prescribed height group')
    queue = root/'QUEUE_SOURCE.json'
    B.save(queue,dict(rows=rows),exclusive=True)
    B.save(root/'DRAWS.json',draws,exclusive=True)
    p = paired.F.freeze(root/'base_freeze',environments=2*len(rows),source_protocol=queue)
    out = root/'collection'
    keys = ('policy','cgr_planning_limit','rescue_planning_limit','candidate_selection','evaluation_only','no_return','video_recording')
    p.update({key:previous[key] for key in keys})
    p.update(schema='anydex_mjx_cross_object_closure_pair_v1',seed=SEED,max_pairs=len(rows),max_grasp_attempts=2*len(rows),
        max_candidate_plans=len(rows)*p['cgr_planning_limit'],max_global_control_ticks=p['max_settle']+p['max_execution_controls'])
    p['max_native_environment_steps'] = p['max_global_control_ticks']*p['environments']*8
    p['stage_native_budget'] = p['max_native_environment_steps']
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT'] = str(out/'contact_spool')
    p['inputs'].update(scene_catalog=source,strata=B.binding(root/'STRATA.json'),draws=B.binding(root/'DRAWS.json'),**derived_refs)
    p['inputs'].update({f'draw:{item["index"]}':item['spec'] for item in draws})
    p['sources'].update(previous['sources'])
    p['sources'][Path(__file__).name] = B.binding(__file__)
    B.verify(p['inputs']); B.verify(p['sources'])
    archive = out/'source_archive'
    archive.mkdir(parents=True)
    for name, ref in p['sources'].items():
        shutil.copyfile(ref['path'],archive/name)
    B.save(out/'PROTOCOL.json',p,exclusive=True)
    shutil.copytree(FRESH/'collection/warp_cache',out/'warp_cache')
    ready = dict(status='frozen',protocol=B.binding(out/'PROTOCOL.json'),seed=SEED,draws=len(draws),
        placement_ready=dict(Counter(row['stratum'] for row in rows)),grasp_budget=p['max_grasp_attempts'],
        max_native_environment_steps=p['max_native_environment_steps'],video_recording=False)
    B.save(root/'READY.json',ready,exclusive=True)
    print(ready,flush=True)


def review(root):
    collection = root/'collection'
    p = B.read(collection/'PROTOCOL.json')
    B.verify(p['inputs']); B.verify(p['sources'])
    stages = B.read(root/'PIPELINE_FINISHED.json')
    process_groups = [stages[key]['process']['pgid'] for key in ('run','audit')]
    for path in collection.glob('planning/*/PROCESS.json'):
        process_groups.append(B.read(path)['identity']['pgid'])
    assert all(not R.group_alive(group) for group in process_groups)
    R.assert_quiescent(collection)
    lock = B.ROOT/'outputs/author_collection_gpu0.lock'
    locks = [lock,collection/'DRIVER.lock',*(collection/'resource_slots').glob('*/*.lock')]
    for path in locks:
        with R.lease(path):
            if path == lock:
                assert B.read(lock.with_suffix('.wave_active.json'))['state'] == 'released'
    B.verify({'result':stages['result']})
    result = B.read(stages['result']['path'])
    assert result['status'] == 'complete'
    B.verify({'selection':result['selection']})
    selection = B.read(result['selection']['path'])
    groups = B.read(p['inputs']['strata']['path'])['internal_groups']
    if set.intersection(*(set(ids) for ids in groups.values())) or any(EXCLUDED in ids for ids in groups.values()):
        raise ValueError('The height strata must contain distinct objects outside the 021 control')
    strata_results = {name:dict(layout_status=Counter(),outcomes=Counter(),held_lifts=set(),rescued_objects=set(),cases=[]) for name in groups}
    for row,selected in zip(p['rows'],selection['records'],strict=True):
        assert row['index']==selected['index'] and row['scene']==selected['scene']
        assert set(row['source_object_ids']) <= set(groups[row['stratum']])
        strata_results[row['stratum']]['layout_status'][selected['status']] += 1
    for case in result['cases']:
        row = p['rows'][case['index']]
        summary = strata_results[row['stratum']]
        a,b = [case['branches'][mode]['label'] for mode in ('original','contact_hold')]
        lifts = {mode:[item['object'] for item in branch['lifts']] for mode,branch in case['branches'].items()}
        assert all(set(items) <= set(row['source_object_ids']) for items in lifts.values())
        outcome = 'invalid' if None in (a,b) or not case['preclose_state_equality_verified'] else {
            (1,1):'both_pass',(0,0):'both_fail',(0,1):'rescued',(1,0):'regressed'}[a,b]
        summary['outcomes'][outcome] += 1
        summary['held_lifts'].update(lifts['contact_hold'])
        if outcome == 'rescued':
            summary['rescued_objects'].update(lifts['contact_hold'])
        summary['cases'].append(dict(index=row['index'],draw=row['draw'],outcome=outcome,lifted_objects=lifts))
    rescued = sorted(set.union(*(item['rescued_objects'] for item in strata_results.values())))
    for item in strata_results.values():
        for key in ('held_lifts','rescued_objects'):
            item[key] = sorted(item[key])
    report = dict(status='verified_complete',comparison=stages['result'],protocol=B.binding(collection/'PROTOCOL.json'),
        strata=strata_results,distinct_rescued_objects=rescued,counts=result['counts'],CGR_forwards=result['CGR_forwards'],
        native_environment_steps=result['native_environment_steps'],runtime_wall_s=stages['run']['wall_s'],
        audit_wall_s=stages['audit']['wall_s'],exited_process_groups=len(set(process_groups)),available_locks=len(locks),
        registry_released=True,video_recording=False,training_updates=0,exclude_training=True)
    B.save(root/'CROSS_OBJECT_REVIEW.json',report,exclusive=True)
    print(report,flush=True)


def run(root):
    paired.run(root)
    review(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','review'))
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    {'freeze':freeze,'run':run,'review':review}[args.action](args.root.resolve())
