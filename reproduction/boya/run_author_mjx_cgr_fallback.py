"""Run a small fresh-layout CGR baseline with measured-start planning fallback."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import author_collection_batch as B
import author_collection_scenes as S
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F
from author_mjx_cgr_fallback import MAX_CANDIDATES
from author_scene_prepare_cache import PreparationCache
from run_author_mjx_expansion import CATALOG, dispatch


SEED = 2026111905
ENVIRONMENTS = 16
FIT_DRAWS = 12
INTERNAL_DRAWS = 4


def freeze(root):
    root.mkdir(parents=True, exist_ok=False)
    catalog = B.read(CATALOG)
    S.verify(catalog['input_sha256']); S.verify(catalog['source_sha256'])
    rows = []
    cache = PreparationCache()
    with cache.installed():
        for split, count in [('fit', FIT_DRAWS), ('internal', INTERNAL_DRAWS)]:
            for index in range(count):
                spec = S.sample_scene(catalog, split, index, SEED)
                spec['catalog'] = B.binding(CATALOG)
                slot = len(rows)
                spec_path = root/'specs'/f'{slot:02d}.json'
                B.save(spec_path, spec, exclusive=True)
                scene = None
                if spec['status'] == 'ready':
                    build = root/'layouts'/f'{slot:02d}'
                    S.build_scene(spec_path, build)
                    scene = B.binding(build/'scene.json')
                rows.append(dict(index=slot, split=split, scene_id=spec['scene_id'], status=spec['status'], scene=scene,
                    scene_spec=B.binding(spec_path), source_object_ids=spec['source_object_ids'],
                    source_object_digests=spec['source_object_digests']))
    S.verify(catalog['input_sha256']); S.verify(catalog['source_sha256'])
    ready = sum(row['scene'] is not None for row in rows)
    if not ready:
        raise ValueError('Fixed fresh draws yielded no placement-ready layouts')
    queue = root/'QUEUE_SOURCE.json'
    B.save(queue, dict(rows=rows), exclusive=True)
    p = F.freeze(root/'base_freeze', environments=ENVIRONMENTS, source_protocol=queue)
    p.update(schema='anydex_mjx_cgr_bounded_planner_baseline_v1', seed=SEED,
             cgr_planning_limit=MAX_CANDIDATES, max_actual_arm_plans=ready*MAX_CANDIDATES,
             max_grasp_attempts=ready, max_CGR_forwards=ready, video_recording=False,
             candidate_policy='Original CGR after unchanged hand collision screen. First feasible arm plan among ranks 0..4.',
             formal_training_eligible=False, physical_retry_limit=0)
    collection = root/'collection'
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT'] = str(collection/'contact_spool')
    p['inputs']['scene_catalog'] = B.binding(CATALOG)
    p['inputs'].update({f"scene_spec:{row['index']}": row['scene_spec'] for row in rows})
    names = (Path(__file__).name, 'author_mjx_cgr_fallback.py', 'author_mjx_policy_compare_collection.py',
             'author_mjx_hour_collection.py', 'author_hour_audit.py', 'audit_author_mjx_wave_snapshot.py',
             'author_expansion_progress.py', 'author_hour_metrics.py', 'author_scene_prepare_cache.py')
    p['sources'].update({name:B.binding(B.HERE/name) for name in names})
    archive = collection/'source_archive'
    archive.mkdir(parents=True)
    for name, ref in p['sources'].items():
        shutil.copyfile(ref['path'], archive/name)
    B.save(collection/'PROTOCOL.json', p, exclusive=True)
    shutil.copytree(B.ROOT/'outputs/boya_mjx_fresh_expansion_20261118_v1/collection/warp_cache', collection/'warp_cache')
    receipt = dict(status='frozen', protocol=B.binding(collection/'PROTOCOL.json'), draws=len(rows), placement_ready=ready,
        max_plans=ready*MAX_CANDIDATES, max_grasps=ready, max_native_environment_steps=p['max_native_environment_steps'],
        video_recording=False)
    B.save(root/'READY.json', receipt, exclusive=True)
    print(receipt, flush=True)


def summarize(root):
    c = root/'collection';p = B.read(c/'PROTOCOL.json');r = B.read(c/'RESULTS.json');a = B.read(c/'READ_ONLY_AUDIT.json')
    if r['status'] != 'complete' or a['status'] != 'read_only_audit_pass' or a['original_result'] != B.binding(c/'RESULTS.json'):
        raise ValueError('Completed closed-wave audit required')
    audit = {row['index']:row for row in a['rounds']}
    scenes = []
    counts = Counter()
    for spec, record in zip(p['rows'], r['records'], strict=True):
        index = spec['index'];directory = c/'scenes'/f'{index:02d}'
        counts[record['status']] += 1
        selected = directory/'cgr_attempts/SELECTION.json'
        scene = dict(index=index, status=record['status'], label=record['label'])
        if selected.exists():
            s = B.read(selected);B.verify({'pool':s['shared_pool'],'observation':s['observation']})
            candidates = B.read(s['shared_pool']['path'])['candidates'];attempts = s['attempts']
            assert s['candidate_count'] == len(candidates)
            assert 1 <= len(attempts) <= p['cgr_planning_limit']
            ready = s['status'] == 'ready'
            if candidates:
                for rank, attempt in enumerate(attempts):
                    B.verify({'plan':attempt['plan']})
                    assert attempt['rank'] == rank and attempt['candidate_sha256'] == candidates[rank]['candidate_sha256']
                    assert attempt['status'] == B.read(attempt['plan']['path'])['status']
                assert all(attempt['status']=='null_geometry' for attempt in attempts[:-1])
            if ready:
                assert s['selected_rank'] == len(attempts)-1 and audit[index]['original_controls_exact'] and audit[index]['return_pass']
                actual = B.read(record['bindings']['candidate']['path'])
                assert actual['candidate_sha256'] == attempts[-1]['candidate_sha256']
            else:
                assert s['selected_rank'] is None and 'execution' not in record['bindings']
            scene.update(plan_ready=ready, selected_rank=s['selected_rank'],
                         candidate_count=len(candidates), first_plan_ready=attempts[0]['status']=='ready',
                         actual_arm_plans=len(attempts) if candidates else 0, selection=B.binding(selected))
        scenes.append(scene)
    plan_ready = sum(s.get('plan_ready',False) for s in scenes)
    top1_ready = sum(s.get('first_plan_ready',False) for s in scenes)
    arm_plans = sum(s.get('actual_arm_plans',0) for s in scenes)
    assert plan_ready == r['cost']['grasp_attempts'] <= p['max_grasp_attempts']
    assert arm_plans <= p['max_actual_arm_plans']
    B.verify(p['sources']);B.verify(p['inputs'])
    R.assert_quiescent(c)
    report = dict(status='complete', scenes=scenes, drawn_scenes=len(scenes), placement_ready=sum(bool(s['scene']) for s in p['rows']),
        observed_scenes=r['cost']['CGR_forwards'], top1_plan_ready=top1_ready, bounded_cgr_plan_ready=plan_ready,
        recovered_plan_ready=plan_ready-top1_ready, successful_lifts=counts['grasp_success'], actual_arm_plans=arm_plans,
        statuses=dict(counts), native_environment_steps=a['native_environment_integrations'], auditable_original_result=a['original_result'],
        native_audit=B.binding(c/'READ_ONLY_AUDIT.json'), controller_wall_s=r['parent_wall_s'],
        video_recording=False, training_updates=0, exclude_training=True)
    B.save(root/'BASELINE_RESULTS.json', report, exclusive=True)
    print({k:v for k,v in report.items() if k!='scenes'}, flush=True)


def run(root):
    B.verify({'protocol':B.read(root/'READY.json')['protocol']})
    if subprocess.check_output(['git','status','--porcelain'], cwd=B.ROOT).strip():
        raise ValueError('Start the fixed wave from a committed clean tree')
    B.save(root/'PIPELINE_STARTED.json', dict(timestamp=time.time(), git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'], cwd=B.ROOT,text=True).strip()), exclusive=True)
    try:
        result = dispatch([sys.executable,'-B',str(B.HERE/'author_mjx_cgr_fallback.py'),'collect',str(root/'collection')],
                          root/'timing', 3600)
        summarize(root)
        B.save(root/'PIPELINE_FINISHED.json', dict(stage=result, result=B.binding(root/'BASELINE_RESULTS.json')), exclusive=True)
    except BaseException:
        B.save(root/'PIPELINE_FAILURE.json', dict(error=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'summarize'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    {'freeze':freeze, 'run':run, 'summarize':summarize}[args.action](args.root.resolve())
