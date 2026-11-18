"""Start a bounded resident CGR queue and stop at 100 cumulative audited fit labels."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import author_collection_batch as B
import author_collection_scenes as S
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F
from author_scene_prepare_cache import PreparationCache
from author_mjx_quota_data import promote,merge,same_domain,read_formal
from run_author_mjx_expansion import dispatch

SEED=2026111902
CATALOG=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/manual_start_20261116_v2/scene_queue/CATALOG.json')
BASELINE=B.ROOT/'outputs/boya_mjx_fresh_expansion_20261118_v1/formal/MANIFEST.json'
ENTRY=B.HERE/'author_mjx_quota_collection.py'
MAX_SCENES=800


def freeze(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    baseline,old_data=read_formal(BASELINE)
    existing_fit=int((old_data['split']=='train').sum());existing_internal=int((old_data['split']=='internal_check').sum())
    if existing_fit!=16 or existing_internal!=4:raise ValueError('This campaign binds the current 16-fit/4-internal formal baseline')
    catalog=B.read(CATALOG);S.verify(catalog['input_sha256']);S.verify(catalog['source_sha256'])
    B.save(root/'CAMPAIGN_INTENT.json',dict(schema='anydex_mjx_to100_intent_v1',seed=SEED,
        existing_formal_manifest=B.binding(BASELINE),existing_fit_records=16,existing_internal_records=4,
        target_fit_records=100,remaining_fit_records=84,max_new_scenes=800,
        new_scene_draw_counts=dict(fit=640,internal=160),environments=16,
        max_native_environment_steps=119040000,max_collection_wall_s=64800,
        candidate_policy='Original CGR descending after unchanged collision filtering, fixed first candidate',
        gestures=[1],retries=0,video_recording=False,training_updates=0,
        stopping='Stop admitting waves once returned fit count reaches100, drain and verify all audits; preserve any in-flight surplus',
        failure_labels='Normal qualified failures count toward fit quota; no candidate/incomplete/invalid remain null',
        robot_initialization='Once per environment for the entire resident queue',
        diagnostic_comparison_records_included=False),exclusive=True)
    cache=PreparationCache();specs={};rows=[]
    with cache.installed():
        for split,count in [('fit',640),('internal',160)]:
            for index in range(count):
                spec=S.sample_scene(catalog,split,index,SEED);spec['catalog']=B.binding(CATALOG);specs[split,index]=spec
        for block in range(160):
            ordered=[specs['fit',4*block+j]for j in range(4)]+[specs['internal',block]]
            for spec in ordered:
                index=len(rows);path=root/'queue/specs'/f'{index:03d}.json';B.save(path,spec,exclusive=True)
                row=dict(index=index,scene_id=spec['scene_id'],split=spec['split'],status=spec['status'],scene=None,
                    scene_spec=B.binding(path),source_object_ids=spec['source_object_ids'],source_object_digests=spec['source_object_digests'],
                    origin='fresh_fixed_seed_scene_no_outcome_resampling')
                if spec['status']=='ready':
                    S.build_scene(path,root/'queue/build'/f'{index:03d}')
                    row['scene']=B.binding(root/'queue/build'/f'{index:03d}'/'scene.json')
                rows.append(row)
                if len(rows)%25==0:print(dict(prepared=len(rows),maximum=800),flush=True)
    # Verify every original file again without cache before any GPU dispatch.
    S.verify(catalog['input_sha256']);S.verify(catalog['source_sha256'])
    B.save(root/'PREPARATION_CACHE.json',cache.receipt(),exclusive=True)
    if len({r['scene_id']for r in rows})!=800:raise ValueError('Repeated new scene identity')
    if any(any(r['scene_id']in str(t)for t in old_data['trial_id'])for r in rows):raise ValueError('Old label replay')
    queue=root/'QUEUE_SOURCE.json';B.save(queue,dict(rows=rows),exclusive=True)
    template_queue=root/'TEMPLATE_QUEUE.json';B.save(template_queue,dict(rows=rows[:16]),exclusive=True)
    base=F.freeze(root/'base_freeze',environments=16,source_protocol=template_queue)
    directory=root/'collection';directory.mkdir();p=copy.deepcopy(base)
    for key in list(p['inputs']):
        if key.startswith('scene:'):del p['inputs'][key]
    for row in rows:
        p['inputs']['scene_spec:'+str(row['index'])]=row['scene_spec']
        if row['scene']:p['inputs']['scene:'+str(row['index'])]=row['scene']
    p.update(schema='anydex_mjx_cgr_fit_quota_v1',rows=rows,seed=SEED,
        existing_fit_records=16,existing_internal_records=4,target_fit_records=100,
        max_global_control_ticks=930000,max_native_environment_steps=119040000,stage_native_budget=119040000,
        max_CGR_forwards=800,max_planning_jobs=800,max_grasp_attempts=800,
        minimum_free_disk_bytes=80*1024**3,formal_collection=True,formal_production_quota_frozen=True,
        formal_scene_budget=800,formal_fit_target=100,video_recording=False,training_updates=0,
        comparison_underfilled=False,background_closed_wave_audits=True,
        staging_records_exclude_training_until_audit=True)
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT']=str(directory/'contact_spool')
    p['inputs'].update(full_queue=B.binding(queue),campaign_intent=B.binding(root/'CAMPAIGN_INTENT.json'),
        expansion_catalog=B.binding(CATALOG),existing_formal_manifest=B.binding(BASELINE),
        existing_dataset=baseline['dataset'],baseline_protocol=B.binding(baseline['source_protocol']['path']),
        base_freeze_protocol=B.binding(root/'base_freeze/PROTOCOL.json'))
    for name in (Path(__file__).name,ENTRY.name,'author_mjx_quota_progress.py','author_mjx_quota_data.py',
        'author_scene_prepare_cache.py','author_mjx_hour_collection.py','author_expansion_progress.py',
        'author_hour_metrics.py','author_hour_audit.py','audit_author_mjx_wave_snapshot.py',
        'finish_author_mjx_collection.py','finish_author_expansion_metadata.py','author_training.py','run_author_mjx_expansion.py'):
        p['sources'][name]=B.binding(B.HERE/name)
    same_domain(B.read(baseline['source_protocol']['path']),p)
    archive=directory/'source_archive';archive.mkdir()
    for name,binding in p['sources'].items():shutil.copyfile(binding['path'],archive/name)
    B.save(directory/'PROTOCOL.json',p,exclusive=True)
    shutil.copytree(B.ROOT/'outputs/boya_mjx_fresh_expansion_20261118_v1/collection/warp_cache',directory/'warp_cache')
    result=dict(status='quota_campaign_frozen_not_started',target_fit_records=100,existing_fit_records=16,
        existing_internal_records=4,remaining_fit_records=84,max_new_scenes=800,
        ready_layouts=sum(r['scene']is not None for r in rows),placement_nulls=sum(r['scene']is None for r in rows),
        environments=16,max_native_environment_steps=119040000,video_recording=False,training_updates=0,
        collection_protocol=B.binding(directory/'PROTOCOL.json'))
    B.save(root/'CAMPAIGN_READY.json',result,exclusive=True);print(result,flush=True);return result


def run(root):
    root=Path(root).resolve();ready=B.read(root/'CAMPAIGN_READY.json');B.verify({'protocol':ready['collection_protocol']})
    p=B.read(root/'collection/PROTOCOL.json');B.verify(p['sources']);B.verify(p['inputs'])
    if (root/'PIPELINE_STARTED.json').exists():raise ValueError('No dispatched-root retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=B.ROOT).strip():raise ValueError('Clean worktree required')
    B.save(root/'PIPELINE_STARTED.json',dict(process=R.process_identity(__import__('os').getpid()),unix_s=time.time(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=B.ROOT,text=True).strip(),
        worktree_clean=True,target_fit_records=100),exclusive=True)
    began=time.monotonic();stages={}
    try:
        stages['collect_and_audit']=dispatch([sys.executable,'-B',str(ENTRY),'--root',str(root/'collection')],
            root/'timing/collect_and_audit',64800)
        stages['staging_export']=dispatch([sys.executable,'-B',str(B.HERE/'finish_author_mjx_collection.py'),
            '--root',str(root/'collection')],root/'timing/staging_export',1800)
        promote(root);dataset=merge(root);R.assert_quiescent(root/'collection')
        status='target_complete'if dataset['target_met']else'finite_capacity_complete_target_not_met'
        B.save(root/'CAMPAIGN_RESULTS.json',dict(status=status,wall_s=time.monotonic()-began,stages=stages,
            merged_manifest=B.binding(root/'merged/MANIFEST.json'),fit=dataset['fit'],internal=dataset['internal'],
            target_fit_records=100,video_recording=False,training_updates=0),exclusive=True)
        print(dict(status=status,fit=dataset['fit'],internal=dataset['internal']),flush=True)
    except BaseException:
        import traceback
        B.save(root/'PIPELINE_FAILURE.json',dict(error=traceback.format_exc(),wall_s=time.monotonic()-began,stages=stages),exclusive=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    if args.action=='freeze':freeze(args.root)
    else:run(args.root)
