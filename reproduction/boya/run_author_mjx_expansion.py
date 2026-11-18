"""Freeze, run and audit-promote the first 100 fresh g1 scene attempts.

80 fit / 20 internal scene draws use the existing deterministic source sampler.
Nulls consume their slots; no relabeling, replacement, or training is performed.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import author_collection_batch as B
import author_collection_scenes as S
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F

SEED = 2026111802
CATALOG = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/manual_start_20261116_v2/scene_queue/CATALOG.json')
ENTRY = B.HERE/'author_mjx_expansion_collection.py'


def freeze(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    catalog = B.read(CATALOG)
    S.verify(catalog['input_sha256'])
    S.verify(catalog['source_sha256'])
    B.save(root/'BATCH_INTENT.json', dict(scene_budget=100, split_scene_counts=dict(fit=80, internal=20),
        seed=SEED, catalog=B.binding(CATALOG), gesture_ids=[1], environments=16,
        candidate_attempts_per_scene=1, retries=0, replace_nulls=False,
        object_split_sizes=dict(fit=32, internal=8), training_updates=0,
        previous_diagnostic_data_merged=False), exclusive=True)
    specs = {}
    cache = {}
    for split,count in [('fit',80),('internal',20)]:
        for index in range(count):
            spec = S.sample_scene(catalog,split,index,SEED,geometry_cache=cache)
            spec['catalog'] = B.binding(CATALOG)
            specs[split,index] = spec
    # Interleave four fit and one internal draw, without looking at geometry or outcomes.
    ordered = []
    for block in range(20):
        ordered.extend(specs['fit',4*block+j] for j in range(4))
        ordered.append(specs['internal',block])
    rows = []
    for index,spec in enumerate(ordered):
        path = root/'queue/specs'/f'{index:03d}.json'
        B.save(path,spec,exclusive=True)
        row = dict(index=index,scene_id=spec['scene_id'],split=spec['split'],status=spec['status'],
            scene=None,scene_spec=B.binding(path),source_object_ids=spec['source_object_ids'],
            source_object_digests=spec['source_object_digests'],origin='fresh_fixed_seed_no_replacement')
        if spec['status']=='ready':
            scene = S.build_scene(path,root/'queue/build'/f'{index:03d}')
            row['scene'] = B.binding(root/'queue/build'/f'{index:03d}'/'scene.json')
        rows.append(row)
        print(dict(prepared=index+1,total=100,status=spec['status']),flush=True)
    drawn = {k for r in rows for k in r['source_object_ids']}
    if drawn != set(catalog['objects']):
        raise ValueError('Fixed first-batch draws do not cover the 40-object catalog; do not redraw silently')
    queue = root/'QUEUE_SOURCE.json'
    B.save(queue,dict(rows=rows,seed=SEED,new_independent_scene_draws=100,repeated_layouts=False),exclusive=True)
    base = F.freeze(root/'base_freeze',environments=16,source_protocol=queue)
    directory = root/'collection'
    directory.mkdir()
    p = copy.deepcopy(base)
    p.update(schema='anydex_mjx_fresh_expansion_v1',seed=SEED,new_independent_scene_draws=100,
        formal_collection=True,formal_production_quota_frozen=True,formal_scene_budget=100,
        target_valid_label_count=None,background_closed_wave_audits=True,
        comparison_underfilled=False,repeated_layouts=False,training_updates=0,
        staging_records_exclude_training_until_audit=True)
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT'] = str(directory/'contact_spool')
    p['inputs'].update(base_freeze_protocol=B.binding(root/'base_freeze/PROTOCOL.json'),
        expansion_catalog=B.binding(CATALOG),expansion_intent=B.binding(root/'BATCH_INTENT.json'))
    for row in rows:
        p['inputs']['scene_spec:'+str(row['index'])] = row['scene_spec']
    for name in ('author_mjx_hour_collection.py','author_hour_metrics.py','author_hour_audit.py',
        'audit_author_mjx_wave_snapshot.py','author_expansion_progress.py',ENTRY.name,
        'finish_author_mjx_collection.py',Path(__file__).name):
        p['sources'][name] = B.binding(B.HERE/name)
    archive = directory/'source_archive'
    archive.mkdir()
    for name,binding in p['sources'].items():
        shutil.copyfile(binding['path'],archive/name)
    B.save(directory/'PROTOCOL.json',p,exclusive=True)
    cache_source = B.ROOT/'outputs/boya_mjx_actual_one_hour_20261118_v1/collection/warp_cache'
    shutil.copytree(cache_source,directory/'warp_cache')
    ready_ids = {k for r in rows if r['scene'] for k in r['source_object_ids']}
    receipt = dict(status='frozen_not_started',scene_draws=100,
        ready_scenes=sum(r['scene'] is not None for r in rows),
        placement_nulls=sum(r['scene'] is None for r in rows),
        drawn_object_types=len(drawn),ready_scene_object_types=len(ready_ids),
        missing_from_ready_scenes=sorted(set(catalog['objects'])-ready_ids),
        environments=16,max_native_environment_steps=p['max_native_environment_steps'],
        collection_protocol=B.binding(directory/'PROTOCOL.json'),physics_steps=0,training_updates=0)
    B.save(root/'EXPANSION_READY.json',receipt,exclusive=True)
    print(receipt,flush=True)
    return receipt


def dispatch(command,directory,timeout_s):
    directory.mkdir(parents=True,exist_ok=False)
    B.save(directory/'DISPATCH.json',dict(command=command,timeout_s=timeout_s),exclusive=True)
    began = time.monotonic()
    with (directory/'process.log').open('x') as log:
        process = subprocess.Popen(command,cwd=B.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity = R.process_identity(process.pid)
        B.save(directory/'PROCESS.json',dict(process=identity,started_unix_s=time.time()),exclusive=True)
        try:
            code = process.wait(timeout=timeout_s)
        except BaseException:
            if R.process_identity(process.pid)==identity:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=20)
            raise
    alive = R.group_alive(identity['pgid'])
    result = dict(returncode=code,wall_s=time.monotonic()-began,process=identity,process_group_exited=not alive)
    B.save(directory/'FINISHED.json',result,exclusive=True)
    if code or alive:
        raise RuntimeError('Expansion stage failed: '+str(directory))
    return result


def coverage(root):
    """Report scene presence, observed hand load, and actual lifts separately."""
    p = B.read(root/'PROTOCOL.json')
    catalog = B.read(p['inputs']['expansion_catalog']['path'])
    rows = {k:dict(split=('fit' if k in catalog['split_object_ids']['fit'] else 'internal'),
        drawn_scenes=0,ready_scenes=0,executed_scenes=0,eligible_scenes=0,
        observed_robot_contact_trials=0,observed_hand_load_trials=0,successful_lift_trials=0,
        contact_without_successful_lift_trials=0) for k in catalog['objects']}
    for spec in p['rows']:
        for k in spec['source_object_ids']:
            rows[k]['drawn_scenes'] += 1
            rows[k]['ready_scenes'] += int(spec['scene'] is not None)
        directory = root/'scenes'/f"{spec['index']:02d}"
        execution_path = directory/'execution/RESULTS.json'
        if not execution_path.exists():
            continue
        record = B.read(directory/'RECORD.json')
        execution = B.read(execution_path)
        loaded,contacted = set(),set()
        # Read complete existing traces; this is observation, never a new control gate.
        with Path(execution['trace']['path']).open() as stream:
            for line in stream:
                state = json.loads(line)
                for k,value in state['objects'].items():
                    if sum(value['finger_loads_n'])>0:
                        loaded.add(k)
                for contact in state['contacts']:
                    bodies = contact['bodies']
                    if any(b.startswith('/World/Robot/') for b in bodies):
                        for k in spec['source_object_ids']:
                            if '/World/'+k in bodies:
                                contacted.add(k)
        outcomes = {v['object_id']:v for v in execution['evaluation']['object_outcomes']}
        for k in spec['source_object_ids']:
            row = rows[k]
            success = bool(outcomes[k]['task_success'])
            row['executed_scenes'] += 1
            row['eligible_scenes'] += int(record['eligible'])
            row['observed_robot_contact_trials'] += int(k in contacted)
            row['observed_hand_load_trials'] += int(k in loaded)
            row['successful_lift_trials'] += int(record['eligible'] and success)
            row['contact_without_successful_lift_trials'] += int(record['eligible'] and k in contacted and not success)
    return dict(objects=rows,object_types=40,
        scene_cooccurrence_is_not_per_object_positive_negative_label=True,
        contact_without_lift_is_diagnostic_not_a_negative_training_label=True,
        untouched_distractors_are_not_negative_attempts=True,
        global_label='Original any-of-three lift success',additional_physics_steps=0)


def promote(root):
    """Create new formal records only after the complete original raw audit/export."""
    directory = root/'collection'
    p = B.read(directory/'PROTOCOL.json')
    B.verify(p['inputs']);B.verify(p['sources'])
    audit = B.read(directory/'READ_ONLY_AUDIT.json')
    manifest = B.read(directory/'export/MANIFEST.json')
    if (audit['status']!='read_only_audit_pass'
        or audit['original_result']!=B.binding(directory/'RESULTS.json')
        or manifest['source_audit']!=B.binding(directory/'READ_ONLY_AUDIT.json')):
        raise ValueError('Formal eligibility requires a bound complete audit')
    B.verify({'dataset':manifest['dataset']})
    checked = {r['index']:r for r in audit['rounds']}
    out = root/'formal'
    out.mkdir(exist_ok=False)
    with np.load(manifest['dataset']['path'],allow_pickle=False) as data:
        arrays = {k:data[k].copy() for k in data.files}
    refs,hashes = [],[]
    for old_path in arrays['record_json']:
        path = Path(str(old_path));record = B.read(path);index = int(path.parent.name)
        if not record['eligible'] or not checked[index].get('return_pass'):
            raise ValueError('No promotion of nulls or failed returns')
        result = dict(record,schema='anydex_mjx_audited_expansion_record_v1',
            exclude_training=False,formal_training_eligible=True,source_staged_record=B.binding(path),
            independent_audit=B.binding(directory/'READ_ONLY_AUDIT.json'),
            cohort='fresh_g1_expansion_20261118_v1',training_updates=0)
        dest = out/'records'/f'{index:03d}.json'
        B.save(dest,result,exclusive=True);refs.append(str(dest));hashes.append(B.sha(dest))
    arrays['record_json'] = np.asarray(refs,str)
    arrays['record_sha256'] = np.asarray(hashes,str)
    target = out/'gesture_1.npz'
    np.savez_compressed(target,**arrays)
    with np.load(target,allow_pickle=False) as data:
        for k,v in arrays.items():
            np.testing.assert_array_equal(data[k],v)
    B.save(out/'OBJECT_COVERAGE.json',coverage(directory),exclusive=True)
    result = dict(manifest,status='audited_fresh_expansion_complete',dataset=B.binding(target),
        exclude_training=False,formal_training_eligible=True,source_staging_export=B.binding(directory/'export/MANIFEST.json'),
        object_coverage=B.binding(out/'OBJECT_COVERAGE.json'),scene_budget=100,new_independent_scene_draws=100,
        training_updates=0,old_data_merged=False,
        note='Eligibility is for this separate MJX functional domain; no training is dispatched. Legacy target mesh ID is not actual grasped-object attribution.')
    B.save(out/'MANIFEST.json',result,exclusive=True)
    return result


def run(root):
    root = Path(root).resolve()
    ready = B.read(root/'EXPANSION_READY.json')
    B.verify({'protocol':ready['collection_protocol']})
    if (root/'PIPELINE_STARTED.json').exists():
        raise ValueError('No dispatched-root retry')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=B.ROOT,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain'],cwd=B.ROOT,text=True)
    if dirty:
        raise ValueError('Commit and push the reviewed source before starting expansion')
    B.save(root/'PIPELINE_STARTED.json',dict(process=R.process_identity(os.getpid()),unix_s=time.time(),
        git_commit=revision,git_worktree_clean=True),exclusive=True)
    stages = {}
    began = time.monotonic()
    try:
        stages['collect_and_audit'] = dispatch([sys.executable,'-B',str(ENTRY),'--root',str(root/'collection')],
            root/'timing/collect_and_audit',21600)
        stages['staging_export'] = dispatch([sys.executable,'-B',str(B.HERE/'finish_author_mjx_collection.py'),
            '--root',str(root/'collection')],root/'timing/staging_export',1800)
        formal = promote(root)
        R.assert_quiescent(root/'collection')
        B.save(root/'EXPANSION_RESULTS.json',dict(status='complete',wall_s=time.monotonic()-began,
            stages=stages,formal_manifest=B.binding(root/'formal/MANIFEST.json'),
            labels=formal['records'],positive=formal['positive'],negative=formal['negative'],
            git_commit=revision,training_updates=0),exclusive=True)
        print(dict(status='complete',labels=formal['records'],positive=formal['positive'],negative=formal['negative']),flush=True)
    except BaseException:
        import traceback
        B.save(root/'PIPELINE_FAILURE.json',dict(error=traceback.format_exc(),stages=stages,
            elapsed_s=time.monotonic()-began),exclusive=True)
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    if args.action=='freeze':
        freeze(args.root)
    else:
        run(args.root)
