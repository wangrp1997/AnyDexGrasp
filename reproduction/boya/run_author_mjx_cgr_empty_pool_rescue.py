"""Freeze, run and audit a fresh MJX trial of the empty-pool CGR recovery."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import author_collection_batch as B
import author_collection_wave_resources as R
from author_mjx_cgr_empty_pool_rescue import RESCUE_LIMIT, VARIANT, original_pool_empty
import run_author_mjx_cgr_fallback as baseline
from run_author_mjx_expansion import dispatch


SEED = 2026111906
WORKER = 'author_mjx_cgr_empty_pool_rescue.py'


def freeze(root):
    with patch.object(baseline, 'SEED', SEED):
        baseline.freeze(root/'baseline_freeze')
    base = root/'baseline_freeze/collection'
    collection = root/'collection'
    protocol = B.read(base/'PROTOCOL.json')
    protocol.update(schema='anydex_boya_mjx_cgr_empty_pool_rescue_v1', rescue_planning_limit=RESCUE_LIMIT,
                    rescue_variant=VARIANT, rescue_trigger='original_post_collision_candidate_count_zero',
                    candidate_policy='First feasible arm plan from up to five original CGR ranks. An empty post-collision pool allows up to two zero-extra-depth roll recovery ranks.')
    protocol['env']['ANYDEX_CONTACT_SPOOL_ROOT'] = str(collection/'contact_spool')
    protocol['inputs']['baseline_freeze'] = B.binding(base/'PROTOCOL.json')
    names = (Path(__file__).name, WORKER, 'audit_cgr_approach_depth_recovery.py')
    protocol['sources'].update({name: B.binding(B.HERE/name) for name in names})
    B.verify(protocol['sources'])
    B.verify(protocol['inputs'])
    archive = collection/'source_archive'
    archive.mkdir(parents=True)
    for name, reference in protocol['sources'].items():
        shutil.copyfile(reference['path'], archive/name)
    B.save(collection/'PROTOCOL.json', protocol, exclusive=True)
    shutil.copytree(base/'warp_cache', collection/'warp_cache')
    B.save(root/'READY.json', dict(status='frozen', protocol=B.binding(collection/'PROTOCOL.json'),
        seed=SEED, draws=len(protocol['rows']), placement_ready=protocol['max_grasp_attempts'],
        max_primary_ranks=protocol['cgr_planning_limit'], max_rescue_ranks=RESCUE_LIMIT,
        max_plans=protocol['max_actual_arm_plans'], max_grasps=protocol['max_grasp_attempts'],
        max_native_environment_steps=protocol['max_native_environment_steps'], video_recording=False), exclusive=True)
    print(B.read(root/'READY.json'), flush=True)


def audit_selection(reference, limit):
    B.verify({'selection': reference})
    selection = B.read(reference['path'])
    B.verify({'pool': selection['shared_pool'], 'observation': selection['observation']})
    candidates = B.read(selection['shared_pool']['path'])['candidates']
    attempts = selection['attempts']
    empty = original_pool_empty(selection)
    ready = selection['status'] == 'ready'
    assert selection['candidate_limit'] == limit and len(candidates) == selection['candidate_count']
    assert 1 <= len(attempts) <= limit and selection['physics_steps'] == 0
    assert selection['selected_rank'] == (len(attempts)-1 if ready else None)
    assert attempts[-1]['status'] == selection['status']
    for rank, attempt in enumerate(attempts):
        B.verify({'plan': attempt['plan']})
        assert attempt['status'] == B.read(attempt['plan']['path'])['status']
        assert attempt['rank'] == (None if empty else rank)
        assert attempt['candidate_sha256'] == (None if empty else candidates[rank]['candidate_sha256'])
    assert all(attempt['status'] == 'null_geometry' for attempt in attempts[:-1])
    return dict(status=selection['status'], candidate_count=len(candidates), observation=selection['observation'],
                arm_plans=0 if empty else len(attempts), selected_rank=selection['selected_rank'],
                candidate_sha256=attempts[-1]['candidate_sha256'] if ready else None, selection=reference)


def summarize(root):
    collection = root/'collection'
    protocol = B.read(collection/'PROTOCOL.json')
    result_file = collection/'RESULTS.json'
    result = B.read(result_file)
    audit_file = collection/'READ_ONLY_AUDIT.json'
    audit = B.read(audit_file)
    if result['status'] != 'complete' or audit['status'] != 'read_only_audit_pass' or audit['original_result'] != B.binding(result_file):
        raise ValueError('Completed closed-wave audit required')
    audited = {row['index']: row for row in audit['rounds']}
    scenes = []
    totals = Counter(primary_empty=0, primary_ready=0, primary_lifted=0, rescue_nonempty=0,
                     rescue_ready=0, rescue_lifted=0, arm_plans=0)
    for spec, record in zip(protocol['rows'], result['records'], strict=True):
        index = spec['index']
        directory = collection/'scenes'/f'{index:02d}'
        row = dict(index=index, status=record['status'], label=record['label'])
        scenes.append(row)
        selection_path = directory/'cgr_attempts/SELECTION.json'
        receipt_path = directory/'EMPTY_RESCUE.json'
        if not selection_path.exists():
            assert not receipt_path.exists() and 'execution' not in record['bindings']
            continue
        primary = audit_selection(B.binding(selection_path), protocol['cgr_planning_limit'])
        row['primary'] = primary
        empty = original_pool_empty(primary)
        assert receipt_path.exists() == empty
        chosen = primary
        mode = 'primary'
        if empty:
            totals['primary_empty'] += 1
            receipt = B.read(receipt_path)
            B.verify({'live_state': receipt['live_state']})
            assert receipt['original_selection'] == primary['selection']
            assert receipt['variant'] == VARIANT and receipt['physics_steps'] == 0
            assert receipt['live_state'] == B.binding(directory/'capture/LIVE_STATE.json')
            chosen = audit_selection(receipt['recovery_selection'], RESCUE_LIMIT)
            assert chosen['observation'] == primary['observation'] == receipt['observation']
            assert chosen['status'] == receipt['recovery_status']
            row['rescue'] = chosen
            totals['rescue_nonempty'] += int(chosen['candidate_count'] > 0)
            mode = 'rescue'
        totals['arm_plans'] += chosen['arm_plans']
        assert chosen['arm_plans'] <= protocol['cgr_planning_limit']
        if chosen['status'] != 'ready':
            assert 'execution' not in record['bindings']
            continue
        assert audited[index]['original_controls_exact'] and audited[index]['return_pass']
        actual = B.read(record['bindings']['candidate']['path'])
        assert actual['candidate_sha256'] == chosen['candidate_sha256']
        if empty:
            assert actual['recovery_variant'] == VARIANT and actual['extra_depth_m'] == 0
        totals[f'{mode}_ready'] += 1
        totals[f'{mode}_lifted'] += int(record['status'] == 'grasp_success')
    counts = Counter(row['status'] for row in scenes)
    assert totals['primary_ready'] + totals['rescue_ready'] == result['cost']['grasp_attempts'] <= protocol['max_grasp_attempts']
    assert totals['primary_lifted'] + totals['rescue_lifted'] == counts['grasp_success']
    assert totals['arm_plans'] <= protocol['max_actual_arm_plans']
    B.verify(protocol['sources'])
    B.verify(protocol['inputs'])
    B.verify({name: dict(path=str(collection/'source_archive'/name), sha256=ref['sha256'])
              for name, ref in protocol['sources'].items()})
    R.assert_quiescent(collection)
    report = dict(status='complete', scenes=scenes, totals=dict(totals), statuses=dict(counts),
        drawn_scenes=len(scenes), observed_scenes=result['cost']['CGR_forwards'],
        native_environment_steps=audit['native_environment_integrations'], native_audit=B.binding(audit_file),
        controller_wall_s=result['parent_wall_s'], video_recording=False, training_updates=0, exclude_training=True)
    B.save(root/'RESCUE_RESULTS.json', report, exclusive=True)
    print({key: value for key, value in report.items() if key != 'scenes'}, flush=True)


def run(root):
    B.verify({'protocol': B.read(root/'READY.json')['protocol']})
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=B.ROOT).strip():
        raise ValueError('Start the fixed trial from a committed clean tree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=B.ROOT, text=True).strip()
    B.save(root/'PIPELINE_STARTED.json', dict(timestamp=time.time(), git_commit=commit), exclusive=True)
    try:
        stage = dispatch([sys.executable, '-B', str(B.HERE/WORKER), 'collect', str(root/'collection')], root/'timing', 3600)
        summarize(root)
        B.save(root/'PIPELINE_FINISHED.json', dict(stage=stage, result=B.binding(root/'RESCUE_RESULTS.json')), exclusive=True)
    except BaseException:
        B.save(root/'PIPELINE_FAILURE.json', dict(error=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'summarize'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'summarize': summarize}[args.action](args.root.resolve())
