"""Freeze, run, and review a bounded shared-observation grasp comparison."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

import author_collection_batch as B
import author_collection_scenes as S
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F
from author_scene_prepare_cache import PreparationCache
from run_author_mjx_expansion import dispatch


SEED = 2026111904
DRAWS = 128
READY_LIMIT = 80
TARGET = 8
ENVIRONMENTS = 16
CATALOG = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/manual_start_20261116_v2/scene_queue/CATALOG.json')
DIAGNOSIS = B.ROOT/'outputs/boya_mjx_head_diagnosis_20261119_v1'
SOURCE_CACHE = B.ROOT/'outputs/boya_mjx_fresh_expansion_20261118_v1/collection/warp_cache'
ENTRY = B.HERE/'author_mjx_shared_pair.py'


def freeze(root):
    root.mkdir(parents=True, exist_ok=False)
    catalog = B.read(CATALOG)
    S.verify(catalog['input_sha256']); S.verify(catalog['source_sha256'])
    review = B.read(DIAGNOSIS/'FINAL_REVIEW.json')
    if review['status'] != 'head_diagnosis_final_review_pass':
        raise ValueError('Completed original head diagnosis required')
    B.verify({'diagnosis': review['results']})
    diag = B.read(DIAGNOSIS/'PROTOCOL.json')
    B.verify({'dataset': diag['dataset']})
    with np.load(diag['dataset']['path'], allow_pickle=False) as data:
        fit_objects = set(data['scene_mesh_sha256'][data['split'] == 'train'].ravel())
        old_trials = data['trial_id'].tolist()
    head = B.binding(DIAGNOSIS/'linear_output/final_diagnostic.pt')
    rng = np.random.default_rng(SEED)
    rows, draws = [], []
    with PreparationCache().installed():
        for number in range(DRAWS):
            spec = S.sample_scene(catalog, 'internal', number, SEED)
            spec['catalog'] = B.binding(CATALOG)
            if (any(spec['scene_id'] in trial for trial in old_trials)
                    or fit_objects & set(spec['source_object_digests'])):
                raise ValueError('Fresh layout uses fit data or a previous trial identity')
            spec_file = root/'draws'/f'{number:03d}.json'
            B.save(spec_file, spec, exclusive=True)
            draw = dict(number=number, scene_id=spec['scene_id'], placement=spec['status'], spec=B.binding(spec_file))
            draws.append(draw)
            if spec['status'] != 'ready' or len(rows) >= 2*READY_LIMIT:
                continue
            pair = len(rows)//2
            build = root/'layouts'/f'{pair:03d}'
            S.build_scene(spec_file, build)
            scene = B.binding(build/'scene.json')
            policies = ('cgr', 'head') if rng.integers(2) == 0 else ('head', 'cgr')
            for policy in policies:
                rows.append(dict(index=len(rows), pair_index=pair, policy=policy, draw=number,
                    scene_id=spec['scene_id'], scene=scene, scene_spec=draw['spec'], split='internal', status='ready',
                    source_object_ids=spec['source_object_ids'], source_object_digests=spec['source_object_digests']))
    S.verify(catalog['input_sha256']); S.verify(catalog['source_sha256'])
    if not rows:
        raise ValueError('No placement-ready paired layouts in the fixed draws')
    B.save(root/'DRAWS.json', draws, exclusive=True)
    queue = root/'QUEUE_SOURCE.json'
    B.save(queue, dict(rows=rows[:ENVIRONMENTS]), exclusive=True)
    directory = root/'collection'
    p = F.freeze(root/'base_freeze', environments=ENVIRONMENTS, source_protocol=queue)
    p['inputs'] = {key: value for key, value in p['inputs'].items() if not key.startswith('scene:')}
    for row in rows[::2]:
        p['inputs'][f"paired_scene:{row['pair_index']}"] = row['scene']
        p['inputs'][f"scene_spec:{row['pair_index']}"] = row['scene_spec']
    p['inputs'].update(comparison_head=head, diagnosis_review=B.binding(DIAGNOSIS/'FINAL_REVIEW.json'),
                       scene_catalog=B.binding(CATALOG), draws=B.binding(root/'DRAWS.json'))
    for name in (Path(__file__).name, ENTRY.name, 'author_mjx_shared_pair_worker.py', 'author_mjx_policy_compare_worker.py',
                 'author_training.py', 'author_scene_prepare_cache.py', 'author_expansion_progress.py', 'author_hour_metrics.py',
                 'author_hour_audit.py', 'audit_author_mjx_wave_snapshot.py', 'run_author_mjx_expansion.py'):
        p['sources'][name] = B.binding(B.HERE/name)
    for name, binding in p['sources'].items():
        dest = directory/'source_archive'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(binding['path'], dest)
    waves = (len(rows) + ENVIRONMENTS - 1)//ENVIRONMENTS
    ticks = waves * (2500 + 8000 + 8100)
    p.update(schema='anydex_mjx_shared_pair_comparison_v1', rows=rows, seed=SEED,
             target_pairs=TARGET, max_new_layout_draws=DRAWS, max_CGR_forwards=len(rows)//2,
             max_planning_jobs=len(rows), max_grasp_attempts=2*TARGET, max_global_control_ticks=ticks,
             max_native_environment_steps=ticks*ENVIRONMENTS*8, stage_native_budget=ticks*ENVIRONMENTS*8,
             minimum_free_disk_bytes=80*1024**3, video_recording=False, comparison_only=True,
             formal_training_eligible=False,
             admission='First eight different choices in draw order, including any no-plan outcomes',
             selection='Unchanged post-collision top10, original CGR vs frozen preactivation-MSE head using published ReLU inference',
             options='Each ranking keeps its fixed top1, with CGR order used to break head-score ties')
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT'] = str(directory/'contact_spool')
    B.save(directory/'PROTOCOL.json', p, exclusive=True)
    shutil.copytree(SOURCE_CACHE, directory/'warp_cache')
    ready = dict(status='frozen', protocol=B.binding(directory/'PROTOCOL.json'), draws=DRAWS,
                 placement_ready=sum(draw['placement']=='ready' for draw in draws), available_pairs=len(rows)//2,
                 target_pairs=TARGET, maximum_physical_attempts=2*TARGET, max_native_environment_steps=p['max_native_environment_steps'],
                 video_recording=False)
    B.save(root/'COMPARISON_READY.json', ready, exclusive=True)
    print(ready, flush=True)


def summarize(root):
    from author_mjx_shared_pair import STATE
    from author_mjx_shared_pair_worker import Scorer, choice_receipt

    directory = root/'collection'
    p = B.read(directory/'PROTOCOL.json')
    result = B.read(directory/'RESULTS.json')
    audit = B.read(directory/'READ_ONLY_AUDIT.json')
    if audit['status'] != 'read_only_audit_pass' or audit['original_result'] != B.binding(directory/'RESULTS.json'):
        raise ValueError('Original complete native audit required')
    B.verify(p['inputs']); B.verify(p['sources'])
    by_index = {row['index']: row for row in audit['rounds']}
    scorer = Scorer(p['inputs']['comparison_head'])
    pairs, coverage = [], Counter()
    for i, spec in enumerate(p['rows'][::2]):
        folder = directory/'pairs'/f'{i:03d}'
        if not (folder/'ADMISSION.json').exists():
            coverage[result['records'][spec['index']]['status']] += 1
            continue
        admission = B.read(folder/'ADMISSION.json')
        B.verify({k: admission[k] for k in ('start', 'ranking')})
        start = B.read(folder/'SHARED_START.json')
        ranking = B.read(folder/'RANKING.json')
        B.verify({k: start[k] for k in ('state', 'observation', 'measured_scene', 'live_state', 'representation')})
        B.verify({'pool': ranking['shared_pool']})
        declared = choice_receipt(B.read(folder/'ORIGINAL_POOL.json'), scorer)
        if any(ranking[k] != value for k, value in declared.items()):
            raise ValueError('Independent score replay disagrees with execution ranking')
        if ranking['observation'] != start['observation'] or ranking['head'] != p['inputs']['comparison_head']:
            raise ValueError('Checkpoint or observation changed')
        indices = [spec['index'], spec['index'] + 1]
        if start['indices'] != indices or [start['donor_lane'], start['receiver_lane']] != [i % p['environments'] for i in indices]:
            raise ValueError('Native integration-state evidence belongs to another pair')
        live = B.read(start['live_state']['path'])
        observation = B.read(start['observation']['path'])
        if not live['scene_id'] == observation['scene_id'] == spec['scene_id'] or not live['sim_step'] == observation['world_step'] == start['global_tick']:
            raise ValueError('Shared image and measured state belong to different frames')
        with np.load(start['state']['path'], allow_pickle=False) as state:
            if set(state.files) != set(STATE):
                raise ValueError('Missing integration buffer')
            for values in state.values():
                np.testing.assert_array_equal(values[0], values[1])
        expected_admission = ranking['disagreement'] and len(pairs) < p['target_pairs']
        if admission['admitted'] != expected_admission or admission['disagreement'] != ranking['disagreement']:
            raise ValueError('Pairs were not admitted in fixed draw order')
        coverage['observed_pairs'] += 1
        coverage['multiple_candidates'] += int(ranking['candidate_count'] >= 2)
        coverage['different_choices'] += int(ranking['disagreement'])
        if not admission['admitted']:
            continue
        policies = {}
        for index in start['indices']:
            row = p['rows'][index]
            scene = directory/'scenes'/f'{index:02d}'
            record = result['records'][index]
            if (B.binding(scene/'capture/observation.json') != start['observation']
                    or B.binding(scene/'representation/RESULTS.json') != start['representation']):
                raise ValueError('Policy used a different observation')
            choice = B.read(scene/'contact/SHARED_CHOICE.json')
            chosen = ranking['hashes'][ranking[row['policy']+'_rank']]
            if choice['selected_candidate_sha256'] != chosen or choice['ranking'] != admission['ranking']:
                raise ValueError('Policy did not consume its original pool slot')
            executed = 'execution' in record['bindings']
            if executed:
                candidate = B.read(record['bindings']['candidate']['path'])
                if candidate['candidate_sha256'] != chosen or not by_index[index].get('return_pass'):
                    raise ValueError('Execution binding or physical return not verified')
            if not by_index[index].get('CGR_input_and_receipt_exact'):
                raise ValueError('Shared RGB-D input lacks original native audit')
            policies[row['policy']] = dict(status=record['status'], label=record['label'], executed=executed,
                candidate_sha256=chosen, index=index)
        pairs.append(dict(draw=spec['draw'], scene_id=spec['scene_id'], candidate_count=ranking['candidate_count'],
                          policies=policies, receipt=B.binding(folder/'ADMISSION.json')))
    if len(pairs) != result['cost']['disagreement_pairs'] or len(pairs) > p['target_pairs']:
        raise ValueError('Pair accounting differs from controller')
    if coverage['observed_pairs'] != result['cost']['CGR_forwards'] or coverage['observed_pairs'] != result['world']['camera_frames']:
        raise ValueError('Each observed pair must have exactly one camera frame and one CGR forward')
    if sum(trial['executed'] for pair in pairs for trial in pair['policies'].values()) != result['cost']['grasp_attempts']:
        raise ValueError('Execution accounting differs from controller')
    matched = [pair for pair in pairs if all(t['label'] is not None for t in pair['policies'].values())]
    wins = Counter('head' if pair['policies']['head']['label'] > pair['policies']['cgr']['label'] else 'cgr'
                   if pair['policies']['head']['label'] < pair['policies']['cgr']['label'] else 'tie' for pair in matched)
    counts = {name: dict(Counter(pair['policies'][name]['status'] for pair in pairs)) for name in ('head', 'cgr')}
    report = dict(status='target_complete' if len(pairs) == p['target_pairs'] else 'finite_draws_complete_underfilled',
        stop_reason=result['stop_reason'], pairs=pairs, counts=counts, coverage=dict(coverage),
        both_valid_labels=len(matched), paired_physical_wins={key: wins[key] for key in ('head', 'cgr', 'tie')},
        cost=result['cost'], wall_s=result['wall_s'], native_environment_steps=audit['native_environment_integrations'],
        source_audit=B.binding(directory/'READ_ONLY_AUDIT.json'), video_recording=False, training_updates=0)
    B.save(root/'COMPARISON_RESULTS.json', report, exclusive=True)
    print({k: v for k, v in report.items() if k != 'pairs'}, flush=True)
    return report


def run(root):
    B.verify({'protocol': B.read(root/'COMPARISON_READY.json')['protocol']})
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=B.ROOT).strip():
        raise ValueError('Commit the experiment source before dispatch')
    B.save(root/'PIPELINE_STARTED.json', dict(process=R.process_identity(__import__('os').getpid()),
         git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=B.ROOT, text=True).strip(),
         timestamp=time.time()), exclusive=True)
    try:
        stage = dispatch([sys.executable, '-B', str(ENTRY), '--root', str(root/'collection')], root/'timing', 10800)
        summarize(root)
        R.assert_quiescent(root/'collection')
        B.save(root/'PIPELINE_FINISHED.json', dict(stage=stage, result=B.binding(root/'COMPARISON_RESULTS.json')), exclusive=True)
    except BaseException:
        B.save(root/'PIPELINE_FAILURE.json', dict(error=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'summarize'))
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'summarize': summarize}[args.action](args.root.resolve())
