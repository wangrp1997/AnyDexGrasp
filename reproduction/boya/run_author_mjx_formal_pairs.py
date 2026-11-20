"""Evaluate the frozen formal g1 head against CGR on fresh paired MJX frames."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from unittest.mock import patch

import numpy as np

import author_collection_batch as B
import author_collection_scenes as S
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F
import author_mjx_shared_pair as shared
from audit_author_mjx_head_holdout import MAINLINE, formal_score, layout_id, ranking_order, training_layouts
from author_scene_prepare_cache import PreparationCache
from run_author_mjx_expansion import dispatch
from run_author_mjx_shared_pair import CATALOG, SOURCE_CACHE


SEED = 2026112005
DRAWS = 32
LAYOUTS_PER_SPLIT = 8
ENVIRONMENTS = 16
TARGET = 4
RANK_LIMIT = 2
HERE = Path(__file__).resolve()


def load_head(binding):
    import torch
    import author_training as training

    B.verify({'head': binding})
    checkpoint = torch.load(binding['path'], map_location='cpu', weights_only=True)
    if checkpoint['schema'] != 'anydex_boya_safe_author_head_v1' or checkpoint['gesture_id'] != 1:
        raise ValueError('The formal g1 checkpoint is required')
    torch.set_num_threads(1)
    model = training.components()[1]().eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model


def freeze(root):
    main = B.read(MAINLINE)
    B.verify({key: main[key] for key in ('dataset', 'formal_training')})
    head = B.read(main['formal_training']['path'])['inference_check']['checkpoint']
    B.verify({'head': head})
    with np.load(main['dataset']['path'], allow_pickle=False) as data:
        seen, used_ids = training_layouts(data)
        fit_objects = set(data['scene_mesh_sha256'][data['split'] == 'train'].ravel())
    catalog = B.read(CATALOG)
    S.verify(catalog['input_sha256'])
    S.verify(catalog['source_sha256'])
    catalog_ref = B.binding(CATALOG)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    draws, layouts = [], {}
    with PreparationCache().installed():
        for split in ('fit', 'internal'):
            accepted = []
            for number in range(DRAWS):
                spec = S.sample_scene(catalog, split, number, SEED)
                spec['catalog'] = catalog_ref
                if spec['scene_id'] in used_ids or (split == 'internal'
                        and set(spec['source_object_digests'].values()) & fit_objects):
                    raise ValueError('Evaluation draw overlaps a forbidden identity')
                path = root / 'draws' / split / f'{number:02d}.json'
                B.save(path, spec, exclusive=True)
                row = dict(draw=number, split=split, scene_id=spec['scene_id'],
                           scene_spec=B.binding(path), status=spec['status'],
                           source_object_ids=spec['source_object_ids'],
                           source_object_digests=spec['source_object_digests'])
                draws.append(row)
                if spec['status'] != 'ready':
                    continue
                out = root / 'layouts' / split / f'{number:02d}'
                S.build_scene(path, out)
                scene = out / 'scene.json'
                digest = layout_id(scene)
                if digest in seen:
                    raise ValueError('Repeated source layout')
                seen.add(digest)
                accepted.append(dict(row, scene=B.binding(scene)))
                if len(accepted) == LAYOUTS_PER_SPLIT:
                    break
            layouts[split] = accepted
    # Interleave the two splits and balance which policy starts on the donor lane.
    rows = []
    offsets = {split: rng.permutation(np.arange(len(layouts[split])) % 2) for split in layouts}
    for i in range(LAYOUTS_PER_SPLIT):
        for split in ('internal', 'fit'):
            if i >= len(layouts[split]):
                continue
            layout = layouts[split][i]
            pair = len(rows) // 2
            policies = ('cgr','head') if not offsets[split][i] else ('head','cgr')
            for policy in policies:
                rows.append(dict(layout, index=len(rows), pair_index=pair, policy=policy))
    if not rows:
        raise ValueError('No placement-ready evaluation draws')
    B.save(root / 'DRAWS.json', draws, exclusive=True)
    queue = root / 'QUEUE_SOURCE.json'
    B.save(queue, dict(rows=rows), exclusive=True)
    protocol = F.freeze(root / 'base_freeze', environments=ENVIRONMENTS, source_protocol=queue)
    out = root / 'collection'
    protocol['rows'] = rows
    protocol['inputs'].update(comparison_head=head, dataset=main['dataset'], scene_catalog=catalog_ref,
                              draws=B.binding(root/'DRAWS.json'))
    for row in rows[::2]:
        protocol['inputs'][f"layout:{row['pair_index']}"] = row['scene']
        protocol['inputs'][f"draw:{row['pair_index']}"] = row['scene_spec']
    for name in (HERE.name, 'author_mjx_shared_pair.py', 'author_mjx_policy_compare_worker.py',
                 'author_training.py', 'audit_author_mjx_head_holdout.py', 'author_hour_audit.py',
                 'author_expansion_progress.py', 'author_hour_metrics.py', 'audit_author_mjx_wave_snapshot.py',
                 'author_scene_prepare_cache.py', 'run_author_mjx_expansion.py', 'run_author_mjx_shared_pair.py'):
        protocol['sources'][name] = B.binding(B.HERE / name)
    archive = out / 'source_archive'
    archive.mkdir(parents=True)
    for name, ref in protocol['sources'].items():
        shutil.copyfile(ref['path'], archive / name)
    protocol.update(schema='anydex_mjx_formal_pair_evaluation_v1', seed=SEED, target_pairs=TARGET,
                    max_CGR_forwards=len(rows)//2, max_planning_jobs=len(rows)*RANK_LIMIT,
                    max_grasp_attempts=2*TARGET, comparison_only=True, formal_training_eligible=False,
                    max_rank_per_policy=RANK_LIMIT, video_recording=False, minimum_free_disk_bytes=80*1024**3)
    protocol['env']['ANYDEX_CONTACT_SPOOL_ROOT'] = str(out / 'contact_spool')
    B.verify(protocol['inputs'])
    B.verify(protocol['sources'])
    B.save(out / 'PROTOCOL.json', protocol, exclusive=True)
    shutil.copytree(SOURCE_CACHE, out / 'warp_cache')
    ready = dict(status='frozen', protocol=B.binding(out/'PROTOCOL.json'), draws=len(draws),
                 placement_ready={split: len(entries) for split, entries in layouts.items()},
                 max_attempts=protocol['max_grasp_attempts'], max_candidate_plans=protocol['max_planning_jobs'],
                 max_native_environment_steps=protocol['max_native_environment_steps'], video_recording=False)
    B.save(root / 'READY.json', ready, exclusive=True)
    print(ready, flush=True)


def orders_for(pool, model):
    options = pool['candidates']
    scores = formal_score(options, model)
    cgr = list(range(len(options)))
    assert all(a['score'] >= b['score'] for a, b in zip(options, options[1:]))
    head = ranking_order(scores).tolist()
    return {'cgr': cgr[:RANK_LIMIT], 'head': head[:RANK_LIMIT]}, scores.tolist()


def first_ready(order, statuses):
    for rank in order:
        if rank not in statuses:
            return None
        if statuses[rank] == 'ready':
            return rank
    return None


def worker(requests, output):
    import author_resident_prepare as resident

    resident.bind_backend(requests[0])
    import author_collect as collector
    import author_contact_pipeline as contact
    import author_mjx_measured_planning as measured

    head = requests[0]['comparison_head']
    model = load_head(head)
    original_prepare = resident.prepare
    original_collection = collector.prepare_collection
    original_mapping = contact.create_mapping_bundle
    cache = {}

    def prepare(request, session):
        frame = Path(request['pair_root'])
        observation = B.binding(request['observation'])
        assert request['comparison_head'] == head
        if frame not in cache:
            context = {}
            plans, receipts = {}, {}

            def mapping(*args, **kwargs):
                if 'mapping' not in context:
                    context['mapping'] = original_mapping(*args, **kwargs)
                return context['mapping']

            def select(*args, **kwargs):
                if 'pool' not in context:
                    pool, trace = original_collection(*args, **kwargs)
                    context.update(pool=pool, trace=trace)
                    context['orders'], context['scores'] = orders_for(pool, model)
                orders = context['orders']
                if orders['cgr'] == orders['head']:
                    context['current_rank'] = None
                elif 'current_rank' not in context:
                    context['current_rank'] = orders['cgr'][0]
                rank = context['current_rank']
                return selected_pool(context['pool'], rank), context['trace']

            def attempt(number):
                path = frame / 'arm_plans' / f'{number:02d}'
                path.parent.mkdir(exist_ok=True)
                with patch.object(contact, 'create_mapping_bundle', mapping), patch.object(
                        collector, 'prepare_collection', select):
                    result = original_prepare(dict(request, output_dir=str(path)), session)
                rank = context['current_rank']
                expected = ('no_candidate',) if rank is None else ('ready', 'null_geometry')
                if result['status'] not in expected:
                    raise RuntimeError(f'Unexpected alternate plan result: {result["status"]}')
                if rank is not None:
                    plan = B.read(result['planning_result']['path'])
                    assert plan['candidate_sha256'] == context['pool']['candidates'][rank]['candidate_sha256']
                    plans[rank] = dict(status=result['status'], plan=result['planning_result'])
                    receipts[rank] = result
                return result

            first = attempt(0)
            orders = context['orders']
            ranks = list(dict.fromkeys(a for pair in zip(orders['cgr'], orders['head']) for a in pair))
            if orders['cgr'] != orders['head']:
                for position, rank in enumerate(ranks[1:], 1):
                    statuses = {r: item['status'] for r, item in plans.items()}
                    if all(first_ready(order, statuses) is not None for order in orders.values()):
                        break
                    context['current_rank'] = rank
                    attempt(position)
            statuses = {rank: item['status'] for rank, item in plans.items()}
            chosen = {policy: first_ready(order, statuses) for policy, order in orders.items()}
            hashes = [candidate['candidate_sha256'] for candidate in context['pool']['candidates']]
            B.save(frame/'ORIGINAL_POOL.json', context['pool'], exclusive=True)
            ranking = dict(candidate_count=len(hashes), hashes=hashes, scores=context['scores'], orders=orders,
                cgr_rank=chosen['cgr'], head_rank=chosen['head'],
                disagreement=all(rank is not None for rank in chosen.values()) and chosen['cgr'] != chosen['head'],
                rank_plans=plans, observation=observation, shared_pool=B.binding(frame/'ORIGINAL_POOL.json'), head=head)
            B.save(frame/'RANKING.json', ranking, exclusive=True)
            cache[frame] = ranking, receipts, first
        ranking, receipts, first = cache[frame]
        assert observation == ranking['observation']
        policy = request['comparison_policy']
        rank = ranking[policy+'_rank']
        if not ranking['disagreement']:
            return dict(first, status='no_candidate')
        result = receipts[rank]
        path = Path(request['output_dir'])
        path.symlink_to(Path(result['planning_result']['path']).parent.parent, target_is_directory=True)
        return result

    with patch.object(resident, 'prepare', prepare):
        measured.run(requests, output)


def selected_pool(pool, rank):
    result = copy.deepcopy(pool)
    candidates = [] if rank is None else [result['candidates'][rank]]
    result.update(status='prepared' if candidates else 'no_geometry', candidates=candidates,
                  selected_candidate_sha256=candidates[0]['candidate_sha256'] if candidates else None,
                  selection='Frozen shared-pool candidate for paired planning')
    result['counts']['top10'] = len(candidates)
    return result


def plan_pairs(output, groups, protocol, resources):
    if not groups:
        return {}

    def plan_group(number):
        directory = output / f'worker_{number}'
        directory.mkdir(parents=True)
        items = [item for group in groups[number::3] for item in group]
        requests = directory / 'REQUESTS.json'
        result = directory / 'RESULTS.json'
        B.save(requests, [req for _,req in items], exclusive=True)
        command = [sys.executable, '-B', str(HERE), 'plan', str(requests), '--output', str(result)]
        B.save(directory/'DISPATCH.json', dict(command=command, requests=B.binding(requests)), exclusive=True)
        with resources.stage(('prepare',), directory), (directory/'process.log').open('x') as log:
            process = resources.popen(command, cwd=B.ROOT, env=B._environment(protocol),
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
                raise RuntimeError(f'Formal-head planning failed in {directory}')
        return dict(zip((idx for idx,_ in items), B.read(result)['jobs'], strict=True))

    with ThreadPoolExecutor(max_workers=3) as executor:
        return {key: value for result in executor.map(plan_group, range(min(3, len(groups))))
                for key, value in result.items()}


def collect(root):
    p = B.read(root/'PROTOCOL.json')
    marker = B.sha(root/'PROTOCOL.json')
    if os.environ.get('ANYDEX_FORMAL_PAIR_ENV') != marker:
        env = B._environment(p)
        env.update(PYTHONPATH=os.pathsep.join((str(F.WARP), str(F.DEPS), env['PYTHONPATH'])),
                   CUDA_VISIBLE_DEVICES=F.GPU, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   WARP_CACHE_PATH=str(root/'warp_cache'), ANYDEX_FORMAL_PAIR_ENV=marker)
        os.execve(sys.executable, [sys.executable, '-B', str(HERE), 'collect', str(root)], env)
    with patch.object(shared, 'plan_pairs', plan_pairs):
        result = shared.run(root)
    if result['status'] != 'complete':
        print(result['error'], flush=True)
        raise SystemExit(1)


def review(root):
    directory = root/'collection'
    p = B.read(directory/'PROTOCOL.json')
    result = B.read(directory/'RESULTS.json')
    audit_path = directory/'READ_ONLY_AUDIT.json'
    audit = B.read(audit_path)
    assert audit['status'] == 'read_only_audit_pass' and audit['original_result'] == B.binding(directory/'RESULTS.json')
    B.verify(p['sources'])
    B.verify(p['inputs'])
    model = load_head(p['inputs']['comparison_head'])
    verified = {row['index']: row for row in audit['rounds']}
    cost = Counter()
    pairs, frames = [], []
    for spec in p['rows'][::2]:
        frame = directory/'pairs'/f"{spec['pair_index']:03d}"
        admission_path = frame/'ADMISSION.json'
        if not admission_path.exists():
            status = result['records'][spec['index']]['status']
            cost[status] += 1
            frames.append(dict(pair=spec['pair_index'], split=spec['split'], status=status))
            continue
        admission = B.read(admission_path)
        B.verify({key: admission[key] for key in ('start','ranking')})
        start = B.read(admission['start']['path'])
        ranking = B.read(admission['ranking']['path'])
        B.verify({key: start[key] for key in ('observation','state','measured_scene','live_state','representation')})
        B.verify({'pool': ranking['shared_pool']})
        assert ranking['observation'] == start['observation'] and ranking['head'] == p['inputs']['comparison_head']
        with np.load(start['state']['path'], allow_pickle=False) as arrays:
            assert set(arrays.files) == set(shared.STATE)
            for value in arrays.values():
                np.testing.assert_array_equal(value[0], value[1])
        pool = B.read(ranking['shared_pool']['path'])
        orders, scores = orders_for(pool, model)
        assert orders == ranking['orders'] and scores == ranking['scores']
        assert ranking['hashes'] == [candidate['candidate_sha256'] for candidate in pool['candidates']]
        statuses = {}
        for key, outcome in ranking['rank_plans'].items():
            rank = int(key)
            B.verify({'plan': outcome['plan']})
            attempt = B.read(outcome['plan']['path'])
            assert attempt['status'] == outcome['status'] and attempt['candidate_sha256'] == ranking['hashes'][rank]
            statuses[rank] = attempt['status']
        chosen = {policy: first_ready(order, statuses) for policy,order in orders.items()}
        assert all(rank == ranking[policy+'_rank'] for policy,rank in chosen.items())
        different = None not in chosen.values() and chosen['cgr'] != chosen['head']
        assert different == ranking['disagreement']
        assert admission['admitted'] == (different and len(pairs) < p['target_pairs'])
        cost['observed'] += 1
        cost['multi_candidate'] += int(len(ranking['hashes']) >= 2)
        cost['actual_candidate_plans'] += len(statuses)
        cost['different_ready_choices'] += int(different)
        frames.append(dict(pair=spec['pair_index'], split=spec['split'], candidates=len(ranking['hashes']),
                           checked=statuses, chosen=chosen, admitted=admission['admitted']))
        if not admission['admitted']:
            continue
        outcomes = {}
        for index in start['indices']:
            policy = p['rows'][index]['policy']
            record = result['records'][index]
            assert verified[index]['return_pass'] and verified[index]['original_controls_exact']
            assert verified[index]['CGR_input_and_receipt_exact']
            ready = record['bindings']
            B.verify({'candidate': ready['candidate'], 'execution': ready['execution']})
            candidate = B.read(ready['candidate']['path'])
            assert candidate['candidate_sha256'] == ranking['hashes'][chosen[policy]]
            execution = B.read(ready['execution']['path'])
            outcomes[policy] = dict(label=record['label'], status=record['status'], stop=execution['stop'],
                                    execution=ready['execution'], candidate=ready['candidate'])
        pairs.append(dict(pair=spec['pair_index'], split=spec['split'], outcomes=outcomes,
                          admission=B.binding(admission_path)))
    assert cost['observed'] == result['cost']['CGR_forwards']
    assert 2*len(pairs) == result['cost']['grasp_attempts'] <= p['max_grasp_attempts']
    assert cost['actual_candidate_plans'] <= p['max_planning_jobs']
    outcomes = Counter()
    for pair in pairs:
        labels = {policy: item['label'] for policy,item in pair['outcomes'].items()}
        if None in labels.values():
            outcomes['one_or_both_invalid'] += 1
        elif labels['head'] == labels['cgr']:
            outcomes['both_lift' if labels['head'] else 'neither_lifts'] += 1
        else:
            outcomes['head_only_lifts' if labels['head'] else 'cgr_only_lifts'] += 1
    review = dict(status='formal_pair_review_complete', pairs=pairs, frames=frames,
                  counts=dict(cost), outcomes=dict(outcomes), wall_s=result['wall_s'],
                  native_environment_integrations=audit['native_environment_integrations'],
                  original_audit=B.binding(audit_path), protocol=B.binding(directory/'PROTOCOL.json'),
                  no_video=True, new_training_labels=0, training_updates=0)
    B.save(root/'FINAL_REVIEW.json', review, exclusive=True)
    print(dict(pairs=len(pairs), counts=cost, outcomes=outcomes), flush=True)


def run(root):
    B.verify({'protocol': B.read(root/'READY.json')['protocol']})
    stage = dispatch([sys.executable, '-B', str(HERE), 'collect', str(root/'collection')], root/'timing', 10800)
    review(root)
    R.assert_quiescent(root/'collection')
    B.save(root/'FINISHED.json', dict(stage=stage, review=B.binding(root/'FINAL_REVIEW.json')), exclusive=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','run','collect','plan','review'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'plan':
        worker(B.read(args.root), args.output)
    else:
        {'freeze': freeze, 'run': run, 'collect': collect, 'review': review}[args.action](args.root.resolve())
