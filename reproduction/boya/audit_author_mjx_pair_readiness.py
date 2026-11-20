"""Check alternate arm plans on held-out archived camera frames, without simulation."""
import argparse
from collections import Counter
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import author_collection_batch as B
import author_training as T
from audit_author_mjx_head_holdout import MAINLINE, formal_score, layout_id, ranking_order, training_layouts


SOURCE = B.ROOT / 'outputs/boya_mjx_cgr_to100_20261119_v1/collection'
WORKERS = 3
MAX_FIT_FRAMES = 5
MAX_INTERNAL_FRAMES = 4


def freeze(root, reuse=None):
    mainline = B.read(MAINLINE)
    B.verify({key: mainline[key] for key in ('dataset', 'formal_training')})
    with np.load(mainline['dataset']['path'], allow_pickle=False) as data:
        excluded_layouts, excluded_ids = training_layouts(data)
    head = B.read(mainline['formal_training']['path'])['inference_check']['checkpoint']
    B.verify({'head': head})
    torch.set_num_threads(1)
    model = T.components()[1]().eval()
    model.load_state_dict(torch.load(head['path'], map_location='cpu', weights_only=True)['model_state_dict'])

    source = B.read(SOURCE / 'PROTOCOL.json')
    backend = source['inputs']['backend']
    B.verify({'backend': backend})
    seen, counts = set(), Counter()
    frames, jobs = [], []
    selected = Counter()
    for row in source['rows']:
        directory = SOURCE / 'scenes' / f"{row['index']:02d}"
        path = directory / 'contact/pipeline/collection/PREPARED.json'
        if not path.is_file():
            continue
        scene = row['scene']
        B.verify({'scene': scene})
        identity = layout_id(scene['path'])
        if identity in excluded_layouts or row['scene_id'] in excluded_ids or identity in seen:
            counts['excluded_training_selection_or_duplicate_pool'] += 1
            continue
        seen.add(identity)
        prepared = B.read(path)
        options = prepared['candidates']
        counts['saved_holdout_pools'] += 1
        counts[f'candidates:{len(options)}'] += 1
        if len(options) < 2:
            continue
        counts[f"multi_candidate:{row['split']}"] += 1
        reference = B.read(directory / 'contact/MEASURED_START_RECEIPT.json')
        B.verify({'plan': reference['planning_result'], 'state': reference['live_state']})
        baseline = B.read(reference['planning_result']['path'])
        assert baseline['candidate_sha256'] == options[0]['candidate_sha256']
        assert baseline['status'] == reference['status'] in ('ready', 'null_geometry')
        scores = formal_score(options, model)
        head_order = ranking_order(scores).tolist()
        cgr_scores = [option['score'] for option in options]
        assert all(a >= b for a, b in zip(cgr_scores, cgr_scores[1:]))
        assert len({option['candidate_sha256'] for option in options}) == len(options)
        if row['split'] == 'internal' and selected['internal'] < MAX_INTERNAL_FRAMES:
            ranks = list(range(1, len(options)))
        elif (row['split'] == 'fit' and selected['fit'] < MAX_FIT_FRAMES
              and baseline['status'] == 'ready' and head_order[0] != 0):
            ranks = [head_order[0]]
        else:
            continue
        if len(jobs) + len(ranks) > 8 * WORKERS:
            raise ValueError('Frozen CPU planning budget exceeded')
        selected[row['split']] += 1
        capture = B.read(directory / 'capture/RESULTS.json')
        original = B.read(directory / 'contact/CONTACT_INTEGRATION_PROTOCOL.json')
        mapping = B.read(directory / 'contact/mapping/BUNDLE.json')
        refs = dict(pool=B.binding(path), baseline_plan=reference['planning_result'], trace=prepared['trace'],
                    scene=original['scene'], observation=capture['observation'], state=capture['live_state'],
                    representation=prepared['source_and_input_bindings']['representation_receipt'],
                    calibration=original['calibration'], cloud_manifest=original['manifest'],
                    mapping=mapping['calibration'], surface_manifest=mapping['surface_manifest'])
        B.verify(refs)
        assert capture['live_state'] == reference['live_state'] and capture['observation'] == original['observation']
        assert prepared['scene_binding']['scene_sha256'] == refs['scene']['sha256']
        assert prepared['source_and_input_bindings']['observation'] == refs['observation']
        assert prepared['seed'] == original['seed']
        frames.append(dict(index=row['index'], split=row['split'], scene_id=row['scene_id'],
                           layout_sha256=identity, inputs=refs, head_order=head_order, scores=scores.tolist(),
                           baseline_status=baseline['status'], seed=original['seed'],
                           candidate_sha256=[option['candidate_sha256'] for option in options]))
        jobs.extend(dict(index=row['index'], rank=rank) for rank in ranks)

    if not jobs:
        raise ValueError('No alternate arm plans in the fixed readiness cohort')
    retained = []
    prior = None
    if reuse:
        prior = B.binding(reuse / 'PROTOCOL.json')
        previous = B.read(prior['path'])
        assert previous['frames'] == frames and previous['jobs'] == jobs
        assert previous['formal_head'] == head and previous['backend'] == backend
        assert B.sha(reuse / 'SOURCE_v1.py') == previous['source_script']['sha256']
        for position, job in enumerate(jobs):
            name = f"{job['index']:03d}_rank{job['rank']:02d}"
            receipt = reuse / f'worker_{position % WORKERS}' / name / 'MEASURED_START_RECEIPT.json'
            if receipt.is_file():
                retained.append(dict(job, receipt=B.binding(receipt)))
    retained_jobs = {(row['index'], row['rank']) for row in retained}
    pending = [job for job in jobs if (job['index'], job['rank']) not in retained_jobs]
    root.mkdir(parents=True, exist_ok=False)
    B.save(root / 'DOMAIN.json', source['domain'], exclusive=True)
    protocol = dict(status='frozen', source=B.binding(SOURCE / 'PROTOCOL.json'), backend=backend,
                    domain=B.binding(root / 'DOMAIN.json'), prior_protocol=prior, retained_plans=retained,
                    formal_head=head, dataset=mainline['dataset'], scope='offline_planning_only',
                    selection='All first four multi-candidate internal frames, plus first five fit frames with CGR rank 0 ready and a different formal top rank',
                    counts=dict(counts), frames=frames, jobs=jobs, pending=pending, physics_steps=0,
                    source_script=B.binding(__file__))
    B.save(root / 'PROTOCOL.json', protocol, exclusive=True)
    print(dict(status='frozen', counts=counts, selected=selected, remaining_plans=len(pending),
               previous_plans=len(retained)), flush=True)


def plan(root, worker):
    import author_resident_prepare as resident

    protocol = B.read(root / 'PROTOCOL.json')
    B.verify({key: protocol[key] for key in ('source', 'backend', 'domain', 'source_script')})
    jobs = protocol['pending'][worker::WORKERS]
    frames = {row['index']: row for row in protocol['frames']}
    if not jobs:
        return
    backend = protocol['backend']['path']
    resident.bind_backend({'backend': backend})
    import author_collect as collector
    import author_contact_pipeline as contact
    import author_mjx_measured_planning as measured

    destination = root / f'worker_{worker}'
    destination.mkdir(exist_ok=False)
    requests = []
    for job in jobs:
        frame = frames[job['index']]
        refs = frame['inputs']
        B.verify(refs)
        job_directory = destination / f"{job['index']:03d}_rank{job['rank']:02d}"
        job_directory.mkdir()
        output = job_directory / 'contact'
        requests.append(dict(job, scene=refs['scene']['path'], observation=refs['observation']['path'],
            representation_dir=str(Path(refs['representation']['path']).parent), seed=frame['seed'],
            output_dir=str(output), domain=str(root / 'DOMAIN.json'), calibration=refs['calibration']['path'],
            cloud_manifest=refs['cloud_manifest']['path'], backend=backend))

    original_prepare = resident.prepare

    def prepare(request, session):
        frame = frames[request['index']]
        refs = frame['inputs']
        prepared = B.read(refs['pool']['path'])
        candidate = prepared['candidates'][request['rank']]
        assert candidate['candidate_sha256'] == frame['candidate_sha256'][request['rank']]
        with np.load(refs['trace']['path'], allow_pickle=False) as arrays:
            trace = {key: value.copy() for key, value in arrays.items()}
        selected = copy.deepcopy(prepared)
        selected.update(candidates=[candidate], selected_candidate_sha256=candidate['candidate_sha256'])
        selected['counts']['top10'] = 1
        with patch.object(collector, 'prepare_collection', return_value=(selected, trace)), patch.object(
                contact, 'create_mapping_bundle', return_value=(Path(refs['mapping']['path']),
                                                               Path(refs['surface_manifest']['path']))):
            receipt = original_prepare(request, session)
        result = B.read(receipt['planning_result']['path'])
        assert receipt['status'] in ('ready', 'null_geometry')
        assert result['candidate_sha256'] == candidate['candidate_sha256']
        assert receipt['physics_steps'] == receipt['new_CGR_forwards'] == 0
        print(request['index'], request['rank'], receipt['status'], flush=True)
        return receipt

    B.save(destination / 'REQUESTS.json', requests, exclusive=True)
    with patch.object(resident, 'prepare', prepare):
        measured.run(requests, destination / 'RESULTS.json')


def review(root):
    protocol = B.read(root / 'PROTOCOL.json')
    results = {}
    for item in protocol['retained_plans']:
        B.verify({'receipt': item['receipt']})
        results[item['index'], item['rank']] = B.read(item['receipt']['path'])
    for worker in range(WORKERS):
        jobs = protocol['pending'][worker::WORKERS]
        if not jobs:
            continue
        rows = B.read(root / f'worker_{worker}/RESULTS.json')['jobs']
        for job, item in zip(jobs, rows, strict=True):
            key = (job['index'], job['rank'])
            results[key] = item
    assert set(results) == {(job['index'], job['rank']) for job in protocol['jobs']}
    summary = []
    for frame in protocol['frames']:
        statuses = {0: frame['baseline_status']}
        for rank, digest in enumerate(frame['candidate_sha256'][1:], 1):
            item = results.get((frame['index'], rank))
            if item:
                B.verify({'plan': item['planning_result'], 'state': item['live_state']})
                assert item['status'] in ('ready', 'null_geometry')
                assert item['physics_steps'] == item['new_CGR_forwards'] == item['training_updates'] == 0
                assert item['live_state'] == frame['inputs']['state']
                assert B.read(item['planning_result']['path'])['candidate_sha256'] == digest
                if item['status'] == 'ready':
                    ready = B.read(Path(item['planning_result']['path']).parent / 'READY.json')
                    B.verify({key: ready[key] for key in ('candidate', 'plan', 'commands')})
                    assert B.read(ready['candidate']['path'])['candidate_sha256'] == digest
                statuses[rank] = item['status']

        def first_executable(order):
            for rank in order:
                if rank not in statuses:
                    return None
                if statuses[rank] == 'ready':
                    return rank
            return None

        cgr = first_executable(range(len(frame['candidate_sha256'])))
        head = first_executable(frame['head_order'])
        summary.append(dict(index=frame['index'], split=frame['split'], arm_plans=statuses,
                            cgr_first_ready=cgr, head_first_ready=head,
                            different_ready_choices=cgr is not None and head is not None and cgr != head))
    report = dict(status='planning_review_complete', protocol=B.binding(root / 'PROTOCOL.json'),
                  frames=summary, alternate_plans=len(results),
                  different_ready_choices=sum(row['different_ready_choices'] for row in summary),
                  physics_steps=0, new_CGR_forwards=0, training_updates=0)
    B.save(root / 'RESULTS.json', report, exclusive=True)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'plan', 'review'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--worker', type=int, choices=range(WORKERS))
    parser.add_argument('--reuse', type=Path, help='Completed receipts from the initial run')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'plan':
        if args.worker is None:
            parser.error('plan requires a worker index')
        plan(root, args.worker)
    elif args.action == 'freeze':
        freeze(root, args.reuse)
    else:
        review(root)
