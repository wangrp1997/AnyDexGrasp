"""Run original measured-start arm checks on a finite set of offline recoveries."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

import author_collection_batch as B
from audit_cgr_approach_depth_recovery import candidate_rows


def requests_for(scene_root, root, observation, p):
    index = observation['index']
    directory = scene_root/'scenes'/f'{index:02d}'
    ordered = [item for variant in observation['variants'] for item in variant['admitted']
               if item['orientation'] != 'recorded_flip' and item['approach_world'][2] < 0]
    ordered.sort(key=lambda x: (-x['score'], x['source_local_index'], x['extra_depth_m'],
                                x['orientation'] != 'preserve_approach'))
    picked = []
    for item in ordered:
        if any(old['source_row'] == item['source_row'] for old in picked):
            continue
        picked.append(item)
        if len(picked) == 2:
            break
    result = []
    for ordinal, item in enumerate(picked):
        dest = root/f"scene_{index:02d}_candidate_{ordinal}"
        dest.mkdir(parents=True)
        result.append(dict(index=index, choice=item, scene=str(directory/'capture/observed_scene.json'),
            observation=str(directory/'capture/observation.json'), representation_dir=str(directory/'representation'),
            seed=p['seed'], output_dir=str(dest/'contact'), domain=str(directory/'DOMAIN.json'),
            calibration=p['inputs']['calibration']['path'], cloud_manifest=p['inputs']['cloud_manifest']['path'],
            backend=p['inputs']['backend']['path']))
    return result


def run(baseline, diagnostic, output, indices):
    from author_resident_prepare import bind_backend

    data = B.read(diagnostic); p = B.read(baseline/'collection/PROTOCOL.json')
    B.verify({'initial':data['original_results'], 'diagnostic_source':data['source']})
    bind_backend({'backend':p['inputs']['backend']['path']})
    import author_collect as collection
    import author_protocol as author
    import author_mjx_measured_planning as planning
    import author_resident_prepare as resident

    original_source = author.collection_candidates
    original_collection = collection.prepare_collection
    original_prepare = resident.prepare
    requests = []
    for observation in data['observations']:
        if observation['index'] in indices and not observation['original_count']:
            B.verify(observation['inputs'])
            requests.extend(requests_for(baseline/'collection', output, observation, p))
    if not requests:
        raise ValueError('No recovered rows in the selected observations')
    B.save(output/'REQUESTS.json', requests, exclusive=True)

    def prepare(request, session):
        wanted = request['choice']

        def orient(rows, features, **kwargs):
            source = original_source(rows, features, **kwargs)
            local = source['source_indices']
            variants = candidate_rows(source['rows'], np.c_[rows[local, :2], rows[local, 3:16]], source['if_flip'])
            source['rows'] = variants[wanted['orientation']]
            source['extra_depth_m'] = np.full(len(local), wanted['extra_depth_m'])
            if wanted['orientation'] == 'unflipped':
                source['if_flip'] = np.zeros(len(local), dtype=bool)
            return source

        def choose(*args, **kwargs):
            prepared, arrays = original_collection(*args, **kwargs)
            candidates = [x for x in prepared['candidates'] if x['source_row_index'] == wanted['source_row']]
            if len(candidates) != 1 or candidates[0]['slot_id'] != wanted['slot']:
                raise ValueError('The recovered voxel result must replay under the normal mapper')
            selected = candidates[0]
            selected['canonicalization']['source_collection_flip_columns'] = [1, 2] if wanted['orientation'] == 'preserve_approach' else []
            selected['candidate_sha256'] = collection._candidate_id(selected)
            prepared.update(candidates=[selected], selected_candidate_sha256=selected['candidate_sha256'],
                            selection='One explicitly bound offline recovery candidate')
            prepared['counts']['top10'] = 1
            return prepared, arrays

        with patch.object(author, 'collection_candidates', orient), patch.object(collection, 'prepare_collection', choose):
            row = original_prepare(request, session)
        print(request['index'], wanted['source_row'], wanted['orientation'], wanted['extra_depth_m'], row['status'], flush=True)
        return row

    with patch.object(resident, 'prepare', prepare):
        planning.run(requests, output/'PLANNING_RESULTS.json')
    jobs = B.read(output/'PLANNING_RESULTS.json')['jobs']
    rows = [dict(scene=q['index'], choice=q['choice'], status=a['status'], plan=a['planning_result'])
            for q,a in zip(requests,jobs,strict=True)]
    B.save(output/'RESULTS.json', dict(status='complete', source=B.binding(__file__), diagnostics=B.binding(diagnostic),
        attempts=rows, physics_steps=0, CGR_forwards=0, training_updates=0, video_recording=False), exclusive=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('diagnostic', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('indices', nargs='+', type=int)
    args = parser.parse_args()
    run(args.baseline.resolve(), args.diagnostic.resolve(), args.output.resolve(), set(args.indices))
