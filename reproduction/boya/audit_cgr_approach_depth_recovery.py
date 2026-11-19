"""Use saved camera candidates to measure geometric recovery without simulation."""
import argparse
from pathlib import Path

import numpy as np

import author_collection_batch as B


APPROACHES = ('recorded_flip', 'preserve_approach', 'unflipped')
EXTRA_DEPTH_M = (0., .01, .02, .03)


def candidate_rows(recorded, raw15, flips):
    rotations = np.asarray(raw15[:, 3:12]).reshape(-1, 3, 3)
    source = rotations.copy()
    source[flips, :, :2] *= -1
    np.testing.assert_array_equal(source, recorded[:, 4:13].reshape(-1, 3, 3))
    roll = rotations.copy()
    roll[flips, :, 1:] *= -1
    result = {'recorded_flip': recorded.copy()}
    for name, replacement in [('preserve_approach', roll), ('unflipped', rotations)]:
        result[name] = recorded.copy()
        result[name][:, 4:13] = replacement.reshape(-1, 9)
    return result


def examine(directory, row, contact, library_cls):
    selected = B.read(row['selection']['path'])
    pool = B.read(selected['shared_pool']['path'])
    B.verify({'selection':row['selection'], 'pool':selected['shared_pool']})
    original = directory/'cgr_attempts/rank_00'
    prepared = B.read(original/'pipeline/collection/PREPARED.json')
    bundle = B.read(original/'mapping/BUNDLE.json')
    rep_dir = directory/'representation'
    rep = B.read(rep_dir/'RESULTS.json')
    bindings = dict(trace=prepared['trace'], calibration=bundle['calibration'], surface=bundle['surface_manifest'],
                    raw=rep['output_files']['raw_camera_candidates.npz'], observation=B.binding(directory/'capture/observation.json'))
    B.verify(bindings)
    cal = B.read(bindings['calibration']['path'])
    library = library_cls(bindings['surface']['path'])
    camera = np.asarray(B.read(bindings['observation']['path'])['T_WC'])
    with np.load(bindings['trace']['path'], allow_pickle=False) as trace, np.load(bindings['raw']['path'], allow_pickle=False) as cache:
        recorded = trace['actual_rows17_camera'].copy()
        ids = trace['source_indices'].copy()
        stocks = candidate_rows(recorded, cache['candidates'][ids], trace['source_if_flip'].astype(bool))
        points = trace['source_scene_camera'].copy()
        original_depth = trace['extra_depth_m'].copy()
        original_accepted = trace['collision_accepted_top500_indices'].tolist()
    gesture = np.ones(len(ids), dtype=np.int64)

    def screen(rows, depth, label, extra_label):
        cal['contact_mapping']['rows_identity'] = contact.COLLECT.array_ref(rows)
        result = contact.contact_detect(rows, gesture, depth, points, calibration=cal, library=library,
                                        profile='dh3_collection')
        admitted = result['accepted_source_indices'].tolist()
        joints = dict(zip(result['hand'].ids, result['hand'].mapped))
        cases = [dict(source_row=int(ids[i]), source_local_index=i, score=float(rows[i, 0]),
                      extra_depth_m=float(depth[i]), orientation=label, slot=joints[i].slot['slot_id'],
                      approach_world=(camera[:3, :3] @ rows[i, 4:13].reshape(3, 3)[:, 0]).tolist()) for i in admitted]
        return dict(orientation=label, extra_depth=extra_label, source_indices=admitted, admitted=cases)

    baseline = screen(recorded, original_depth, 'recorded_flip', 'recorded')
    if baseline['source_indices'] != original_accepted or len(baseline['admitted']) != pool['counts']['collision_admitted']:
        raise ValueError('Replayed collision baseline differs from collection')
    variants = []
    if not baseline['admitted']:
        for label in APPROACHES:
            rows = stocks[label]
            if label != 'recorded_flip' and np.any(rows[:, 4:13].reshape(-1, 3, 3)[:, :, 0] @ camera[2, :3] >= 0):
                raise ValueError('Alternative lost the source downward approach')
            for depth in EXTRA_DEPTH_M:
                variants.append(screen(rows, np.full(len(ids), depth), label, depth))
    B.verify(bindings)
    return dict(index=row['index'], original_status=row['status'], original_raw_rows=len(ids),
                original_count=len(original_accepted), original_baseline_exact=True, inputs=bindings, variants=variants)


def audit(root, output, indices):
    import author_resident_prepare as resident

    p = B.read(root/'collection/PROTOCOL.json')
    finish = B.read(root/'PIPELINE_FINISHED.json')
    B.verify({'results':finish['result']})
    data = B.read(finish['result']['path'])
    resident.bind_backend({'backend':p['inputs']['backend']['path']})
    import author_contact_pipeline as contact
    from author_hand_clouds import HandCloudLibrary

    observations = []
    for row in data['scenes']:
        if row['index'] not in indices:
            continue
        directory = root/'collection/scenes'/f"{row['index']:02d}"
        receipt = examine(directory, row, contact, HandCloudLibrary)
        observations.append(receipt)
        print(dict(scene=row['index'], baseline=receipt['original_count'],
                   recovered=[(x['orientation'], x['extra_depth'], len(x['admitted']))
                              for x in receipt['variants'] if x['admitted']]), flush=True)
    if len(observations) != len(indices):
        raise ValueError('Requested observations not found')
    B.save(output, dict(status='complete', source=B.binding(__file__), original_results=finish['result'],
        observations=observations, collision_gate_unchanged=True, physics_steps=0, CGR_forwards=0,
        audit_only=True, training_updates=0, video_recording=False), exclusive=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('indices', nargs='+', type=int)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve(), set(args.indices))
