"""Read-only dataset audit; writes one fresh, lightweight receipt, no simulation."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'outputs/s40_stable_supplement_20261107_v2/training_delivery_v3/merged_dataset'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest_path = DATA / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    protected_path = (ROOT / 'reproduction/boya/evidence/'
                      'author_home_release_integration_20261114_v1/UNCHANGED_DATA_MODELS.json')
    protected = json.loads(protected_path.read_text())['verified_files']
    for path, expected in protected.items():
        assert sha(path) == expected, path
    stats, ids, objects, split_objects = [], set(), set(), {}
    by_trial = {r['trial_id']: r for r in manifest['samples']}
    assert len(by_trial) == len(manifest['samples']) == 435
    inputs = {str(manifest_path): sha(manifest_path)}
    for gesture in (1, 6, 7, 8):
        path = DATA / f'gesture_{gesture}.npz'
        inputs[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            labels = data['labels']
            assert data['features'].shape == (len(labels), 480)
            assert set(labels.tolist()) <= {0, 1}
            assert np.all(data['gesture_ids'] == gesture)
            split_stats = {}
            for split in ('train', 'internal_check'):
                mask = data['split'] == split
                split_stats[split] = dict(count=int(mask.sum()),
                                         positive=int(labels[mask].sum()),
                                         negative=int(mask.sum() - labels[mask].sum()),
                                         objects=len(set(data['object_id'][mask].tolist())))
                split_objects.setdefault(split, set()).update(data['object_id'][mask].tolist())
            for row, trial in enumerate(data['trial_id'].tolist()):
                assert trial not in ids
                ids.add(trial)
                sample = by_trial[trial]
                assert sample['npz_row'] == row and sample['gesture_id'] == gesture
                assert sample['label'] == int(labels[row])
                assert sample['target_index'] == int(data['target_index'][row])
                assert sample['object_id'] == data['object_id'][row]
                feature_sha = hashlib.sha256(data['features'][row].tobytes()).hexdigest()
                assert feature_sha == sample['canonical_CGR480_identity']['c_order_bytes_sha256']
                for key in ('record', 'evaluation'):
                    assert sha(sample[key + '_json']) == sample[key + '_sha256']
                record = json.loads(Path(sample['record_json']).read_text())
                assert record['label_eligible'] is True and record['result'] == sample['label']
                assert record['outcome_status'] == ('executed_lift_success' if sample['label']
                                                    else 'executed_lift_failure')
            objects.update(data['object_id'].tolist())
            stats.append(dict(gesture_id=gesture, count=len(labels), positive=int(labels.sum()),
                              negative=int(len(labels) - labels.sum()),
                              objects=len(set(data['object_id'].tolist())), splits=split_stats))
    assert len(ids) == manifest['sample_count'] == 435
    assert sum(r['positive'] for r in stats) == manifest['positive'] == 302
    assert sum(r['negative'] for r in stats) == manifest['negative'] == 133
    assert split_objects['train'].isdisjoint(split_objects['internal_check'])
    paper = ROOT / 'outputs/paper_code_audit/source/sections/results.tex'
    lines = paper.read_text().splitlines()
    assert '144' in lines[123] and '1,000' in lines[123]
    assert '100 attempts' in lines[166] and '50 attempts' in lines[166]
    assert '1,000 and 800' in lines[171]
    inputs[str(paper)] = sha(paper)
    inputs[str(protected_path)] = sha(protected_path)
    receipt = dict(
        schema='anydex_boya_lift_scope_dataset_status_v1', status='passed',
        dataset_role='Historical single-object lift labels; no new collection in this audit',
        counts=dict(total=435, positive=302, negative=133, fit=364, internal_check=71,
                    distinct_labelled_objects=len(objects), declared_objects=len(manifest['objects']),
                    fit_objects=len(split_objects['train']),
                    internal_objects=len(split_objects['internal_check'])),
        per_gesture=stats,
        paper=dict(source=str(paper), full_setting_line=124, few_shot_line=167,
                   other_hands_total_line=172, dh3_few_shot=400, dh3_full_approximate=4000,
                   allegro_few_shot=1000, inspire_few_shot=800),
        old_data_and_heads_sha256=protected, input_sha256=inputs,
        audit_source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        record_sha_and_label_checks=len(ids), new_formal_labels=0,
        new_formal_training_updates=0, simulation_steps=0, CGR_forwards=0,
        limits='Hashes and stored labels checked; does not revalidate physical outcomes or prove a new lift pipeline ready')
    destination = Path(__file__).with_name('DATASET_STATUS.json')
    with destination.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=receipt['status'], counts=receipt['counts'],
                          checked_old_files=len(protected), receipt=str(destination)), indent=2))


if __name__ == '__main__':
    main()
