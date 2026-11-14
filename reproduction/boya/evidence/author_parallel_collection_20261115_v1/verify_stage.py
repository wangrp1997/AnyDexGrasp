"""Read-only stage verification; no simulator, audit replay or training."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
PUBLIC = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    with (OUT/name).open('x') as stream:
        stream.write(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def main():
    bound = {}

    def check(path, digest):
        path = str(Path(path).resolve())
        assert sha(path) == digest, path
        assert path not in bound or bound[path] == digest
        bound[path] = digest

    def tree(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                check(value['path'], value['sha256'])
            for item in value.values():
                tree(item)
        elif isinstance(value, list):
            for item in value:
                tree(item)

    old = read(ROOT/'outputs/boya_author_collection_scenes_20261114_v2/COLLECTION_READ_ONLY_SUMMARY.json')
    tree(old)
    batch = PUBLIC/'development/batch_v1'
    wave = PUBLIC/'development_waves/waves_v1'
    all_rows = []
    summaries = {}
    for name, directory, expected in [('batch', batch, 40), ('wave', wave, 24)]:
        rows = [read(path) for path in sorted((directory/'ledger').glob('*.json'))]
        assert len(rows) == expected
        counts = Counter(row['status'] for row in rows)
        labels = [row for row in rows if row.get('label') is not None]
        summaries[name] = dict(ledger_rows=len(rows), statuses=dict(counts),
            labels=len(labels), positives=sum(row['label'] for row in labels))
        all_rows.extend(labels)
        if name == 'wave':
            save('WAVE_LEDGER.json', rows)
    assert summaries['batch']['statuses'] == dict(labelled=14, null_geometry=10,
        placement_null=12, no_candidate=3, infrastructure_failure=1)
    assert summaries['wave']['statuses'] == dict(labelled=1, null_geometry=4,
        no_candidate=7, placement_null=12)
    assert Counter(row['split'] for row in all_rows) == dict(fit=14, internal=1)
    assert sum(row['label'] for row in all_rows) == 11
    assert Counter(row['gesture_id'] for row in all_rows if row['split'] == 'fit') == {1:4, 6:3, 7:3, 8:4}

    export = ROOT/'outputs/boya_author_wave_certified_export_20261115_v1'
    manifest = read(export/'MANIFEST.json')
    assert manifest['exported_labels'] == 1 and manifest['size_ready'] is False
    assert manifest['verification_cost']['numeric_replays'] == 0
    parent = next(row for row in all_rows if row['split'] == 'internal')
    tree(parent)
    trial = parent['trial']
    candidate, record = read(trial['candidate']['path']), read(trial['record']['path'])
    assert parent['gesture_id'] == 7 and parent['label'] == record['label'] == 1
    for gesture in (1, 6, 7, 8):
        binding = manifest['datasets'][str(gesture)]['binding']
        check(binding['path'], binding['sha256'])
        with np.load(binding['path'], allow_pickle=False) as data:
            n = int(gesture == 7)
            assert data['features'].shape == (n, 480)
            assert data['scene_object_ids'].shape == (n, 3)
            assert all(not data[key].dtype.hasobject for key in data.files)
            if not n:
                continue
            np.testing.assert_array_equal(data['features'][0], np.asarray(candidate['canonical_cgr480'], dtype=np.float32))
            assert int(data['labels'][0]) == 1
            assert int(data['target_index'][0]) == candidate['raw_depth_index']*4+candidate['extra_depth_index']
            assert str(data['split'][0]) == 'internal_check'
            assert str(data['trial_id'][0]) == parent['trial_id']
            assert str(data['scene_id'][0]) == parent['scene_id']
            ids = sorted(parent['source_object_ids'])
            assert list(data['scene_object_ids'][0]) == ids
            assert list(data['scene_mesh_sha256'][0]) == [parent['source_object_digests'][key] for key in ids]
            for field, key in [('commands_sha256','commands'), ('evaluation_sha256','evaluation'), ('record_sha256','record')]:
                assert str(data[field][0]) == trial[key]['sha256']
            assert str(data['candidate_sha256'][0]) == trial['candidate_sha256']
            assert str(data['record_json'][0]) == trial['record']['path']
    tree(manifest['audits'])
    cert = manifest['certificate_reuse'][parent['trial_id']]['original_certificate']
    assert cert['sha256'] == manifest['audits'][parent['trial_id']]['sha256']

    old_models = read(ROOT/'reproduction/boya/evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')
    assert len(old_models['verified_files']) == 13
    for path, digest in old_models['verified_files'].items():
        check(path, digest)
    save('UNCHANGED_DATA_MODELS.json', dict(status='pass', verified_files=old_models['verified_files'],
        old_labels=435, old_heads=8, new_formal_training_updates=0))
    for name, source in [
        ('BATCH_COMPLETE_SUMMARY.json', ROOT/'outputs/boya_author_collection_scenes_20261114_v2/COLLECTION_READ_ONLY_SUMMARY.json'),
        ('WAVE_PROTOCOL.json', wave/'WAVE_PROTOCOL.json'),
        ('WAVE_PROGRESS.json', wave/'PROGRESS.json'),
        ('WAVE_HOST_FINAL.json', ROOT/'outputs/boya_parallel_runtime_audit_20261115_v1/FINAL.json'),
        ('WAVE_CERTIFIED_EXPORT_MANIFEST.json', export/'MANIFEST.json')]:
        with (OUT/name).open('xb') as stream:
            stream.write(source.read_bytes())
        check(source, sha(OUT/name))
    save('ROOT_VERIFICATION.json', dict(status='pass', summaries=summaries,
        fit_labels=14, internal_labels=1, positive=11, negative=4,
        wave_independent_array_check='pass', verified_bindings=bound,
        verifier=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        new_numeric_audits=0, new_physics_steps=0, new_training_updates=0))
    print(json.dumps(dict(status='pass', summaries=summaries, verified_bindings=len(bound))))


if __name__ == '__main__':
    main()
