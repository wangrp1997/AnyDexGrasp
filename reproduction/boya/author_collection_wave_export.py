"""Export admitted wave labels; never run a compatible legacy parent batch.

The parent admission order is checked against raw child outcomes. Each exported
trial is verified against its own frozen child protocol and independently
audited again. Quota excess remains raw evidence and never enters these arrays.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

import author_collection_batch as B
import author_collection_export as E
import author_collection_waves as W


def _ordered_rows(rows, protocol, plan, index):
    """Check the parent's class-independent prefix admission, including skips."""
    counts = {s: {g: 0 for g in B.GESTURES} for s in ('fit', 'internal')}
    ordered = sorted(rows, key=lambda r: (r['queue_index'], B.GESTURES.index(r['gesture_id'])))
    keys = [(r['queue_index'], r['gesture_id']) for r in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate parent scene/gesture outcome')
    admitted = []
    for row in ordered:
        i, g = row['queue_index'], row['gesture_id']
        if type(i) is not int or not 0 <= i < len(index['children']):
            raise ValueError('Parent queue index outside frozen budget')
        child, spec = index['children'][i], plan['queue']['scenes'][i]
        if (row.get('child_queue_index') != 0 or row['scene_id'] != spec['scene_id']
                or row['split'] != spec['split'] or row['trial_id'] != f'{spec["scene_id"]}:g{g}'
                or row['source_object_ids'] != spec['source_object_ids']
                or row['source_object_digests'] != spec['source_object_digests']):
            raise ValueError('Parent outcome differs from the full original scene')
        split = row['split']
        raw_binding = row.get('raw_ledger')
        if not raw_binding:
            if row.get('label') is not None or row.get('trial') is not None:
                raise ValueError('No raw execution can become an admitted label')
            if row['status'] == 'quota_reached' and not all(
                    counts[split][other] >= protocol['quotas'][split] for other in B.GESTURES):
                raise ValueError('An unstarted scene was skipped before its split quota')
            continue
        expected_raw = Path(child['protocol']['path']).parent/'ledger'/f'0000000_g{g}.json'
        if Path(raw_binding['path']).resolve() != expected_raw.resolve():
            raise ValueError('Raw ledger belongs to a different frozen child')
        B.verify({'raw': raw_binding})
        raw = B.read(raw_binding['path'])
        cp = B.read(child['protocol']['path'])
        if (raw.get('protocol_sha256') != B.sha(child['protocol']['path'])
                or cp['queue']['scenes'] != [spec] or raw['queue_index'] != 0
                or raw['gesture_id'] != g or raw['scene_id'] != spec['scene_id']
                or raw['split'] != split or raw['trial_id'] != row['trial_id']
                or raw['source_object_ids'] != spec['source_object_ids']
                or raw['source_object_digests'] != spec['source_object_digests']
                or row.get('raw_label') != raw.get('label')):
            raise ValueError('Raw outcome/parent/child binding differs')
        B.verify(raw.get('artifact_bindings', {}))
        raw_label = raw.get('label')
        if raw_label is None:
            if row.get('label') is not None:
                raise ValueError('A raw null is not a negative label')
            continue
        if type(raw_label) is not int or raw_label not in (0, 1) or raw['status'] != 'labelled':
            raise ValueError('Raw eligible binary execution required')
        should_admit = counts[split][g] < protocol['quotas'][split]
        if not should_admit:
            if row.get('label') is not None or row['status'] != 'quota_excess_not_admitted':
                raise ValueError('Quota excess entered the training dataset')
            continue
        if (type(row.get('label')) is not int or row['label'] != raw_label
                or row['status'] != 'labelled' or row.get('trial') != raw.get('trial')):
            raise ValueError('Parent must admit the first eligible labels independently of class')
        counts[split][g] += 1
        admitted.append((row, raw, cp, spec))
    return admitted


def export(run_dir, output_dir):
    root, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    protocol, plan, index = W.verify_protocol(root)
    rows = W.ledger(root, protocol)
    admitted = _ordered_rows(rows, protocol, plan, index)
    data = {g: [] for g in B.GESTURES}
    ids, meshes = ({s: set() for s in ('fit', 'internal')} for _ in range(2))
    audits, trial_ids, record_paths = {}, set(), set()
    for row, raw, child_protocol, spec in admitted:
        path = Path(raw['trial']['record']['path']).parent
        if row['trial_id'] in trial_ids or path.resolve() in record_paths:
            raise ValueError('An actually executed trial was exported more than once')
        trial_ids.add(row['trial_id']); record_paths.add(path.resolve())
        trial = B.validate_trial(path, spec)
        if trial != raw['trial'] or trial != row['trial']:
            raise ValueError('Actual trial changed after parent admission')
        audit = E._audit_trial(path, child_protocol)
        B.certify_audit(audit, trial, spec)
        audits[row['trial_id']] = audit
        ids[row['split']].update(trial['scene_object_ids'])
        meshes[row['split']].update(trial['scene_mesh_sha256'])
        data[row['gesture_id']].append((row, trial))
    if ids['fit'] & ids['internal'] or meshes['fit'] & meshes['internal']:
        raise ValueError('Nonprimary object leakage across fit/internal or gestures')
    # A source change during long independent audits must fail before writing.
    W.verify_protocol(root)
    if W.ledger(root, protocol) != rows:
        raise ValueError('Parent ledger changed during this export; take a new snapshot')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema='anydex_boya_author_collection_wave_export_v1', task=B.TASK,
        source_protocol=B.binding(root/'WAVE_PROTOCOL.json'), old435_merged=False,
        training_updates=0, exported_labels=len(admitted),
        quota_excess_not_exported=sum(r['status'] == 'quota_excess_not_admitted' for r in rows),
        mesh_sha256_semantics='Compatibility primary only; scene_mesh_sha256 binds all three source identities',
        fit_source_object_ids=sorted(ids['fit']), internal_source_object_ids=sorted(ids['internal']),
        fit_source_mesh_sha256=sorted(meshes['fit']), internal_source_mesh_sha256=sorted(meshes['internal']),
        datasets={}, audits={})
    for trial_id, audit in audits.items():
        path = out/'audits'/f'{hashlib.sha256(trial_id.encode()).hexdigest()}.json'
        B.save(path, audit, exclusive=True); manifest['audits'][trial_id] = B.binding(path)
    for g, entries in data.items():
        arrays = dict(features=np.asarray([t['features'] for _, t in entries], dtype=np.float32).reshape(-1, 480),
            labels=np.asarray([t['label'] for _, t in entries], dtype=np.int64),
            target_index=np.asarray([t['target_index'] for _, t in entries], dtype=np.int64),
            gesture_ids=np.full(len(entries), g, dtype=np.int64),
            trial_id=np.asarray([r['trial_id'] for r, _ in entries], dtype=str),
            split=np.asarray(['train' if r['split'] == 'fit' else 'internal_check' for r, _ in entries], dtype=str),
            mesh_sha256=np.asarray([t['mesh_sha256'] for _, t in entries], dtype=str),
            scene_object_ids=np.asarray([t['scene_object_ids'] for _, t in entries], dtype=str).reshape(-1, 3),
            scene_mesh_sha256=np.asarray([t['scene_mesh_sha256'] for _, t in entries], dtype=str).reshape(-1, 3),
            scene_id=np.asarray([r['scene_id'] for r, _ in entries], dtype=str),
            candidate_sha256=np.asarray([t['candidate_sha256'] for _, t in entries], dtype=str),
            commands_sha256=np.asarray([t['commands']['sha256'] for _, t in entries], dtype=str),
            evaluation_sha256=np.asarray([t['evaluation']['sha256'] for _, t in entries], dtype=str),
            record_json=np.asarray([t['record']['path'] for _, t in entries], dtype=str),
            record_sha256=np.asarray([t['record']['sha256'] for _, t in entries], dtype=str))
        if any(value.dtype.hasobject for value in arrays.values()):
            raise ValueError('Pickled training arrays forbidden')
        path = out/f'gesture_{g}.npz'; np.savez_compressed(path, **arrays)
        counts = {s: int(np.count_nonzero(arrays['split'] == v))
                  for s, v in (('fit', 'train'), ('internal', 'internal_check'))}
        manifest['datasets'][str(g)] = dict(binding=B.binding(path), counts=counts,
            positive=int(arrays['labels'].sum()), negative=len(entries)-int(arrays['labels'].sum()))
    manifest['size_ready'] = all(d['counts'][s] == protocol['quotas'][s]
                               for d in manifest['datasets'].values() for s in ('fit', 'internal'))
    B.save(out/'MANIFEST.json', manifest, exclusive=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(export(args.run_dir, args.output_dir))


if __name__ == '__main__':
    main()
