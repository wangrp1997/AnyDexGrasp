"""Check actual labelled fit counts; never collect data or start training.

This is a size and split gate, not certification of physical label semantics.
The full-data Boya target has 1,000 fit trials per gesture, plus separate
internal validation. Geometry points and raw candidates are not inputs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from author_training import validate_arrays, source_object_set

GESTURES = (1, 6, 7, 8)
TARGET = dict(fit_per_gesture=1000, total_fit=4000,
              internal_validation='Additional labels from disjoint source objects',
              nested_fit_prefixes=[100, 300, 1000],
              count_unit='One unique actually executed and eligible binary-labelled trial',
              collection_started=False, formal_training_started=False)


def assess(data_by_gesture, *, fit_per_gesture=1000):
    if type(fit_per_gesture) is not int or fit_per_gesture not in (100, 300, 1000):
        raise ValueError('Explicit 100/300/1000 fit-label target required')
    if set(data_by_gesture) != set(GESTURES):
        raise ValueError('All four declared Boya gesture datasets required')
    rows, trial_ids, fit_objects, check_objects = [], set(), set(), set()
    for gesture in GESTURES:
        data = data_by_gesture[gesture]
        fit, check = validate_arrays(data)
        if 'gesture_ids' in data and (data['gesture_ids'].shape != data['labels'].shape
                                     or not np.all(data['gesture_ids'] == gesture)):
            raise ValueError('Gesture dataset contains another gesture')
        ids = set(map(str, data['trial_id']))
        if trial_ids & ids:
            raise ValueError('A trial cannot count for more than one gesture')
        trial_ids.update(ids)
        fit_objects.update(source_object_set(data,fit))
        check_objects.update(source_object_set(data,check))
        rows.append(dict(gesture_id=gesture, total_labels=len(ids), fit_labels=len(fit),
                         internal_labels=len(check), fit_positive=int(data['labels'][fit].sum()),
                         fit_negative=int(len(fit)-data['labels'][fit].sum()),
                         fit_target=fit_per_gesture,
                         fit_deficit=max(0, fit_per_gesture-len(fit)),
                         full_fit_count_met=len(fit)>=fit_per_gesture))
    if fit_objects & check_objects:
        raise ValueError('Source object leakage across gestures between fit and internal validation')
    return dict(schema='anydex_boya_author_dataset_scale_v1',
                status='size_ready' if all(r['full_fit_count_met'] for r in rows) else 'insufficient_fit_labels',
                target_fit_per_gesture=fit_per_gesture, target_total_fit=4*fit_per_gesture,
                actual_total_labels=len(trial_ids), actual_fit_labels=sum(r['fit_labels'] for r in rows),
                actual_internal_labels=sum(r['internal_labels'] for r in rows),
                fit_deficit=sum(r['fit_deficit'] for r in rows),
                per_gesture=rows, fit_objects=len(fit_objects), internal_objects=len(check_objects),
                nulls_counted_as_labels=False, internal_counted_as_fit=False,
                physical_semantics_certified=False, sampling_distribution_certified=False,
                physics_steps=0, CGR_forwards=0, optimizer_updates=0)


def audit_directory(directory, *, fit_per_gesture=1000):
    directory = Path(directory).resolve()
    datasets, bindings = {}, {}
    for gesture in GESTURES:
        path = directory/f'gesture_{gesture}.npz'
        with np.load(path, allow_pickle=False) as data:
            datasets[gesture] = {name: data[name].copy() for name in data.files}
        bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = assess(datasets, fit_per_gesture=fit_per_gesture)
    result['input_sha256'] = bindings
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_directory', type=Path)
    parser.add_argument('--fit-per-gesture', type=int, choices=(100, 300, 1000), default=1000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_directory(args.dataset_directory, fit_per_gesture=args.fit_per_gesture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('status','actual_fit_labels',
                       'actual_internal_labels','target_total_fit','fit_deficit')}, indent=2))
    return 0 if result['status']=='size_ready' else 2


if __name__ == '__main__':
    raise SystemExit(main())
