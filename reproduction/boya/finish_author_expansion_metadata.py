"""Read-only closure for incomplete-trial coverage, preserving frozen raw sources.

Partial executions intentionally have no evaluated object_outcomes and remain
null. This separate postprocessor can finish only missing derived metadata after
that precise KeyError; it never restarts collection or overwrites old results.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import author_collection_batch as B
import author_collection_wave_resources as R


def coverage(root):
    """Report scene presence, observed hand load, and actual lifts separately."""
    p = B.read(root/'PROTOCOL.json')
    catalog = B.read(p['inputs']['expansion_catalog']['path'])
    rows = {k:dict(split=('fit' if k in catalog['split_object_ids']['fit'] else 'internal'),
        drawn_scenes=0,ready_scenes=0,executed_scenes=0,eligible_scenes=0,
        observed_robot_contact_trials=0,observed_hand_load_trials=0,successful_lift_trials=0,
        contact_without_successful_lift_trials=0,unavailable_lift_outcome_trials=0) for k in catalog['objects']}
    for spec in p['rows']:
        for k in spec['source_object_ids']:
            rows[k]['drawn_scenes'] += 1
            rows[k]['ready_scenes'] += int(spec['scene'] is not None)
        directory = root/'scenes'/f"{spec['index']:02d}"
        execution_path = directory/'execution/RESULTS.json'
        if not execution_path.exists():
            continue
        record = B.read(directory/'RECORD.json')
        execution = B.read(execution_path)
        loaded,contacted = set(),set()
        # Read complete existing traces; this is observation, never a new control gate.
        with Path(execution['trace']['path']).open() as stream:
            for line in stream:
                state = json.loads(line)
                for k,value in state['objects'].items():
                    if sum(value['finger_loads_n'])>0:
                        loaded.add(k)
                for contact in state['contacts']:
                    bodies = contact['bodies']
                    if any(b.startswith('/World/Robot/') for b in bodies):
                        for k in spec['source_object_ids']:
                            if '/World/'+k in bodies:
                                contacted.add(k)
        outcomes = {v['object_id']:v for v in execution['evaluation'].get('object_outcomes',[])}
        for k in spec['source_object_ids']:
            row = rows[k]
            success = bool(outcomes.get(k,{}).get('task_success',False))
            row['unavailable_lift_outcome_trials'] += int(k not in outcomes)
            if record['eligible'] and k not in outcomes:
                raise ValueError('An eligible sample must have complete lift outcomes')
            row['executed_scenes'] += 1
            row['eligible_scenes'] += int(record['eligible'])
            row['observed_robot_contact_trials'] += int(k in contacted)
            row['observed_hand_load_trials'] += int(k in loaded)
            row['successful_lift_trials'] += int(record['eligible'] and success)
            row['contact_without_successful_lift_trials'] += int(record['eligible'] and k in contacted and not success)
    return dict(objects=rows,object_types=len(rows),
        scene_cooccurrence_is_not_per_object_positive_negative_label=True,
        contact_without_lift_is_diagnostic_not_a_negative_training_label=True,
        untouched_distractors_are_not_negative_attempts=True,
        global_label='Original any-of-three lift success',additional_physics_steps=0)


def finish_missing_metadata(root):
    root=Path(root).resolve();directory=root/'collection';out=root/'formal'
    failure=B.read(root/'PIPELINE_FAILURE.json')
    if "KeyError: 'object_outcomes'" not in failure['error']:
        raise ValueError('Only the known partial-outcome metadata failure can be completed')
    p=B.read(directory/'PROTOCOL.json');B.verify(p['inputs']);B.verify(p['sources'])
    raw=B.read(directory/'RESULTS.json');audit=B.read(directory/'READ_ONLY_AUDIT.json')
    manifest=B.read(directory/'export/MANIFEST.json')
    if (raw['status']!='complete' or audit['status']!='read_only_audit_pass'
        or audit['original_result']!=B.binding(directory/'RESULTS.json')
        or manifest['source_audit']!=B.binding(directory/'READ_ONLY_AUDIT.json')):
        raise ValueError('Complete bound physics, audit, and staging export required')
    B.verify({'staging_dataset':manifest['dataset']})
    target=out/'gesture_1.npz'
    with np.load(manifest['dataset']['path'],allow_pickle=False) as z:
        staged={k:z[k].copy() for k in z.files}
    with np.load(target,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    if set(arrays)!=set(staged):raise ValueError('Dataset schema changed')
    for key in arrays:
        if key not in ('record_json','record_sha256'):
            np.testing.assert_array_equal(arrays[key],staged[key])
    checked={r['index']:r for r in audit['rounds']}
    for old_path,new_path,digest in zip(staged['record_json'],arrays['record_json'],arrays['record_sha256']):
        old_path=Path(str(old_path));new_path=Path(str(new_path))
        if new_path.resolve().parent!=(out/'records').resolve() or B.sha(new_path)!=str(digest):
            raise ValueError('Formal record location or hash changed')
        original=B.read(old_path);record=B.read(new_path);index=int(old_path.parent.name)
        if (not original['eligible'] or not checked[index].get('return_pass')
            or record['source_staged_record']!=B.binding(old_path)
            or record['independent_audit']!=B.binding(directory/'READ_ONLY_AUDIT.json')
            or not record['formal_training_eligible'] or record['exclude_training']):
            raise ValueError('Unverified promoted record')
        for key,value in original.items():
            if key not in ('schema','exclude_training','formal_training_eligible') and record[key]!=value:
                raise ValueError('Promoted scientific record changed')
    # Exclusive writes: existing partial dataset/records and failure remain intact.
    B.save(out/'OBJECT_COVERAGE.json',coverage(directory),exclusive=True)
    result=dict(manifest,status='audited_fresh_expansion_complete',dataset=B.binding(target),
        exclude_training=False,formal_training_eligible=True,
        source_staging_export=B.binding(directory/'export/MANIFEST.json'),
        object_coverage=B.binding(out/'OBJECT_COVERAGE.json'),scene_budget=100,new_independent_scene_draws=100,
        training_updates=0,old_data_merged=False,postprocessing_only_completion=True,
        preserved_pipeline_failure=B.binding(root/'PIPELINE_FAILURE.json'),implementation=B.binding(__file__),
        note='Incomplete executions remain null; missing object outcomes are unavailable diagnostics, never negative labels.')
    B.save(out/'MANIFEST.json',result,exclusive=True)
    R.assert_quiescent(directory)
    B.save(root/'EXPANSION_METADATA_COMPLETION.json',dict(status='complete',
        formal_manifest=B.binding(out/'MANIFEST.json'),source_pipeline_failure=B.binding(root/'PIPELINE_FAILURE.json'),
        labels=result['records'],positive=result['positive'],negative=result['negative'],
        additional_physics_steps=0,training_updates=0,implementation=B.binding(__file__)),exclusive=True)
    return result


def watch(root):
    root=Path(root).resolve()
    identity=B.read(root/'PIPELINE_STARTED.json')['process']
    implementation=B.binding(__file__)
    began=time.monotonic()
    B.save(root/'METADATA_WATCH_STARTED.json',dict(process=R.process_identity(__import__('os').getpid()),
        watched_process=identity,implementation=implementation,additional_physics_steps=0),exclusive=True)
    while R.process_identity(identity['pid'])==identity:
        if time.monotonic()-began>23400:raise TimeoutError('Bounded metadata watch elapsed')
        time.sleep(10)
    B.verify({'implementation':implementation})
    if (root/'EXPANSION_RESULTS.json').exists():
        result=dict(status='original_pipeline_complete_no_recovery_needed',additional_physics_steps=0)
    elif (root/'PIPELINE_FAILURE.json').exists():
        failure=B.read(root/'PIPELINE_FAILURE.json')
        if "KeyError: 'object_outcomes'" not in failure['error']:
            result=dict(status='unrelated_failure_preserved_no_retry',failure=B.binding(root/'PIPELINE_FAILURE.json'),additional_physics_steps=0)
        else:
            formal=finish_missing_metadata(root)
            result=dict(status='derived_metadata_completed',labels=formal['records'],additional_physics_steps=0)
    else:
        raise ValueError('Collector exited without a completion or failure receipt')
    B.save(root/'METADATA_WATCH_FINISHED.json',result,exclusive=True)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    try:watch(args.root)
    except BaseException:
        B.save(args.root/'METADATA_WATCH_FAILURE.json',dict(error=traceback.format_exc(),additional_physics_steps=0),exclusive=True)
        raise
