"""Freeze original indices17..31 only; metadata and dependency probe, no Kit."""
import argparse
import copy
from pathlib import Path
import os

import author_production_tail as D
import author_production_continuation as PREVIOUS
import author_storage_path_validation as STORAGE
from production_dependency_environment import restore_and_probe

B=D.B
BASE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')


def freeze(output):
    initial=BASE/'production_pilot_20261115_v1/run'
    previous=BASE/'production_pilot_20261115_continuation_v1/run'
    original=D.ORIGINAL.verify(initial);prior=PREVIOUS.verify(previous)
    result=B.read(previous/'PRODUCTION_RESULTS.json')
    if result['status']!='stopped'or result['terminal_slots']!=64:raise ValueError('Expected original completed16-scene prefix')
    old_rows=PREVIOUS.ledger(previous,prior)
    if {r['queue_index']for r in old_rows}!=set(range(16)):raise ValueError('Original prefix differs')
    remaining=prior['queue']['scenes'][16:]
    if len(remaining)!=15 or remaining!=original['queue']['scenes'][17:32]:raise ValueError('Only exact unstarted draws may continue')
    for row in remaining:
        for root in (initial,previous):
            for marker in ('stages/*/STAGE.json','stages/*/STARTED.json','g*/RESIDENT_EXECUTION_ENTRY.json'):
                if any(root.glob(f'windows/*/scenes/*_{row["scene_id"]}/'+marker)):
                    raise ValueError('A remaining scene already has a started stage: '+row['scene_id'])
    readiness=BASE/'verified_saved_audits_20261115_v1/RESULTS.json';ready=B.read(readiness)
    if ready['status']!='saved_full_integrity_pass'or len(ready['audits'])!=4:
        raise ValueError('Four passing full tmpfs executions and original audits required')
    for a in ready['audits']:
        B.verify({'audit':a['audit']})
        if B.read(a['audit']['path'])['status']!='full_integrity_pass':raise ValueError('Full verification failed')
    if STORAGE.filesystem(D.SPOOL)['fstype']!='tmpfs'or Path(D.SPOOL).exists():raise ValueError('Fresh tmpfs spool required')
    stat=os.statvfs(Path(D.SPOOL).parent)
    if stat.f_bavail*stat.f_frsize<128*(1<<30):raise ValueError('Insufficient free tmpfs for retained finite batch logs')
    registry=B.read(D.P.OLD.REGISTRY)
    if registry['state']!='released':raise ValueError('Previous physical run not released')
    if registry.get('review'):B.verify({'review':registry['review']})
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    plan=dict(original_protocol=B.binding(initial/D.PROTOCOL),previous_protocol=B.binding(previous/D.PROTOCOL),
        previous_result=B.binding(previous/'PRODUCTION_RESULTS.json'),storage_verification=B.binding(readiness),
        original_scene_indices=list(range(17,32)),remaining_scenes=15,remaining_slots=60,
        seed=original['queue']['seed'],sampler_redraws=0,failed_scene_retries=0,
        combined_original_scene_budget=32,combined_original_slot_budget=128,source_adapter=D.AST_ADAPTATIONS)
    B.save(output/'TAIL_PLAN.json',plan,exclusive=True)
    queue=copy.deepcopy(original['queue']);queue['scenes']=copy.deepcopy(remaining)
    queue.update(scene_counts=dict(fit=7,internal=8),maximum_gesture_scene_attempts=60,
        original_scene_indices=list(range(17,32)),tail_plan=B.binding(output/'TAIL_PLAN.json'))
    queue['inputs']=B.normalized_bindings(queue['inputs'])
    queue['inputs'].update(tail_plan=B.binding(output/'TAIL_PLAN.json'),original_protocol=plan['original_protocol'],
        previous_protocol=plan['previous_protocol'],previous_result=plan['previous_result'],storage_verification=plan['storage_verification'])
    B.save(output/'QUEUE.json',queue,exclusive=True)
    reference=BASE/'development/batch_v1/BATCH_PROTOCOL.json';ref=B.read(reference)
    options={k:ref['inputs'][k]['path']for k in ('calibration','cloud_manifest','backend','checkpoint','python')}
    B.freeze(output/'QUEUE.json',output/'source_batch',**options,domain=ref['domain'],native_deps=ref['native_deps'],
        fit_per_gesture=1000,internal_per_gesture=200,stage_timeout_s=1200.)
    restore_and_probe(output/'source_batch',reference,output/'DEPENDENCY_PREFLIGHT.json')
    return D.freeze(output/'source_batch',output/'run')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,required=True)
    a=parser.parse_args();p=freeze(a.output_root)
    print(dict(status='frozen_not_started',scenes=15,slots=60,seed=p['queue']['seed'],protocol=B.binding(a.output_root/'run'/D.PROTOCOL)))
