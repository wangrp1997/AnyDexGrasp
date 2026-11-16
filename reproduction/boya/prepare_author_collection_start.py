"""Prepare a NEW bounded PhysX collection queue; never start collection.

Reuse paths/domain/dependency environment from the successful previous batch,
generate fresh scene specifications, then freeze the existing serial collector.
Run the printed command yourself to start physics. Old batches are untouched.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import author_collection_batch as B
import author_collection_scenes as SCENES

BASE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')
REFERENCE=BASE/'production_tail_20261115_v1/source_batch/BATCH_PROTOCOL.json'


def prepare(output, *, fit_scenes=2, internal_scenes=2, seed=2026111601):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; never overwrite an old queue')
    if (type(fit_scenes)is not int or type(internal_scenes)is not int
            or not 1<=fit_scenes<=8 or not 1<=internal_scenes<=8):
        raise ValueError('This starter supports 1..8 fixed scenes per split, not a formal full-scale queue')
    reference=B.read(REFERENCE)
    B.verify(reference['inputs']);B.verify(reference['sources'])
    source_queue=B.read(reference['inputs']['queue']['path'])
    qinputs=source_queue['inputs']
    paths={key:qinputs[key]['path']for key in ('source_manifest','native_summary','backend','split_manifest')}
    inputs={key:reference['inputs'][key]['path']for key in ('calibration','cloud_manifest','backend','checkpoint','python')}
    entry=B.HERE/'author_verified_collection_execute.py'
    deps=[B.HERE/n for n in ('author_verified_execution.py','author_verified_contact_journal.py','author_contact_compressor.py')]
    output.mkdir(parents=True,exist_ok=False)
    B.save(output/'STARTER_INTENT.json',dict(reference=B.binding(REFERENCE),sources={str(p):B.binding(p)for p in [Path(__file__),entry,*deps]},
        scene_budget=dict(fit=fit_scenes,internal=internal_scenes),maximum_gesture_attempts=4*(fit_scenes+internal_scenes),
        retries=0,seed=seed,backend='Existing CPU PhysX',physics_started=False,training_started=False),exclusive=True)
    queue=SCENES.generate_queue(paths['source_manifest'],paths['native_summary'],paths['backend'],paths['split_manifest'],
        output/'scene_queue',seed=seed,fit_scene_count=fit_scenes,internal_scene_count=internal_scenes)
    old_env=B.DEFAULT_ENV
    inherited={k:os.environ.get(k)for k in reference['inherited_env']}
    try:
        B.DEFAULT_ENV=dict(copy.deepcopy(reference['env']),ANYDEX_CONTACT_SPOOL_ROOT=str(output/'contact_spool'))
        for key,value in reference['inherited_env'].items():os.environ[key]=value
        protocol=B.freeze(output/'scene_queue/QUEUE.json',output/'batch',calibration=inputs['calibration'],
            cloud_manifest=inputs['cloud_manifest'],backend=inputs['backend'],checkpoint=inputs['checkpoint'],
            python=inputs['python'],native_deps=reference['native_deps'],domain=reference['domain'],
            execution_entry=entry,prepare_workers=3,fit_per_gesture=1000,internal_per_gesture=200,
            stage_timeout_s=reference['stage_timeout_s'])
    finally:
        B.DEFAULT_ENV=old_env
        for key,value in inherited.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    receipt=dict(status='prepared_not_started',batch=str(output/'batch'),
        queue=B.binding(output/'scene_queue/QUEUE.json'),protocol=B.binding(output/'batch/BATCH_PROTOCOL.json'),
        fixed_scenes=len(queue['scenes']),placement_nulls=queue['placement_null_count'],
        maximum_gesture_attempts=protocol['total_gesture_attempt_budget'],
        formal_target_labels=dict(fit_per_gesture=1000,internal_per_gesture=200),
        target_counts_are_not_attempt_budget=True,Kit_starts=0,physics_steps=0,new_labels=0,training_updates=0,
        execution_storage='Verified raw journal plus isolated lossless compression/full readback; raw retained on disk',
        startup_command=[inputs['python'],'-B',str(B.HERE/'author_collection_batch.py'),'run',
                         '--batch-dir',str(output/'batch'),'--max-scenes','1'])
    B.save(output/'STARTER_READY.json',receipt,exclusive=True)
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--fit-scenes',type=int,default=2);p.add_argument('--internal-scenes',type=int,default=2)
    p.add_argument('--seed',type=int,default=2026111601)
    a=p.parse_args();prepare(a.output_dir,fit_scenes=a.fit_scenes,internal_scenes=a.internal_scenes,seed=a.seed)
