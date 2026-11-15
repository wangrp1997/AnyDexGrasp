"""Freeze the authorized16fit+16internal pilot with one predeclared fresh seed."""
import argparse
import copy
from pathlib import Path
import time

import author_production_collection as D
import author_collection_scenes as S

B=D.B
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,required=True)
args=parser.parse_args();root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=False)
reference=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development')
queue_protocol=B.read(reference/'scene_queue/QUEUE_PROTOCOL.json')
base=B.read(reference/'batch_v1/BATCH_PROTOCOL.json')
B.verify(queue_protocol['inputs']);B.verify(base['inputs']);B.verify(base['sources'])
B.save(root/'REQUESTED_BUDGET.json',dict(seed=2026111501,fit_scenes=16,internal_scenes=16,
    max_gesture_slots=128,zero_retries=True,order='fit0,internal0,fit1,internal1,...,fit15,internal15',
    no_replacement_after_null=True,automatic_training=False,created_unix_s=time.time(),
    freezer=B.binding(__file__)),exclusive=True)
if D.P.OLD.REGISTRY.exists():
    previous=B.read(D.P.OLD.REGISTRY)
    if previous['state']!='released':raise ValueError('Prior run not released')
    D.R.assert_quiescent(previous['run_dir'])
with D.R.lease(D.P.OLD.LEGACY_LOCK):pass
inputs={k:v['path']for k,v in queue_protocol['inputs'].items()}
queue=S.generate_queue(inputs['source_manifest'],inputs['native_summary'],inputs['backend'],inputs['split_manifest'],
    root/'scene_queue',seed=2026111501,fit_scene_count=16,internal_scene_count=16)
ordered=copy.deepcopy(queue);raw=queue['scenes']
ordered['scenes']=[raw[j]for i in range(16)for j in(i,16+i)]
ordered['inputs']=B.normalized_bindings(ordered['inputs'])
ordered['inputs']['original_draw_queue']=B.binding(root/'scene_queue/QUEUE.json')
ordered.update(order_policy='Predeclared alternating fit/internal, preserving each split original sampled index order',
    original_draw_queue=B.binding(root/'scene_queue/QUEUE.json'))
B.save(root/'ORDERED_QUEUE.json',ordered,exclusive=True)
options={k:base['inputs'][k]['path']for k in('calibration','cloud_manifest','backend','checkpoint','python')}
B.freeze(root/'ORDERED_QUEUE.json',root/'source_batch',**options,domain=base['domain'],
    native_deps=base['native_deps'],fit_per_gesture=1000,internal_per_gesture=200,stage_timeout_s=1200.)
protocol=D.freeze(root/'source_batch',root/'run')
print(dict(status='frozen',scenes=32,slots=128,placement_null=sum(r['status']=='placement_null'for r in ordered['scenes']),
    run_dir=str(root/'run'),protocol=B.binding(root/'run'/D.PROTOCOL),seed=2026111501))
