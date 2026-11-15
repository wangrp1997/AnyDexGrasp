"""Corrected bootstrap for a future separately authorized32-scene pilot; no run."""
import argparse
import copy
from pathlib import Path

import author_collection_scenes as S
import author_production_control_v2 as D
from production_dependency_environment import restore_and_probe

B=D.B
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,required=True)
parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
if args.seed<0 or args.seed in(2026111401,2026111402,2026111501):
    raise ValueError('Declare a fresh seed; these historical/current draws must not be rerun')
root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=False)
reference=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development')
q=B.read(reference/'scene_queue/QUEUE_PROTOCOL.json');base=B.read(reference/'batch_v1/BATCH_PROTOCOL.json')
B.verify(q['inputs']);B.verify(base['inputs']);B.verify(base['sources'])
B.save(root/'REQUESTED_BUDGET.json',dict(seed=args.seed,fit_scenes=16,internal_scenes=16,max_gesture_slots=128,
    retries=0,automatic_training=False,source=B.binding(__file__),physics_dispatches=0),exclusive=True)
inputs={k:v['path']for k,v in q['inputs'].items()}
queue=S.generate_queue(inputs['source_manifest'],inputs['native_summary'],inputs['backend'],inputs['split_manifest'],
    root/'scene_queue',seed=args.seed,fit_scene_count=16,internal_scene_count=16)
ordered=copy.deepcopy(queue);ordered['scenes']=[queue['scenes'][j]for i in range(16)for j in(i,16+i)]
ordered['inputs']=B.normalized_bindings(ordered['inputs'])
ordered['inputs']['original_draw_queue']=B.binding(root/'scene_queue/QUEUE.json')
B.save(root/'ORDERED_QUEUE.json',ordered,exclusive=True)
options={k:base['inputs'][k]['path']for k in('calibration','cloud_manifest','backend','checkpoint','python')}
B.freeze(root/'ORDERED_QUEUE.json',root/'source_batch',**options,domain=base['domain'],
    native_deps=base['native_deps'],fit_per_gesture=1000,internal_per_gesture=200,stage_timeout_s=1200.)
restore_and_probe(root/'source_batch',reference/'batch_v1/BATCH_PROTOCOL.json',root/'DEPENDENCY_PREFLIGHT.json')
p=D.freeze(root/'source_batch',root/'run')
print(dict(status='frozen_not_started',protocol=B.binding(root/'run'/D.PROTOCOL),seed=args.seed))
