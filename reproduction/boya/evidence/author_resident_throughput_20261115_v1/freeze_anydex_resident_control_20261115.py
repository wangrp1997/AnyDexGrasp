from pathlib import Path
import json, shutil, time
import author_collection_batch as B
import author_resident_collection as C
import author_resident_benchmark as M
base=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')
out=base/'resident_collection_control_20261115_v1'
if out.exists(): raise FileExistsError(out)
fit=base/'development/batch_v1'
internal=base/'development_waves/waves_v1/children/0000004_s40_lift_internal_00001_s2026111402/batch'
a=C.freeze_from_batch(fit,[1,4],out/'fit',resident_workers=2)
b=C.freeze_from_batch(internal,[0],out/'internal',resident_workers=1)
M.freeze_benchmark(fit,out/'fit'/C.PROTOCOL,out/'fit_benchmark',scene_indices=[1,4])
M.freeze_benchmark(internal,out/'internal'/C.PROTOCOL,out/'internal_benchmark',scene_indices=[0])
old=B.read('reproduction/boya/evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')['verified_files']
B.verify(old)
B.save(out/'UNCHANGED_DATA_MODELS_BEFORE.json',dict(status='pass',verified_files=old))
for file in ('author_resident_execution.py','author_resident_collection.py','author_resident_benchmark.py',
             'test_author_resident_execution.py','test_author_resident_collection.py','test_author_resident_benchmark.py'):
 p=B.HERE/file; target=out/'source_snapshot'/file; target.parent.mkdir(exist_ok=True); shutil.copyfile(p,target)
B.save(out/'MASTER_PROTOCOL.json',dict(schema='anydex_resident_small_control_master_v1',
 created_unix_s=time.time(),fit_protocol=B.binding(out/'fit'/C.PROTOCOL),
 internal_protocol=B.binding(out/'internal'/C.PROTOCOL),total_scenes=3,total_gesture_slots=12,
 max_actual_executions=12,expected_ready_from_historical_reference=9,selection='known_scene_controls',
 exclude_training=True,formal_training_labels_added=0,training_updates=0,
 bulk_collection_authorized=False,automatic_large_collection=False,retry_budget=0,
 source_files={p.name:B.binding(p) for p in (out/'source_snapshot').iterdir()},
 instruction='Measure complete bounded lifecycle and original independent audits. User decides whether to begin 4000fit+800internal afterward.'))
print(json.dumps({'root':str(out),'fit_protocol':B.binding(out/'fit'/C.PROTOCOL),
 'internal_protocol':B.binding(out/'internal'/C.PROTOCOL),'total_slots':12,'max_executions':12},indent=2))
